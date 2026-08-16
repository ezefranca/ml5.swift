@preconcurrency import CoreML
@preconcurrency import CoreVideo
import Foundation
@preconcurrency import ImageIO
@preconcurrency import Vision

/// EXIF-compatible orientation supplied when Vision interprets image pixels.
public enum VisionImageOrientation: String, CaseIterable, Sendable, Hashable, Codable {
    /// The first stored row is the visual top and the first column is the visual left.
    case up
    /// The image is mirrored horizontally while otherwise upright.
    case upMirrored
    /// The image is rotated 180 degrees.
    case down
    /// The image is rotated 180 degrees and mirrored horizontally.
    case downMirrored
    /// The image is rotated 90 degrees counterclockwise.
    case left
    /// The image is rotated 90 degrees counterclockwise and mirrored horizontally.
    case leftMirrored
    /// The image is rotated 90 degrees clockwise.
    case right
    /// The image is rotated 90 degrees clockwise and mirrored horizontally.
    case rightMirrored

    fileprivate var coreGraphicsOrientation: CGImagePropertyOrientation {
        switch self {
        case .up:
            .up
        case .upMirrored:
            .upMirrored
        case .down:
            .down
        case .downMirrored:
            .downMirrored
        case .left:
            .left
        case .leftMirrored:
            .leftMirrored
        case .right:
            .right
        case .rightMirrored:
            .rightMirrored
        }
    }
}

/// Policy Vision uses to reconcile an image's aspect ratio with a model input.
public enum VisionImageCropAndScale: String, CaseIterable, Sendable, Hashable, Codable {
    /// Scale the image to fill the model input and crop equally around its center.
    case centerCrop
    /// Scale the whole image inside the model input, preserving all source pixels.
    case scaleFit
    /// Scale the image independently in each dimension to fill the model input.
    case scaleFill

    fileprivate var visionOption: VNImageCropAndScaleOption {
        switch self {
        case .centerCrop:
            .centerCrop
        case .scaleFit:
            .scaleFit
        case .scaleFill:
            .scaleFill
        }
    }
}

/// A framework-independent label and score returned by a Vision classifier.
public struct VisionClassification: Sendable, Hashable, Codable {
    /// Model-provided class identifier.
    public let identifier: String
    /// Finite score reported by Vision without additional normalization.
    public let confidence: Double

    /// Creates a validated classification result.
    ///
    /// Vision forwards model scores, so this type requires a finite value but does not force the
    /// score into the range from zero through one.
    ///
    /// - Throws: ``ML5Error/unsupportedVisionResult(reason:)`` for an empty identifier or
    ///   nonfinite score.
    public init(identifier: String, confidence: Double) throws {
        guard !identifier.isEmpty else {
            throw ML5Error.unsupportedVisionResult(
                reason: "A classification observation has an empty identifier."
            )
        }
        guard confidence.isFinite else {
            throw ML5Error.unsupportedVisionResult(
                reason: "A classification observation has a nonfinite score."
            )
        }
        self.identifier = identifier
        self.confidence = confidence
    }

    /// Decodes and revalidates a classification result.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            identifier: container.decode(String.self, forKey: .identifier),
            confidence: container.decode(Double.self, forKey: .confidence)
        )
    }
}

/// A named, framework-independent Core ML feature returned through Vision.
public struct VisionCoreMLFeature: Sendable, Hashable, Codable {
    /// Model output name associated with the feature.
    public let name: OutputName
    /// Converted Core ML scalar, collection, tensor, sequence, or image value.
    public let value: FeatureValue

    /// Creates and validates a named Vision feature.
    ///
    /// - Throws: An ``ML5Error`` when the feature value violates its public invariants.
    public init(name: OutputName, value: FeatureValue) throws {
        try value.validate(field: name.rawValue)
        self.name = name
        self.value = value
    }

    /// Decodes and revalidates a named Vision feature.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            name: container.decode(OutputName.self, forKey: .name),
            value: container.decode(FeatureValue.self, forKey: .value)
        )
    }
}

enum VisionCoreMLRawResult: @unchecked Sendable {
    case classification(identifier: String, confidence: Double)
    case feature(name: String, value: MLFeatureValue)
    case unsupported
}

struct VisionCoreMLModelOperation: @unchecked Sendable {
    var perform:
        (CVPixelBuffer, CGImagePropertyOrientation, VNImageCropAndScaleOption) throws ->
            [VisionCoreMLRawResult]
}

struct VisionCoreMLModelLoader: @unchecked Sendable {
    var load: (URL, MLModelConfiguration) throws -> VisionCoreMLModelOperation

    static let system = Self { modelURL, configuration in
        let model = try MLModel(contentsOf: modelURL, configuration: configuration)
        let visionModel = try VNCoreMLModel(for: model)
        return VisionCoreMLModelOperation { pixelBuffer, orientation, cropAndScale in
            let request = VNCoreMLRequest(model: visionModel)
            request.imageCropAndScaleOption = cropAndScale
            let handler = VNImageRequestHandler(
                cvPixelBuffer: pixelBuffer,
                orientation: orientation,
                options: [:]
            )
            try handler.perform([request])
            return results(request.results)
        }
    }

    static func results(_ observations: [VNObservation]?) -> [VisionCoreMLRawResult] {
        observations?.map { observation in
            if let classification = observation as? VNClassificationObservation {
                return .classification(
                    identifier: classification.identifier,
                    confidence: Double(classification.confidence)
                )
            }
            if let feature = observation as? VNCoreMLFeatureValueObservation {
                return .feature(name: feature.featureName, value: feature.featureValue)
            }
            return .unsupported
        } ?? []
    }
}

/// An actor-isolated Core ML image model evaluated through Vision.
///
/// Use ``classify(_:orientation:cropAndScale:maximumResults:)`` for classifier models and
/// ``extractFeatures(_:orientation:cropAndScale:)`` for models whose outputs become
/// `VNCoreMLFeatureValueObservation` values. Calling the wrong method produces a typed error.
public actor VisionCoreMLImageModel {
    private let operation: VisionCoreMLModelOperation

    /// Loads a compiled `.mlmodelc` model for Vision image requests.
    ///
    /// - Throws: ``ML5Error/modelLoadingFailed(path:message:)`` when Core ML or Vision rejects
    ///   the model.
    public init(
        contentsOf modelURL: URL,
        configuration: CoreMLModelConfiguration = .init()
    ) throws {
        try self.init(contentsOf: modelURL, configuration: configuration, loader: .system)
    }

    init(
        contentsOf modelURL: URL,
        configuration: CoreMLModelConfiguration = .init(),
        loader: VisionCoreMLModelLoader
    ) throws {
        do {
            operation = try loader.load(modelURL, configuration.makeCoreMLConfiguration())
        } catch {
            throw ML5Error.modelLoadingFailed(
                path: modelURL.path,
                message: error.localizedDescription
            )
        }
    }

    /// Resolves an integrity-checked model through a cache and loads it for Vision.
    ///
    /// - Returns: An actor that owns the resolved Core ML image model.
    public static func load(
        from source: ML5ModelSource,
        using cache: ML5ModelCache,
        configuration: CoreMLModelConfiguration = .init()
    ) async throws -> VisionCoreMLImageModel {
        let cached = try await cache.model(for: source)
        return try VisionCoreMLImageModel(
            contentsOf: cached.modelURL,
            configuration: configuration
        )
    }

    /// Classifies an immutable image and returns deterministic score ordering.
    ///
    /// - Parameters:
    ///   - image: Owned grayscale, RGBA, or BGRA pixels.
    ///   - orientation: Visual orientation of the stored pixels.
    ///   - cropAndScale: Aspect-ratio policy applied before model evaluation.
    ///   - maximumResults: Optional positive limit applied after descending score sorting.
    /// - Returns: Classification observations sorted by descending score and identifier.
    /// - Throws: ``ML5Error/invalidVisionConfiguration(reason:)``,
    ///   ``ML5Error/unsupportedVisionResult(reason:)``,
    ///   ``ML5Error/visionRequestFailed(message:)``, or `CancellationError`.
    public func classify(
        _ image: ML5Image,
        orientation: VisionImageOrientation = .up,
        cropAndScale: VisionImageCropAndScale = .centerCrop,
        maximumResults: Int? = nil
    ) async throws -> [VisionClassification] {
        if let maximumResults, maximumResults <= 0 {
            throw ML5Error.invalidVisionConfiguration(
                reason: "maximumResults must be positive when supplied."
            )
        }
        let observations = try perform(
            image,
            orientation: orientation,
            cropAndScale: cropAndScale
        )
        let classifications = try Self.classifications(from: observations)
        if let maximumResults {
            return Array(classifications.prefix(maximumResults))
        }
        return classifications
    }

    /// Extracts named Core ML feature values from an immutable image.
    ///
    /// - Returns: Feature values sorted by model output name.
    /// - Throws: ``ML5Error/unsupportedVisionResult(reason:)``,
    ///   ``ML5Error/visionRequestFailed(message:)``, or `CancellationError`.
    public func extractFeatures(
        _ image: ML5Image,
        orientation: VisionImageOrientation = .up,
        cropAndScale: VisionImageCropAndScale = .centerCrop
    ) async throws -> [VisionCoreMLFeature] {
        let observations = try perform(
            image,
            orientation: orientation,
            cropAndScale: cropAndScale
        )
        return try Self.features(from: observations)
    }

    private func perform(
        _ image: ML5Image,
        orientation: VisionImageOrientation,
        cropAndScale: VisionImageCropAndScale
    ) throws -> [VisionCoreMLRawResult] {
        do {
            try Task.checkCancellation()
            let pixelBuffer = try image.makePixelBuffer()
            try Task.checkCancellation()
            let observations = try operation.perform(
                pixelBuffer,
                orientation.coreGraphicsOrientation,
                cropAndScale.visionOption
            )
            try Task.checkCancellation()
            return observations
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as ML5Error {
            throw error
        } catch {
            throw ML5Error.visionRequestFailed(message: error.localizedDescription)
        }
    }

    static func classifications(
        from observations: [VisionCoreMLRawResult]
    ) throws -> [VisionClassification] {
        guard !observations.isEmpty else {
            throw ML5Error.unsupportedVisionResult(
                reason: "The image classifier returned no observations."
            )
        }
        let values = try observations.map { observation in
            guard case let .classification(identifier, confidence) = observation else {
                throw ML5Error.unsupportedVisionResult(
                    reason: "The model did not return classification observations."
                )
            }
            return try VisionClassification(
                identifier: identifier,
                confidence: confidence
            )
        }
        return values.sorted {
            if $0.confidence == $1.confidence {
                return $0.identifier < $1.identifier
            }
            return $0.confidence > $1.confidence
        }
    }

    static func features(
        from observations: [VisionCoreMLRawResult]
    ) throws -> [VisionCoreMLFeature] {
        guard !observations.isEmpty else {
            throw ML5Error.unsupportedVisionResult(
                reason: "The feature extractor returned no observations."
            )
        }
        return try observations.map { observation in
            guard case let .feature(name, coreMLValue) = observation else {
                throw ML5Error.unsupportedVisionResult(
                    reason: "The model did not return Core ML feature-value observations."
                )
            }
            let outputName = try OutputName(name)
            let value = try FeatureValue(
                coreMLValue: coreMLValue,
                outputName: name
            )
            return try VisionCoreMLFeature(name: outputName, value: value)
        }.sorted { $0.name.rawValue < $1.name.rawValue }
    }
}

/// Vision feature-print algorithm revision used for image embeddings.
public enum VisionFeaturePrintRevision: Int, CaseIterable, Sendable, Hashable, Codable {
    /// Original Vision image feature-print algorithm.
    case revision1 = 1
    /// Current Vision image feature-print algorithm on the supported platform baseline.
    case revision2 = 2

    fileprivate var visionRevision: Int {
        switch self {
        case .revision1:
            VNGenerateImageFeaturePrintRequestRevision1
        case .revision2:
            VNGenerateImageFeaturePrintRequestRevision2
        }
    }
}

/// Numeric storage used by a Vision image feature print.
public enum VisionFeaturePrintElementType: String, CaseIterable, Sendable, Hashable, Codable {
    /// IEEE 754 single-precision elements.
    case float32
    /// IEEE 754 double-precision elements.
    case float64
}

/// Configuration for native Vision image feature extraction.
public struct VisionFeaturePrintConfiguration: Sendable, Hashable, Codable {
    /// Feature-print algorithm revision.
    public var revision: VisionFeaturePrintRevision
    /// Aspect-ratio policy applied before extraction.
    public var cropAndScale: VisionImageCropAndScale

    /// Creates feature-print request settings.
    public init(
        revision: VisionFeaturePrintRevision = .revision2,
        cropAndScale: VisionImageCropAndScale = .centerCrop
    ) {
        self.revision = revision
        self.cropAndScale = cropAndScale
    }
}

/// A serializable Vision image embedding suitable for similarity search.
public struct VisionFeaturePrint: Sendable, Hashable, Codable {
    /// Feature-print algorithm revision.
    public let revision: VisionFeaturePrintRevision
    /// Original framework element storage type.
    public let elementType: VisionFeaturePrintElementType
    /// Finite feature elements converted to `Double` for portable processing.
    public let values: [Double]

    /// Creates and validates a portable Vision feature print.
    ///
    /// - Throws: ``ML5Error/unsupportedVisionResult(reason:)`` when values are empty or
    ///   nonfinite.
    public init(
        revision: VisionFeaturePrintRevision,
        elementType: VisionFeaturePrintElementType,
        values: [Double]
    ) throws {
        guard !values.isEmpty else {
            throw ML5Error.unsupportedVisionResult(
                reason: "A Vision feature print must contain at least one element."
            )
        }
        guard values.allSatisfy(\.isFinite) else {
            throw ML5Error.unsupportedVisionResult(
                reason: "A Vision feature print contains a nonfinite element."
            )
        }
        self.revision = revision
        self.elementType = elementType
        self.values = values
    }

    /// Decodes and revalidates a portable Vision feature print.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            revision: container.decode(VisionFeaturePrintRevision.self, forKey: .revision),
            elementType: container.decode(
                VisionFeaturePrintElementType.self,
                forKey: .elementType
            ),
            values: container.decode([Double].self, forKey: .values)
        )
    }

    /// Computes Euclidean distance to a compatible feature print.
    ///
    /// Smaller distances indicate more similar images. Only feature prints with identical
    /// revisions, element types, and element counts are comparable.
    ///
    /// - Returns: Euclidean distance between the two feature vectors.
    /// - Throws: ``ML5Error/invalidVisionConfiguration(reason:)`` for incompatible prints or
    ///   arithmetic overflow.
    public func distance(to other: VisionFeaturePrint) throws -> Double {
        guard
            revision == other.revision,
            elementType == other.elementType,
            values.count == other.values.count
        else {
            throw ML5Error.invalidVisionConfiguration(
                reason: "Feature prints must have matching revisions, types, and element counts."
            )
        }
        var distance = 0.0
        for (left, right) in zip(values, other.values) {
            let difference = left - right
            guard difference.isFinite else {
                throw ML5Error.invalidVisionConfiguration(
                    reason: "Feature-print distance overflowed Double."
                )
            }
            distance = hypot(distance, difference)
        }
        return distance
    }
}

struct VisionFeaturePrintOperation: @unchecked Sendable {
    var perform:
        (
            CVPixelBuffer, CGImagePropertyOrientation, VNImageCropAndScaleOption, Int
        ) throws -> VisionFeaturePrintRawResult

    static let system = Self { pixelBuffer, orientation, cropAndScale, revision in
        let request = VNGenerateImageFeaturePrintRequest()
        request.revision = revision
        request.imageCropAndScaleOption = cropAndScale
        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: orientation,
            options: [:]
        )
        try handler.perform([request])
        return try requireResult(request.results)
    }

    static func requireResult(
        _ results: [VNFeaturePrintObservation]?
    ) throws -> VisionFeaturePrintRawResult {
        guard let observation = results?.first else {
            throw ML5Error.unsupportedVisionResult(
                reason: "Vision returned no image feature print."
            )
        }
        return VisionFeaturePrintRawResult(
            elementType: observation.elementType,
            elementCount: observation.elementCount,
            data: observation.data
        )
    }
}

struct VisionFeaturePrintRawResult: @unchecked Sendable {
    let elementType: VNElementType
    let elementCount: Int
    let data: Data
}

/// An actor-isolated native Vision image feature-print extractor.
public actor VisionImageFeatureExtractor {
    /// Immutable request configuration used for each image.
    public nonisolated let configuration: VisionFeaturePrintConfiguration

    private let operation: VisionFeaturePrintOperation

    /// Creates an extractor backed by `VNGenerateImageFeaturePrintRequest`.
    public init(configuration: VisionFeaturePrintConfiguration = .init()) {
        self.init(configuration: configuration, operation: .system)
    }

    init(
        configuration: VisionFeaturePrintConfiguration,
        operation: VisionFeaturePrintOperation
    ) {
        self.configuration = configuration
        self.operation = operation
    }

    /// Extracts a serializable image feature print.
    ///
    /// - Returns: A portable copy of Vision's feature-print elements and metadata.
    /// - Throws: ``ML5Error/unsupportedVisionResult(reason:)``,
    ///   ``ML5Error/visionRequestFailed(message:)``, or `CancellationError`.
    public func extract(
        _ image: ML5Image,
        orientation: VisionImageOrientation = .up
    ) async throws -> VisionFeaturePrint {
        do {
            try Task.checkCancellation()
            let pixelBuffer = try image.makePixelBuffer()
            try Task.checkCancellation()
            let result = try operation.perform(
                pixelBuffer,
                orientation.coreGraphicsOrientation,
                configuration.cropAndScale.visionOption,
                configuration.revision.visionRevision
            )
            try Task.checkCancellation()
            return try Self.featurePrint(
                from: result,
                revision: configuration.revision
            )
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as ML5Error {
            throw error
        } catch {
            throw ML5Error.visionRequestFailed(message: error.localizedDescription)
        }
    }

    static func featurePrint(
        from result: VisionFeaturePrintRawResult,
        revision: VisionFeaturePrintRevision
    ) throws -> VisionFeaturePrint {
        let count = result.elementCount
        let data = result.data
        switch result.elementType {
        case .float:
            return try VisionFeaturePrint(
                revision: revision,
                elementType: .float32,
                values: try decode(data, count: count, as: Float.self).map(Double.init)
            )
        case .double:
            return try VisionFeaturePrint(
                revision: revision,
                elementType: .float64,
                values: try decode(data, count: count, as: Double.self)
            )
        default:
            throw ML5Error.unsupportedVisionResult(
                reason: "Vision returned an unknown feature-print element type."
            )
        }
    }

    private static func decode<Element>(
        _ data: Data,
        count: Int,
        as type: Element.Type
    ) throws -> [Element] {
        let expectedCount = count.multipliedReportingOverflow(by: MemoryLayout<Element>.stride)
        guard count > 0, !expectedCount.overflow, data.count == expectedCount.partialValue else {
            throw ML5Error.unsupportedVisionResult(
                reason: "Vision feature-print storage does not match its element count."
            )
        }
        return data.withUnsafeBytes { bytes in
            (0..<count).map { index in
                bytes.loadUnaligned(
                    fromByteOffset: index * MemoryLayout<Element>.stride,
                    as: Element.self
                )
            }
        }
    }
}
