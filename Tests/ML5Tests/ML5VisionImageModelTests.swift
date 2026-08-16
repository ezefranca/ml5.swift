@preconcurrency import CoreML
import Foundation
@preconcurrency import ImageIO
import Testing
@preconcurrency import Vision

@testable import ML5

private enum VisionFixtureFailure: Error {
    case request
}

@Suite("ML5 Vision and Core ML image adapters", .serialized)
struct ML5VisionImageModelTests {
    @Test("Orientations, crop policies, and compute settings reach the Vision boundary")
    func requestOptions() async throws {
        let orientations: [(VisionImageOrientation, CGImagePropertyOrientation)] = [
            (.up, .up),
            (.upMirrored, .upMirrored),
            (.down, .down),
            (.downMirrored, .downMirrored),
            (.left, .left),
            (.leftMirrored, .leftMirrored),
            (.right, .right),
            (.rightMirrored, .rightMirrored),
        ]
        let cropPolicies: [(VisionImageCropAndScale, VNImageCropAndScaleOption)] = [
            (.centerCrop, .centerCrop),
            (.scaleFit, .scaleFit),
            (.scaleFill, .scaleFill),
        ]
        let observation = VisionCoreMLRawResult.classification(
            identifier: "class",
            confidence: 1
        )

        for (orientation, expectedOrientation) in orientations {
            for (cropPolicy, expectedCropPolicy) in cropPolicies {
                let model = try VisionCoreMLImageModel(
                    contentsOf: URL(fileURLWithPath: "/injected/image.mlmodelc"),
                    configuration: .init(computeUnits: .cpuAndNeuralEngine),
                    loader: VisionCoreMLModelLoader { _, configuration in
                        #expect(configuration.computeUnits == .cpuAndNeuralEngine)
                        return VisionCoreMLModelOperation { _, actualOrientation, actualCrop in
                            #expect(actualOrientation == expectedOrientation)
                            #expect(actualCrop == expectedCropPolicy)
                            return [observation]
                        }
                    }
                )
                #expect(
                    try await model.classify(
                        image(),
                        orientation: orientation,
                        cropAndScale: cropPolicy
                    ).count == 1
                )
            }
        }

        let configuration = VisionFeaturePrintConfiguration(
            revision: .revision1,
            cropAndScale: .scaleFit
        )
        #expect(
            try JSONDecoder().decode(
                VisionFeaturePrintConfiguration.self,
                from: JSONEncoder().encode(configuration)
            ) == configuration
        )
        #expect(VisionImageOrientation.allCases.count == 8)
        #expect(VisionImageCropAndScale.allCases.count == 3)
        #expect(VisionFeaturePrintRevision.allCases == [.revision1, .revision2])
        #expect(VisionFeaturePrintElementType.allCases == [.float32, .float64])
    }

    @Test("Classifications validate, sort, limit, serialize, and reject wrong observations")
    func classifications() async throws {
        let dog = classification("dog", 0.25)
        let ant = classification("ant", 0.75)
        let cat = classification("cat", 0.75)
        let model = try injectedModel(results: [dog, cat, ant])

        let results = try await model.classify(image(), maximumResults: 2)
        #expect(results.map(\.identifier) == ["ant", "cat"])
        #expect(results.map(\.confidence) == [0.75, 0.75])
        #expect(
            try JSONDecoder().decode(
                [VisionClassification].self,
                from: JSONEncoder().encode(results)
            ) == results
        )
        #expect(try await model.classify(image()).map(\.identifier) == ["ant", "cat", "dog"])

        for limit in [0, -1] {
            await #expect(throws: ML5Error.self) {
                _ = try await model.classify(image(), maximumResults: limit)
            }
        }
        await #expect(throws: ML5Error.self) {
            _ = try await injectedModel(results: []).classify(image())
        }
        await #expect(throws: ML5Error.self) {
            _ = try await injectedModel(results: [.unsupported]).classify(image())
        }

        await #expect(throws: ML5Error.self) {
            _ = try await injectedModel(
                results: [.classification(identifier: "", confidence: 1)]
            ).classify(image())
        }
        await #expect(throws: ML5Error.self) {
            _ = try await injectedModel(
                results: [.classification(identifier: "class", confidence: .nan)]
            ).classify(image())
        }
        #expect(throws: ML5Error.self) {
            _ = try VisionClassification(identifier: "", confidence: 1)
        }
        #expect(throws: ML5Error.self) {
            _ = try VisionClassification(identifier: "class", confidence: .infinity)
        }
        #expect(throws: ML5Error.self) {
            _ = try JSONDecoder().decode(
                VisionClassification.self,
                from: Data(#"{"identifier":"","confidence":1}"#.utf8)
            )
        }
    }

    @Test("Named Core ML features convert, validate, sort, and serialize")
    func coreMLFeatures() async throws {
        let second = VisionCoreMLRawResult.feature(
            name: "z",
            value: MLFeatureValue(string: "value")
        )
        let first = VisionCoreMLRawResult.feature(
            name: "a",
            value: MLFeatureValue(double: 2.5)
        )

        let model = try injectedModel(results: [second, first])
        let features = try await model.extractFeatures(
            image(),
            orientation: .right,
            cropAndScale: .scaleFill
        )
        #expect(features.map(\.name.rawValue) == ["a", "z"])
        #expect(features.map(\.value) == [.number(2.5), .string("value")])
        #expect(
            try JSONDecoder().decode(
                [VisionCoreMLFeature].self,
                from: JSONEncoder().encode(features)
            ) == features
        )

        await #expect(throws: ML5Error.self) {
            _ = try await injectedModel(results: []).extractFeatures(image())
        }
        await #expect(throws: ML5Error.self) {
            _ = try await injectedModel(results: [.unsupported]).extractFeatures(image())
        }
        await #expect(throws: ML5Error.self) {
            _ = try await injectedModel(
                results: [.feature(name: "", value: MLFeatureValue(double: 1))]
            ).extractFeatures(image())
        }
        #expect(throws: ML5Error.self) {
            _ = try VisionCoreMLFeature(name: "bad", value: .array([]))
        }
        #expect(throws: ML5Error.self) {
            _ = try JSONDecoder().decode(
                VisionCoreMLFeature.self,
                from: Data(#"{"name":"bad","value":{"kind":"array","array":[]}}"#.utf8)
            )
        }
    }

    @Test("Vision requests preserve typed failures, wrap framework errors, and cancel")
    func requestFailuresAndCancellation() async throws {
        let frameworkFailure = try injectedModel { _, _, _ in
            throw VisionFixtureFailure.request
        }
        await #expect(throws: ML5Error.self) {
            _ = try await frameworkFailure.classify(image())
        }

        let typedFailure = try injectedModel { _, _, _ in
            throw ML5Error.invalidImage(reason: "injected")
        }
        await #expect(throws: ML5Error.invalidImage(reason: "injected")) {
            _ = try await typedFailure.extractFeatures(image())
        }

        let preCancelled = try injectedModel(results: [classification("class", 1)])
        let preCancelledTask = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await preCancelled.classify(image())
        }
        await #expect(throws: CancellationError.self) {
            _ = try await preCancelledTask.value
        }

        let postCancelled = try injectedModel { _, _, _ in
            withUnsafeCurrentTask { $0?.cancel() }
            return [classification("class", 1)]
        }
        let postCancelledTask = Task { try await postCancelled.classify(image()) }
        await #expect(throws: CancellationError.self) {
            _ = try await postCancelledTask.value
        }
    }

    @Test("Model loading wraps failures and cache-backed loading uses trusted resources")
    func loading() async throws {
        #expect(throws: ML5Error.self) {
            _ = try VisionCoreMLImageModel(
                contentsOf: URL(fileURLWithPath: "/missing/image.mlmodelc")
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try VisionCoreMLImageModel(
                contentsOf: URL(fileURLWithPath: "/injected/image.mlmodelc"),
                loader: VisionCoreMLModelLoader { _, _ in
                    throw VisionFixtureFailure.request
                }
            )
        }

        let fixture = try CompiledVisionModelFixture.classifier()
        defer { fixture.remove() }
        let digest = try ML5ModelDigest.sha256(contentsOf: fixture.compiledModelURL)
        let source = try ML5ModelSource(
            fileURL: fixture.compiledModelURL,
            integrityDigest: digest,
            metadata: modelMetadata()
        )
        let cache = try ML5ModelCache(
            configuration: ML5ModelCacheConfiguration(
                directory: fixture.directoryURL.appendingPathComponent("cache")
            )
        )
        let model = try await VisionCoreMLImageModel.load(from: source, using: cache)
        let result = try await model.classify(image())
        #expect(result.map(\.identifier) == ["cat", "dog"])
        #expect(result[0].confidence > result[1].confidence)
        #expect(VisionCoreMLModelLoader.results(nil).isEmpty)
        let unsupportedResults = VisionCoreMLModelLoader.results([VNObservation()])
        #expect(unsupportedResults.count == 1)
        guard case .unsupported = unsupportedResults[0] else {
            Issue.record("Expected an unsupported raw observation.")
            return
        }

        let featureFixture = try CompiledVisionModelFixture.featureExtractor()
        defer { featureFixture.remove() }
        let featureModel = try VisionCoreMLImageModel(contentsOf: featureFixture.compiledModelURL)
        let features = try await featureModel.extractFeatures(image())
        #expect(features.count == 1)
        #expect(features[0].name.rawValue == "features")
        guard case let .tensor(tensor) = features[0].value else {
            Issue.record("Expected a tensor feature output.")
            return
        }
        #expect(tensor.shape.dimensions == [2])
    }

    @Test("Feature prints validate, serialize, and calculate stable distances")
    func featurePrintValues() throws {
        let first = try VisionFeaturePrint(
            revision: .revision2,
            elementType: .float32,
            values: [0, 3, 4]
        )
        let second = try VisionFeaturePrint(
            revision: .revision2,
            elementType: .float32,
            values: [0, 0, 0]
        )
        #expect(try first.distance(to: second) == 5)
        #expect(
            try JSONDecoder().decode(
                VisionFeaturePrint.self,
                from: JSONEncoder().encode(first)
            ) == first
        )
        #expect(throws: ML5Error.self) {
            _ = try VisionFeaturePrint(
                revision: .revision1,
                elementType: .float64,
                values: []
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try VisionFeaturePrint(
                revision: .revision1,
                elementType: .float64,
                values: [.nan]
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try JSONDecoder().decode(
                VisionFeaturePrint.self,
                from: Data(
                    #"{"revision":1,"elementType":"float64","values":[]}"#.utf8
                )
            )
        }

        let incompatibleValues: [VisionFeaturePrint] = [
            try VisionFeaturePrint(
                revision: .revision1,
                elementType: .float32,
                values: [0, 0, 0]
            ),
            try VisionFeaturePrint(
                revision: .revision2,
                elementType: .float64,
                values: [0, 0, 0]
            ),
            try VisionFeaturePrint(
                revision: .revision2,
                elementType: .float32,
                values: [0]
            ),
        ]
        for incompatible in incompatibleValues {
            #expect(throws: ML5Error.self) {
                _ = try first.distance(to: incompatible)
            }
        }
        let positive = try VisionFeaturePrint(
            revision: .revision2,
            elementType: .float32,
            values: [.greatestFiniteMagnitude]
        )
        let negative = try VisionFeaturePrint(
            revision: .revision2,
            elementType: .float32,
            values: [-.greatestFiniteMagnitude]
        )
        #expect(throws: ML5Error.self) {
            _ = try positive.distance(to: negative)
        }
    }

    @Test("Feature-print observations decode both element widths and reject malformed storage")
    func featurePrintObservationDecoding() throws {
        let floatObservation = VisionFeaturePrintRawResult(
            elementType: .float,
            elementCount: 2,
            data: data([Float(1.5), Float(-2)])
        )
        let floatPrint = try VisionImageFeatureExtractor.featurePrint(
            from: floatObservation,
            revision: .revision1
        )
        #expect(floatPrint.values == [1.5, -2])
        #expect(floatPrint.elementType == .float32)

        let doubleObservation = VisionFeaturePrintRawResult(
            elementType: .double,
            elementCount: 2,
            data: data([Double(3), Double(4)])
        )
        let doublePrint = try VisionImageFeatureExtractor.featurePrint(
            from: doubleObservation,
            revision: .revision2
        )
        #expect(doublePrint.values == [3, 4])
        #expect(doublePrint.elementType == .float64)

        let malformed = VisionFeaturePrintRawResult(
            elementType: .float,
            elementCount: 2,
            data: Data(repeating: 0, count: 1)
        )
        #expect(throws: ML5Error.self) {
            _ = try VisionImageFeatureExtractor.featurePrint(
                from: malformed,
                revision: .revision2
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try VisionImageFeatureExtractor.featurePrint(
                from: VisionFeaturePrintRawResult(
                    elementType: .float,
                    elementCount: 0,
                    data: Data()
                ),
                revision: .revision2
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try VisionImageFeatureExtractor.featurePrint(
                from: VisionFeaturePrintRawResult(
                    elementType: .float,
                    elementCount: Int.max,
                    data: Data()
                ),
                revision: .revision2
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try VisionImageFeatureExtractor.featurePrint(
                from: VisionFeaturePrintRawResult(
                    elementType: .unknown,
                    elementCount: 1,
                    data: Data(repeating: 0, count: 1)
                ),
                revision: .revision2
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try VisionFeaturePrintOperation.requireResult(nil)
        }
    }

    @Test("Native Vision feature-print extraction succeeds and handles failures and cancellation")
    func featurePrintExtraction() async throws {
        let system = VisionImageFeatureExtractor()
        #expect(system.configuration == .init())
        let systemPrint = try await system.extract(image(), orientation: .leftMirrored)
        #expect(!systemPrint.values.isEmpty)
        #expect(systemPrint.revision == .revision2)

        let observation = VisionFeaturePrintRawResult(
            elementType: .float,
            elementCount: 1,
            data: data([Float(2)])
        )
        let injected = VisionImageFeatureExtractor(
            configuration: .init(revision: .revision1, cropAndScale: .scaleFill),
            operation: VisionFeaturePrintOperation { _, orientation, crop, revision in
                #expect(orientation == .downMirrored)
                #expect(crop == .scaleFill)
                #expect(revision == VNGenerateImageFeaturePrintRequestRevision1)
                return observation
            }
        )
        #expect(
            try await injected.extract(image(), orientation: .downMirrored).values == [2]
        )

        let typedFailure = VisionImageFeatureExtractor(
            configuration: .init(),
            operation: VisionFeaturePrintOperation { _, _, _, _ in
                throw ML5Error.invalidImage(reason: "injected")
            }
        )
        await #expect(throws: ML5Error.invalidImage(reason: "injected")) {
            _ = try await typedFailure.extract(image())
        }
        let frameworkFailure = VisionImageFeatureExtractor(
            configuration: .init(),
            operation: VisionFeaturePrintOperation { _, _, _, _ in
                throw VisionFixtureFailure.request
            }
        )
        await #expect(throws: ML5Error.self) {
            _ = try await frameworkFailure.extract(image())
        }

        let preCancelled = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await injected.extract(image())
        }
        await #expect(throws: CancellationError.self) {
            _ = try await preCancelled.value
        }
        let postCancellationExtractor = VisionImageFeatureExtractor(
            configuration: .init(),
            operation: VisionFeaturePrintOperation { _, _, _, _ in
                withUnsafeCurrentTask { $0?.cancel() }
                return observation
            }
        )
        let postCancelled = Task { try await postCancellationExtractor.extract(image()) }
        await #expect(throws: CancellationError.self) {
            _ = try await postCancelled.value
        }
    }

    private func injectedModel(
        results: [VisionCoreMLRawResult]
    ) throws -> VisionCoreMLImageModel {
        try injectedModel { _, _, _ in results }
    }

    private func injectedModel(
        operation:
            @escaping (
                CVPixelBuffer, CGImagePropertyOrientation, VNImageCropAndScaleOption
            ) throws -> [VisionCoreMLRawResult]
    ) throws -> VisionCoreMLImageModel {
        try VisionCoreMLImageModel(
            contentsOf: URL(fileURLWithPath: "/injected/image.mlmodelc"),
            loader: VisionCoreMLModelLoader { _, _ in
                VisionCoreMLModelOperation(perform: operation)
            }
        )
    }

    private func classification(
        _ identifier: String,
        _ confidence: Double
    ) -> VisionCoreMLRawResult {
        .classification(identifier: identifier, confidence: confidence)
    }

    private func image() throws -> ML5Image {
        try ML5Image(
            width: 2,
            height: 2,
            pixelFormat: .rgba8,
            data: Data([
                255, 0, 0, 255,
                0, 255, 0, 255,
                0, 0, 255, 255,
                255, 255, 255, 255,
            ])
        )
    }

    private func data<Element>(_ values: [Element]) -> Data {
        values.withUnsafeBytes { Data($0) }
    }

    private func modelMetadata() throws -> ML5ModelMetadata {
        try ML5ModelMetadata(
            name: "Vision Fixture",
            version: "1.0.0",
            author: "p5.swift",
            license: "Test fixture",
            source: URL(string: "https://example.com/vision-fixture")
        )
    }
}

private struct CompiledVisionModelFixture {
    let directoryURL: URL
    let compiledModelURL: URL

    static func classifier() throws -> Self {
        try Self(
            encodedModel:
                "CAESVwoRCgVpbWFnZRoIIgYIAhACGBRSFQoNcHJvYmFiaWxpdGllcxoEMgISAFIQCgpjbGFzc0xhYmVsGgIaAFoKY2xhc3NMYWJlbGINcHJvYmFiaWxpdGllc5oZgQIKGQoHZmxhdHRlbhIFaW1hZ2UaBGZsYXTqEgAKkQEKBnNjb3JlcxIEZmxhdBoGc2NvcmVz4gh4CAwQAlABogFiCmAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACqAQoKCAAAgD8AAAAACiMKB3NvZnRtYXgSBnNjb3JlcxoNcHJvYmFiaWxpdGllc/oKABIOCgVpbWFnZVIFVQAAgD+iBgoKA2NhdAoDZG9nwgwNcHJvYmFiaWxpdGllcw==",
            name: "TinyClassifier"
        )
    }

    static func featureExtractor() throws -> Self {
        try Self(
            encodedModel:
                "CAESKgoRCgVpbWFnZRoIIgYIAhACGBRSFQoIZmVhdHVyZXMaCSoHCgECEMCABKIfwwEKGQoHZmxhdHRlbhIFaW1hZ2UaBGZsYXTqEgAKlQEKCGZlYXR1cmVzEgRmbGF0GghmZWF0dXJlc+IIeAgMEAJQAaIBYgpgAACAPwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAqgEKCggAAAAAAAAAABIOCgVpbWFnZVIFVQAAgD8=",
            name: "TinyFeatures"
        )
    }

    init(encodedModel: String, name: String) throws {
        guard let data = Data(base64Encoded: encodedModel) else {
            throw VisionFixtureFailure.request
        }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "p5-swift-vision-\(UUID().uuidString)",
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let source = directory.appendingPathComponent("\(name).mlmodel")
        try data.write(to: source, options: .atomic)
        directoryURL = directory
        compiledModelURL = try MLModel.compileModel(at: source)
    }

    func remove() {
        try? FileManager.default.removeItem(at: directoryURL)
        try? FileManager.default.removeItem(at: compiledModelURL)
    }
}
