@preconcurrency import CoreML
@preconcurrency import CoreVideo
import Foundation

/// A value-safe selection of Core ML compute units.
public enum ML5ComputeUnits: String, Sendable, Equatable, Codable {
    /// Allows Core ML to choose among the CPU, GPU, and Neural Engine.
    case all
    /// Restricts model execution to the CPU.
    case cpuOnly
    /// Allows the CPU and GPU while excluding the Neural Engine.
    case cpuAndGPU
    /// Allows the CPU and Neural Engine while excluding the GPU.
    case cpuAndNeuralEngine
}

/// Configuration applied when loading a Core ML model.
public struct CoreMLModelConfiguration: Sendable, Equatable {
    /// The processors Core ML may use for model execution.
    public var computeUnits: ML5ComputeUnits

    /// Creates a model-loading configuration.
    public init(computeUnits: ML5ComputeUnits = .all) {
        self.computeUnits = computeUnits
    }

    func makeCoreMLConfiguration() -> MLModelConfiguration {
        let configuration = MLModelConfiguration()
        configuration.computeUnits =
            switch computeUnits {
            case .all:
                .all
            case .cpuOnly:
                .cpuOnly
            case .cpuAndGPU:
                .cpuAndGPU
            case .cpuAndNeuralEngine:
                .cpuAndNeuralEngine
            }
        return configuration
    }
}

struct CoreMLPredictionOperation: @unchecked Sendable {
    var predict: (any MLFeatureProvider) async throws -> any MLFeatureProvider
}

struct CoreMLModelLoader: @unchecked Sendable {
    var load: (URL, MLModelConfiguration) throws -> CoreMLPredictionOperation

    static let system = Self { modelURL, configuration in
        let model = try MLModel(contentsOf: modelURL, configuration: configuration)
        return CoreMLPredictionOperation { provider in
            try await model.prediction(from: provider)
        }
    }
}

/// An actor that owns a Core ML model and converts only value-safe data at its boundary.
///
/// `MLModel` is never exposed or captured by detached work, so the non-Sendable
/// framework object remains isolated to this actor.
public actor CoreMLModelPredictor: ModelPredicting {
    private let predictionOperation: CoreMLPredictionOperation

    /// Loads a compiled `.mlmodelc` model from disk.
    public init(
        contentsOf modelURL: URL,
        configuration: CoreMLModelConfiguration = .init()
    ) throws {
        try self.init(
            contentsOf: modelURL,
            configuration: configuration,
            loader: .system
        )
    }

    init(
        contentsOf modelURL: URL,
        configuration: CoreMLModelConfiguration = .init(),
        loader: CoreMLModelLoader
    ) throws {
        do {
            predictionOperation = try loader.load(
                modelURL,
                configuration.makeCoreMLConfiguration()
            )
        } catch {
            throw ML5Error.modelLoadingFailed(
                path: modelURL.path,
                message: error.localizedDescription
            )
        }
    }

    /// Evaluates the model and returns framework-independent output values.
    public func predict(_ features: FeatureVector) async throws -> ModelOutput {
        try Task.checkCancellation()

        do {
            var inputs: [String: Any] = [:]
            for (name, value) in features.values {
                try Task.checkCancellation()
                inputs[name.rawValue] = try value.makeCoreMLValue()
            }

            let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
            try Task.checkCancellation()
            let prediction = try await predictionOperation.predict(provider)
            try Task.checkCancellation()

            var output: [OutputName: FeatureValue] = [:]
            for name in prediction.featureNames {
                try Task.checkCancellation()
                guard let value = prediction.featureValue(for: name) else {
                    continue
                }
                output[try OutputName(name)] = try FeatureValue(
                    coreMLValue: value, outputName: name)
            }

            return try ModelOutput(output)
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as ML5Error {
            throw error
        } catch {
            throw ML5Error.predictionFailed(message: error.localizedDescription)
        }
    }
}

extension FeatureValue {
    func makeCoreMLValue() throws -> MLFeatureValue {
        try validate(field: "feature")

        switch self {
        case let .number(value):
            return MLFeatureValue(double: value)
        case let .integer(value):
            return MLFeatureValue(int64: value)
        case let .string(value):
            return MLFeatureValue(string: value)
        case let .boolean(value):
            return MLFeatureValue(int64: value ? 1 : 0)
        case let .array(values):
            return MLFeatureValue(
                multiArray: try makeMultiArray(shape: [values.count], values: values))
        case let .dictionary(values):
            return try MLFeatureValue(
                dictionary: values.mapValues { NSNumber(value: $0) }
            )
        case let .tensor(tensor):
            return MLFeatureValue(
                multiArray: try makeMultiArray(
                    shape: tensor.shape.dimensions,
                    values: tensor.values
                )
            )
        case let .sequence(sequence):
            let coreMLSequence =
                switch sequence {
                case let .strings(values):
                    MLSequence(strings: values)
                case let .integers(values):
                    MLSequence(int64s: values.map { NSNumber(value: $0) })
                }
            return MLFeatureValue(sequence: coreMLSequence)
        case let .image(image):
            return MLFeatureValue(pixelBuffer: try image.makePixelBuffer())
        }
    }

    init(coreMLValue: MLFeatureValue, outputName: String) throws {
        guard coreMLValue.type == .invalid || !coreMLValue.isUndefined else {
            throw ML5Error.predictionFailed(
                message: "Output \(outputName.debugDescription) is undefined."
            )
        }
        switch coreMLValue.type {
        case .double:
            self = .number(coreMLValue.doubleValue)
        case .int64:
            self = .integer(coreMLValue.int64Value)
        case .string:
            self = .string(coreMLValue.stringValue)
        case .multiArray:
            let multiArray = try Self.requireCoreMLStorage(
                coreMLValue.multiArrayValue,
                outputName: outputName,
                type: "multi-array"
            )
            let shape = try TensorShape(multiArray.shape.map(\.intValue))
            let values = (0..<multiArray.count).map { multiArray[$0].doubleValue }
            self = .tensor(try Tensor(shape: shape, values: values))
        case .dictionary:
            var values: [String: Double] = [:]
            for (key, value) in coreMLValue.dictionaryValue {
                guard let key = key as? String else {
                    throw ML5Error.predictionFailed(
                        message:
                            "Output \(outputName.debugDescription) contains a non-string dictionary key."
                    )
                }
                values[key] = value.doubleValue
            }
            self = .dictionary(values)
            try validate(field: outputName)
        case .sequence:
            let sequence = try Self.requireCoreMLStorage(
                coreMLValue.sequenceValue,
                outputName: outputName,
                type: "sequence"
            )
            switch sequence.type {
            case .string:
                self = .sequence(.strings(sequence.stringValues))
            case .int64:
                self = .sequence(.integers(sequence.int64Values.map(\.int64Value)))
            default:
                throw ML5Error.predictionFailed(
                    message:
                        "Output \(outputName.debugDescription) has an unsupported Core ML sequence element type."
                )
            }
        case .image:
            let pixelBuffer = try Self.requireCoreMLStorage(
                coreMLValue.imageBufferValue,
                outputName: outputName,
                type: "image"
            )
            self = .image(try ML5Image(pixelBuffer: pixelBuffer))
        default:
            throw ML5Error.predictionFailed(
                message:
                    "Output \(outputName.debugDescription) has unsupported Core ML type \(coreMLValue.type.rawValue)."
            )
        }
    }

    private func makeMultiArray(shape: [Int], values: [Double]) throws -> MLMultiArray {
        let multiArray = try MLMultiArray(
            shape: shape.map { NSNumber(value: $0) },
            dataType: .double
        )
        for (index, value) in values.enumerated() {
            multiArray[index] = NSNumber(value: value)
        }
        return multiArray
    }

    static func requireCoreMLStorage<Value>(
        _ value: Value?,
        outputName: String,
        type: String
    ) throws -> Value {
        guard let value else {
            throw ML5Error.predictionFailed(
                message: "Output \(outputName.debugDescription) has no \(type) storage."
            )
        }
        return value
    }
}
