@preconcurrency import CoreML
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

    /// Evaluates the model and returns a framework-independent scalar output.
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
        }
    }

    init(coreMLValue: MLFeatureValue, outputName: String) throws {
        switch coreMLValue.type {
        case .double:
            self = .number(coreMLValue.doubleValue)
        case .int64:
            self = .integer(coreMLValue.int64Value)
        case .string:
            self = .string(coreMLValue.stringValue)
        default:
            throw ML5Error.predictionFailed(
                message:
                    "Output \(outputName.debugDescription) has unsupported Core ML type \(coreMLValue.type.rawValue)."
            )
        }
    }
}
