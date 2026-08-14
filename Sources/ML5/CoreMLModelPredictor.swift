@preconcurrency import CoreML
import Foundation

/// A value-safe selection of Core ML compute units.
public enum ML5ComputeUnits: String, Sendable, Equatable, Codable {
    case all
    case cpuOnly
    case cpuAndGPU
    case cpuAndNeuralEngine
}

/// Configuration applied when loading a Core ML model.
public struct CoreMLModelConfiguration: Sendable, Equatable {
    public var computeUnits: ML5ComputeUnits

    public init(computeUnits: ML5ComputeUnits = .all) {
        self.computeUnits = computeUnits
    }

    fileprivate func makeCoreMLConfiguration() -> MLModelConfiguration {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = switch computeUnits {
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

/// An actor that owns a Core ML model and converts only value-safe data at its boundary.
///
/// `MLModel` is never exposed or captured by detached work, so the non-Sendable
/// framework object remains isolated to this actor.
public actor CoreMLModelPredictor: ModelPredicting {
    private let model: MLModel

    /// Loads a compiled `.mlmodelc` model from disk.
    public init(
        contentsOf modelURL: URL,
        configuration: CoreMLModelConfiguration = .init()
    ) throws {
        do {
            model = try MLModel(
                contentsOf: modelURL,
                configuration: configuration.makeCoreMLConfiguration()
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
            let prediction = try await model.prediction(from: provider)
            try Task.checkCancellation()

            var output: [OutputName: FeatureValue] = [:]
            for name in prediction.featureNames {
                try Task.checkCancellation()
                guard let value = prediction.featureValue(for: name) else {
                    continue
                }
                output[try OutputName(name)] = try FeatureValue(coreMLValue: value, outputName: name)
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

private extension FeatureValue {
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
                message: "Output \(outputName.debugDescription) has unsupported Core ML type \(coreMLValue.type.rawValue)."
            )
        }
    }
}
