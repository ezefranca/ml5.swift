import Foundation

/// A task that transforms raw model output into a typed prediction.
public protocol NeuralNetworkTask: Sendable {
    associatedtype Prediction: Sendable

    /// The broad model behavior represented by this task.
    var kind: NeuralNetworkTaskKind { get }

    /// Decodes a framework-independent model output into the task's prediction.
    func decode(_ output: ModelOutput) throws -> Prediction
}

/// The supported neural-network task families.
public enum NeuralNetworkTaskKind: String, Sendable, Equatable, Codable {
    case classification
    case regression
}

/// A classification label that can be decoded from a Core ML string output.
public protocol ClassificationLabel: Hashable, Sendable {
    init?(ml5RawValue: String)

    var ml5RawValue: String { get }
}

extension String: ClassificationLabel {
    public init?(ml5RawValue: String) {
        self = ml5RawValue
    }

    public var ml5RawValue: String {
        self
    }
}

/// Output selection rules for a classification task.
public struct ClassificationConfiguration: Sendable, Equatable {
    public let labelOutput: OutputName
    public let confidenceOutput: OutputName?

    public init(labelOutput: OutputName, confidenceOutput: OutputName? = nil) throws {
        guard labelOutput != confidenceOutput else {
            throw ML5Error.invalidConfiguration(
                reason: "A classification label and confidence cannot use the same output."
            )
        }

        self.labelOutput = labelOutput
        self.confidenceOutput = confidenceOutput
    }
}

/// A typed classification task.
public struct ClassificationTask<Label: ClassificationLabel>: NeuralNetworkTask {
    public let configuration: ClassificationConfiguration

    public init(configuration: ClassificationConfiguration) {
        self.configuration = configuration
    }

    public var kind: NeuralNetworkTaskKind {
        .classification
    }

    public func decode(_ output: ModelOutput) throws -> ClassificationPrediction<Label> {
        guard let rawLabel = output[configuration.labelOutput] else {
            throw ML5Error.missingOutput(name: configuration.labelOutput.rawValue)
        }
        guard case let .string(label) = rawLabel else {
            throw ML5Error.unexpectedOutputType(
                name: configuration.labelOutput.rawValue,
                expected: .string,
                actual: rawLabel.kind
            )
        }
        guard let decodedLabel = Label(ml5RawValue: label) else {
            throw ML5Error.invalidClassLabel(label)
        }

        let confidence: Double?
        if let confidenceOutput = configuration.confidenceOutput {
            guard let rawConfidence = output[confidenceOutput] else {
                throw ML5Error.missingOutput(name: confidenceOutput.rawValue)
            }
            guard let value = rawConfidence.numericValue else {
                throw ML5Error.unexpectedOutputType(
                    name: confidenceOutput.rawValue,
                    expected: .number,
                    actual: rawConfidence.kind
                )
            }
            confidence = value
        } else {
            confidence = nil
        }

        return try ClassificationPrediction(label: decodedLabel, confidence: confidence)
    }
}

/// Output selection rules for a regression task.
public struct RegressionConfiguration: Sendable, Equatable {
    public let valueOutput: OutputName

    public init(valueOutput: OutputName) {
        self.valueOutput = valueOutput
    }
}

/// A typed regression task.
public struct RegressionTask: NeuralNetworkTask {
    public let configuration: RegressionConfiguration

    public init(configuration: RegressionConfiguration) {
        self.configuration = configuration
    }

    public var kind: NeuralNetworkTaskKind {
        .regression
    }

    public func decode(_ output: ModelOutput) throws -> RegressionPrediction {
        guard let rawValue = output[configuration.valueOutput] else {
            throw ML5Error.missingOutput(name: configuration.valueOutput.rawValue)
        }
        guard let value = rawValue.numericValue else {
            throw ML5Error.unexpectedOutputType(
                name: configuration.valueOutput.rawValue,
                expected: .number,
                actual: rawValue.kind
            )
        }

        return try RegressionPrediction(value: value)
    }
}

/// A labeled sample suitable for a future classification training adapter.
public struct ClassificationSample<Label: ClassificationLabel>: Sendable, Equatable {
    public let features: FeatureVector
    public let label: Label

    public init(features: FeatureVector, label: Label) {
        self.features = features
        self.label = label
    }
}

/// A labeled sample suitable for a future regression training adapter.
public struct RegressionSample: Sendable, Equatable {
    public let features: FeatureVector
    public let target: Double

    public init(features: FeatureVector, target: Double) throws {
        guard target.isFinite else {
            throw ML5Error.invalidRegressionValue(target)
        }

        self.features = features
        self.target = target
    }
}
