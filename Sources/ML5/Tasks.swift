import Foundation

/// A task that transforms raw model output into a typed prediction.
public protocol NeuralNetworkTask: Sendable {
    /// The value-safe result returned after output decoding.
    associatedtype Prediction: Sendable

    /// The broad model behavior represented by this task.
    var kind: NeuralNetworkTaskKind { get }

    /// Decodes a framework-independent model output into the task's prediction.
    func decode(_ output: ModelOutput) throws -> Prediction
}

/// The supported neural-network task families.
public enum NeuralNetworkTaskKind: String, Sendable, Equatable, Codable {
    /// A task that selects a discrete label and optional confidence.
    case classification
    /// A task that predicts a finite scalar value.
    case regression
}

/// A classification label that can be decoded from a Core ML string output.
public protocol ClassificationLabel: Hashable, Sendable {
    /// Creates a label from the string emitted by a model, or returns `nil` when unsupported.
    init?(ml5RawValue: String)

    /// The stable string representation expected in a model output.
    var ml5RawValue: String { get }
}

extension String: ClassificationLabel {
    /// Creates a string label without additional validation.
    public init?(ml5RawValue: String) {
        self = ml5RawValue
    }

    /// Returns this string as its stable model representation.
    public var ml5RawValue: String {
        self
    }
}

/// Output selection rules for a classification task.
public struct ClassificationConfiguration: Sendable, Equatable {
    /// The output containing the classification label string.
    public let labelOutput: OutputName
    /// The optional numeric output containing probability-like confidence.
    public let confidenceOutput: OutputName?

    /// Creates output-selection rules for a classification task.
    ///
    /// - Throws: ``ML5Error/invalidConfiguration(reason:)`` when both roles use one output.
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
    /// The output names used to decode the model result.
    public let configuration: ClassificationConfiguration

    /// Creates a task with validated output-selection rules.
    public init(configuration: ClassificationConfiguration) {
        self.configuration = configuration
    }

    /// The classification task family.
    public var kind: NeuralNetworkTaskKind {
        .classification
    }

    /// Decodes a string label and optional numeric confidence from model output.
    ///
    /// - Throws: ``ML5Error`` when an output is missing, mistyped, or invalid.
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
    /// The numeric output containing the predicted scalar.
    public let valueOutput: OutputName

    /// Creates output-selection rules for scalar regression.
    public init(valueOutput: OutputName) {
        self.valueOutput = valueOutput
    }
}

/// A typed regression task.
public struct RegressionTask: NeuralNetworkTask {
    /// The output name used to decode the model result.
    public let configuration: RegressionConfiguration

    /// Creates a scalar regression task.
    public init(configuration: RegressionConfiguration) {
        self.configuration = configuration
    }

    /// The regression task family.
    public var kind: NeuralNetworkTaskKind {
        .regression
    }

    /// Decodes a finite number or integer from model output.
    ///
    /// - Throws: ``ML5Error`` when the configured output is missing, mistyped, or nonfinite.
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
    /// The input features associated with the label.
    public let features: FeatureVector
    /// The expected classification label.
    public let label: Label

    /// Creates a labeled classification sample.
    public init(features: FeatureVector, label: Label) {
        self.features = features
        self.label = label
    }
}

/// A labeled sample suitable for a future regression training adapter.
public struct RegressionSample: Sendable, Equatable {
    /// The input features associated with the target.
    public let features: FeatureVector
    /// The finite scalar the model should learn to predict.
    public let target: Double

    /// Creates a validated scalar regression sample.
    ///
    /// - Throws: ``ML5Error/invalidRegressionValue(_:)`` when `target` is nonfinite.
    public init(features: FeatureVector, target: Double) throws {
        guard target.isFinite else {
            throw ML5Error.invalidRegressionValue(target)
        }

        self.features = features
        self.target = target
    }
}
