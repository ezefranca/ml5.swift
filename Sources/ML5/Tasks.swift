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

/// Output selection and score interpretation for ranked classification.
public struct RankedClassificationConfiguration: Sendable, Hashable, Codable {
    /// The dictionary output mapping raw label strings to numeric scores.
    public let scoresOutput: OutputName
    /// How the task converts model scores into calibrated confidence values.
    public let interpretation: ClassificationScoreInterpretation

    /// Creates ranked-classification output rules.
    public init(
        scoresOutput: OutputName,
        interpretation: ClassificationScoreInterpretation = .probabilities
    ) {
        self.scoresOutput = scoresOutput
        self.interpretation = interpretation
    }
}

/// A classification task that returns every label ordered by confidence.
public struct RankedClassificationTask<Label: ClassificationLabel>: NeuralNetworkTask {
    /// The score output and calibration rules used to decode model results.
    public let configuration: RankedClassificationConfiguration

    /// Creates a ranked classification task.
    public init(configuration: RankedClassificationConfiguration) {
        self.configuration = configuration
    }

    /// The classification task family.
    public var kind: NeuralNetworkTaskKind {
        .classification
    }

    /// Decodes, calibrates, and deterministically ranks a string-keyed score dictionary.
    ///
    /// - Throws: ``ML5Error`` when the output is absent, mistyped, invalid, or includes a label
    ///   unsupported by `Label`.
    public func decode(_ output: ModelOutput) throws -> RankedClassificationPrediction<Label> {
        guard let rawScores = output[configuration.scoresOutput] else {
            throw ML5Error.missingOutput(name: configuration.scoresOutput.rawValue)
        }
        guard case let .dictionary(scores) = rawScores else {
            throw ML5Error.unexpectedOutputType(
                name: configuration.scoresOutput.rawValue,
                expected: .dictionary,
                actual: rawScores.kind
            )
        }

        let orderedScores = scores.sorted { $0.key < $1.key }
        var labels: [Label] = []
        var decodedLabels: Set<Label> = []
        for (rawLabel, _) in orderedScores {
            guard let label = Label(ml5RawValue: rawLabel) else {
                throw ML5Error.invalidClassLabel(rawLabel)
            }
            guard decodedLabels.insert(label).inserted else {
                throw ML5Error.invalidClassificationScores(
                    reason: "Distinct model labels decoded to the same classification label."
                )
            }
            labels.append(label)
        }

        let confidences = try Self.confidences(
            scores: orderedScores.map(\.value),
            interpretation: configuration.interpretation
        )
        let rankedIndices = labels.indices.sorted {
            if confidences[$0] == confidences[$1] {
                return labels[$0].ml5RawValue < labels[$1].ml5RawValue
            }
            return confidences[$0] > confidences[$1]
        }
        let predictions = try rankedIndices.map {
            try ClassificationPrediction(label: labels[$0], confidence: confidences[$0])
        }
        return try RankedClassificationPrediction(predictions: predictions)
    }

    private static func confidences(
        scores: [Double],
        interpretation: ClassificationScoreInterpretation
    ) throws -> [Double] {
        switch interpretation {
        case .probabilities:
            guard scores.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
                throw ML5Error.invalidClassificationScores(
                    reason: "Probability weights must be finite and nonnegative."
                )
            }
            let total = scores.reduce(0, +)
            guard total.isFinite, total > 0 else {
                throw ML5Error.invalidClassificationScores(
                    reason: "Probability weights must have a positive finite sum."
                )
            }
            return scores.map { $0 / total }
        case let .logits(calibration):
            // ModelOutput guarantees that numeric dictionaries are nonempty and finite.
            let maximum = scores.reduce(-Double.infinity, max)
            let exponentials = scores.map {
                Foundation.exp(($0 - maximum) / calibration.temperature)
            }
            let total = exponentials.reduce(0, +)
            return exponentials.map { $0 / total }
        }
    }
}

/// Ordered output selection rules for vector regression.
public struct RegressionVectorConfiguration: Sendable, Hashable, Codable {
    /// Numeric model outputs in the vector's public component order.
    public let valueOutputs: [OutputName]

    /// Creates vector-regression output rules.
    ///
    /// - Throws: ``ML5Error/invalidRegressionVector(reason:)`` when no output is supplied or an
    ///   output name appears more than once.
    public init(valueOutputs: [OutputName]) throws {
        guard valueOutputs.isEmpty == false else {
            throw ML5Error.invalidRegressionVector(
                reason: "At least one value output is required."
            )
        }
        guard Set(valueOutputs).count == valueOutputs.count else {
            throw ML5Error.invalidRegressionVector(
                reason: "Value output names must be unique."
            )
        }
        self.valueOutputs = valueOutputs
    }

    /// Decodes and revalidates persisted output ordering.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        try self.init(valueOutputs: container.decode([OutputName].self))
    }

    /// Encodes output names in their declared component order.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(valueOutputs)
    }
}

/// A regression task that decodes multiple scalar outputs in a declared order.
public struct RegressionVectorTask: NeuralNetworkTask {
    /// The ordered output names used to decode the model result.
    public let configuration: RegressionVectorConfiguration

    /// Creates a vector regression task.
    public init(configuration: RegressionVectorConfiguration) {
        self.configuration = configuration
    }

    /// The regression task family.
    public var kind: NeuralNetworkTaskKind {
        .regression
    }

    /// Decodes finite number or integer outputs into an ordered vector.
    ///
    /// - Throws: ``ML5Error`` when an output is missing, mistyped, or nonfinite.
    public func decode(_ output: ModelOutput) throws -> RegressionVectorPrediction {
        let values = try configuration.valueOutputs.map { outputName in
            guard let rawValue = output[outputName] else {
                throw ML5Error.missingOutput(name: outputName.rawValue)
            }
            guard let value = rawValue.numericValue else {
                throw ML5Error.unexpectedOutputType(
                    name: outputName.rawValue,
                    expected: .number,
                    actual: rawValue.kind
                )
            }
            return value
        }
        return try RegressionVectorPrediction(values: values)
    }
}

/// A labeled sample suitable for a future classification training adapter.
public struct ClassificationSample<Label: ClassificationLabel>: Sendable, Hashable, Codable {
    /// The input features associated with the label.
    public let features: FeatureVector
    /// The expected classification label.
    public let label: Label

    /// Creates a labeled classification sample.
    public init(features: FeatureVector, label: Label) {
        self.features = features
        self.label = label
    }

    /// Decodes features and reconstructs the label from its stable ML5 string.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let rawLabel = try container.decode(String.self, forKey: .label)
        guard let label = Label(ml5RawValue: rawLabel) else {
            throw ML5Error.invalidClassLabel(rawLabel)
        }
        self.init(
            features: try container.decode(FeatureVector.self, forKey: .features),
            label: label
        )
    }

    /// Encodes features and the label's stable ML5 string.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(features, forKey: .features)
        try container.encode(label.ml5RawValue, forKey: .label)
    }

    private enum CodingKeys: String, CodingKey {
        case features
        case label
    }
}

/// A labeled sample suitable for a future regression training adapter.
public struct RegressionSample: Sendable, Hashable, Codable {
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

    /// Decodes a sample and revalidates its finite target.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            features: container.decode(FeatureVector.self, forKey: .features),
            target: container.decode(Double.self, forKey: .target)
        )
    }

    /// Encodes features and the finite scalar target.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(features, forKey: .features)
        try container.encode(target, forKey: .target)
    }

    private enum CodingKeys: String, CodingKey {
        case features
        case target
    }
}
