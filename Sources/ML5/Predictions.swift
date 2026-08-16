import Foundation

/// A classification result with an optional probability-like confidence.
public struct ClassificationPrediction<Label: ClassificationLabel>: Sendable, Hashable {
    /// The label decoded by the classification task.
    public let label: Label
    /// An optional finite probability-like value in the closed range from zero to one.
    public let confidence: Double?

    /// Creates a validated classification result.
    ///
    /// - Throws: ``ML5Error/invalidConfidence(_:)`` for nonfinite or out-of-range confidence.
    public init(label: Label, confidence: Double? = nil) throws {
        if let confidence, confidence.isFinite == false || !(0...1).contains(confidence) {
            throw ML5Error.invalidConfidence(confidence)
        }

        self.label = label
        self.confidence = confidence
    }
}

/// A finite scalar regression result.
public struct RegressionPrediction: Sendable, Hashable, Codable {
    /// The finite scalar predicted by the model.
    public let value: Double

    /// Creates a finite scalar regression prediction.
    ///
    /// - Throws: ``ML5Error/invalidRegressionValue(_:)`` when `value` is nonfinite.
    public init(value: Double) throws {
        guard value.isFinite else {
            throw ML5Error.invalidRegressionValue(value)
        }

        self.value = value
    }

    /// Decodes and revalidates a persisted scalar regression prediction.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        try self.init(value: container.decode(Double.self))
    }

    /// Encodes the finite prediction as a single numeric value.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(value)
    }
}

/// A validated temperature used to calibrate classification logits before softmax.
public struct TemperatureScaling: Sendable, Hashable, Codable {
    /// The finite, strictly positive divisor applied to logits.
    public let temperature: Double

    /// Creates a temperature-scaling calibration.
    ///
    /// Values below one sharpen a distribution and values above one soften it.
    ///
    /// - Throws: ``ML5Error/invalidClassificationScores(reason:)`` when `temperature`
    ///   is nonfinite or not positive.
    public init(temperature: Double = 1) throws {
        guard temperature.isFinite, temperature > 0 else {
            throw ML5Error.invalidClassificationScores(
                reason: "Temperature must be finite and greater than zero."
            )
        }
        self.temperature = temperature
    }

    /// Decodes and revalidates a persisted temperature.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        try self.init(temperature: container.decode(Double.self))
    }

    /// Encodes the temperature as a single numeric value.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(temperature)
    }
}

/// The meaning of scores emitted by a ranked classification model.
public enum ClassificationScoreInterpretation: Sendable, Hashable, Codable {
    /// Scores are already nonnegative probability-like weights and are normalized to sum to one.
    case probabilities
    /// Scores are unbounded logits calibrated by temperature scaling and stable softmax.
    case logits(TemperatureScaling)
}

/// A complete, confidence-ordered classification distribution.
public struct RankedClassificationPrediction<Label: ClassificationLabel>: Sendable, Equatable {
    /// Every decoded label ordered by descending confidence and then stable raw label.
    public let predictions: [ClassificationPrediction<Label>]

    /// The highest-confidence prediction.
    public var best: ClassificationPrediction<Label> {
        // The initializer guarantees a nonempty collection.
        predictions[0]
    }

    /// Creates and validates a complete probability distribution.
    ///
    /// Predictions must be nonempty, uniquely labeled, confidence-bearing, sorted by descending
    /// confidence, and normalized to approximately one.
    ///
    /// - Throws: ``ML5Error/invalidClassificationScores(reason:)`` when an invariant is violated.
    public init(predictions: [ClassificationPrediction<Label>]) throws {
        guard predictions.isEmpty == false else {
            throw ML5Error.invalidClassificationScores(reason: "A ranking cannot be empty.")
        }

        var labels: Set<Label> = []
        var previousConfidence = Double.infinity
        var confidenceSum = 0.0
        for prediction in predictions {
            guard let confidence = prediction.confidence else {
                throw ML5Error.invalidClassificationScores(
                    reason: "Every ranked prediction requires confidence."
                )
            }
            guard labels.insert(prediction.label).inserted else {
                throw ML5Error.invalidClassificationScores(
                    reason: "A ranking cannot contain duplicate labels."
                )
            }
            guard confidence <= previousConfidence else {
                throw ML5Error.invalidClassificationScores(
                    reason: "Predictions must be sorted by descending confidence."
                )
            }
            previousConfidence = confidence
            confidenceSum += confidence
        }

        guard abs(confidenceSum - 1) <= 1e-9 else {
            throw ML5Error.invalidClassificationScores(
                reason: "Ranked confidence values must sum to one."
            )
        }
        self.predictions = predictions
    }

    /// Returns at most the first `maximumCount` predictions.
    ///
    /// - Throws: ``ML5Error/invalidClassificationScores(reason:)`` when the count is not positive.
    public func top(_ maximumCount: Int) throws -> [ClassificationPrediction<Label>] {
        guard maximumCount > 0 else {
            throw ML5Error.invalidClassificationScores(
                reason: "A top-k count must be greater than zero."
            )
        }
        return Array(predictions.prefix(maximumCount))
    }
}

/// An ordered, finite vector-valued regression result.
public struct RegressionVectorPrediction: Sendable, Hashable, Codable {
    /// Predicted components in the task configuration's declared order.
    public let values: [Double]

    /// Creates a nonempty finite regression vector.
    ///
    /// - Throws: ``ML5Error/invalidRegressionVector(reason:)`` when the vector is empty, or
    ///   ``ML5Error/invalidRegressionValue(_:)`` when a component is nonfinite.
    public init(values: [Double]) throws {
        guard values.isEmpty == false else {
            throw ML5Error.invalidRegressionVector(reason: "A prediction cannot be empty.")
        }
        if let invalidValue = values.first(where: { $0.isFinite == false }) {
            throw ML5Error.invalidRegressionValue(invalidValue)
        }
        self.values = values
    }

    /// Decodes and revalidates a persisted regression vector.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        try self.init(values: container.decode([Double].self))
    }

    /// Encodes the ordered components as a numeric array.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(values)
    }

    /// Accesses a component by its declared output position.
    public subscript(index: Int) -> Double {
        values[index]
    }
}
