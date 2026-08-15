import Foundation

/// A classification result with an optional probability-like confidence.
public struct ClassificationPrediction<Label: ClassificationLabel>: Sendable, Equatable {
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
public struct RegressionPrediction: Sendable, Equatable {
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
}
