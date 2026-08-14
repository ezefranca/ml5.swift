import Foundation

/// A classification result with an optional probability-like confidence.
public struct ClassificationPrediction<Label: ClassificationLabel>: Sendable, Equatable {
    public let label: Label
    public let confidence: Double?

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
    public let value: Double

    public init(value: Double) throws {
        guard value.isFinite else {
            throw ML5Error.invalidRegressionValue(value)
        }

        self.value = value
    }
}
