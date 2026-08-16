import Foundation

/// Per-label counts and derived metrics from a classification evaluation.
public struct ClassificationLabelMetrics: Sendable, Hashable {
    /// Number of evaluated samples whose expected label was this label.
    public let support: Int
    /// Fraction of predictions for this label that were correct, or zero when never predicted.
    public let precision: Double
    /// Fraction of expected samples for this label that were correctly predicted.
    public let recall: Double
    /// Harmonic mean of precision and recall, or zero when both are zero.
    public let f1Score: Double
}

/// Accuracy, top-k accuracy, confusion counts, and macro metrics for classification results.
public struct ClassificationEvaluation<Label: ClassificationLabel>: Sendable, Equatable {
    /// Number of evaluated samples.
    public let sampleCount: Int
    /// Number of samples whose best prediction matched the expected label.
    public let correctCount: Int
    /// Fraction of best predictions that matched the expected label.
    public let accuracy: Double
    /// Mean per-label precision across labels present in expected or predicted values.
    public let macroPrecision: Double
    /// Mean per-label recall across labels present in expected or predicted values.
    public let macroRecall: Double
    /// Mean per-label F1 score across labels present in expected or predicted values.
    public let macroF1Score: Double
    /// Per-label metrics keyed by the typed classification label.
    public let perLabel: [Label: ClassificationLabelMetrics]

    private let expected: [Label]
    private let rankings: [RankedClassificationPrediction<Label>]
    private let confusion: [Label: [Label: Int]]

    /// Evaluates expected labels against complete ranked predictions.
    ///
    /// - Throws: ``ML5Error/invalidEvaluation(reason:)`` for empty input or unequal counts.
    public init(
        expected: [Label],
        predictions: [RankedClassificationPrediction<Label>]
    ) throws {
        guard expected.isEmpty == false else {
            throw ML5Error.invalidEvaluation(reason: "Classification input cannot be empty.")
        }
        guard expected.count == predictions.count else {
            throw ML5Error.invalidEvaluation(
                reason: "Expected labels and predictions must have equal counts."
            )
        }

        var confusion: [Label: [Label: Int]] = [:]
        for (expectedLabel, prediction) in zip(expected, predictions) {
            let predictedLabel = prediction.best.label
            confusion[expectedLabel, default: [:]][predictedLabel, default: 0] += 1
        }
        let labels = Set(expected).union(predictions.map(\.best.label))
        var perLabel: [Label: ClassificationLabelMetrics] = [:]
        for label in labels {
            let truePositive = confusion[label]?[label, default: 0] ?? 0
            let support = confusion[label]?.values.reduce(0, +) ?? 0
            let predictedCount = confusion.values.reduce(0) { count, row in
                count + (row[label] ?? 0)
            }
            let precision = predictedCount == 0 ? 0 : Double(truePositive) / Double(predictedCount)
            let recall = support == 0 ? 0 : Double(truePositive) / Double(support)
            let f1Score =
                precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall)
            perLabel[label] = ClassificationLabelMetrics(
                support: support,
                precision: precision,
                recall: recall,
                f1Score: f1Score
            )
        }

        let correctCount = zip(expected, predictions).count { $0 == $1.best.label }
        let divisor = Double(labels.count)
        self.sampleCount = expected.count
        self.correctCount = correctCount
        self.accuracy = Double(correctCount) / Double(expected.count)
        self.macroPrecision = perLabel.values.reduce(0) { $0 + $1.precision } / divisor
        self.macroRecall = perLabel.values.reduce(0) { $0 + $1.recall } / divisor
        self.macroF1Score = perLabel.values.reduce(0) { $0 + $1.f1Score } / divisor
        self.perLabel = perLabel
        self.expected = expected
        self.rankings = predictions
        self.confusion = confusion
    }

    /// Returns the number of samples with the specified expected and best-predicted labels.
    public func confusionCount(expected: Label, predicted: Label) -> Int {
        confusion[expected]?[predicted] ?? 0
    }

    /// Returns the fraction of expected labels found within the first `maximumCount` predictions.
    ///
    /// - Throws: ``ML5Error/invalidClassificationScores(reason:)`` when the count is not positive.
    public func topKAccuracy(_ maximumCount: Int) throws -> Double {
        let correct = try zip(expected, rankings).count { expectedLabel, ranking in
            try ranking.top(maximumCount).contains { $0.label == expectedLabel }
        }
        return Double(correct) / Double(sampleCount)
    }
}

/// Error metrics for one component of a regression result.
public struct RegressionComponentMetrics: Sendable, Hashable {
    /// Mean absolute error.
    public let meanAbsoluteError: Double
    /// Mean squared error.
    public let meanSquaredError: Double
    /// Square root of mean squared error.
    public let rootMeanSquaredError: Double
    /// Coefficient of determination, or `nil` when every expected value is equal.
    public let rSquared: Double?
}

/// Aggregate and per-component metrics for scalar or vector regression predictions.
public struct RegressionEvaluation: Sendable, Hashable {
    /// Number of evaluated samples.
    public let sampleCount: Int
    /// Number of values in each prediction.
    public let componentCount: Int
    /// Metrics computed across every sample and component.
    public let aggregate: RegressionComponentMetrics
    /// Metrics computed independently in component order.
    public let perComponent: [RegressionComponentMetrics]

    /// Evaluates scalar regression predictions.
    ///
    /// - Throws: ``ML5Error/invalidEvaluation(reason:)`` for empty input or unequal counts.
    public init(expected: [RegressionPrediction], predictions: [RegressionPrediction]) throws {
        try self.init(
            expectedValues: expected.map { [$0.value] },
            predictedValues: predictions.map { [$0.value] }
        )
    }

    /// Evaluates vector regression predictions.
    ///
    /// - Throws: ``ML5Error/invalidEvaluation(reason:)`` for empty input, unequal counts, or
    ///   inconsistent component dimensions.
    public init(
        expected: [RegressionVectorPrediction],
        predictions: [RegressionVectorPrediction]
    ) throws {
        try self.init(
            expectedValues: expected.map(\.values),
            predictedValues: predictions.map(\.values)
        )
    }

    private init(expectedValues: [[Double]], predictedValues: [[Double]]) throws {
        guard expectedValues.isEmpty == false else {
            throw ML5Error.invalidEvaluation(reason: "Regression input cannot be empty.")
        }
        guard expectedValues.count == predictedValues.count else {
            throw ML5Error.invalidEvaluation(
                reason: "Expected values and predictions must have equal counts."
            )
        }
        let componentCount = expectedValues[0].count
        guard
            expectedValues.allSatisfy({ $0.count == componentCount }),
            predictedValues.allSatisfy({ $0.count == componentCount })
        else {
            throw ML5Error.invalidEvaluation(
                reason: "Every regression vector must use the same component count."
            )
        }

        let perComponent = (0..<componentCount).map { component in
            Self.metrics(
                expected: expectedValues.map { $0[component] },
                predicted: predictedValues.map { $0[component] }
            )
        }
        self.sampleCount = expectedValues.count
        self.componentCount = componentCount
        self.aggregate = Self.metrics(
            expected: expectedValues.flatMap { $0 },
            predicted: predictedValues.flatMap { $0 }
        )
        self.perComponent = perComponent
    }

    private static func metrics(expected: [Double], predicted: [Double])
        -> RegressionComponentMetrics
    {
        let count = Double(expected.count)
        let errors = zip(expected, predicted).map(-)
        let meanAbsoluteError = errors.reduce(0) { $0 + abs($1) } / count
        let meanSquaredError = errors.reduce(0) { $0 + $1 * $1 } / count
        let expectedMean = expected.reduce(0, +) / count
        let residualSquares = errors.reduce(0) { $0 + $1 * $1 }
        let totalSquares = expected.reduce(0) { $0 + ($1 - expectedMean) * ($1 - expectedMean) }
        return RegressionComponentMetrics(
            meanAbsoluteError: meanAbsoluteError,
            meanSquaredError: meanSquaredError,
            rootMeanSquaredError: sqrt(meanSquaredError),
            rSquared: totalSquares == 0 ? nil : 1 - residualSquares / totalSquares
        )
    }
}
