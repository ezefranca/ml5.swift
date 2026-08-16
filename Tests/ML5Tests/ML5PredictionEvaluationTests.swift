import Foundation
import Testing

@testable import ML5

private enum AliasLabel: String, ClassificationLabel {
    case accepted

    init?(ml5RawValue: String) {
        switch ml5RawValue {
        case "accepted", "yes":
            self = .accepted
        default:
            return nil
        }
    }

    var ml5RawValue: String { rawValue }
}

@Suite("ML5 ranked predictions and evaluation")
struct ML5PredictionEvaluationTests {
    @Test("Temperature scaling validates and round-trips")
    func temperatureScaling() throws {
        let calibration = try TemperatureScaling(temperature: 2.5)
        let data = try JSONEncoder().encode(calibration)

        #expect(calibration.temperature == 2.5)
        #expect(try JSONDecoder().decode(TemperatureScaling.self, from: data) == calibration)
        #expect(throws: ML5Error.self) {
            _ = try TemperatureScaling(temperature: 0)
        }
        #expect(throws: ML5Error.self) {
            _ = try TemperatureScaling(temperature: .infinity)
        }
        #expect(throws: ML5Error.self) {
            _ = try JSONDecoder().decode(TemperatureScaling.self, from: Data("-1".utf8))
        }
    }

    @Test("Probability dictionaries normalize and rank with stable ties")
    func probabilityRanking() throws {
        let configuration = RankedClassificationConfiguration(scoresOutput: "scores")
        let task = RankedClassificationTask<String>(configuration: configuration)
        let result = try task.decode(
            ModelOutput([
                "scores": .dictionary([
                    "zebra": 1,
                    "ant": 2,
                    "bee": 2,
                ])
            ])
        )

        #expect(task.kind == .classification)
        #expect(result.best.label == "ant")
        #expect(result.predictions.map(\.label) == ["ant", "bee", "zebra"])
        #expect(abs((result.predictions[0].confidence ?? 0) - 0.4) < 1e-12)
        #expect(try result.top(2).map(\.label) == ["ant", "bee"])
        #expect(try result.top(20) == result.predictions)

        let encoded = try JSONEncoder().encode(configuration)
        #expect(
            try JSONDecoder().decode(RankedClassificationConfiguration.self, from: encoded)
                == configuration
        )
    }

    @Test("Temperature-scaled logits use stable softmax")
    func logitRanking() throws {
        let interpretation = ClassificationScoreInterpretation.logits(
            try TemperatureScaling(temperature: 2)
        )
        let task = RankedClassificationTask<String>(
            configuration: RankedClassificationConfiguration(
                scoresOutput: "logits",
                interpretation: interpretation
            )
        )
        let result = try task.decode(
            ModelOutput([
                "logits": .dictionary([
                    "far": -1_000,
                    "near": 1_000,
                ])
            ])
        )

        #expect(result.best.label == "near")
        #expect(result.best.confidence == 1)
        #expect(result.predictions[1].confidence == 0)

        let encoded = try JSONEncoder().encode(interpretation)
        #expect(
            try JSONDecoder().decode(ClassificationScoreInterpretation.self, from: encoded)
                == interpretation
        )
    }

    @Test("Ranked task reports missing, mistyped, unsupported, duplicate, and invalid scores")
    func rankingFailures() throws {
        let task = RankedClassificationTask<AliasLabel>(
            configuration: RankedClassificationConfiguration(scoresOutput: "scores")
        )

        #expect(throws: ML5Error.missingOutput(name: "scores")) {
            _ = try task.decode(ModelOutput(["other": .number(1)]))
        }
        #expect(throws: ML5Error.self) {
            _ = try task.decode(ModelOutput(["scores": .number(1)]))
        }
        #expect(throws: ML5Error.invalidClassLabel("unknown")) {
            _ = try task.decode(ModelOutput(["scores": .dictionary(["unknown": 1])]))
        }
        #expect(throws: ML5Error.self) {
            _ = try task.decode(
                ModelOutput(["scores": .dictionary(["accepted": 0.5, "yes": 0.5])])
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try task.decode(ModelOutput(["scores": .dictionary(["accepted": -1])]))
        }
        #expect(throws: ML5Error.self) {
            _ = try task.decode(ModelOutput(["scores": .dictionary(["accepted": 0])]))
        }
        #expect(throws: ML5Error.self) {
            _ = try task.decode(
                ModelOutput([
                    "scores": .dictionary([
                        "accepted": Double.greatestFiniteMagnitude,
                        "yes": Double.greatestFiniteMagnitude,
                    ])
                ])
            )
        }
    }

    @Test("Ranked prediction validates complete distributions and top-k counts")
    func rankingValidation() throws {
        let half = try ClassificationPrediction(label: "a", confidence: 0.5)
        let otherHalf = try ClassificationPrediction(label: "b", confidence: 0.5)

        #expect(throws: ML5Error.self) {
            _ = try RankedClassificationPrediction<String>(predictions: [])
        }
        #expect(throws: ML5Error.self) {
            _ = try RankedClassificationPrediction(
                predictions: [try ClassificationPrediction(label: "a")]
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try RankedClassificationPrediction(predictions: [half, half])
        }
        #expect(throws: ML5Error.self) {
            _ = try RankedClassificationPrediction(
                predictions: [
                    try ClassificationPrediction(label: "a", confidence: 0.4),
                    try ClassificationPrediction(label: "b", confidence: 0.6),
                ]
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try RankedClassificationPrediction(
                predictions: [
                    try ClassificationPrediction(label: "a", confidence: 0.6),
                    try ClassificationPrediction(label: "b", confidence: 0.2),
                ]
            )
        }

        let ranking = try RankedClassificationPrediction(predictions: [half, otherHalf])
        #expect(throws: ML5Error.self) {
            _ = try ranking.top(0)
        }
    }

    @Test("Regression vectors preserve order, Codable, and task validation")
    func regressionVectors() throws {
        let configuration = try RegressionVectorConfiguration(valueOutputs: ["x", "y"])
        let task = RegressionVectorTask(configuration: configuration)
        let prediction = try task.decode(
            ModelOutput([
                "x": .integer(2),
                "y": .number(3.5),
            ])
        )

        #expect(task.kind == .regression)
        #expect(prediction.values == [2, 3.5])
        #expect(prediction[1] == 3.5)
        #expect(
            try JSONDecoder().decode(
                RegressionVectorPrediction.self,
                from: JSONEncoder().encode(prediction)
            ) == prediction
        )
        #expect(
            try JSONDecoder().decode(
                RegressionVectorConfiguration.self,
                from: JSONEncoder().encode(configuration)
            ) == configuration
        )
        let scalar = try RegressionPrediction(value: 4)
        #expect(
            try JSONDecoder().decode(
                RegressionPrediction.self,
                from: JSONEncoder().encode(scalar)
            ) == scalar
        )
    }

    @Test("Regression vectors reject empty, nonfinite, duplicate, missing, and mistyped values")
    func regressionVectorFailures() throws {
        #expect(throws: ML5Error.self) {
            _ = try RegressionVectorPrediction(values: [])
        }
        #expect(throws: ML5Error.invalidRegressionValue(.infinity)) {
            _ = try RegressionVectorPrediction(values: [1, .infinity])
        }
        #expect(throws: ML5Error.self) {
            _ = try JSONDecoder().decode(
                RegressionVectorPrediction.self,
                from: Data("[]".utf8)
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try RegressionVectorConfiguration(valueOutputs: [])
        }
        #expect(throws: ML5Error.self) {
            _ = try JSONDecoder().decode(
                RegressionVectorConfiguration.self,
                from: Data("[]".utf8)
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try RegressionVectorConfiguration(valueOutputs: ["x", "x"])
        }
        #expect(throws: Error.self) {
            _ = try JSONDecoder().decode(RegressionPrediction.self, from: Data("1e999".utf8))
        }

        let task = RegressionVectorTask(
            configuration: try RegressionVectorConfiguration(valueOutputs: ["x", "y"])
        )
        #expect(throws: ML5Error.missingOutput(name: "y")) {
            _ = try task.decode(ModelOutput(["x": .number(1)]))
        }
        #expect(throws: ML5Error.self) {
            _ = try task.decode(ModelOutput(["x": .number(1), "y": .string("bad")]))
        }
    }

    @Test("Classification evaluation reports confusion, macro, and top-k metrics")
    func classificationEvaluation() throws {
        let rankings = try [
            ranking("cat", "dog"),
            ranking("dog", "cat"),
            ranking("dog", "cat"),
            ranking("fox", "bird"),
        ]
        let evaluation = try ClassificationEvaluation(
            expected: ["cat", "cat", "dog", "bird"],
            predictions: rankings
        )

        #expect(evaluation.sampleCount == 4)
        #expect(evaluation.correctCount == 2)
        #expect(evaluation.accuracy == 0.5)
        #expect(evaluation.confusionCount(expected: "cat", predicted: "dog") == 1)
        #expect(evaluation.confusionCount(expected: "bird", predicted: "cat") == 0)
        #expect(evaluation.perLabel["cat"]?.support == 2)
        #expect(evaluation.perLabel["bird"]?.precision == 0)
        #expect(evaluation.perLabel["fox"]?.recall == 0)
        #expect(abs(evaluation.macroPrecision - 0.375) < 1e-12)
        #expect(abs(evaluation.macroRecall - 0.375) < 1e-12)
        #expect(abs(evaluation.macroF1Score - 1.0 / 3.0) < 1e-12)
        #expect(try evaluation.topKAccuracy(1) == 0.5)
        #expect(try evaluation.topKAccuracy(2) == 1)
        #expect(throws: ML5Error.self) {
            _ = try evaluation.topKAccuracy(0)
        }
    }

    @Test("Classification evaluation rejects empty and mismatched collections")
    func classificationEvaluationFailures() throws {
        #expect(throws: ML5Error.self) {
            _ = try ClassificationEvaluation<String>(expected: [], predictions: [])
        }
        #expect(throws: ML5Error.self) {
            _ = try ClassificationEvaluation(expected: ["cat"], predictions: [])
        }
    }

    @Test("Regression evaluation supports scalar, vector, and constant targets")
    func regressionEvaluation() throws {
        let scalar = try RegressionEvaluation(
            expected: [RegressionPrediction(value: 2), RegressionPrediction(value: 2)],
            predictions: [RegressionPrediction(value: 1), RegressionPrediction(value: 3)]
        )
        #expect(scalar.sampleCount == 2)
        #expect(scalar.componentCount == 1)
        #expect(scalar.aggregate.meanAbsoluteError == 1)
        #expect(scalar.aggregate.meanSquaredError == 1)
        #expect(scalar.aggregate.rootMeanSquaredError == 1)
        #expect(scalar.aggregate.rSquared == nil)

        let vectors = try RegressionEvaluation(
            expected: [
                RegressionVectorPrediction(values: [1, 2]),
                RegressionVectorPrediction(values: [3, 4]),
            ],
            predictions: [
                RegressionVectorPrediction(values: [2, 2]),
                RegressionVectorPrediction(values: [2, 6]),
            ]
        )
        #expect(vectors.sampleCount == 2)
        #expect(vectors.componentCount == 2)
        #expect(vectors.aggregate.meanAbsoluteError == 1)
        #expect(vectors.aggregate.meanSquaredError == 1.5)
        #expect(abs(vectors.aggregate.rootMeanSquaredError - sqrt(1.5)) < 1e-12)
        #expect(abs((vectors.aggregate.rSquared ?? 0) - -0.2) < 1e-12)
        #expect(vectors.perComponent[0].rSquared == 0)
        #expect(vectors.perComponent[1].rSquared == -1)
    }

    @Test("Regression evaluation rejects empty, unequal, and inconsistent dimensions")
    func regressionEvaluationFailures() throws {
        #expect(throws: ML5Error.self) {
            _ = try RegressionEvaluation(expected: [RegressionPrediction](), predictions: [])
        }
        #expect(throws: ML5Error.self) {
            _ = try RegressionEvaluation(
                expected: [try RegressionPrediction(value: 1)],
                predictions: []
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try RegressionEvaluation(
                expected: [
                    try RegressionVectorPrediction(values: [1]),
                    try RegressionVectorPrediction(values: [1, 2]),
                ],
                predictions: [
                    try RegressionVectorPrediction(values: [1]),
                    try RegressionVectorPrediction(values: [1]),
                ]
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try RegressionEvaluation(
                expected: [
                    try RegressionVectorPrediction(values: [1]),
                    try RegressionVectorPrediction(values: [1]),
                ],
                predictions: [
                    try RegressionVectorPrediction(values: [1]),
                    try RegressionVectorPrediction(values: [1, 2]),
                ]
            )
        }
    }

    private func ranking(_ first: String, _ second: String) throws
        -> RankedClassificationPrediction<String>
    {
        try RankedClassificationPrediction(
            predictions: [
                ClassificationPrediction(label: first, confidence: 0.6),
                ClassificationPrediction(label: second, confidence: 0.4),
            ]
        )
    }
}
