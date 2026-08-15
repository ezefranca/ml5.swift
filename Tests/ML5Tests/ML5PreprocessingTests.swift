import Foundation
import Testing

@testable import ML5

private enum PreprocessingLabel: String, ClassificationLabel {
    case sample

    init?(ml5RawValue: String) {
        self.init(rawValue: ml5RawValue)
    }

    var ml5RawValue: String { rawValue }
}

@Suite("ML5 preprocessing")
struct ML5PreprocessingTests {
    @Test("Normalization ranges validate, hash, and serialize their bounds")
    func ranges() throws {
        let standard = try NormalizationRange()
        let custom = try NormalizationRange(lowerBound: -1, upperBound: 1)

        #expect(standard.lowerBound == 0)
        #expect(standard.upperBound == 1)
        #expect(Set([custom, custom]).count == 1)
        #expect(
            try JSONDecoder().decode(
                NormalizationRange.self,
                from: JSONEncoder().encode(custom)
            ) == custom
        )

        for (lower, upper) in [
            (Double.nan, 1),
            (0, Double.infinity),
            (1, 1),
            (2, 1),
        ] {
            #expect(throws: ML5Error.self) {
                _ = try NormalizationRange(lowerBound: lower, upperBound: upper)
            }
        }
    }

    @Test("Fitted statistics validate and expose population standard deviation")
    func statistics() throws {
        let statistics = try NumericFeatureStatistics(
            count: 4,
            minimum: 1,
            maximum: 7,
            mean: 4,
            variance: 5
        )

        #expect(statistics.standardDeviation == Double(5).squareRoot())
        #expect(
            try JSONDecoder().decode(
                NumericFeatureStatistics.self,
                from: JSONEncoder().encode(statistics)
            ) == statistics
        )

        let invalid: [(Int, Double, Double, Double, Double)] = [
            (0, 0, 0, 0, 0),
            (1, .nan, 0, 0, 0),
            (1, 0, .infinity, 0, 0),
            (1, 0, 0, .nan, 0),
            (1, 0, 0, 0, .infinity),
            (1, 2, 3, 1, 0),
            (1, 1, 2, 3, 0),
            (1, 0, 0, 0, -1),
        ]
        for value in invalid {
            #expect(throws: ML5Error.self) {
                _ = try NumericFeatureStatistics(
                    count: value.0,
                    minimum: value.1,
                    maximum: value.2,
                    mean: value.3,
                    variance: value.4
                )
            }
        }
        #expect(throws: ML5Error.self) {
            _ = try NumericFeatureStatistics.fit([], feature: "empty")
        }
    }

    @Test("Pipelines fit and reverse every supported numeric feature kind")
    func fitAndTransform() throws {
        let shape = try TensorShape([2])
        let first = try FeatureVector([
            "number": .number(1),
            "array": .array([1, 3]),
            "dictionary": .dictionary(["b": 3, "a": 1]),
            "tensor": .tensor(try Tensor(shape: shape, values: [1, 3])),
            "untouched": .string("first"),
        ])
        let second = try FeatureVector([
            "number": .number(3),
            "array": .array([5, 7]),
            "dictionary": .dictionary(["a": 5, "b": 7]),
            "tensor": .tensor(try Tensor(shape: shape, values: [5, 7])),
            "untouched": .string("second"),
        ])
        let signedRange = try NormalizationRange(lowerBound: -1, upperBound: 1)
        let pipeline = try FeaturePreprocessingPipeline.fit(
            samples: [first, second],
            rules: [
                FeatureNormalizationRule(feature: "number", strategy: .standardScore),
                FeatureNormalizationRule(feature: "array", strategy: .minMax(signedRange)),
                FeatureNormalizationRule(feature: "dictionary", strategy: .standardScore),
                FeatureNormalizationRule(
                    feature: "tensor",
                    strategy: .minMax(try NormalizationRange())
                ),
            ]
        )

        #expect(pipeline.normalizations.count == 4)
        #expect(pipeline.normalizations[0].statistics.count == 2)
        #expect(pipeline.normalizations[0].statistics.mean == 2)
        #expect(pipeline.normalizations[0].statistics.variance == 1)
        #expect(pipeline.normalizations[1].statistics.count == 4)
        #expect(pipeline.normalizations[1].statistics.mean == 4)
        #expect(pipeline.normalizations[1].statistics.variance == 5)

        let normalized = try pipeline.normalize(first)
        #expect(normalized["number"] == .number(-1))
        #expect(normalized["untouched"] == .string("first"))
        guard case let .array(array) = normalized["array"] else {
            Issue.record("Expected normalized array storage.")
            return
        }
        #expect(array[0] == -1)
        #expect(abs(array[1] - (-1.0 / 3.0)) < 1e-12)
        guard case let .dictionary(dictionary) = normalized["dictionary"] else {
            Issue.record("Expected normalized dictionary storage.")
            return
        }
        #expect(
            abs(dictionary["a", default: 0] - (-3 / Double(5).squareRoot())) < 1e-12
        )
        guard case let .tensor(tensor) = normalized["tensor"] else {
            Issue.record("Expected normalized tensor storage.")
            return
        }
        #expect(tensor.shape == shape)
        #expect(tensor.values[0] == 0)
        #expect(abs(tensor.values[1] - (1.0 / 3.0)) < 1e-12)

        let restored = try pipeline.denormalize(normalized)
        for name: FeatureName in ["number", "array", "dictionary", "tensor"] {
            let expectedValue = try #require(first[name])
            let actualValue = try #require(restored[name])
            let expected = try FeaturePreprocessingPipeline.components(
                of: expectedValue,
                feature: name
            )
            let actual = try FeaturePreprocessingPipeline.components(
                of: actualValue,
                feature: name
            )
            #expect(zip(expected, actual).allSatisfy { abs($0 - $1) < 1e-12 })
        }

        let decoded = try JSONDecoder().decode(
            FeaturePreprocessingPipeline.self,
            from: JSONEncoder().encode(pipeline)
        )
        #expect(decoded == pipeline)
        #expect(Set([pipeline, decoded]).count == 1)
    }

    @Test("Constant features use reversible deterministic representatives")
    func constantFeatures() throws {
        let sample = try FeatureVector(["value": .number(5)])
        let standard = try FeaturePreprocessingPipeline.fit(
            samples: [sample],
            rules: [FeatureNormalizationRule(feature: "value", strategy: .standardScore)]
        )
        let range = try NormalizationRange(lowerBound: 2, upperBound: 4)
        let minMax = try FeaturePreprocessingPipeline.fit(
            samples: [sample],
            rules: [FeatureNormalizationRule(feature: "value", strategy: .minMax(range))]
        )

        let standardValue = try standard.normalize(sample)
        #expect(standardValue["value"] == .number(0))
        #expect(try standard.denormalize(standardValue) == sample)

        let minMaxValue = try minMax.normalize(sample)
        #expect(minMaxValue["value"] == .number(3))
        #expect(try minMax.denormalize(minMaxValue) == sample)
    }

    @Test("Classification and regression adapters preserve outcomes")
    func sampleAdapters() throws {
        let first = try FeatureVector(["value": .number(1)])
        let second = try FeatureVector(["value": .number(3)])
        let rule = FeatureNormalizationRule(feature: "value", strategy: .standardScore)
        let classifications = [
            ClassificationSample(features: first, label: PreprocessingLabel.sample),
            ClassificationSample(features: second, label: PreprocessingLabel.sample),
        ]
        let regressions = [
            try RegressionSample(features: first, target: 10),
            try RegressionSample(features: second, target: 20),
        ]

        let classificationPipeline = try FeaturePreprocessingPipeline.fit(
            samples: classifications,
            rules: [rule]
        )
        let regressionPipeline = try FeaturePreprocessingPipeline.fit(
            samples: regressions,
            rules: [rule]
        )
        let normalizedClassification = try classificationPipeline.normalize(classifications[0])
        let normalizedRegression = try regressionPipeline.normalize(regressions[1])

        #expect(normalizedClassification.features["value"] == .number(-1))
        #expect(normalizedClassification.label == .sample)
        #expect(normalizedRegression.features["value"] == .number(1))
        #expect(normalizedRegression.target == 20)
    }

    @Test("Fitting rejects absent, duplicate, inconsistent, and unsupported data")
    func fittingFailures() throws {
        let number = try FeatureVector(["value": .number(1)])
        let array = try FeatureVector(["value": .array([1])])
        let missing = try FeatureVector(["other": .number(1)])
        let integer = try FeatureVector(["value": .integer(1)])
        let rule = FeatureNormalizationRule(feature: "value", strategy: .standardScore)

        #expect(throws: ML5Error.invalidTrainingSamples) {
            _ = try FeaturePreprocessingPipeline.fit(
                samples: [FeatureVector](),
                rules: [rule]
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try FeaturePreprocessingPipeline.fit(samples: [number], rules: [])
        }
        #expect(throws: ML5Error.self) {
            _ = try FeaturePreprocessingPipeline.fit(samples: [number], rules: [rule, rule])
        }
        #expect(throws: ML5Error.missingFeature("value")) {
            _ = try FeaturePreprocessingPipeline.fit(samples: [missing], rules: [rule])
        }
        #expect(throws: ML5Error.self) {
            _ = try FeaturePreprocessingPipeline.fit(samples: [number, missing], rules: [rule])
        }
        #expect(throws: ML5Error.self) {
            _ = try FeaturePreprocessingPipeline.fit(samples: [number, array], rules: [rule])
        }
        #expect(throws: ML5Error.self) {
            _ = try FeaturePreprocessingPipeline.fit(samples: [integer], rules: [rule])
        }

        let extreme = [
            try FeatureVector(["value": .number(-Double.greatestFiniteMagnitude)]),
            try FeatureVector(["value": .number(Double.greatestFiniteMagnitude)]),
        ]
        #expect(throws: ML5Error.self) {
            _ = try FeaturePreprocessingPipeline.fit(samples: extreme, rules: [rule])
        }
    }

    @Test("Pipeline construction and application reject invalid state")
    func pipelineFailures() throws {
        let statistics = try NumericFeatureStatistics(
            count: 1,
            minimum: 0,
            maximum: 0,
            mean: 0,
            variance: 0
        )
        let normalization = try FittedFeatureNormalization(
            feature: "value",
            valueKind: .number,
            strategy: .standardScore,
            statistics: statistics
        )

        #expect(throws: ML5Error.self) {
            _ = try FittedFeatureNormalization(
                feature: "value",
                valueKind: .string,
                strategy: .standardScore,
                statistics: statistics
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try FeaturePreprocessingPipeline(normalizations: [])
        }
        #expect(throws: ML5Error.self) {
            _ = try FeaturePreprocessingPipeline(
                normalizations: [normalization, normalization]
            )
        }

        let pipeline = try FeaturePreprocessingPipeline(normalizations: [normalization])
        #expect(throws: ML5Error.missingFeature("value")) {
            try pipeline.normalize(FeatureVector(["other": .number(1)]))
        }
        #expect(throws: ML5Error.self) {
            try pipeline.normalize(FeatureVector(["value": .array([1])]))
        }
        #expect(throws: ML5Error.self) {
            _ = try FeaturePreprocessingPipeline.transform(
                .integer(1),
                using: normalization,
                direction: .normalize
            )
        }

        let encodedDuplicates = try JSONEncoder().encode([normalization, normalization])
        #expect(throws: ML5Error.self) {
            _ = try JSONDecoder().decode(
                FeaturePreprocessingPipeline.self,
                from: encodedDuplicates
            )
        }
    }

    @Test("Normalization and denormalization detect arithmetic overflow")
    func transformOverflow() throws {
        let maximum = Double.greatestFiniteMagnitude
        let normalizeStatistics = try NumericFeatureStatistics(
            count: 1,
            minimum: -maximum,
            maximum: -maximum,
            mean: -maximum,
            variance: Double.leastNonzeroMagnitude
        )
        let normalizeStage = try FittedFeatureNormalization(
            feature: "value",
            valueKind: .number,
            strategy: .standardScore,
            statistics: normalizeStatistics
        )
        let normalizePipeline = try FeaturePreprocessingPipeline(
            normalizations: [normalizeStage]
        )
        #expect(throws: ML5Error.self) {
            try normalizePipeline.normalize(FeatureVector(["value": .number(maximum)]))
        }

        let denormalizeStatistics = try NumericFeatureStatistics(
            count: 1,
            minimum: 0,
            maximum: 0,
            mean: 0,
            variance: maximum
        )
        let denormalizeStage = try FittedFeatureNormalization(
            feature: "value",
            valueKind: .number,
            strategy: .standardScore,
            statistics: denormalizeStatistics
        )
        let denormalizePipeline = try FeaturePreprocessingPipeline(
            normalizations: [denormalizeStage]
        )
        #expect(throws: ML5Error.self) {
            try denormalizePipeline.denormalize(
                FeatureVector(["value": .number(maximum)])
            )
        }
    }
}
