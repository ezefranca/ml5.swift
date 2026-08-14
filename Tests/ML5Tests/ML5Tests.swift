import ML5
import Testing

private enum Animal: String, ClassificationLabel {
    case cat
    case dog

    init?(ml5RawValue: String) {
        self.init(rawValue: ml5RawValue)
    }

    var ml5RawValue: String {
        rawValue
    }
}

private actor StubPredictor: ModelPredicting {
    private let result: Result<ModelOutput, ML5Error>

    init(result: Result<ModelOutput, ML5Error>) {
        self.result = result
    }

    func predict(_: FeatureVector) async throws -> ModelOutput {
        try Task.checkCancellation()
        return try result.get()
    }
}

struct ML5Tests {
    @Test(arguments: ["", " label", "label ", "\nlabel"])
    func fieldNamesRejectEmptyAndUntrimmedValues(_ value: String) {
        do {
            _ = try FeatureName(value)
            Issue.record("Expected an invalid field name error.")
        } catch let error as ML5Error {
            #expect(error == .invalidFieldName(value))
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test
    func featureVectorsRejectNonFiniteNumbers() {
        do {
            _ = try FeatureVector(["x": .number(.infinity)])
            Issue.record("Expected non-finite feature validation to fail.")
        } catch let error as ML5Error {
            #expect(error == .invalidNumericValue(field: "x"))
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test
    func classificationDecodesLabelAndConfidence() throws {
        let task = ClassificationTask<Animal>(
            configuration: try ClassificationConfiguration(
                labelOutput: "label",
                confidenceOutput: "confidence"
            )
        )
        let output = try ModelOutput([
            "label": .string("cat"),
            "confidence": .number(0.875),
        ])

        let prediction = try task.decode(output)

        #expect(prediction.label == .cat)
        #expect(prediction.confidence == 0.875)
    }

    @Test
    func regressionRequiresNumericOutput() throws {
        let task = RegressionTask(configuration: RegressionConfiguration(valueOutput: "estimate"))
        let output = try ModelOutput(["estimate": .string("not a number")])

        do {
            _ = try task.decode(output)
            Issue.record("Expected a type mismatch.")
        } catch let error as ML5Error {
            #expect(
                error == .unexpectedOutputType(
                    name: "estimate",
                    expected: .number,
                    actual: .string
                )
            )
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test
    func neuralNetworkPredictsThroughInjectedBackend() async throws {
        let output = try ModelOutput([
            "label": .string("dog"),
            "confidence": .number(0.75),
        ])
        let predictor = StubPredictor(result: .success(output))
        let task = ClassificationTask<Animal>(
            configuration: try ClassificationConfiguration(
                labelOutput: "label",
                confidenceOutput: "confidence"
            )
        )
        let network = NeuralNetwork(task: task, predictor: predictor)
        let features = try FeatureVector(["distance": .number(4.0)])

        let prediction = try await network.predict(features)

        #expect(prediction.label == .dog)
        #expect(prediction.confidence == 0.75)
    }

    @Test
    func trainingReportsCoreMLLimitation() async throws {
        let output = try ModelOutput(["estimate": .number(1)])
        let network = NeuralNetwork(
            task: RegressionTask(configuration: RegressionConfiguration(valueOutput: "estimate")),
            predictor: StubPredictor(result: .success(output))
        )
        let sample = try RegressionSample(
            features: FeatureVector(["x": .number(1)]),
            target: 2
        )

        do {
            try await network.train([sample])
            Issue.record("Expected arbitrary on-device training to be unsupported.")
        } catch let error as ML5Error {
            #expect(error == .unsupportedOperation(.onDeviceTraining))
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }
}
