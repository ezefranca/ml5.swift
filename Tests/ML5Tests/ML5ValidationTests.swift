import Foundation
import Testing

@testable import ML5

private enum RestrictedLabel: String, ClassificationLabel {
    case accepted

    init?(ml5RawValue: String) {
        self.init(rawValue: ml5RawValue)
    }

    var ml5RawValue: String { rawValue }
}

private actor ThrowingPredictor: ModelPredicting {
    func predict(_: FeatureVector) async throws -> ModelOutput {
        throw ML5Error.predictionFailed(message: "Synthetic predictor failure.")
    }
}

private actor SelfCancellingPredictor: ModelPredicting {
    let output: ModelOutput

    init(output: ModelOutput) {
        self.output = output
    }

    func predict(_: FeatureVector) async throws -> ModelOutput {
        withUnsafeCurrentTask { task in
            task?.cancel()
        }
        return output
    }
}

@Suite("ML5 values and validation")
struct ML5ValidationTests {
    @Test("Localized errors describe every failure case")
    func localizedErrors() {
        let errors: [ML5Error] = [
            .invalidFieldName(" bad "),
            .emptyFeatureVector,
            .emptyModelOutput,
            .invalidNumericValue(field: "x"),
            .invalidTensorShape([0]),
            .invalidTensorElementCount(expected: 2, actual: 1),
            .emptyCollection(field: "array"),
            .invalidDictionaryKey(field: "probabilities"),
            .invalidImage(reason: "bad pixels"),
            .duplicateFeatureName("x"),
            .missingFeature("x"),
            .unexpectedFeature("y"),
            .featureKindMismatch(name: "x", expected: .number, actual: .string),
            .tensorShapeMismatch(name: "x", expected: [2], actual: [1, 2]),
            .invalidDatasetSampleID(0),
            .invalidDatasetSnapshot(reason: "duplicate identifier"),
            .datasetIdentifierExhausted,
            .invalidDatasetSplit(reason: "bad fractions"),
            .invalidNormalization(reason: "bad statistics"),
            .invalidConfiguration(reason: "duplicate output"),
            .invalidTrainingSamples,
            .missingOutput(name: "label"),
            .unexpectedOutputType(name: "label", expected: .string, actual: .number),
            .invalidClassLabel("unknown"),
            .invalidConfidence(2),
            .invalidRegressionValue(.infinity),
            .unsupportedOperation(.onDeviceTraining),
            .modelLoadingFailed(path: "/missing.mlmodelc", message: "missing"),
            .predictionFailed(message: "failed"),
        ]

        for error in errors {
            #expect(error.errorDescription?.isEmpty == false)
        }
        #expect(UnsupportedOperation.onDeviceTraining.description.contains("not supported"))
    }

    @Test("Feature and output names support validated construction and Codable")
    func names() throws {
        let featureText = "temperature"
        let outputText = "label"
        let feature = try FeatureName(featureText)
        let rawFeature = FeatureName(rawValue: "humidity")
        let literalFeature: FeatureName = "pressure"
        let output = try OutputName(outputText)
        let rawOutput = OutputName(rawValue: "confidence")
        let literalOutput: OutputName = "score"

        #expect(feature.rawValue == "temperature")
        #expect(rawFeature.rawValue == "humidity")
        #expect(literalFeature.rawValue == "pressure")
        #expect(output.rawValue == "label")
        #expect(rawOutput.rawValue == "confidence")
        #expect(literalOutput.rawValue == "score")
        #expect(
            try JSONDecoder().decode(FeatureName.self, from: JSONEncoder().encode(feature))
                == feature)
        #expect(
            try JSONDecoder().decode(OutputName.self, from: JSONEncoder().encode(output)) == output)
    }

    @Test(
        "Both name types reject empty and untrimmed text",
        arguments: ["", " leading", "trailing ", "\nname"]
    )
    func invalidNames(_ value: String) {
        #expect(throws: ML5Error.invalidFieldName(value)) {
            _ = try FeatureName(value)
        }
        #expect(throws: ML5Error.invalidFieldName(value)) {
            _ = try OutputName(value)
        }
    }

    @Test("Feature values expose kinds, numeric projections, and Codable")
    func featureValues() throws {
        let values: [FeatureValue] = [
            .number(1.5),
            .integer(2),
            .string("three"),
            .boolean(true),
        ]

        #expect(values.map(\.kind) == [.number, .integer, .string, .boolean])
        #expect(values[0].numericValue == 1.5)
        #expect(values[1].numericValue == 2)
        #expect(values[2].numericValue == nil)
        #expect(values[3].numericValue == nil)

        for value in values {
            let decoded = try JSONDecoder().decode(
                FeatureValue.self,
                from: JSONEncoder().encode(value)
            )
            #expect(decoded == value)
        }
    }

    @Test("Feature vectors require finite, nonempty values")
    func featureVectors() throws {
        #expect(throws: ML5Error.emptyFeatureVector) {
            _ = try FeatureVector([:])
        }
        #expect(throws: ML5Error.invalidNumericValue(field: "x")) {
            _ = try FeatureVector(["x": .number(.nan)])
        }

        let vector = try FeatureVector([
            "x": .number(1),
            "enabled": .boolean(true),
        ])
        #expect(vector["x"] == .number(1))
        #expect(vector["missing"] == nil)
    }

    @Test("Model outputs require finite, nonempty values")
    func modelOutputs() throws {
        #expect(throws: ML5Error.emptyModelOutput) {
            _ = try ModelOutput([:])
        }
        #expect(throws: ML5Error.invalidNumericValue(field: "score")) {
            _ = try ModelOutput(["score": .number(.infinity)])
        }

        let output = try ModelOutput([
            "score": .integer(7),
            "label": .string("accepted"),
        ])
        #expect(output["score"] == .integer(7))
        #expect(output["missing"] == nil)
    }

    @Test("Predictions validate confidence and finite regression values")
    func predictionValidation() throws {
        #expect(try ClassificationPrediction(label: RestrictedLabel.accepted).confidence == nil)
        #expect(
            try ClassificationPrediction(label: RestrictedLabel.accepted, confidence: 0).confidence
                == 0)
        #expect(
            try ClassificationPrediction(label: RestrictedLabel.accepted, confidence: 1).confidence
                == 1)

        for confidence in [-0.1, 1.1, Double.infinity, Double.nan] {
            #expect(throws: ML5Error.self) {
                _ = try ClassificationPrediction(
                    label: RestrictedLabel.accepted,
                    confidence: confidence
                )
            }
        }

        #expect(try RegressionPrediction(value: 3.5).value == 3.5)
        #expect(throws: ML5Error.self) {
            _ = try RegressionPrediction(value: .nan)
        }
    }

    @Test("String labels preserve their raw representation")
    func stringLabels() throws {
        let label = try #require(String(ml5RawValue: "swift"))
        #expect(label.ml5RawValue == "swift")
    }

    @Test("Classification configuration rejects duplicate output names")
    func classificationConfiguration() throws {
        #expect(
            throws: ML5Error.invalidConfiguration(
                reason: "A classification label and confidence cannot use the same output."
            )
        ) {
            _ = try ClassificationConfiguration(
                labelOutput: "result",
                confidenceOutput: "result"
            )
        }

        let configuration = try ClassificationConfiguration(
            labelOutput: "label",
            confidenceOutput: "confidence"
        )
        #expect(configuration.labelOutput == "label")
        #expect(configuration.confidenceOutput == "confidence")
    }

    @Test("Classification tasks decode valid optional confidence")
    func classificationSuccess() throws {
        let withoutConfidence = ClassificationTask<RestrictedLabel>(
            configuration: try ClassificationConfiguration(labelOutput: "label")
        )
        #expect(withoutConfidence.kind == .classification)
        let first = try withoutConfidence.decode(
            ModelOutput(["label": .string("accepted")])
        )
        #expect(first.label == .accepted)
        #expect(first.confidence == nil)

        let withConfidence = ClassificationTask<RestrictedLabel>(
            configuration: try ClassificationConfiguration(
                labelOutput: "label",
                confidenceOutput: "confidence"
            )
        )
        let second = try withConfidence.decode(
            ModelOutput([
                "label": .string("accepted"),
                "confidence": .integer(1),
            ])
        )
        #expect(second.confidence == 1)
    }

    @Test("Classification tasks report every malformed output")
    func classificationFailures() throws {
        let simple = ClassificationTask<RestrictedLabel>(
            configuration: try ClassificationConfiguration(labelOutput: "label")
        )
        #expect(throws: ML5Error.missingOutput(name: "label")) {
            try simple.decode(ModelOutput(["other": .string("accepted")]))
        }
        #expect(
            throws: ML5Error.unexpectedOutputType(
                name: "label",
                expected: .string,
                actual: .number
            )
        ) {
            try simple.decode(ModelOutput(["label": .number(1)]))
        }
        #expect(throws: ML5Error.invalidClassLabel("rejected")) {
            try simple.decode(ModelOutput(["label": .string("rejected")]))
        }

        let confidence = ClassificationTask<RestrictedLabel>(
            configuration: try ClassificationConfiguration(
                labelOutput: "label",
                confidenceOutput: "confidence"
            )
        )
        #expect(throws: ML5Error.missingOutput(name: "confidence")) {
            try confidence.decode(ModelOutput(["label": .string("accepted")]))
        }
        #expect(
            throws: ML5Error.unexpectedOutputType(
                name: "confidence",
                expected: .number,
                actual: .boolean
            )
        ) {
            try confidence.decode(
                ModelOutput([
                    "label": .string("accepted"),
                    "confidence": .boolean(true),
                ]))
        }
    }

    @Test("Regression tasks decode numeric outputs and reject malformed output")
    func regressionTask() throws {
        let configuration = RegressionConfiguration(valueOutput: "estimate")
        let task = RegressionTask(configuration: configuration)
        #expect(configuration.valueOutput == "estimate")
        #expect(task.configuration == configuration)
        #expect(task.kind == .regression)
        #expect(try task.decode(ModelOutput(["estimate": .integer(4)])).value == 4)

        #expect(throws: ML5Error.missingOutput(name: "estimate")) {
            try task.decode(ModelOutput(["other": .number(1)]))
        }
        #expect(
            throws: ML5Error.unexpectedOutputType(
                name: "estimate",
                expected: .number,
                actual: .string
            )
        ) {
            try task.decode(ModelOutput(["estimate": .string("four")]))
        }
    }

    @Test("Training sample values preserve labels and finite targets")
    func samples() throws {
        let features = try FeatureVector(["x": .number(1)])
        let classification = ClassificationSample(
            features: features,
            label: RestrictedLabel.accepted
        )
        let regression = try RegressionSample(features: features, target: 2)

        #expect(classification.features == features)
        #expect(classification.label == .accepted)
        #expect(regression.features == features)
        #expect(regression.target == 2)
        #expect(throws: ML5Error.self) {
            _ = try RegressionSample(features: features, target: .infinity)
        }
    }

    @Test("Networks preserve backend errors and training validation")
    func networkFailures() async throws {
        let task = RegressionTask(
            configuration: RegressionConfiguration(valueOutput: "estimate")
        )
        let network = NeuralNetwork(task: task, predictor: ThrowingPredictor())
        let features = try FeatureVector(["x": .number(1)])

        await #expect(
            throws: ML5Error.predictionFailed(
                message: "Synthetic predictor failure."
            )
        ) {
            try await network.predict(features)
        }
        await #expect(throws: ML5Error.invalidTrainingSamples) {
            try await network.train([Int]())
        }
    }

    @Test("Networks check cancellation before and after backend prediction")
    func networkCancellation() async throws {
        let task = RegressionTask(
            configuration: RegressionConfiguration(valueOutput: "estimate")
        )
        let features = try FeatureVector(["x": .number(1)])
        let output = try ModelOutput(["estimate": .number(2)])
        let network = NeuralNetwork(
            task: task,
            predictor: SelfCancellingPredictor(output: output)
        )

        await #expect(throws: CancellationError.self) {
            try await network.predict(features)
        }

        let preCancelled = Task {
            try await NeuralNetwork(task: task, predictor: ThrowingPredictor())
                .predict(features)
        }
        preCancelled.cancel()
        await #expect(throws: CancellationError.self) {
            try await preCancelled.value
        }

        let cancelledTraining = Task {
            try await NeuralNetwork(task: task, predictor: ThrowingPredictor())
                .train([1])
        }
        cancelledTraining.cancel()
        await #expect(throws: CancellationError.self) {
            try await cancelledTraining.value
        }
    }

    @Test("A network reports failure while loading a missing Core ML model")
    func networkModelLoadingFailure() {
        let task = RegressionTask(
            configuration: RegressionConfiguration(valueOutput: "estimate")
        )
        #expect(throws: ML5Error.self) {
            _ = try NeuralNetwork(
                task: task,
                modelAt: URL(fileURLWithPath: "/definitely/missing/model.mlmodelc"),
                configuration: CoreMLModelConfiguration(computeUnits: .cpuOnly)
            )
        }
    }
}
