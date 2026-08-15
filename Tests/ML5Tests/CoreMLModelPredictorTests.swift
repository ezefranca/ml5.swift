@preconcurrency import CoreML
import Foundation
import Testing

@testable import ML5

private enum SyntheticCoreMLError: LocalizedError {
    case expected

    var errorDescription: String? { "Synthetic Core ML failure." }
}

private final class MissingValueProvider: NSObject, MLFeatureProvider, @unchecked Sendable {
    var featureNames: Set<String> { ["missing"] }

    func featureValue(for _: String) -> MLFeatureValue? { nil }
}

@Suite("ML5 Core ML bridge")
struct CoreMLModelPredictorTests {
    @Test("A compiled Core ML model loads and predicts end to end")
    func compiledModelPrediction() async throws {
        let fixture = try CompiledIdentityModelFixture()
        defer { fixture.remove() }
        let features = try FeatureVector(["x": .number(2.5)])

        let predictor = try CoreMLModelPredictor(
            contentsOf: fixture.compiledModelURL,
            configuration: .init(computeUnits: .cpuOnly)
        )
        let directOutput = try await predictor.predict(features)
        #expect(directOutput["prediction"] == .number(2.5))

        let network = try NeuralNetwork(
            task: RegressionTask(
                configuration: RegressionConfiguration(valueOutput: "prediction")
            ),
            modelAt: fixture.compiledModelURL,
            configuration: .init(computeUnits: .cpuOnly)
        )
        let typedOutput = try await network.predict(features)
        #expect(typedOutput.value == 2.5)
    }

    @Test("Compute-unit configuration maps every value to Core ML")
    func computeUnits() throws {
        let mappings: [(ML5ComputeUnits, MLComputeUnits)] = [
            (.all, .all),
            (.cpuOnly, .cpuOnly),
            (.cpuAndGPU, .cpuAndGPU),
            (.cpuAndNeuralEngine, .cpuAndNeuralEngine),
        ]

        for (value, expected) in mappings {
            let configuration = CoreMLModelConfiguration(computeUnits: value)
            #expect(configuration.computeUnits == value)
            #expect(configuration.makeCoreMLConfiguration().computeUnits == expected)

            let decoded = try JSONDecoder().decode(
                ML5ComputeUnits.self,
                from: JSONEncoder().encode(value)
            )
            #expect(decoded == value)
        }

        #expect(CoreMLModelConfiguration().computeUnits == .all)
    }

    @Test("Scalar feature values convert to and from Core ML")
    func scalarConversions() throws {
        let number = try FeatureValue.number(1.25).makeCoreMLValue()
        let integer = try FeatureValue.integer(7).makeCoreMLValue()
        let string = try FeatureValue.string("swift").makeCoreMLValue()
        let boolean = try FeatureValue.boolean(true).makeCoreMLValue()
        let falseBoolean = try FeatureValue.boolean(false).makeCoreMLValue()

        #expect(number.doubleValue == 1.25)
        #expect(integer.int64Value == 7)
        #expect(string.stringValue == "swift")
        #expect(boolean.int64Value == 1)
        #expect(falseBoolean.int64Value == 0)
        #expect(try FeatureValue(coreMLValue: number, outputName: "number") == .number(1.25))
        #expect(try FeatureValue(coreMLValue: integer, outputName: "integer") == .integer(7))
        #expect(try FeatureValue(coreMLValue: string, outputName: "string") == .string("swift"))

        #expect(throws: ML5Error.invalidNumericValue(field: "feature")) {
            try FeatureValue.number(.infinity).makeCoreMLValue()
        }

        let undefined = MLFeatureValue(undefined: .double)
        #expect(throws: ML5Error.self) {
            _ = try FeatureValue(coreMLValue: undefined, outputName: "value")
        }
        let invalid = MLFeatureValue(undefined: .invalid)
        #expect(throws: ML5Error.self) {
            _ = try FeatureValue(coreMLValue: invalid, outputName: "value")
        }
        #expect(throws: ML5Error.self) {
            _ = try FeatureValue.requireCoreMLStorage(
                Optional<Int>.none,
                outputName: "value",
                type: "test"
            )
        }
    }

    @Test("Collections, tensors, and sequences bridge to and from Core ML")
    func structuredConversions() throws {
        let arrayValue = try FeatureValue.array([1, 2]).makeCoreMLValue()
        let dictionaryValue = try FeatureValue.dictionary(["yes": 0.75]).makeCoreMLValue()
        let tensor = try Tensor(shape: TensorShape([1, 2]), values: [3, 4])
        let tensorValue = try FeatureValue.tensor(tensor).makeCoreMLValue()
        let stringSequenceValue = try FeatureValue.sequence(.strings(["a", "b"]))
            .makeCoreMLValue()
        let integerSequenceValue = try FeatureValue.sequence(.integers([5, 6]))
            .makeCoreMLValue()

        #expect(arrayValue.multiArrayValue?.shape.map(\.intValue) == [2])
        #expect(arrayValue.multiArrayValue?[0].doubleValue == 1)
        #expect(dictionaryValue.dictionaryValue["yes"]?.doubleValue == 0.75)
        #expect(tensorValue.multiArrayValue?.shape.map(\.intValue) == [1, 2])
        #expect(tensorValue.multiArrayValue?[1].doubleValue == 4)
        #expect(stringSequenceValue.sequenceValue?.stringValues == ["a", "b"])
        #expect(integerSequenceValue.sequenceValue?.int64Values.map(\.int64Value) == [5, 6])

        #expect(
            try FeatureValue(coreMLValue: arrayValue, outputName: "array")
                == .tensor(try Tensor(shape: TensorShape([2]), values: [1, 2]))
        )
        #expect(
            try FeatureValue(coreMLValue: dictionaryValue, outputName: "dictionary")
                == .dictionary(["yes": 0.75])
        )
        #expect(
            try FeatureValue(coreMLValue: stringSequenceValue, outputName: "strings")
                == .sequence(.strings(["a", "b"]))
        )
        #expect(
            try FeatureValue(coreMLValue: integerSequenceValue, outputName: "integers")
                == .sequence(.integers([5, 6]))
        )

        let numericDictionary = try MLFeatureValue(dictionary: [NSNumber(value: 7): 1.0])
        #expect(throws: ML5Error.self) {
            _ = try FeatureValue(coreMLValue: numericDictionary, outputName: "dictionary")
        }
        let unsupportedSequence = MLFeatureValue(sequence: MLSequence(empty: .double))
        #expect(throws: ML5Error.self) {
            _ = try FeatureValue(coreMLValue: unsupportedSequence, outputName: "sequence")
        }
    }

    @Test("Image values bridge to and from Core ML")
    func imageConversions() throws {
        let image = try ML5Image(
            width: 2,
            height: 1,
            pixelFormat: .bgra8,
            data: Data([1, 2, 3, 4, 5, 6, 7, 8])
        )
        let coreMLValue = try FeatureValue.image(image).makeCoreMLValue()
        let decoded = try FeatureValue(coreMLValue: coreMLValue, outputName: "image")

        guard case let .image(decodedImage) = decoded else {
            Issue.record("Expected an image value.")
            return
        }
        #expect(decodedImage.width == image.width)
        #expect(decodedImage.height == image.height)
        #expect(decodedImage.pixelFormat == image.pixelFormat)
        #expect(decodedImage.data.prefix(image.data.count) == image.data)
    }

    @Test("A predictor converts inputs and supported scalar outputs")
    func predictionSuccess() async throws {
        let loader = CoreMLModelLoader { _, configuration in
            #expect(configuration.computeUnits == .cpuOnly)
            return CoreMLPredictionOperation { provider in
                #expect(provider.featureValue(for: "number")?.doubleValue == 1.5)
                #expect(provider.featureValue(for: "integer")?.int64Value == 2)
                #expect(provider.featureValue(for: "string")?.stringValue == "three")
                #expect(provider.featureValue(for: "boolean")?.int64Value == 1)
                return try MLDictionaryFeatureProvider(dictionary: [
                    "score": MLFeatureValue(double: 0.75),
                    "count": MLFeatureValue(int64: 4),
                    "label": MLFeatureValue(string: "accepted"),
                ])
            }
        }
        let predictor = try CoreMLModelPredictor(
            contentsOf: URL(fileURLWithPath: "/injected/model.mlmodelc"),
            configuration: .init(computeUnits: .cpuOnly),
            loader: loader
        )
        let features = try FeatureVector([
            "number": .number(1.5),
            "integer": .integer(2),
            "string": .string("three"),
            "boolean": .boolean(true),
        ])

        let output = try await predictor.predict(features)

        #expect(output["score"] == .number(0.75))
        #expect(output["count"] == .integer(4))
        #expect(output["label"] == .string("accepted"))
    }

    @Test("A missing provider value is ignored before output validation")
    func missingProviderValue() async throws {
        let predictor = try makePredictor { _ in MissingValueProvider() }
        let features = try FeatureVector(["x": .number(1)])

        await #expect(throws: ML5Error.emptyModelOutput) {
            try await predictor.predict(features)
        }
    }

    @Test("Core ML bridge preserves ML5 conversion failures")
    func conversionFailure() async throws {
        let predictor = try makePredictor { _ in
            try MLDictionaryFeatureProvider(dictionary: [
                "undefined": MLFeatureValue(undefined: .double)
            ])
        }
        let features = try FeatureVector(["x": .number(1)])

        await #expect(throws: ML5Error.self) {
            try await predictor.predict(features)
        }
    }

    @Test("Core ML bridge wraps framework prediction failures")
    func frameworkFailure() async throws {
        let predictor = try makePredictor { _ in
            throw SyntheticCoreMLError.expected
        }
        let features = try FeatureVector(["x": .number(1)])

        await #expect(
            throws: ML5Error.predictionFailed(
                message: "Synthetic Core ML failure."
            )
        ) {
            try await predictor.predict(features)
        }
    }

    @Test("Core ML bridge preserves cancellation failures")
    func cancellationFailure() async throws {
        let predictor = try makePredictor { _ in throw CancellationError() }
        let features = try FeatureVector(["x": .number(1)])

        await #expect(throws: CancellationError.self) {
            try await predictor.predict(features)
        }
    }

    @Test("Core ML bridge checks cancellation after framework prediction")
    func cancellationAfterPrediction() async throws {
        let predictor = try makePredictor { _ in
            withUnsafeCurrentTask { task in
                task?.cancel()
            }
            return try MLDictionaryFeatureProvider(dictionary: ["score": 1.0])
        }
        let features = try FeatureVector(["x": .number(1)])

        await #expect(throws: CancellationError.self) {
            try await predictor.predict(features)
        }
    }

    @Test("Core ML bridge checks cancellation before conversion")
    func cancellationBeforePrediction() async throws {
        let predictor = try makePredictor { _ in
            try MLDictionaryFeatureProvider(dictionary: ["score": 1.0])
        }
        let features = try FeatureVector(["x": .number(1)])
        let task = Task { try await predictor.predict(features) }
        task.cancel()

        await #expect(throws: CancellationError.self) {
            try await task.value
        }
    }

    private func makePredictor(
        operation: @escaping (any MLFeatureProvider) async throws -> any MLFeatureProvider
    ) throws -> CoreMLModelPredictor {
        try CoreMLModelPredictor(
            contentsOf: URL(fileURLWithPath: "/injected/model.mlmodelc"),
            loader: CoreMLModelLoader { _, _ in
                CoreMLPredictionOperation(predict: operation)
            }
        )
    }
}

private struct CompiledIdentityModelFixture {
    let directoryURL: URL
    let compiledModelURL: URL

    init() throws {
        // A 68-byte Core ML GLM regressor specification generated by this project.
        // It maps one Double input named `x` to an identical `prediction` output.
        let encodedModel =
            "CAESJwoHCgF4GgISAFIQCgpwcmVkaWN0aW9uGgISAFoKcHJlZGljdGlvbuISFgoKCggAAAAAAADwPxIIAAAAAAAAAAA="
        let modelData = try #require(Data(base64Encoded: encodedModel))
        let directoryURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("p5-swift-ml5-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directoryURL,
            withIntermediateDirectories: true
        )
        let sourceURL = directoryURL.appendingPathComponent("Identity.mlmodel")
        try modelData.write(to: sourceURL, options: .atomic)

        self.directoryURL = directoryURL
        self.compiledModelURL = try MLModel.compileModel(at: sourceURL)
    }

    func remove() {
        try? FileManager.default.removeItem(at: directoryURL)
        try? FileManager.default.removeItem(at: compiledModelURL)
    }
}
