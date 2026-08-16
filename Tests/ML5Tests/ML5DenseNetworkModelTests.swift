import Foundation
import Testing

@testable import ML5

@Suite("ML5 immutable dense models")
struct ML5DenseNetworkModelTests {
    @Test("Layer parameters validate dimensions, storage, and finiteness")
    func layerParameterValidation() throws {
        #expect(throws: ML5Error.self) {
            _ = try DenseLayerParameters(inputCount: 0, outputCount: 1, weights: [], biases: [0])
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseLayerParameters(inputCount: 1, outputCount: 0, weights: [], biases: [])
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseLayerParameters(
                inputCount: Int.max,
                outputCount: 2,
                weights: [],
                biases: [0, 0]
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseLayerParameters(
                inputCount: 2, outputCount: 2, weights: [1], biases: [0, 0])
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseLayerParameters(
                inputCount: 1,
                outputCount: 2,
                weights: [1, 2],
                biases: [0]
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseLayerParameters(
                inputCount: 1,
                outputCount: 1,
                weights: [.infinity],
                biases: [0]
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseLayerParameters(
                inputCount: 1,
                outputCount: 1,
                weights: [0],
                biases: [.nan]
            )
        }
    }

    @Test("Linear inference preserves configured feature and output order")
    func linearInference() async throws {
        let configuration = try DenseNetworkConfiguration(
            inputFeatures: ["x", "y"],
            outputNames: ["sum", "difference"]
        )
        let layer = try DenseLayerParameters(
            inputCount: 2,
            outputCount: 2,
            weights: [1, 1, 1, -1],
            biases: [0.5, -0.5]
        )
        let model = try DenseNetworkModel(configuration: configuration, layers: [layer])
        let features = try FeatureVector(["y": .integer(2), "x": .number(5)])
        let output = try await model.predict(features)

        #expect(output["sum"] == .number(7.5))
        #expect(output["difference"] == .number(2.5))
        #expect(model.configuration == configuration)
        #expect(model.layers == [layer])
    }

    @Test("Every activation has stable finite forward semantics")
    func activations() async throws {
        let rectified = try await oneLayerModel(
            outputActivation: .rectifiedLinear,
            loss: .meanSquaredError,
            weights: [1, -1],
            biases: [-2, -2]
        ).predict(FeatureVector(["x": .number(1)]))
        #expect(rectified["a"] == .number(0))
        #expect(rectified["b"] == .number(0))

        let sigmoid = try await oneLayerModel(
            outputActivation: .sigmoid,
            loss: .binaryCrossEntropy,
            weights: [1, -1],
            biases: [1_000, -1_000]
        ).predict(FeatureVector(["x": .number(1)]))
        #expect(sigmoid["a"] == .number(1))
        #expect(sigmoid["b"] == .number(0))

        let hyperbolic = try await oneLayerModel(
            outputActivation: .hyperbolicTangent,
            loss: .meanSquaredError,
            weights: [0, 0],
            biases: [1, -1]
        ).predict(FeatureVector(["x": .number(0)]))
        #expect(hyperbolic["a"] == .number(tanh(1)))
        #expect(hyperbolic["b"] == .number(tanh(-1)))

        let softmax = try await oneLayerModel(
            outputActivation: .softmax,
            loss: .categoricalCrossEntropy,
            weights: [1, -1],
            biases: [1_000, 1_000]
        ).predict(FeatureVector(["x": .number(1)]))
        let a = try #require(softmax["a"]?.numericValue)
        let b = try #require(softmax["b"]?.numericValue)
        #expect(a > b)
        #expect(abs(a + b - 1) < 1e-12)
    }

    @Test("Hidden layers feed their configured activation into the output layer")
    func hiddenLayer() async throws {
        let configuration = try DenseNetworkConfiguration(
            inputFeatures: ["x", "y"],
            outputNames: ["value"],
            hiddenLayers: [try DenseLayerConfiguration(neuronCount: 2)],
            outputActivation: .linear
        )
        let hidden = try DenseLayerParameters(
            inputCount: 2,
            outputCount: 2,
            weights: [1, 0, 0, 1],
            biases: [-2, 1]
        )
        let output = try DenseLayerParameters(
            inputCount: 2,
            outputCount: 1,
            weights: [3, 4],
            biases: [5]
        )
        let model = try DenseNetworkModel(configuration: configuration, layers: [hidden, output])
        let result = try await model.predict(FeatureVector(["x": .number(1), "y": .number(2)]))

        #expect(result["value"] == .number(17))
    }

    @Test("Models reject missing and nonnumeric configured inputs")
    func invalidInputs() async throws {
        let model = try oneLayerModel(
            outputActivation: .linear,
            loss: .meanSquaredError,
            weights: [1, 1],
            biases: [0, 0]
        )
        await #expect(throws: ML5Error.missingFeature("x")) {
            _ = try await model.predict(FeatureVector(["other": .number(1)]))
        }
        await #expect(throws: ML5Error.self) {
            _ = try await model.predict(FeatureVector(["x": .string("bad")]))
        }
    }

    @Test("Model construction rejects layer-count and connectivity mismatches")
    func modelValidation() throws {
        let configuration = try DenseNetworkConfiguration(
            inputFeatures: ["x"],
            outputNames: ["y"],
            hiddenLayers: [try DenseLayerConfiguration(neuronCount: 2)]
        )
        let oneByOne = try DenseLayerParameters(
            inputCount: 1,
            outputCount: 1,
            weights: [1],
            biases: [0]
        )
        #expect(throws: ML5Error.self) {
            _ = try DenseNetworkModel(configuration: configuration, layers: [oneByOne])
        }

        let wrongInput = try DenseLayerParameters(
            inputCount: 2,
            outputCount: 2,
            weights: [1, 0, 0, 1],
            biases: [0, 0]
        )
        let output = try DenseLayerParameters(
            inputCount: 2,
            outputCount: 1,
            weights: [1, 1],
            biases: [0]
        )
        #expect(throws: ML5Error.self) {
            _ = try DenseNetworkModel(configuration: configuration, layers: [wrongInput, output])
        }

        let wrongHiddenWidth = try DenseLayerParameters(
            inputCount: 1,
            outputCount: 1,
            weights: [1],
            biases: [0]
        )
        #expect(throws: ML5Error.self) {
            _ = try DenseNetworkModel(
                configuration: configuration,
                layers: [wrongHiddenWidth, oneByOne]
            )
        }

        let hidden = try DenseLayerParameters(
            inputCount: 1,
            outputCount: 2,
            weights: [1, 1],
            biases: [0, 0]
        )
        let disconnectedOutput = try DenseLayerParameters(
            inputCount: 3,
            outputCount: 1,
            weights: [1, 1, 1],
            biases: [0]
        )
        #expect(throws: ML5Error.self) {
            _ = try DenseNetworkModel(
                configuration: configuration,
                layers: [hidden, disconnectedOutput]
            )
        }

        let wrongOutputWidth = try DenseLayerParameters(
            inputCount: 2,
            outputCount: 2,
            weights: [1, 0, 0, 1],
            biases: [0, 0]
        )
        #expect(throws: ML5Error.self) {
            _ = try DenseNetworkModel(
                configuration: configuration,
                layers: [hidden, wrongOutputWidth]
            )
        }
    }

    @Test("Models and layers use validated Codable restoration")
    func codable() throws {
        let model = try oneLayerModel(
            outputActivation: .linear,
            loss: .meanSquaredError,
            weights: [2, 3],
            biases: [4, 5]
        )
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        let data = try encoder.encode(model)

        #expect(try decoder.decode(DenseNetworkModel.self, from: data) == model)
        #expect(
            try decoder.decode(
                DenseLayerParameters.self,
                from: encoder.encode(model.layers[0])
            ) == model.layers[0]
        )

        var object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        object["layers"] = []
        #expect(throws: ML5Error.self) {
            _ = try decoder.decode(
                DenseNetworkModel.self,
                from: JSONSerialization.data(withJSONObject: object)
            )
        }
    }

    @Test("Async batches and synchronous snapshots preserve immutable results")
    func batchAndSnapshot() async throws {
        let model = try oneLayerModel(
            outputActivation: .linear,
            loss: .meanSquaredError,
            weights: [1, 2],
            biases: [0, 0]
        )
        let features = try [
            FeatureVector(["x": .number(1)]),
            FeatureVector(["x": .number(2)]),
        ]
        let batch = try await model.predict(features)
        let snapshot = try await model.makeInferenceSnapshot()

        #expect(batch.map { $0["a"] } == [.number(1), .number(2)])
        #expect(try snapshot.predict(features) == batch)
        #expect(try snapshot.predict(features[0]) == batch[0])
    }

    @Test("Dense model operations cooperate with preexisting cancellation")
    func cancellation() async throws {
        let model = try oneLayerModel(
            outputActivation: .linear,
            loss: .meanSquaredError,
            weights: [1, 1],
            biases: [0, 0]
        )
        let features = try FeatureVector(["x": .number(1)])

        let prediction = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await model.predict(features)
        }
        await #expect(throws: CancellationError.self) {
            _ = try await prediction.value
        }

        let snapshot = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await model.makeInferenceSnapshot()
        }
        await #expect(throws: CancellationError.self) {
            _ = try await snapshot.value
        }
    }

    private func oneLayerModel(
        outputActivation: ActivationFunction,
        loss: TrainingLoss,
        weights: [Double],
        biases: [Double]
    ) throws -> DenseNetworkModel {
        let configuration = try DenseNetworkConfiguration(
            inputFeatures: ["x"],
            outputNames: ["a", "b"],
            outputActivation: outputActivation,
            loss: loss
        )
        let layer = try DenseLayerParameters(
            inputCount: 1,
            outputCount: 2,
            weights: weights,
            biases: biases
        )
        return try DenseNetworkModel(configuration: configuration, layers: [layer])
    }
}
