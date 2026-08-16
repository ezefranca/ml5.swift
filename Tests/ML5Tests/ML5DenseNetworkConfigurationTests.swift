import Foundation
import Testing

@testable import ML5

@Suite("ML5 dense-network configuration")
struct ML5DenseNetworkConfigurationTests {
    @Test("Default and complete configurations preserve every option")
    func validConfigurations() throws {
        let defaults = try DenseNetworkConfiguration(
            inputFeatures: ["x"],
            outputNames: ["value"]
        )
        #expect(defaults.hiddenLayers.isEmpty)
        #expect(defaults.outputActivation == .linear)
        #expect(defaults.weightInitialization == .glorotUniform)
        #expect(defaults.loss == .meanSquaredError)
        #expect(defaults.optimizer == .adam)
        #expect(defaults.learningRate == 0.001)
        #expect(defaults.batchSize == 32)
        #expect(defaults.epochs == 100)
        #expect(defaults.validationFraction == 0.2)
        #expect(defaults.seed == 0)

        let optimizer = try OptimizerConfiguration(
            kind: .stochasticGradientDescent,
            momentum: 0.75,
            beta1: 0.8,
            beta2: 0.95,
            epsilon: 1e-6
        )
        let complete = try DenseNetworkConfiguration(
            inputFeatures: ["x", "y"],
            outputNames: ["cold", "hot"],
            hiddenLayers: [
                try DenseLayerConfiguration(neuronCount: 8),
                try DenseLayerConfiguration(neuronCount: 4, activation: .hyperbolicTangent),
            ],
            outputActivation: .softmax,
            weightInitialization: .heNormal,
            loss: .categoricalCrossEntropy,
            optimizer: optimizer,
            learningRate: 0.01,
            batchSize: 4,
            epochs: 25,
            validationFraction: 0,
            seed: 42
        )

        #expect(complete.inputFeatures.map(\.rawValue) == ["x", "y"])
        #expect(complete.outputNames.map(\.rawValue) == ["cold", "hot"])
        #expect(complete.hiddenLayers.map(\.neuronCount) == [8, 4])
        #expect(complete.optimizer == optimizer)
        #expect(OptimizerConfiguration.stochasticGradientDescent.kind == .stochasticGradientDescent)
    }

    @Test("Configuration values round-trip through validated Codable paths")
    func codable() throws {
        let layer = try DenseLayerConfiguration(neuronCount: 3, activation: .sigmoid)
        let optimizer = try OptimizerConfiguration(kind: .adam)
        let configuration = try DenseNetworkConfiguration(
            inputFeatures: ["a", "b"],
            outputNames: ["result"],
            hiddenLayers: [layer],
            outputActivation: .sigmoid,
            weightInitialization: .zeros,
            loss: .binaryCrossEntropy,
            optimizer: optimizer,
            learningRate: 0.02,
            batchSize: 2,
            epochs: 7,
            validationFraction: 0.25,
            seed: 9
        )
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        #expect(
            try decoder.decode(DenseLayerConfiguration.self, from: encoder.encode(layer)) == layer
        )
        #expect(
            try decoder.decode(OptimizerConfiguration.self, from: encoder.encode(optimizer))
                == optimizer
        )
        #expect(
            try decoder.decode(DenseNetworkConfiguration.self, from: encoder.encode(configuration))
                == configuration
        )
        #expect(ActivationFunction.allCases.count == 5)
        #expect(WeightInitialization.allCases.count == 3)
        #expect(TrainingLoss.allCases.count == 3)
        #expect(OptimizerKind.allCases.count == 2)
    }

    @Test("Optimizer validation covers every hyperparameter domain")
    func invalidOptimizers() {
        for momentum in [-0.1, 1, Double.infinity] {
            #expect(throws: ML5Error.self) {
                _ = try OptimizerConfiguration(momentum: momentum)
            }
        }
        for beta1 in [-0.1, 1, Double.nan] {
            #expect(throws: ML5Error.self) {
                _ = try OptimizerConfiguration(beta1: beta1)
            }
        }
        for beta2 in [-0.1, 1, Double.infinity] {
            #expect(throws: ML5Error.self) {
                _ = try OptimizerConfiguration(beta2: beta2)
            }
        }
        for epsilon in [0, -1, Double.nan] {
            #expect(throws: ML5Error.self) {
                _ = try OptimizerConfiguration(epsilon: epsilon)
            }
        }
    }

    @Test("Hidden layers require positive neurons and element-wise activations")
    func invalidLayers() {
        #expect(throws: ML5Error.self) {
            _ = try DenseLayerConfiguration(neuronCount: 0)
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseLayerConfiguration(neuronCount: 1, activation: .softmax)
        }
    }

    @Test("Architectures require nonempty unique feature and output names")
    func invalidNames() throws {
        #expect(throws: ML5Error.self) {
            _ = try DenseNetworkConfiguration(inputFeatures: [], outputNames: ["y"])
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseNetworkConfiguration(inputFeatures: ["x", "x"], outputNames: ["y"])
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseNetworkConfiguration(inputFeatures: ["x"], outputNames: [])
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseNetworkConfiguration(
                inputFeatures: ["x"],
                outputNames: ["y", "y"]
            )
        }
    }

    @Test("Training loop controls require finite positive values")
    func invalidLoopControls() throws {
        for learningRate in [0, -1, Double.infinity] {
            #expect(throws: ML5Error.self) {
                _ = try DenseNetworkConfiguration(
                    inputFeatures: ["x"],
                    outputNames: ["y"],
                    learningRate: learningRate
                )
            }
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseNetworkConfiguration(
                inputFeatures: ["x"],
                outputNames: ["y"],
                batchSize: 0
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseNetworkConfiguration(
                inputFeatures: ["x"],
                outputNames: ["y"],
                epochs: 0
            )
        }
        for validationFraction in [-0.1, 1, Double.nan] {
            #expect(throws: ML5Error.self) {
                _ = try DenseNetworkConfiguration(
                    inputFeatures: ["x"],
                    outputNames: ["y"],
                    validationFraction: validationFraction
                )
            }
        }
    }

    @Test("Losses require compatible output activations and dimensions")
    func lossCompatibility() throws {
        #expect(throws: ML5Error.self) {
            _ = try DenseNetworkConfiguration(
                inputFeatures: ["x"],
                outputNames: ["a", "b"],
                outputActivation: .softmax
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseNetworkConfiguration(
                inputFeatures: ["x"],
                outputNames: ["a", "b"],
                outputActivation: .linear,
                loss: .categoricalCrossEntropy
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseNetworkConfiguration(
                inputFeatures: ["x"],
                outputNames: ["only"],
                outputActivation: .softmax,
                loss: .categoricalCrossEntropy
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseNetworkConfiguration(
                inputFeatures: ["x"],
                outputNames: ["binary"],
                loss: .binaryCrossEntropy
            )
        }

        let binary = try DenseNetworkConfiguration(
            inputFeatures: ["x"],
            outputNames: ["binary"],
            outputActivation: .sigmoid,
            loss: .binaryCrossEntropy
        )
        #expect(binary.loss == .binaryCrossEntropy)
    }

    @Test("Decoded configurations cannot bypass validation")
    func invalidCodable() throws {
        let layer = Data(#"{"activation":"softmax","neuronCount":2}"#.utf8)
        #expect(throws: ML5Error.self) {
            _ = try JSONDecoder().decode(DenseLayerConfiguration.self, from: layer)
        }

        let invalidOptimizer = Data(
            #"{"beta1":0.9,"beta2":0.999,"epsilon":0,"kind":"adam","momentum":0}"#.utf8
        )
        #expect(throws: ML5Error.self) {
            _ = try JSONDecoder().decode(OptimizerConfiguration.self, from: invalidOptimizer)
        }

        let valid = try DenseNetworkConfiguration(inputFeatures: ["x"], outputNames: ["y"])
        let object = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(valid)) as? [String: Any]
        )
        var invalid = object
        invalid["batchSize"] = 0
        #expect(throws: ML5Error.self) {
            _ = try JSONDecoder().decode(
                DenseNetworkConfiguration.self,
                from: JSONSerialization.data(withJSONObject: invalid)
            )
        }
    }
}
