import Foundation
import Testing

@testable import ML5

private enum TrainingLabel: String, ClassificationLabel {
    case negative
    case positive

    init?(ml5RawValue: String) {
        self.init(rawValue: ml5RawValue)
    }

    var ml5RawValue: String { rawValue }
}

@Suite("ML5 deterministic CPU training")
struct ML5DenseTrainingTests {
    @Test("Dense samples validate targets, conveniences, and Codable")
    func samples() throws {
        let features = try FeatureVector(["x": .number(1)])
        #expect(throws: ML5Error.self) {
            _ = try DenseTrainingSample(features: features, targets: [])
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseTrainingSample(features: features, targets: [.nan])
        }

        let regression = DenseTrainingSample(
            regression: try RegressionSample(features: features, target: 3)
        )
        #expect(regression.targets == [3])

        let classification = try DenseTrainingSample.classification(
            features: features,
            label: TrainingLabel.positive,
            labels: [.negative, .positive]
        )
        #expect(classification.targets == [0, 1])
        #expect(
            try JSONDecoder().decode(
                DenseTrainingSample.self,
                from: JSONEncoder().encode(classification)
            ) == classification
        )
        #expect(throws: ML5Error.self) {
            _ = try JSONDecoder().decode(
                DenseTrainingSample.self,
                from: Data(#"{"features":{"x":{"kind":"number","number":1}},"targets":[]}"#.utf8)
            )
        }
    }

    @Test("Classification sample construction rejects invalid label sets")
    func classificationSamples() throws {
        let features = try FeatureVector(["x": .number(1)])
        #expect(throws: ML5Error.self) {
            _ = try DenseTrainingSample.classification(
                features: features,
                label: TrainingLabel.positive,
                labels: []
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseTrainingSample.classification(
                features: features,
                label: TrainingLabel.positive,
                labels: [.positive, .positive]
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseTrainingSample.classification(
                features: features,
                label: TrainingLabel.positive,
                labels: [.negative]
            )
        }
    }

    @Test("Adam learns a canonical affine regression reproducibly")
    func affineRegression() async throws {
        let samples = try (-4...4).map { value in
            try DenseTrainingSample(
                features: FeatureVector(["x": .number(Double(value))]),
                targets: [2 * Double(value) + 1]
            )
        }
        let configuration = try DenseNetworkConfiguration(
            inputFeatures: ["x"],
            outputNames: ["y"],
            weightInitialization: .glorotUniform,
            optimizer: try OptimizerConfiguration(kind: .adam),
            learningRate: 0.05,
            batchSize: 3,
            epochs: 160,
            validationFraction: 0.22,
            seed: 7
        )
        let trainer = DenseCPUTrainer()
        let first = try await trainer.train(samples, configuration: configuration)
        let second = try await trainer.train(samples, configuration: configuration)
        let prediction = try await first.model.predict(FeatureVector(["x": .number(5)]))
        let value = try #require(prediction["y"]?.numericValue)

        #expect(first == second)
        #expect(first.history.count == 160)
        #expect(first.history[0].epoch == 1)
        #expect(first.history.last?.epoch == 160)
        #expect(first.history[0].validationLoss != nil)
        #expect(first.history.last?.trainingLoss ?? .infinity < 0.001)
        #expect(abs(value - 11) < 0.05)
    }

    @Test("He initialization and SGD learn a small softmax classifier")
    func categoricalClassification() async throws {
        let labels = [TrainingLabel.negative, .positive]
        let raw: [(Double, TrainingLabel)] = [
            (-2, .negative),
            (-1, .negative),
            (1, .positive),
            (2, .positive),
        ]
        let samples = try raw.map { value, label in
            try DenseTrainingSample.classification(
                features: FeatureVector(["x": .number(value)]),
                label: label,
                labels: labels
            )
        }
        let configuration = try DenseNetworkConfiguration(
            inputFeatures: ["x"],
            outputNames: ["negative", "positive"],
            hiddenLayers: [
                try DenseLayerConfiguration(
                    neuronCount: 3,
                    activation: .hyperbolicTangent
                )
            ],
            outputActivation: .softmax,
            weightInitialization: .heNormal,
            loss: .categoricalCrossEntropy,
            optimizer: try OptimizerConfiguration(
                kind: .stochasticGradientDescent,
                momentum: 0.8
            ),
            learningRate: 0.1,
            batchSize: 2,
            epochs: 120,
            validationFraction: 0,
            seed: 11
        )
        let result = try await DenseCPUTrainer().train(samples, configuration: configuration)
        let negative = try await result.model.predict(FeatureVector(["x": .number(-3)]))
        let positive = try await result.model.predict(FeatureVector(["x": .number(3)]))

        #expect(result.history.last?.validationLoss == nil)
        #expect((negative["negative"]?.numericValue ?? 0) > 0.95)
        #expect((positive["positive"]?.numericValue ?? 0) > 0.95)
    }

    @Test("Zero initialization and binary cross entropy update sigmoid bias")
    func binaryClassification() async throws {
        let sample = try DenseTrainingSample(
            features: FeatureVector(["x": .number(0)]),
            targets: [1]
        )
        let configuration = try DenseNetworkConfiguration(
            inputFeatures: ["x"],
            outputNames: ["probability"],
            outputActivation: .sigmoid,
            weightInitialization: .zeros,
            loss: .binaryCrossEntropy,
            optimizer: .stochasticGradientDescent,
            learningRate: 0.5,
            batchSize: 4,
            epochs: 20,
            validationFraction: 0,
            seed: 0
        )
        let result = try await DenseCPUTrainer().train([sample], configuration: configuration)
        let prediction = try await result.model.predict(sample.features)

        #expect((prediction["probability"]?.numericValue ?? 0) > 0.85)
        #expect(result.history.allSatisfy { $0.trainingLoss.isFinite })
    }

    @Test("Training validates sample dimensions, features, and loss domains")
    func invalidTrainingSamples() async throws {
        let trainer = DenseCPUTrainer()
        let regression = try DenseNetworkConfiguration(
            inputFeatures: ["x"],
            outputNames: ["y"]
        )
        await #expect(throws: ML5Error.invalidTrainingSamples) {
            _ = try await trainer.train([], configuration: regression)
        }
        await #expect(throws: ML5Error.self) {
            _ = try await trainer.train(
                [
                    try DenseTrainingSample(
                        features: FeatureVector(["x": .number(1)]),
                        targets: [1, 2]
                    )
                ],
                configuration: regression
            )
        }
        await #expect(throws: ML5Error.missingFeature("x")) {
            _ = try await trainer.train(
                [
                    try DenseTrainingSample(
                        features: FeatureVector(["other": .number(1)]),
                        targets: [1]
                    )
                ],
                configuration: regression
            )
        }
        await #expect(throws: ML5Error.self) {
            _ = try await trainer.train(
                [
                    try DenseTrainingSample(
                        features: FeatureVector(["x": .string("bad")]),
                        targets: [1]
                    )
                ],
                configuration: regression
            )
        }

        let categorical = try DenseNetworkConfiguration(
            inputFeatures: ["x"],
            outputNames: ["a", "b"],
            outputActivation: .softmax,
            loss: .categoricalCrossEntropy
        )
        for targets in [[-1.0, 2], [0.2, 0.2]] {
            await #expect(throws: ML5Error.self) {
                _ = try await trainer.train(
                    [
                        try DenseTrainingSample(
                            features: FeatureVector(["x": .number(1)]),
                            targets: targets
                        )
                    ],
                    configuration: categorical
                )
            }
        }

        let binary = try DenseNetworkConfiguration(
            inputFeatures: ["x"],
            outputNames: ["y"],
            outputActivation: .sigmoid,
            loss: .binaryCrossEntropy
        )
        await #expect(throws: ML5Error.self) {
            _ = try await trainer.train(
                [
                    try DenseTrainingSample(
                        features: FeatureVector(["x": .number(1)]),
                        targets: [2]
                    )
                ],
                configuration: binary
            )
        }
    }

    @Test("Every activation derivative uses its documented local slope")
    func activationDerivatives() {
        #expect(
            DenseNetworkMath.derivative(preactivation: 2, activation: 2, using: .linear) == 1
        )
        #expect(
            DenseNetworkMath.derivative(
                preactivation: 2,
                activation: 2,
                using: .rectifiedLinear
            ) == 1
        )
        #expect(
            DenseNetworkMath.derivative(
                preactivation: -1,
                activation: 0,
                using: .rectifiedLinear
            ) == 0
        )
        #expect(
            DenseNetworkMath.derivative(preactivation: 0, activation: 0.5, using: .sigmoid)
                == 0.25
        )
        #expect(
            DenseNetworkMath.derivative(
                preactivation: 0,
                activation: 0.5,
                using: .hyperbolicTangent
            ) == 0.75
        )
        #expect(
            DenseNetworkMath.derivative(preactivation: 0, activation: 0.5, using: .softmax)
                == 0.25
        )
    }

    @Test("Training reports dimension overflow and numerical instability")
    func numericalFailures() async throws {
        let overflowConfiguration = try DenseNetworkConfiguration(
            inputFeatures: ["x", "z"],
            outputNames: ["y"],
            hiddenLayers: [try DenseLayerConfiguration(neuronCount: Int.max)],
            epochs: 1,
            validationFraction: 0
        )
        let overflowSample = try DenseTrainingSample(
            features: FeatureVector(["x": .number(0), "z": .number(0)]),
            targets: [0]
        )
        await #expect(throws: ML5Error.self) {
            _ = try await DenseCPUTrainer().train(
                [overflowSample],
                configuration: overflowConfiguration
            )
        }

        let unstableConfiguration = try DenseNetworkConfiguration(
            inputFeatures: ["x"],
            outputNames: ["y"],
            weightInitialization: .zeros,
            optimizer: .stochasticGradientDescent,
            learningRate: .greatestFiniteMagnitude,
            batchSize: 1,
            epochs: 1,
            validationFraction: 0
        )
        let unstableSample = try DenseTrainingSample(
            features: FeatureVector(["x": .number(1)]),
            targets: [1]
        )
        await #expect(throws: ML5Error.self) {
            _ = try await DenseCPUTrainer().train(
                [unstableSample],
                configuration: unstableConfiguration
            )
        }

        let nonfiniteLoss = try DenseTrainingSample(
            features: FeatureVector(["x": .number(0)]),
            targets: [.greatestFiniteMagnitude / 2]
        )
        let moderateConfiguration = try DenseNetworkConfiguration(
            inputFeatures: ["x"],
            outputNames: ["y"],
            weightInitialization: .zeros,
            optimizer: .stochasticGradientDescent,
            learningRate: 1e-300,
            batchSize: 1,
            epochs: 1,
            validationFraction: 0
        )
        await #expect(throws: ML5Error.self) {
            _ = try await DenseCPUTrainer().train(
                [nonfiniteLoss],
                configuration: moderateConfiguration
            )
        }
    }

    @Test("Training cooperatively observes cancellation")
    func cancellation() async throws {
        let sample = try DenseTrainingSample(
            features: FeatureVector(["x": .number(1)]),
            targets: [1]
        )
        let configuration = try DenseNetworkConfiguration(
            inputFeatures: ["x"],
            outputNames: ["y"],
            epochs: 2,
            validationFraction: 0
        )
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await DenseCPUTrainer().train([sample], configuration: configuration)
        }
        await #expect(throws: CancellationError.self) {
            _ = try await task.value
        }
    }
}
