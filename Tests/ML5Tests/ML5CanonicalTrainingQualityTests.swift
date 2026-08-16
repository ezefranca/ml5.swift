import Foundation
import Testing

@testable import ML5

#if canImport(Metal) && canImport(MetalPerformanceShadersGraph)
    import Metal
    import MetalPerformanceShadersGraph
#endif

@Suite("ML5 canonical trained-model quality", .serialized)
struct ML5CanonicalTrainingQualityTests {
    @Test("A dense nonlinear network learns canonical XOR")
    func xorClassification() async throws {
        let observations: [(Double, Double, Int)] = [
            (0, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
        ]
        let samples = try observations.map { x, y, label in
            try DenseTrainingSample(
                features: FeatureVector(["x": .number(x), "y": .number(y)]),
                targets: label == 0 ? [1, 0] : [0, 1]
            )
        }
        let configuration = try DenseNetworkConfiguration(
            inputFeatures: ["x", "y"],
            outputNames: ["equal", "different"],
            hiddenLayers: [
                try DenseLayerConfiguration(
                    neuronCount: 4,
                    activation: .hyperbolicTangent
                )
            ],
            outputActivation: .softmax,
            weightInitialization: .glorotUniform,
            loss: .categoricalCrossEntropy,
            optimizer: .adam,
            learningRate: 0.05,
            batchSize: 4,
            epochs: 500,
            validationFraction: 0,
            seed: 42
        )

        let result = try await DenseCPUTrainer().train(
            samples,
            configuration: configuration
        )
        var correct = 0
        var minimumExpectedProbability = 1.0
        for (x, y, label) in observations {
            let prediction = try await result.model.predict(
                FeatureVector(["x": .number(x), "y": .number(y)])
            )
            let values = [
                try #require(prediction["equal"]?.numericValue),
                try #require(prediction["different"]?.numericValue),
            ]
            let predictedLabel = values[0] > values[1] ? 0 : 1
            correct += predictedLabel == label ? 1 : 0
            minimumExpectedProbability = min(minimumExpectedProbability, values[label])
        }

        #expect(correct == observations.count)
        #expect(minimumExpectedProbability > 0.98)
        #expect((result.history.last?.trainingLoss ?? .infinity) < 0.02)
    }

    @Test("Affine regression generalizes to a held-out lattice")
    func affineRegression() async throws {
        let trainingCoordinates = (-2...2).flatMap { x in
            (-2...2).map { y in (Double(x), Double(y)) }
        }
        let samples = try trainingCoordinates.map { x, y in
            try DenseTrainingSample(
                features: FeatureVector(["x": .number(x), "y": .number(y)]),
                targets: [target(x: x, y: y)]
            )
        }
        let configuration = try DenseNetworkConfiguration(
            inputFeatures: ["x", "y"],
            outputNames: ["value"],
            weightInitialization: .glorotUniform,
            loss: .meanSquaredError,
            optimizer: .adam,
            learningRate: 0.03,
            batchSize: 5,
            epochs: 250,
            validationFraction: 0,
            seed: 314_159
        )
        let result = try await DenseCPUTrainer().train(
            samples,
            configuration: configuration
        )
        let evaluationCoordinates: [(Double, Double)] = [
            (-3, -2.5),
            (-1.5, 2.75),
            (0.25, -3),
            (1.75, 2.5),
            (3, 3),
        ]
        var squaredErrors: [Double] = []
        for (x, y) in evaluationCoordinates {
            let prediction = try await result.model.predict(
                FeatureVector(["x": .number(x), "y": .number(y)])
            )
            let value = try #require(prediction["value"]?.numericValue)
            squaredErrors.append(pow(value - target(x: x, y: y), 2))
        }
        let rootMeanSquaredError = sqrt(
            squaredErrors.reduce(0, +) / Double(squaredErrors.count)
        )

        #expect(rootMeanSquaredError < 0.01)
        #expect((result.history.last?.trainingLoss ?? .infinity) < 0.0001)
    }

    #if canImport(Metal) && canImport(MetalPerformanceShadersGraph) && !targetEnvironment(simulator)
        @Test("CPU and Metal remain numerically conformant on a canonical full batch")
        func cpuMetalConformance() async throws {
            let samples = try (-2...2).map { value in
                let x = Double(value)
                return try DenseTrainingSample(
                    features: FeatureVector(["x": .number(x)]),
                    targets: [1.5 * x - 0.25]
                )
            }
            let configuration = try DenseNetworkConfiguration(
                inputFeatures: ["x"],
                outputNames: ["y"],
                weightInitialization: .zeros,
                loss: .meanSquaredError,
                optimizer: .stochasticGradientDescent,
                learningRate: 0.05,
                batchSize: samples.count,
                epochs: 12,
                validationFraction: 0,
                seed: 9
            )

            let cpu = try await DenseCPUTrainer().train(
                samples,
                configuration: configuration
            )
            let metal = try await DenseMPSGraphTrainer().train(
                samples,
                configuration: configuration
            )
            for value in [-2.5, -0.5, 1.5, 3.0] {
                let features = try FeatureVector(["x": .number(value)])
                let cpuValue = try #require(
                    try await cpu.model.predict(features)["y"]?.numericValue
                )
                let metalValue = try #require(
                    try await metal.model.predict(features)["y"]?.numericValue
                )
                #expect(abs(cpuValue - metalValue) < 1e-5)
            }
            #expect(
                zip(cpu.history, metal.history).allSatisfy {
                    abs($0.trainingLoss - $1.trainingLoss) < 1e-5
                }
            )
        }
    #endif

    private func target(x: Double, y: Double) -> Double {
        0.75 * x - 1.25 * y + 0.5
    }
}
