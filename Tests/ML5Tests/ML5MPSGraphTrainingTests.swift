#if canImport(Metal) && canImport(MetalPerformanceShadersGraph)
    import Metal
    import MetalPerformanceShadersGraph
    import Testing

    @testable import ML5

    @Suite("ML5 MPSGraph training")
    struct ML5MPSGraphTrainingTests {
        @Test("Trainer construction exposes selected devices and explicit failures")
        func construction() throws {
            let device = try #require(MTLCreateSystemDefaultDevice())
            let automatic = try DenseMPSGraphTrainer()
            let explicit = try DenseMPSGraphTrainer(device: device)

            #expect(automatic.deviceName.isEmpty == false)
            #expect(explicit.deviceName == device.name)
            #expect(throws: ML5Error.self) {
                _ = try DenseMPSGraphTrainer(testDevice: nil)
            }
            #expect(throws: ML5Error.self) {
                _ = try DenseMPSGraphTrainer(testDevice: device) { _ in nil }
            }
        }

        @Test("MPSGraph learns linear MSE through an explicit Metal command queue")
        func meanSquaredError() async throws {
            let trainer = try DenseMPSGraphTrainer()
            let samples = try [
                DenseTrainingSample(
                    features: FeatureVector(["x": .number(1)]),
                    targets: [3]
                ),
                DenseTrainingSample(
                    features: FeatureVector(["x": .number(2)]),
                    targets: [5]
                ),
            ]
            let configuration = try DenseNetworkConfiguration(
                inputFeatures: ["x"],
                outputNames: ["y"],
                weightInitialization: .zeros,
                optimizer: .stochasticGradientDescent,
                learningRate: 0.1,
                batchSize: 1,
                epochs: 8,
                validationFraction: 0.5,
                seed: 3
            )
            let result = try await trainer.train(samples, configuration: configuration)

            #expect(result.history.count == 8)
            #expect(result.history[0].validationLoss != nil)
            #expect(
                (result.history.last?.trainingLoss ?? .infinity)
                    < result.history[0].trainingLoss
            )
            #expect(result.model.layers[0].weights.allSatisfy { $0.isFinite })
        }

        @Test("MPSGraph differentiates categorical softmax with tanh hidden state")
        func categoricalCrossEntropy() async throws {
            let samples = try [
                DenseTrainingSample(
                    features: FeatureVector(["x": .number(-1)]),
                    targets: [1, 0]
                ),
                DenseTrainingSample(
                    features: FeatureVector(["x": .number(1)]),
                    targets: [0, 1]
                ),
            ]
            let configuration = try DenseNetworkConfiguration(
                inputFeatures: ["x"],
                outputNames: ["negative", "positive"],
                hiddenLayers: [
                    try DenseLayerConfiguration(
                        neuronCount: 2,
                        activation: .hyperbolicTangent
                    )
                ],
                outputActivation: .softmax,
                weightInitialization: .heNormal,
                loss: .categoricalCrossEntropy,
                optimizer: .adam,
                learningRate: 0.05,
                batchSize: 2,
                epochs: 2,
                validationFraction: 0,
                seed: 4
            )
            let result = try await DenseMPSGraphTrainer().train(
                samples,
                configuration: configuration
            )
            let output = try await result.model.predict(samples[1].features)
            let total =
                (output["negative"]?.numericValue ?? 0)
                + (output["positive"]?.numericValue ?? 0)

            #expect(abs(total - 1) < 1e-9)
            #expect(result.history.allSatisfy { $0.validationLoss == nil })
        }

        @Test("MPSGraph covers ReLU, sigmoid, and binary cross entropy")
        func binaryCrossEntropy() async throws {
            let sample = try DenseTrainingSample(
                features: FeatureVector(["x": .number(1)]),
                targets: [1]
            )
            let configuration = try DenseNetworkConfiguration(
                inputFeatures: ["x"],
                outputNames: ["probability"],
                hiddenLayers: [
                    try DenseLayerConfiguration(neuronCount: 2, activation: .rectifiedLinear),
                    try DenseLayerConfiguration(neuronCount: 2, activation: .sigmoid),
                ],
                outputActivation: .sigmoid,
                loss: .binaryCrossEntropy,
                learningRate: 0.01,
                batchSize: 1,
                epochs: 1,
                validationFraction: 0,
                seed: 8
            )
            let result = try await DenseMPSGraphTrainer().train(
                [sample],
                configuration: configuration
            )

            #expect(result.history[0].trainingLoss.isFinite)
            #expect(result.model.layers.count == 3)
        }

        @Test("MPSGraph trainer rejects empty input and observes cancellation")
        func requestValidation() async throws {
            let trainer = try DenseMPSGraphTrainer()
            let configuration = try DenseNetworkConfiguration(
                inputFeatures: ["x"],
                outputNames: ["y"],
                epochs: 1,
                validationFraction: 0
            )
            await #expect(throws: ML5Error.invalidTrainingSamples) {
                _ = try await trainer.train([], configuration: configuration)
            }

            let sample = try DenseTrainingSample(
                features: FeatureVector(["x": .number(1)]),
                targets: [1]
            )
            let task = Task {
                withUnsafeCurrentTask { $0?.cancel() }
                return try await trainer.train([sample], configuration: configuration)
            }
            await #expect(throws: CancellationError.self) {
                _ = try await task.value
            }
        }

        @Test("MPSGraph result validation preserves missing-gradient failures")
        func graphResultValidation() throws {
            let graph = MPSGraph()
            let parameter = graph.placeholder(shape: [1], dataType: .float32, name: "parameter")

            #expect(throws: ML5Error.self) {
                _ = try MPSGraphDenseBatch.requiredGradients(for: [parameter], from: [:])
            }
            #expect(
                try MPSGraphDenseBatch.requiredGradients(
                    for: [parameter],
                    from: [parameter: parameter]
                ) == [parameter]
            )
            #expect(throws: ML5Error.self) {
                _ = try MPSGraphDenseBatch.values(for: parameter, count: 1, results: [:])
            }
        }
    }
#endif
