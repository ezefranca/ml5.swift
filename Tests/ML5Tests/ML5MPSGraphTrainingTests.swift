#if canImport(Metal) && canImport(MetalPerformanceShadersGraph)
    import Metal
    import MetalPerformanceShadersGraph
    import Testing

    @testable import ML5

    private enum MPSPause: Error {
        case captured
    }

    private actor MPSCheckpointRecorder {
        private(set) var checkpoint: DenseTrainingCheckpoint?

        func record(_ value: DenseTrainingCheckpoint?) {
            checkpoint = value
        }
    }

    @Suite("ML5 MPSGraph training")
    struct ML5MPSGraphTrainingTests {
        @Test("Trainer construction exposes selected devices and explicit failures")
        func construction() throws {
            let device = try #require(MTLCreateSystemDefaultDevice())
            #if targetEnvironment(simulator)
                #expect(
                    throws: ML5Error.trainingAcceleratorUnavailable(
                        reason: "MPSGraph training requires macOS or a physical Apple device.")
                ) {
                    _ = try DenseMPSGraphTrainer()
                }
                #expect(throws: ML5Error.self) {
                    _ = try DenseMPSGraphTrainer(device: device)
                }
            #else
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
            #endif
        }

        #if !targetEnvironment(simulator)
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
        #endif

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

        #if !targetEnvironment(simulator)
            @Test("Metal checkpoints resume only on the same device")
            func checkpointResume() async throws {
                let trainer = try DenseMPSGraphTrainer()
                let sample = try DenseTrainingSample(
                    features: FeatureVector(["x": .number(1)]),
                    targets: [2]
                )
                let configuration = try DenseNetworkConfiguration(
                    inputFeatures: ["x"],
                    outputNames: ["y"],
                    weightInitialization: .zeros,
                    optimizer: .stochasticGradientDescent,
                    learningRate: 0.05,
                    batchSize: 1,
                    epochs: 2,
                    validationFraction: 0,
                    seed: 2
                )
                let recorder = MPSCheckpointRecorder()
                do {
                    _ = try await trainer.train(
                        [sample],
                        configuration: configuration,
                        options: DenseTrainingOptions(checkpointInterval: 1)
                    ) { update in
                        await recorder.record(update.checkpoint)
                        throw MPSPause.captured
                    }
                    Issue.record("The progress handler should stop the first run.")
                } catch MPSPause.captured {
                }

                let checkpoint = try #require(await recorder.checkpoint)
                #expect(checkpoint.backend == .metal(deviceName: trainer.deviceName))
                let resumed = try await trainer.resume(checkpoint, samples: [sample])
                #expect(resumed.history.count == 2)
                #expect(resumed.stopReason == .completed)

                let highLevel = DenseTrainer(
                    executionPolicy: DenseTrainingExecutionPolicy(
                        preference: .metal,
                        fallback: .none
                    ),
                    makeMetalTrainer: { trainer }
                )
                #expect(
                    try await highLevel.resume(checkpoint, samples: [sample]).backend
                        == checkpoint.backend
                )
                #expect(
                    try await highLevel.train([sample], configuration: configuration).backend
                        == checkpoint.backend
                )
                #expect(
                    try await DenseTrainer().train([sample], configuration: configuration).backend
                        == checkpoint.backend
                )

                let cpuResult = try await DenseCPUTrainer().train(
                    [sample],
                    configuration: configuration
                )
                await #expect(throws: ML5Error.self) {
                    _ = try await trainer.resume(cpuResult.checkpoint, samples: [sample])
                }
                let wrongDevice = try DenseTrainingCheckpoint(
                    configuration: checkpoint.configuration,
                    options: checkpoint.options,
                    completedEpochs: checkpoint.completedEpochs,
                    history: checkpoint.history,
                    sampleCount: checkpoint.sampleCount,
                    sampleFingerprint: checkpoint.sampleFingerprint,
                    backend: .metal(deviceName: "Different Metal Device"),
                    optimizerStep: checkpoint.optimizerStep,
                    trainingIndices: checkpoint.trainingIndices,
                    validationIndices: checkpoint.validationIndices,
                    layers: checkpoint.layers,
                    bestLoss: checkpoint.bestLoss,
                    bestLayers: checkpoint.bestLayers,
                    staleEpochCount: checkpoint.staleEpochCount
                )
                await #expect(throws: ML5Error.self) {
                    _ = try await trainer.resume(wrongDevice, samples: [sample])
                }
            }
        #endif
    }
#endif
