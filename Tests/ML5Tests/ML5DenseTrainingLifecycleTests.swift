import Foundation
import Testing

@testable import ML5

private enum PauseTraining: Error {
    case checkpointCaptured
    case progressRejected
}

private actor DenseProgressRecorder {
    private(set) var updates: [DenseTrainingProgress] = []
    private(set) var checkpoint: DenseTrainingCheckpoint?

    func record(_ update: DenseTrainingProgress) {
        updates.append(update)
        if let emitted = update.checkpoint {
            checkpoint = emitted
        }
    }
}

@Suite("ML5 dense-training lifecycle")
struct ML5DenseTrainingLifecycleTests {
    @Test("Lifecycle values validate and round-trip")
    func lifecycleValues() throws {
        #expect(throws: ML5Error.self) {
            _ = try DenseEpochMetrics(epoch: 0, trainingLoss: 1, validationLoss: nil)
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseEpochMetrics(epoch: 1, trainingLoss: .infinity, validationLoss: nil)
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseEpochMetrics(epoch: 1, trainingLoss: 1, validationLoss: .nan)
        }
        let metrics = try DenseEpochMetrics(epoch: 2, trainingLoss: 0.5, validationLoss: 0.75)
        #expect(
            try JSONDecoder().decode(
                DenseEpochMetrics.self,
                from: JSONEncoder().encode(metrics)
            ) == metrics
        )
        #expect(throws: ML5Error.self) {
            _ = try JSONDecoder().decode(
                DenseEpochMetrics.self,
                from: Data(#"{"epoch":0,"trainingLoss":1}"#.utf8)
            )
        }

        #expect(throws: ML5Error.self) {
            _ = try DenseEarlyStoppingConfiguration(patience: 0)
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseEarlyStoppingConfiguration(patience: 1, minimumImprovement: -.infinity)
        }
        let stopping = try DenseEarlyStoppingConfiguration(
            patience: 3,
            minimumImprovement: 0.01,
            restoresBestModel: false
        )
        #expect(
            try JSONDecoder().decode(
                DenseEarlyStoppingConfiguration.self,
                from: JSONEncoder().encode(stopping)
            ) == stopping
        )
        #expect(throws: ML5Error.self) {
            _ = try JSONDecoder().decode(
                DenseEarlyStoppingConfiguration.self,
                from: Data(
                    #"{"patience":0,"minimumImprovement":0,"restoresBestModel":true}"#.utf8
                )
            )
        }

        #expect(throws: ML5Error.self) {
            _ = try DenseTrainingOptions(checkpointInterval: 0)
        }
        let options = try DenseTrainingOptions(
            earlyStopping: stopping,
            checkpointInterval: 2
        )
        #expect(
            try JSONDecoder().decode(
                DenseTrainingOptions.self,
                from: JSONEncoder().encode(options)
            ) == options
        )
        #expect(DenseTrainingOptions.standard.earlyStopping == nil)
        #expect(DenseTrainingOptions.standard.checkpointInterval == nil)
        #expect(try DenseTrainingOptions().checkpointInterval == nil)

        let policy = DenseTrainingExecutionPolicy(preference: .metal, fallback: .none)
        #expect(
            try JSONDecoder().decode(
                DenseTrainingExecutionPolicy.self,
                from: JSONEncoder().encode(policy)
            ) == policy
        )
        #expect(DenseTrainingDevicePreference.allCasesForTesting == [.automatic, .cpu, .metal])
        #expect(DenseTrainingFallback.allCasesForTesting == [.none, .cpu])
        #expect(
            ML5Error.invalidTrainingCheckpoint(reason: "Malformed.").errorDescription
                == "Invalid training checkpoint: Malformed."
        )
    }

    @Test("CPU progress emits exact resumable checkpoints")
    func cpuResume() async throws {
        let samples = try regressionSamples()
        let configuration = try regressionConfiguration(epochs: 8)
        let options = try DenseTrainingOptions(checkpointInterval: 2)
        let recorder = DenseProgressRecorder()

        do {
            _ = try await DenseCPUTrainer().train(
                samples,
                configuration: configuration,
                options: options
            ) { update in
                await recorder.record(update)
                if update.metrics.epoch == 2 {
                    throw PauseTraining.checkpointCaptured
                }
            }
            Issue.record("Training should have paused after capturing epoch two.")
        } catch PauseTraining.checkpointCaptured {
        }

        let checkpoint = try #require(await recorder.checkpoint)
        #expect(checkpoint.backend == .cpu)
        #expect(checkpoint.formatVersion == DenseTrainingCheckpoint.currentFormatVersion)
        #expect(checkpoint.completedEpochs == 2)
        #expect(checkpoint.history.count == 2)
        #expect(checkpoint.sampleCount == samples.count)
        #expect(checkpoint.sampleFingerprint != 0)
        #expect(try checkpoint.makeModel().layers.isEmpty == false)
        #expect(
            try JSONDecoder().decode(
                DenseTrainingCheckpoint.self,
                from: JSONEncoder().encode(checkpoint)
            ) == checkpoint
        )

        let resumedRecorder = DenseProgressRecorder()
        let resumed = try await DenseCPUTrainer().resume(
            checkpoint,
            samples: samples
        ) { update in
            await resumedRecorder.record(update)
        }
        let uninterrupted = try await DenseCPUTrainer().train(
            samples,
            configuration: configuration,
            options: options
        )
        #expect(resumed == uninterrupted)
        #expect(resumed.backend == .cpu)
        #expect(resumed.stopReason == .completed)
        #expect(resumed.checkpoint.completedEpochs == configuration.epochs)
        let resumedUpdates = await resumedRecorder.updates
        #expect(resumedUpdates.map(\.metrics.epoch) == [3, 4, 5, 6, 7, 8])
        #expect(resumedUpdates.allSatisfy { $0.totalEpochs == 8 && $0.backend == .cpu })
        #expect(resumedUpdates.compactMap(\.checkpoint).map(\.completedEpochs) == [4, 6, 8])
    }

    @Test("Early stopping monitors losses and optionally restores the best model")
    func earlyStopping() async throws {
        let sample = try DenseTrainingSample(
            features: FeatureVector(["x": .number(0)]),
            targets: [0]
        )
        let configuration = try DenseNetworkConfiguration(
            inputFeatures: ["x"],
            outputNames: ["y"],
            weightInitialization: .zeros,
            optimizer: .stochasticGradientDescent,
            learningRate: 0.1,
            batchSize: 1,
            epochs: 10,
            validationFraction: 0,
            seed: 1
        )
        for restoresBestModel in [false, true] {
            let options = try DenseTrainingOptions(
                earlyStopping: DenseEarlyStoppingConfiguration(
                    patience: 2,
                    restoresBestModel: restoresBestModel
                ),
                checkpointInterval: 1
            )
            let result = try await DenseCPUTrainer().train(
                [sample],
                configuration: configuration,
                options: options
            )
            #expect(result.stopReason == .earlyStopping)
            #expect(result.history.count == 3)
            #expect(result.checkpoint.completedEpochs == 3)
            #expect(result.checkpoint.bestLoss == 0)
            #expect(result.checkpoint.staleEpochCount == 2)
            #expect(result.model == (try result.checkpoint.makeModel()))
            let continued = try await DenseCPUTrainer().resume(
                result.checkpoint,
                samples: [sample]
            )
            #expect(continued.stopReason == .earlyStopping)
        }
    }

    @Test("Checkpoint validation rejects malformed optimizer and partition state")
    func checkpointValidation() async throws {
        let samples = try regressionSamples()
        let result = try await DenseCPUTrainer().train(
            samples,
            configuration: regressionConfiguration(epochs: 2),
            options: DenseTrainingOptions.standard
        )
        let value = result.checkpoint
        let layer = try #require(value.layers.first)

        #expect(throws: ML5Error.self) {
            _ = try DenseLayerTrainingState(
                parameters: layer.parameters,
                weightVelocity: [],
                biasVelocity: layer.biasVelocity,
                weightFirstMoment: layer.weightFirstMoment,
                biasFirstMoment: layer.biasFirstMoment,
                weightSecondMoment: layer.weightSecondMoment,
                biasSecondMoment: layer.biasSecondMoment
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseLayerTrainingState(
                parameters: layer.parameters,
                weightVelocity: layer.weightVelocity.map { _ in .nan },
                biasVelocity: layer.biasVelocity,
                weightFirstMoment: layer.weightFirstMoment,
                biasFirstMoment: layer.biasFirstMoment,
                weightSecondMoment: layer.weightSecondMoment,
                biasSecondMoment: layer.biasSecondMoment
            )
        }

        func rebuild(
            formatVersion: Int? = nil,
            completedEpochs: Int? = nil,
            history: [DenseEpochMetrics]? = nil,
            sampleCount: Int? = nil,
            optimizerStep: Int? = nil,
            trainingIndices: [Int]? = nil,
            validationIndices: [Int]? = nil,
            layers: [DenseLayerTrainingState]? = nil,
            bestLoss: Double?? = nil,
            bestLayers: [DenseLayerTrainingState]?? = nil,
            staleEpochCount: Int? = nil
        ) throws -> DenseTrainingCheckpoint {
            try DenseTrainingCheckpoint(
                formatVersion: formatVersion ?? value.formatVersion,
                configuration: value.configuration,
                options: value.options,
                completedEpochs: completedEpochs ?? value.completedEpochs,
                history: history ?? value.history,
                sampleCount: sampleCount ?? value.sampleCount,
                sampleFingerprint: value.sampleFingerprint,
                backend: value.backend,
                optimizerStep: optimizerStep ?? value.optimizerStep,
                trainingIndices: trainingIndices ?? value.trainingIndices,
                validationIndices: validationIndices ?? value.validationIndices,
                layers: layers ?? value.layers,
                bestLoss: bestLoss ?? value.bestLoss,
                bestLayers: bestLayers ?? value.bestLayers,
                staleEpochCount: staleEpochCount ?? value.staleEpochCount
            )
        }

        #expect(throws: ML5Error.self) { _ = try rebuild(formatVersion: 2) }
        #expect(throws: ML5Error.self) { _ = try rebuild(completedEpochs: 3) }
        #expect(throws: ML5Error.self) { _ = try rebuild(history: []) }
        #expect(throws: ML5Error.self) { _ = try rebuild(sampleCount: 0) }
        #expect(throws: ML5Error.self) {
            _ = try rebuild(trainingIndices: [0, 0, 1, 2, 3], validationIndices: [])
        }
        #expect(throws: ML5Error.self) {
            _ = try rebuild(trainingIndices: [0, 1, 2, 3, 4], validationIndices: [])
        }
        #expect(throws: ML5Error.self) { _ = try rebuild(optimizerStep: -1) }
        #expect(throws: ML5Error.self) { _ = try rebuild(staleEpochCount: -1) }
        #expect(throws: ML5Error.self) { _ = try rebuild(bestLoss: .some(.nan)) }
        #expect(throws: ML5Error.self) {
            _ = try rebuild(bestLoss: .some(0), bestLayers: .some(nil))
        }
        #expect(throws: ML5Error.self) { _ = try rebuild(layers: []) }
        #expect(throws: ML5Error.self) {
            _ = try rebuild(bestLoss: .some(0), bestLayers: .some([]))
        }
    }

    @Test("Resume rejects backend and ordered-dataset changes")
    func resumeValidation() async throws {
        let samples = try regressionSamples()
        let result = try await DenseCPUTrainer().train(
            samples,
            configuration: regressionConfiguration(epochs: 1)
        )
        let checkpoint = result.checkpoint
        let metalCheckpoint = try DenseTrainingCheckpoint(
            configuration: checkpoint.configuration,
            options: checkpoint.options,
            completedEpochs: checkpoint.completedEpochs,
            history: checkpoint.history,
            sampleCount: checkpoint.sampleCount,
            sampleFingerprint: checkpoint.sampleFingerprint,
            backend: .metal(deviceName: "Other"),
            optimizerStep: checkpoint.optimizerStep,
            trainingIndices: checkpoint.trainingIndices,
            validationIndices: checkpoint.validationIndices,
            layers: checkpoint.layers,
            bestLoss: checkpoint.bestLoss,
            bestLayers: checkpoint.bestLayers,
            staleEpochCount: checkpoint.staleEpochCount
        )
        await #expect(throws: ML5Error.self) {
            _ = try await DenseCPUTrainer().resume(metalCheckpoint, samples: samples)
        }
        await #expect(throws: ML5Error.self) {
            _ = try await DenseCPUTrainer().resume(checkpoint, samples: Array(samples.reversed()))
        }
        await #expect(throws: ML5Error.self) {
            _ = try await DenseCPUTrainer().resume(checkpoint, samples: Array(samples.dropLast()))
        }
    }

    @Test("High-level trainer selects CPU and makes fallback explicit")
    func trainerSelection() async throws {
        let samples = try regressionSamples()
        let configuration = try regressionConfiguration(epochs: 1)
        let cpu = DenseTrainer(
            executionPolicy: DenseTrainingExecutionPolicy(
                preference: .cpu,
                fallback: .none
            )
        )
        let cpuResult = try await cpu.train(samples, configuration: configuration)
        #expect(cpu.executionPolicy.preference == .cpu)
        #expect(cpuResult.backend == .cpu)
        #expect(
            try await cpu.resume(cpuResult.checkpoint, samples: samples).model
                == cpuResult.model
        )

        #if canImport(Metal) && canImport(MetalPerformanceShadersGraph)
            let unavailable: @Sendable () throws -> DenseMPSGraphTrainer = {
                throw ML5Error.trainingAcceleratorUnavailable(reason: "Injected unavailable.")
            }
            let fallback = DenseTrainer(
                executionPolicy: DenseTrainingExecutionPolicy(
                    preference: .automatic,
                    fallback: .cpu
                ),
                makeMetalTrainer: unavailable
            )
            #expect(
                try await fallback.train(samples, configuration: configuration).backend == .cpu
            )

            let required = DenseTrainer(
                executionPolicy: DenseTrainingExecutionPolicy(
                    preference: .metal,
                    fallback: .none
                ),
                makeMetalTrainer: unavailable
            )
            await #expect(throws: ML5Error.self) {
                _ = try await required.train(samples, configuration: configuration)
            }
            await #expect(throws: ML5Error.self) {
                _ = try await required.resume(
                    cpuResult.checkpoint, samples: Array(samples.dropLast()))
            }
        #endif
    }

    @Test("Progress-handler failures are propagated")
    func progressFailure() async throws {
        await #expect(throws: PauseTraining.progressRejected) {
            _ = try await DenseCPUTrainer().train(
                regressionSamples(),
                configuration: regressionConfiguration(epochs: 2),
                options: DenseTrainingOptions.standard
            ) { _ in
                throw PauseTraining.progressRejected
            }
        }
    }

    private func regressionSamples() throws -> [DenseTrainingSample] {
        try (-2...2).map { value in
            try DenseTrainingSample(
                features: FeatureVector(["x": .number(Double(value))]),
                targets: [2 * Double(value) + 1]
            )
        }
    }

    private func regressionConfiguration(epochs: Int) throws -> DenseNetworkConfiguration {
        try DenseNetworkConfiguration(
            inputFeatures: ["x"],
            outputNames: ["y"],
            optimizer: try OptimizerConfiguration(kind: .adam),
            learningRate: 0.02,
            batchSize: 2,
            epochs: epochs,
            validationFraction: 0.2,
            seed: 17
        )
    }
}

extension DenseTrainingDevicePreference {
    fileprivate static var allCasesForTesting: [Self] { [.automatic, .cpu, .metal] }
}

extension DenseTrainingFallback {
    fileprivate static var allCasesForTesting: [Self] { [.none, .cpu] }
}
