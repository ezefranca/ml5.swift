import Foundation

#if canImport(Metal) && canImport(MetalPerformanceShadersGraph)
    import Metal
    import MetalPerformanceShadersGraph
#endif

/// A feature vector paired with finite numeric targets for dense-network training.
public struct DenseTrainingSample: Sendable, Hashable, Codable {
    /// Model input features.
    public let features: FeatureVector
    /// Regression values, one-hot classes, or independent binary targets in output order.
    public let targets: [Double]

    /// Creates a training sample with at least one finite target.
    ///
    /// - Throws: ``ML5Error/invalidTrainingSample(reason:)`` when targets are empty or nonfinite.
    public init(features: FeatureVector, targets: [Double]) throws {
        guard targets.isEmpty == false else {
            throw ML5Error.invalidTrainingSample(reason: "Targets cannot be empty.")
        }
        guard targets.allSatisfy(\.isFinite) else {
            throw ML5Error.invalidTrainingSample(reason: "Every target must be finite.")
        }
        self.features = features
        self.targets = targets
    }

    /// Creates a scalar dense sample from an existing regression sample.
    public init(regression sample: RegressionSample) {
        features = sample.features
        targets = [sample.target]
    }

    /// Creates a one-hot sample for an ordered list of classification labels.
    ///
    /// - Throws: ``ML5Error/invalidTrainingSample(reason:)`` when labels are empty, duplicated, or
    ///   do not contain the sample's expected label.
    public static func classification<Label: ClassificationLabel>(
        features: FeatureVector,
        label: Label,
        labels: [Label]
    ) throws -> Self {
        guard labels.isEmpty == false else {
            throw ML5Error.invalidTrainingSample(
                reason: "Classification labels cannot be empty."
            )
        }
        guard Set(labels).count == labels.count else {
            throw ML5Error.invalidTrainingSample(
                reason: "Classification labels must be unique."
            )
        }
        guard let selectedIndex = labels.firstIndex(of: label) else {
            throw ML5Error.invalidTrainingSample(
                reason: "The expected classification label is not configured."
            )
        }
        var targets = Array(repeating: 0.0, count: labels.count)
        targets[selectedIndex] = 1
        return try Self(features: features, targets: targets)
    }

    /// Decodes and revalidates a persisted dense training sample.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            features: container.decode(FeatureVector.self, forKey: .features),
            targets: container.decode([Double].self, forKey: .targets)
        )
    }

    /// Encodes features and ordered numeric targets.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(features, forKey: .features)
        try container.encode(targets, forKey: .targets)
    }

    private enum CodingKeys: String, CodingKey {
        case features
        case targets
    }
}

/// Loss measurements captured after one complete training epoch.
public struct DenseEpochMetrics: Sendable, Hashable, Codable {
    /// One-based epoch number.
    public let epoch: Int
    /// Mean loss across the epoch's training partition after updates.
    public let trainingLoss: Double
    /// Mean loss across the held-out validation partition, or `nil` when it is empty.
    public let validationLoss: Double?

    /// Creates validated finite measurements for a one-based epoch.
    ///
    /// - Throws: ``ML5Error/invalidTrainingConfiguration(reason:)`` when the epoch is not
    ///   positive or either supplied loss is nonfinite.
    public init(epoch: Int, trainingLoss: Double, validationLoss: Double?) throws {
        guard epoch > 0 else {
            throw ML5Error.invalidTrainingConfiguration(
                reason: "Training metric epochs must be positive."
            )
        }
        guard trainingLoss.isFinite, validationLoss?.isFinite ?? true else {
            throw ML5Error.invalidTrainingConfiguration(
                reason: "Training metrics must contain finite losses."
            )
        }
        self.epoch = epoch
        self.trainingLoss = trainingLoss
        self.validationLoss = validationLoss
    }

    /// Decodes and revalidates persisted epoch metrics.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            epoch: container.decode(Int.self, forKey: .epoch),
            trainingLoss: container.decode(Double.self, forKey: .trainingLoss),
            validationLoss: container.decodeIfPresent(Double.self, forKey: .validationLoss)
        )
    }

    /// Encodes the epoch and its training and optional validation losses.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(epoch, forKey: .epoch)
        try container.encode(trainingLoss, forKey: .trainingLoss)
        try container.encodeIfPresent(validationLoss, forKey: .validationLoss)
    }

    private enum CodingKeys: String, CodingKey {
        case epoch
        case trainingLoss
        case validationLoss
    }
}

/// The concrete compute backend that performed dense training.
public enum DenseTrainingBackend: Sendable, Hashable, Codable {
    /// Deterministic scalar CPU execution.
    case cpu
    /// Metal Performance Shaders Graph execution on the named device.
    case metal(deviceName: String)
}

/// Why a dense training run returned control to its caller.
public enum DenseTrainingStopReason: String, Sendable, Hashable, Codable {
    /// Every configured epoch completed.
    case completed
    /// Validation or training loss stopped improving for the configured patience.
    case earlyStopping
}

/// Validated loss-monitoring behavior for ending training before its epoch limit.
public struct DenseEarlyStoppingConfiguration: Sendable, Hashable, Codable {
    /// Number of consecutive non-improving epochs accepted before stopping.
    public let patience: Int
    /// Smallest absolute loss reduction considered an improvement.
    public let minimumImprovement: Double
    /// Whether the returned model restores the best observed parameters.
    public let restoresBestModel: Bool

    /// Creates an early-stopping policy.
    ///
    /// - Throws: ``ML5Error/invalidTrainingConfiguration(reason:)`` when patience is not
    ///   positive or the minimum improvement is negative or nonfinite.
    public init(
        patience: Int,
        minimumImprovement: Double = 0,
        restoresBestModel: Bool = true
    ) throws {
        guard patience > 0 else {
            throw ML5Error.invalidTrainingConfiguration(
                reason: "Early-stopping patience must be greater than zero."
            )
        }
        guard minimumImprovement.isFinite, minimumImprovement >= 0 else {
            throw ML5Error.invalidTrainingConfiguration(
                reason: "Early-stopping minimum improvement must be finite and nonnegative."
            )
        }
        self.patience = patience
        self.minimumImprovement = minimumImprovement
        self.restoresBestModel = restoresBestModel
    }

    /// Decodes and revalidates a persisted early-stopping policy.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            patience: container.decode(Int.self, forKey: .patience),
            minimumImprovement: container.decode(Double.self, forKey: .minimumImprovement),
            restoresBestModel: container.decode(Bool.self, forKey: .restoresBestModel)
        )
    }

    /// Encodes the complete early-stopping policy.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(patience, forKey: .patience)
        try container.encode(minimumImprovement, forKey: .minimumImprovement)
        try container.encode(restoresBestModel, forKey: .restoresBestModel)
    }

    private enum CodingKeys: String, CodingKey {
        case patience
        case minimumImprovement
        case restoresBestModel
    }
}

/// Lifecycle controls shared by CPU and Metal dense trainers.
public struct DenseTrainingOptions: Sendable, Hashable, Codable {
    /// Optional early-stopping behavior.
    public let earlyStopping: DenseEarlyStoppingConfiguration?
    /// Epoch interval at which progress updates include resumable checkpoints.
    public let checkpointInterval: Int?

    /// Creates lifecycle controls with optional early stopping and checkpoint emission.
    ///
    /// - Throws: ``ML5Error/invalidTrainingConfiguration(reason:)`` when a checkpoint interval
    ///   is present but not positive.
    public init(
        earlyStopping: DenseEarlyStoppingConfiguration? = nil,
        checkpointInterval: Int? = nil
    ) throws {
        guard checkpointInterval.map({ $0 > 0 }) ?? true else {
            throw ML5Error.invalidTrainingConfiguration(
                reason: "Checkpoint intervals must be greater than zero."
            )
        }
        self.earlyStopping = earlyStopping
        self.checkpointInterval = checkpointInterval
    }

    private init(standard: Void) {
        earlyStopping = nil
        checkpointInterval = nil
    }

    /// Lifecycle controls that run every configured epoch without intermediate checkpoints.
    public static let standard = Self(standard: ())

    /// Decodes and revalidates persisted lifecycle controls.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            earlyStopping: container.decodeIfPresent(
                DenseEarlyStoppingConfiguration.self,
                forKey: .earlyStopping
            ),
            checkpointInterval: container.decodeIfPresent(Int.self, forKey: .checkpointInterval)
        )
    }

    /// Encodes early-stopping and checkpoint-emission controls.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(earlyStopping, forKey: .earlyStopping)
        try container.encodeIfPresent(checkpointInterval, forKey: .checkpointInterval)
    }

    private enum CodingKeys: String, CodingKey {
        case earlyStopping
        case checkpointInterval
    }
}

/// A serializable optimizer-complete snapshot that can resume dense training exactly.
public struct DenseTrainingCheckpoint: Sendable, Hashable, Codable {
    /// Checkpoint schema version written by this release.
    public static let currentFormatVersion = 1
    /// Persisted schema version used to reject unsupported future checkpoint layouts.
    public let formatVersion: Int
    /// Training architecture and hyperparameters captured by the checkpoint.
    public let configuration: DenseNetworkConfiguration
    /// Lifecycle controls captured by the checkpoint.
    public let options: DenseTrainingOptions
    /// Number of fully completed epochs.
    public let completedEpochs: Int
    /// Metrics for every completed epoch.
    public let history: [DenseEpochMetrics]
    /// Number of samples whose ordered content is bound to this checkpoint.
    public let sampleCount: Int
    /// Stable identity of the ordered training samples.
    public let sampleFingerprint: UInt64

    /// Compute backend whose numerical state must be used when resuming.
    public let backend: DenseTrainingBackend
    let optimizerStep: Int
    let trainingIndices: [Int]
    let validationIndices: [Int]
    let layers: [DenseLayerTrainingState]
    let bestLoss: Double?
    let bestLayers: [DenseLayerTrainingState]?
    let staleEpochCount: Int

    /// Reconstructs the current immutable model stored in this checkpoint.
    ///
    /// This is the current optimizer state, which can differ from a best model restored in an
    /// early-stopped result.
    public func makeModel() throws -> DenseNetworkModel {
        try DenseNetworkModel(
            configuration: configuration,
            layers: layers.map(\.parameters)
        )
    }

    init(
        formatVersion: Int = Self.currentFormatVersion,
        configuration: DenseNetworkConfiguration,
        options: DenseTrainingOptions,
        completedEpochs: Int,
        history: [DenseEpochMetrics],
        sampleCount: Int,
        sampleFingerprint: UInt64,
        backend: DenseTrainingBackend,
        optimizerStep: Int,
        trainingIndices: [Int],
        validationIndices: [Int],
        layers: [DenseLayerTrainingState],
        bestLoss: Double?,
        bestLayers: [DenseLayerTrainingState]?,
        staleEpochCount: Int
    ) throws {
        guard formatVersion == Self.currentFormatVersion else {
            throw ML5Error.invalidTrainingCheckpoint(
                reason: "Unsupported checkpoint format version \(formatVersion)."
            )
        }
        guard (0...configuration.epochs).contains(completedEpochs) else {
            throw ML5Error.invalidTrainingCheckpoint(
                reason: "Completed epochs must be within the configured epoch range."
            )
        }
        guard history.count == completedEpochs,
            history.enumerated().allSatisfy({ $0.offset + 1 == $0.element.epoch })
        else {
            throw ML5Error.invalidTrainingCheckpoint(
                reason: "History must contain one ordered metric per completed epoch."
            )
        }
        guard sampleCount > 0 else {
            throw ML5Error.invalidTrainingCheckpoint(reason: "Sample count must be positive.")
        }
        let allIndices = trainingIndices + validationIndices
        guard allIndices.sorted() == Array(0..<sampleCount) else {
            throw ML5Error.invalidTrainingCheckpoint(
                reason: "Training and validation indices must partition every sample exactly once."
            )
        }
        let expectedValidationCount = Int(
            Double(sampleCount) * configuration.validationFraction
        )
        guard validationIndices.count == expectedValidationCount else {
            throw ML5Error.invalidTrainingCheckpoint(
                reason: "Validation indices do not match the configured fraction."
            )
        }
        guard optimizerStep >= 0, staleEpochCount >= 0 else {
            throw ML5Error.invalidTrainingCheckpoint(
                reason: "Optimizer and early-stopping counters cannot be negative."
            )
        }
        guard bestLoss?.isFinite ?? true else {
            throw ML5Error.invalidTrainingCheckpoint(reason: "Best loss must be finite.")
        }
        guard (bestLayers == nil) == (bestLoss == nil) else {
            throw ML5Error.invalidTrainingCheckpoint(
                reason: "Best loss and best parameter state must be present together."
            )
        }
        _ = try DenseNetworkModel(
            configuration: configuration,
            layers: layers.map(\.parameters)
        )
        if let bestLayers {
            _ = try DenseNetworkModel(
                configuration: configuration,
                layers: bestLayers.map(\.parameters)
            )
        }
        self.formatVersion = formatVersion
        self.configuration = configuration
        self.options = options
        self.completedEpochs = completedEpochs
        self.history = history
        self.sampleCount = sampleCount
        self.sampleFingerprint = sampleFingerprint
        self.backend = backend
        self.optimizerStep = optimizerStep
        self.trainingIndices = trainingIndices
        self.validationIndices = validationIndices
        self.layers = layers
        self.bestLoss = bestLoss
        self.bestLayers = bestLayers
        self.staleEpochCount = staleEpochCount
    }

    /// Decodes and revalidates a persisted optimizer checkpoint.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            formatVersion: container.decode(Int.self, forKey: .formatVersion),
            configuration: container.decode(
                DenseNetworkConfiguration.self,
                forKey: .configuration
            ),
            options: container.decode(DenseTrainingOptions.self, forKey: .options),
            completedEpochs: container.decode(Int.self, forKey: .completedEpochs),
            history: container.decode([DenseEpochMetrics].self, forKey: .history),
            sampleCount: container.decode(Int.self, forKey: .sampleCount),
            sampleFingerprint: container.decode(UInt64.self, forKey: .sampleFingerprint),
            backend: container.decode(DenseTrainingBackend.self, forKey: .backend),
            optimizerStep: container.decode(Int.self, forKey: .optimizerStep),
            trainingIndices: container.decode([Int].self, forKey: .trainingIndices),
            validationIndices: container.decode([Int].self, forKey: .validationIndices),
            layers: container.decode([DenseLayerTrainingState].self, forKey: .layers),
            bestLoss: container.decodeIfPresent(Double.self, forKey: .bestLoss),
            bestLayers: container.decodeIfPresent(
                [DenseLayerTrainingState].self,
                forKey: .bestLayers
            ),
            staleEpochCount: container.decode(Int.self, forKey: .staleEpochCount)
        )
    }

    /// Encodes model parameters, optimizer moments, partitions, and lifecycle state.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(formatVersion, forKey: .formatVersion)
        try container.encode(configuration, forKey: .configuration)
        try container.encode(options, forKey: .options)
        try container.encode(completedEpochs, forKey: .completedEpochs)
        try container.encode(history, forKey: .history)
        try container.encode(sampleCount, forKey: .sampleCount)
        try container.encode(sampleFingerprint, forKey: .sampleFingerprint)
        try container.encode(backend, forKey: .backend)
        try container.encode(optimizerStep, forKey: .optimizerStep)
        try container.encode(trainingIndices, forKey: .trainingIndices)
        try container.encode(validationIndices, forKey: .validationIndices)
        try container.encode(layers, forKey: .layers)
        try container.encodeIfPresent(bestLoss, forKey: .bestLoss)
        try container.encodeIfPresent(bestLayers, forKey: .bestLayers)
        try container.encode(staleEpochCount, forKey: .staleEpochCount)
    }

    private enum CodingKeys: String, CodingKey {
        case formatVersion
        case configuration
        case options
        case completedEpochs
        case history
        case sampleCount
        case sampleFingerprint
        case backend
        case optimizerStep
        case trainingIndices
        case validationIndices
        case layers
        case bestLoss
        case bestLayers
        case staleEpochCount
    }
}

/// An immutable epoch update delivered by a dense trainer.
public struct DenseTrainingProgress: Sendable, Hashable {
    /// Metrics for the epoch that just completed.
    public let metrics: DenseEpochMetrics
    /// Total epoch limit in the training configuration.
    public let totalEpochs: Int
    /// Compute backend executing the training run.
    public let backend: DenseTrainingBackend
    /// Resumable state when the completed epoch matches the checkpoint interval.
    public let checkpoint: DenseTrainingCheckpoint?
}

/// An asynchronous callback invoked after each completed dense-training epoch.
public typealias DenseTrainingProgressHandler =
    @Sendable (DenseTrainingProgress) async throws -> Void

/// The immutable model and complete loss history produced by dense training.
public struct DenseTrainingResult: Sendable, Hashable {
    /// Trained immutable model.
    public let model: DenseNetworkModel
    /// One metrics value for every completed epoch.
    public let history: [DenseEpochMetrics]
    /// Compute backend that produced the result.
    public let backend: DenseTrainingBackend
    /// Whether the epoch limit completed or early stopping ended the run.
    public let stopReason: DenseTrainingStopReason
    /// Optimizer-complete state at the final executed epoch.
    public let checkpoint: DenseTrainingCheckpoint
}

/// A deterministic reference trainer for small dense classification and regression models.
///
/// This implementation prioritizes readable numerical semantics and reproducible tests. Use an
/// accelerated training device for large datasets after selecting a backend that supports it.
public struct DenseCPUTrainer: Sendable {
    /// Creates a stateless CPU reference trainer.
    public init() {}

    /// Fits an immutable dense model using deterministic mini-batch backpropagation.
    ///
    /// Training validates every sample against the configuration before allocating parameters.
    /// Cancellation is checked between samples, batches, and epochs.
    ///
    /// - Throws: ``ML5Error`` for invalid samples, incompatible targets, numerical instability,
    ///   or model construction failures; `CancellationError` when cancelled.
    public func train(
        _ samples: [DenseTrainingSample],
        configuration: DenseNetworkConfiguration
    ) async throws -> DenseTrainingResult {
        try await train(
            samples,
            configuration: configuration,
            options: .standard,
            progress: nil
        )
    }

    /// Fits an immutable dense model with progress, early stopping, and resumable checkpoints.
    ///
    /// The optional progress handler executes serially after each epoch. Throwing from the
    /// handler immediately ends training with that error.
    ///
    /// - Throws: ``ML5Error`` for invalid inputs or state; `CancellationError` when cancelled;
    ///   or an error propagated by `progress`.
    public func train(
        _ samples: [DenseTrainingSample],
        configuration: DenseNetworkConfiguration,
        options: DenseTrainingOptions,
        progress: DenseTrainingProgressHandler? = nil
    ) async throws -> DenseTrainingResult {
        try Task.checkCancellation()
        guard samples.isEmpty == false else {
            throw ML5Error.invalidTrainingSamples
        }
        _ = try samples.map { try Self.prepare($0, configuration: configuration) }
        let layers = try Self.initialize(configuration: configuration)
        var indices = Array(samples.indices)
        Self.shuffle(&indices, seed: configuration.seed)
        let validationCount = Int(Double(indices.count) * configuration.validationFraction)
        let trainingCount = indices.count - validationCount
        let trainingIndices = Array(indices[..<trainingCount])
        let validationIndices = Array(indices[trainingCount...])
        let checkpoint = try DenseTrainingCheckpoint(
            configuration: configuration,
            options: options,
            completedEpochs: 0,
            history: [],
            sampleCount: samples.count,
            sampleFingerprint: try Self.fingerprint(samples),
            backend: .cpu,
            optimizerStep: 0,
            trainingIndices: trainingIndices,
            validationIndices: validationIndices,
            layers: layers.map(DenseLayerTrainingState.init(layer:)),
            bestLoss: nil,
            bestLayers: nil,
            staleEpochCount: 0
        )
        return try await Self.run(
            samples,
            checkpoint: checkpoint,
            progress: progress,
            gradientProvider: { batch, prepared, layers, configuration in
                var gradients = layers.map(LayerGradients.init(layer:))
                for sampleIndex in batch {
                    try Task.checkCancellation()
                    let pass = Self.forward(
                        prepared[sampleIndex].input,
                        layers: layers,
                        configuration: configuration
                    )
                    try Self.accumulateGradients(
                        pass: pass,
                        target: prepared[sampleIndex].target,
                        layers: layers,
                        configuration: configuration,
                        gradients: &gradients
                    )
                }
                return (gradients, batch.count)
            }
        )
    }

    /// Continues a CPU checkpoint with the exact saved partitions, optimizer moments, and epoch.
    ///
    /// Samples must have the same order and content used to create the checkpoint.
    ///
    /// - Throws: ``ML5Error/invalidTrainingCheckpoint(reason:)`` for a different backend or
    ///   dataset; otherwise the errors documented by ``train(_:configuration:options:progress:)``.
    public func resume(
        _ checkpoint: DenseTrainingCheckpoint,
        samples: [DenseTrainingSample],
        progress: DenseTrainingProgressHandler? = nil
    ) async throws -> DenseTrainingResult {
        guard checkpoint.backend == .cpu else {
            throw ML5Error.invalidTrainingCheckpoint(
                reason: "A CPU trainer can resume only a CPU checkpoint."
            )
        }
        return try await Self.run(
            samples,
            checkpoint: checkpoint,
            progress: progress,
            gradientProvider: { batch, prepared, layers, configuration in
                var gradients = layers.map(LayerGradients.init(layer:))
                for sampleIndex in batch {
                    try Task.checkCancellation()
                    let pass = Self.forward(
                        prepared[sampleIndex].input,
                        layers: layers,
                        configuration: configuration
                    )
                    try Self.accumulateGradients(
                        pass: pass,
                        target: prepared[sampleIndex].target,
                        layers: layers,
                        configuration: configuration,
                        gradients: &gradients
                    )
                }
                return (gradients, batch.count)
            }
        )
    }

    fileprivate static func run(
        _ samples: [DenseTrainingSample],
        checkpoint: DenseTrainingCheckpoint,
        progress: DenseTrainingProgressHandler?,
        gradientProvider:
            @Sendable (
                _ batch: [Int],
                _ samples: [PreparedSample],
                _ layers: [TrainableLayer],
                _ configuration: DenseNetworkConfiguration
            ) throws -> ([LayerGradients], Int)
    ) async throws -> DenseTrainingResult {
        try Task.checkCancellation()
        guard samples.count == checkpoint.sampleCount,
            try Self.fingerprint(samples) == checkpoint.sampleFingerprint
        else {
            throw ML5Error.invalidTrainingCheckpoint(
                reason: "The ordered samples do not match the checkpoint dataset."
            )
        }
        let configuration = checkpoint.configuration
        let prepared = try samples.map { try Self.prepare($0, configuration: configuration) }
        var layers = checkpoint.layers.map(\.trainableLayer)
        var history = checkpoint.history
        history.reserveCapacity(configuration.epochs)
        var optimizerStep = checkpoint.optimizerStep
        var bestLoss = checkpoint.bestLoss
        var bestLayers = checkpoint.bestLayers?.map(\.trainableLayer)
        var staleEpochCount = checkpoint.staleEpochCount
        var stopReason = DenseTrainingStopReason.completed

        for epochIndex in checkpoint.completedEpochs..<configuration.epochs {
            try Task.checkCancellation()
            var epochIndices = checkpoint.trainingIndices
            Self.shuffle(&epochIndices, seed: configuration.seed &+ UInt64(epochIndex) &+ 1)
            for batchStart in stride(
                from: 0,
                to: epochIndices.count,
                by: configuration.batchSize
            ) {
                try Task.checkCancellation()
                let batchEnd = min(batchStart + configuration.batchSize, epochIndices.count)
                let (gradients, gradientSampleCount) = try gradientProvider(
                    Array(epochIndices[batchStart..<batchEnd]),
                    prepared,
                    layers,
                    configuration
                )
                optimizerStep += 1
                try Self.apply(
                    gradients: gradients,
                    sampleCount: gradientSampleCount,
                    step: optimizerStep,
                    configuration: configuration,
                    layers: &layers
                )
            }

            let trainingLoss = try Self.meanLoss(
                indices: checkpoint.trainingIndices,
                samples: prepared,
                layers: layers,
                configuration: configuration
            )
            let validationLoss =
                checkpoint.validationIndices.isEmpty
                ? nil
                : try Self.meanLoss(
                    indices: checkpoint.validationIndices,
                    samples: prepared,
                    layers: layers,
                    configuration: configuration
                )
            let metrics = try DenseEpochMetrics(
                epoch: epochIndex + 1,
                trainingLoss: trainingLoss,
                validationLoss: validationLoss
            )
            history.append(metrics)

            var shouldStop = false
            if let earlyStopping = checkpoint.options.earlyStopping {
                let monitoredLoss = validationLoss ?? trainingLoss
                if bestLoss.map({ $0 - monitoredLoss > earlyStopping.minimumImprovement }) ?? true {
                    bestLoss = monitoredLoss
                    bestLayers = layers
                    staleEpochCount = 0
                } else {
                    staleEpochCount += 1
                    shouldStop = staleEpochCount >= earlyStopping.patience
                }
            }

            let includesCheckpoint =
                checkpoint.options.checkpointInterval.map {
                    metrics.epoch.isMultiple(of: $0)
                } ?? false
            let currentCheckpoint = try Self.checkpoint(
                template: checkpoint,
                completedEpochs: metrics.epoch,
                history: history,
                optimizerStep: optimizerStep,
                layers: layers,
                bestLoss: bestLoss,
                bestLayers: bestLayers,
                staleEpochCount: staleEpochCount
            )
            if let progress {
                try await progress(
                    DenseTrainingProgress(
                        metrics: metrics,
                        totalEpochs: configuration.epochs,
                        backend: checkpoint.backend,
                        checkpoint: includesCheckpoint ? currentCheckpoint : nil
                    )
                )
            }
            if shouldStop {
                stopReason = .earlyStopping
                break
            }
            await Task.yield()
        }
        try Task.checkCancellation()
        let finalCheckpoint = try Self.checkpoint(
            template: checkpoint,
            completedEpochs: history.count,
            history: history,
            optimizerStep: optimizerStep,
            layers: layers,
            bestLoss: bestLoss,
            bestLayers: bestLayers,
            staleEpochCount: staleEpochCount
        )
        let resultLayers: [TrainableLayer]
        if stopReason == .earlyStopping,
            checkpoint.options.earlyStopping?.restoresBestModel == true,
            let bestLayers
        {
            resultLayers = bestLayers
        } else {
            resultLayers = layers
        }
        return DenseTrainingResult(
            model: try Self.model(configuration: configuration, layers: resultLayers),
            history: history,
            backend: checkpoint.backend,
            stopReason: stopReason,
            checkpoint: finalCheckpoint
        )
    }

    fileprivate static func checkpoint(
        template: DenseTrainingCheckpoint,
        completedEpochs: Int,
        history: [DenseEpochMetrics],
        optimizerStep: Int,
        layers: [TrainableLayer],
        bestLoss: Double?,
        bestLayers: [TrainableLayer]?,
        staleEpochCount: Int
    ) throws -> DenseTrainingCheckpoint {
        try DenseTrainingCheckpoint(
            configuration: template.configuration,
            options: template.options,
            completedEpochs: completedEpochs,
            history: history,
            sampleCount: template.sampleCount,
            sampleFingerprint: template.sampleFingerprint,
            backend: template.backend,
            optimizerStep: optimizerStep,
            trainingIndices: template.trainingIndices,
            validationIndices: template.validationIndices,
            layers: layers.map(DenseLayerTrainingState.init(layer:)),
            bestLoss: bestLoss,
            bestLayers: bestLayers?.map(DenseLayerTrainingState.init(layer:)),
            staleEpochCount: staleEpochCount
        )
    }

    fileprivate static func fingerprint(_ samples: [DenseTrainingSample]) throws -> UInt64 {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(samples)
        return data.reduce(0xcbf2_9ce4_8422_2325) { hash, byte in
            (hash ^ UInt64(byte)) &* 0x0000_0100_0000_01B3
        }
    }

    fileprivate static func prepare(
        _ sample: DenseTrainingSample,
        configuration: DenseNetworkConfiguration
    ) throws -> PreparedSample {
        guard sample.targets.count == configuration.outputNames.count else {
            throw ML5Error.invalidTrainingSample(
                reason:
                    "Expected \(configuration.outputNames.count) targets, but received \(sample.targets.count)."
            )
        }
        let input = try configuration.inputFeatures.map { name in
            guard let value = sample.features[name] else {
                throw ML5Error.missingFeature(name.rawValue)
            }
            guard let number = value.numericValue else {
                throw ML5Error.featureKindMismatch(
                    name: name.rawValue,
                    expected: .number,
                    actual: value.kind
                )
            }
            return number
        }
        switch configuration.loss {
        case .meanSquaredError:
            break
        case .categoricalCrossEntropy:
            guard
                sample.targets.allSatisfy({ (0...1).contains($0) }),
                abs(sample.targets.reduce(0, +) - 1) <= 1e-9
            else {
                throw ML5Error.invalidTrainingSample(
                    reason: "Categorical targets must form a probability distribution."
                )
            }
        case .binaryCrossEntropy:
            guard sample.targets.allSatisfy({ (0...1).contains($0) }) else {
                throw ML5Error.invalidTrainingSample(
                    reason: "Binary targets must be in the closed range from zero to one."
                )
            }
        }
        return PreparedSample(input: input, target: sample.targets)
    }

    fileprivate static func initialize(configuration: DenseNetworkConfiguration) throws
        -> [TrainableLayer]
    {
        let widths =
            configuration.hiddenLayers.map(\.neuronCount) + [configuration.outputNames.count]
        var inputCount = configuration.inputFeatures.count
        var generator = DenseSeededGenerator(state: configuration.seed)
        var layers: [TrainableLayer] = []
        layers.reserveCapacity(widths.count)
        for outputCount in widths {
            guard inputCount <= Int.max / outputCount else {
                throw ML5Error.invalidModel(reason: "Layer dimensions overflow their weight count.")
            }
            let weightCount = inputCount * outputCount
            let weights: [Double]
            switch configuration.weightInitialization {
            case .glorotUniform:
                let limit = sqrt(6 / Double(inputCount + outputCount))
                weights = (0..<weightCount).map { _ in generator.uniform(in: -limit...limit) }
            case .heNormal:
                let standardDeviation = sqrt(2 / Double(inputCount))
                weights = (0..<weightCount).map { _ in
                    generator.normal() * standardDeviation
                }
            case .zeros:
                weights = Array(repeating: 0, count: weightCount)
            }
            layers.append(
                TrainableLayer(
                    inputCount: inputCount,
                    outputCount: outputCount,
                    weights: weights,
                    biases: Array(repeating: 0, count: outputCount)
                )
            )
            inputCount = outputCount
        }
        return layers
    }

    private static func forward(
        _ input: [Double],
        layers: [TrainableLayer],
        configuration: DenseNetworkConfiguration
    ) -> ForwardPass {
        var activations = [input]
        var preactivations: [[Double]] = []
        preactivations.reserveCapacity(layers.count)
        for (index, layer) in layers.enumerated() {
            let values = layer.affine(activations[index])
            preactivations.append(values)
            let activation =
                index < configuration.hiddenLayers.count
                ? configuration.hiddenLayers[index].activation
                : configuration.outputActivation
            activations.append(DenseNetworkMath.activate(values, using: activation))
        }
        return ForwardPass(activations: activations, preactivations: preactivations)
    }

    private static func accumulateGradients(
        pass: ForwardPass,
        target: [Double],
        layers: [TrainableLayer],
        configuration: DenseNetworkConfiguration,
        gradients: inout [LayerGradients]
    ) throws {
        let prediction = pass.activations[pass.activations.count - 1]
        var delta: [Double]
        switch configuration.loss {
        case .meanSquaredError:
            delta = zip(prediction, target).map { 2 * ($0 - $1) / Double(target.count) }
            delta = zip(delta, prediction).enumerated().map { index, pair in
                pair.0
                    * DenseNetworkMath.derivative(
                        preactivation: pass.preactivations[pass.preactivations.count - 1][index],
                        activation: pair.1,
                        using: configuration.outputActivation
                    )
            }
        case .categoricalCrossEntropy, .binaryCrossEntropy:
            delta = zip(prediction, target).map { ($0 - $1) / Double(target.count) }
        }

        for layerIndex in layers.indices.reversed() {
            let previousActivation = pass.activations[layerIndex]
            for outputIndex in 0..<layers[layerIndex].outputCount {
                gradients[layerIndex].biases[outputIndex] += delta[outputIndex]
                let rowStart = outputIndex * layers[layerIndex].inputCount
                for inputIndex in 0..<layers[layerIndex].inputCount {
                    gradients[layerIndex].weights[rowStart + inputIndex] +=
                        delta[outputIndex] * previousActivation[inputIndex]
                }
            }

            if layerIndex > 0 {
                var previousDelta = Array(repeating: 0.0, count: layers[layerIndex].inputCount)
                for inputIndex in previousDelta.indices {
                    for outputIndex in 0..<layers[layerIndex].outputCount {
                        previousDelta[inputIndex] +=
                            layers[layerIndex].weights[
                                outputIndex * layers[layerIndex].inputCount + inputIndex
                            ] * delta[outputIndex]
                    }
                    let function = configuration.hiddenLayers[layerIndex - 1].activation
                    previousDelta[inputIndex] *= DenseNetworkMath.derivative(
                        preactivation: pass.preactivations[layerIndex - 1][inputIndex],
                        activation: pass.activations[layerIndex][inputIndex],
                        using: function
                    )
                }
                delta = previousDelta
            }
        }
    }

    fileprivate static func apply(
        gradients: [LayerGradients],
        sampleCount: Int,
        step: Int,
        configuration: DenseNetworkConfiguration,
        layers: inout [TrainableLayer]
    ) throws {
        let scale = 1 / Double(sampleCount)
        for layerIndex in layers.indices {
            switch configuration.optimizer.kind {
            case .stochasticGradientDescent:
                for index in layers[layerIndex].weights.indices {
                    let gradient = gradients[layerIndex].weights[index] * scale
                    layers[layerIndex].weightVelocity[index] =
                        configuration.optimizer.momentum
                        * layers[layerIndex].weightVelocity[index]
                        - configuration.learningRate * gradient
                    layers[layerIndex].weights[index] +=
                        layers[layerIndex].weightVelocity[index]
                }
                for index in layers[layerIndex].biases.indices {
                    let gradient = gradients[layerIndex].biases[index] * scale
                    layers[layerIndex].biasVelocity[index] =
                        configuration.optimizer.momentum * layers[layerIndex].biasVelocity[index]
                        - configuration.learningRate * gradient
                    layers[layerIndex].biases[index] += layers[layerIndex].biasVelocity[index]
                }
            case .adam:
                for index in layers[layerIndex].weights.indices {
                    let gradient = gradients[layerIndex].weights[index] * scale
                    layers[layerIndex].weightFirstMoment[index] =
                        configuration.optimizer.beta1
                        * layers[layerIndex].weightFirstMoment[index]
                        + (1 - configuration.optimizer.beta1) * gradient
                    layers[layerIndex].weightSecondMoment[index] =
                        configuration.optimizer.beta2
                        * layers[layerIndex].weightSecondMoment[index]
                        + (1 - configuration.optimizer.beta2) * gradient * gradient
                    layers[layerIndex].weights[index] -= Self.adamUpdate(
                        firstMoment: layers[layerIndex].weightFirstMoment[index],
                        secondMoment: layers[layerIndex].weightSecondMoment[index],
                        step: step,
                        configuration: configuration
                    )
                }
                for index in layers[layerIndex].biases.indices {
                    let gradient = gradients[layerIndex].biases[index] * scale
                    layers[layerIndex].biasFirstMoment[index] =
                        configuration.optimizer.beta1 * layers[layerIndex].biasFirstMoment[index]
                        + (1 - configuration.optimizer.beta1) * gradient
                    layers[layerIndex].biasSecondMoment[index] =
                        configuration.optimizer.beta2 * layers[layerIndex].biasSecondMoment[index]
                        + (1 - configuration.optimizer.beta2) * gradient * gradient
                    layers[layerIndex].biases[index] -= Self.adamUpdate(
                        firstMoment: layers[layerIndex].biasFirstMoment[index],
                        secondMoment: layers[layerIndex].biasSecondMoment[index],
                        step: step,
                        configuration: configuration
                    )
                }
            }
            guard
                layers[layerIndex].weights.allSatisfy(\.isFinite),
                layers[layerIndex].biases.allSatisfy(\.isFinite)
            else {
                throw ML5Error.invalidModel(
                    reason: "Training produced nonfinite model parameters."
                )
            }
        }
    }

    private static func adamUpdate(
        firstMoment: Double,
        secondMoment: Double,
        step: Int,
        configuration: DenseNetworkConfiguration
    ) -> Double {
        let correctedFirst =
            firstMoment / (1 - pow(configuration.optimizer.beta1, Double(step)))
        let correctedSecond =
            secondMoment / (1 - pow(configuration.optimizer.beta2, Double(step)))
        return configuration.learningRate * correctedFirst
            / (sqrt(correctedSecond) + configuration.optimizer.epsilon)
    }

    fileprivate static func meanLoss(
        indices: [Int],
        samples: [PreparedSample],
        layers: [TrainableLayer],
        configuration: DenseNetworkConfiguration
    ) throws -> Double {
        var total = 0.0
        for index in indices {
            try Task.checkCancellation()
            let pass = Self.forward(
                samples[index].input,
                layers: layers,
                configuration: configuration
            )
            let prediction = pass.activations[pass.activations.count - 1]
            total += Self.loss(
                prediction: prediction,
                target: samples[index].target,
                function: configuration.loss
            )
        }
        let result = total / Double(indices.count)
        guard result.isFinite else {
            throw ML5Error.invalidModel(reason: "Training produced nonfinite loss.")
        }
        return result
    }

    private static func loss(
        prediction: [Double],
        target: [Double],
        function: TrainingLoss
    ) -> Double {
        switch function {
        case .meanSquaredError:
            return zip(prediction, target).reduce(0) { $0 + ($1.0 - $1.1) * ($1.0 - $1.1) }
                / Double(target.count)
        case .categoricalCrossEntropy:
            return -zip(prediction, target).reduce(0) {
                $0 + $1.1 * log(max($1.0, Double.leastNonzeroMagnitude))
            }
        case .binaryCrossEntropy:
            return -zip(prediction, target).reduce(0) {
                let probability = min(max($1.0, Double.leastNonzeroMagnitude), 1 - Double.ulpOfOne)
                return $0 + $1.1 * log(probability) + (1 - $1.1) * log(1 - probability)
            } / Double(target.count)
        }
    }

    fileprivate static func model(
        configuration: DenseNetworkConfiguration,
        layers: [TrainableLayer]
    ) throws -> DenseNetworkModel {
        try DenseNetworkModel(
            configuration: configuration,
            layers: layers.map {
                try DenseLayerParameters(
                    inputCount: $0.inputCount,
                    outputCount: $0.outputCount,
                    weights: $0.weights,
                    biases: $0.biases
                )
            }
        )
    }

    fileprivate static func shuffle(_ values: inout [Int], seed: UInt64) {
        guard values.count > 1 else { return }
        var generator = DenseSeededGenerator(state: seed)
        for index in stride(from: values.count - 1, through: 1, by: -1) {
            let destination = Int(generator.next() % UInt64(index + 1))
            if destination != index {
                values.swapAt(index, destination)
            }
        }
    }
}

#if canImport(Metal) && canImport(MetalPerformanceShadersGraph)
    /// An actor-isolated dense trainer using Metal Performance Shaders Graph automatic
    /// differentiation on a selected Metal device.
    ///
    /// MPSGraph executes batched forward and gradient graphs on Metal. Parameter updates and
    /// epoch metrics intentionally share the CPU reference implementation so both backends emit
    /// the same validated ``DenseNetworkModel`` representation.
    public actor DenseMPSGraphTrainer {
        /// Human-readable name reported by the selected Metal device.
        public nonisolated let deviceName: String

        private let device: any MTLDevice
        private let commandQueue: any MTLCommandQueue

        /// Creates a trainer using the system's default Metal device.
        ///
        /// - Throws: ``ML5Error/trainingAcceleratorUnavailable(reason:)`` when no Metal device or
        ///   command queue can be created.
        public init() throws {
            try self.init(testDevice: MTLCreateSystemDefaultDevice())
        }

        /// Creates a trainer using an explicitly selected Metal device.
        ///
        /// - Throws: ``ML5Error/trainingAcceleratorUnavailable(reason:)`` when the device cannot
        ///   create a command queue.
        public init(device: any MTLDevice) throws {
            try self.init(testDevice: device)
        }

        init(
            testDevice device: (any MTLDevice)?,
            makeCommandQueue: (any MTLDevice) -> (any MTLCommandQueue)? = { $0.makeCommandQueue() }
        ) throws {
            #if targetEnvironment(simulator)
                throw ML5Error.trainingAcceleratorUnavailable(
                    reason: "MPSGraph training requires macOS or a physical Apple device."
                )
            #else
                guard let device else {
                    throw ML5Error.trainingAcceleratorUnavailable(
                        reason: "No Metal device is available."
                    )
                }
                guard let commandQueue = makeCommandQueue(device) else {
                    throw ML5Error.trainingAcceleratorUnavailable(
                        reason: "The Metal device could not create a command queue."
                    )
                }
                self.device = device
                self.commandQueue = commandQueue
                deviceName = device.name
            #endif
        }

        /// Fits a dense model using MPSGraph forward evaluation and automatic differentiation.
        ///
        /// Cancellation is checked between graph executions and epoch metric passes. Results use
        /// the same immutable model and history types as ``DenseCPUTrainer``.
        ///
        /// - Throws: ``ML5Error`` for invalid samples, graph output failures, numerical
        ///   instability, or model construction failures; `CancellationError` when cancelled.
        public func train(
            _ samples: [DenseTrainingSample],
            configuration: DenseNetworkConfiguration
        ) async throws -> DenseTrainingResult {
            try await train(
                samples,
                configuration: configuration,
                options: .standard,
                progress: nil
            )
        }

        /// Fits a dense model on Metal with progress, early stopping, and resumable checkpoints.
        ///
        /// - Throws: ``ML5Error`` for invalid inputs, graph failures, or state; `CancellationError`
        ///   when cancelled; or an error propagated by `progress`.
        public func train(
            _ samples: [DenseTrainingSample],
            configuration: DenseNetworkConfiguration,
            options: DenseTrainingOptions,
            progress: DenseTrainingProgressHandler? = nil
        ) async throws -> DenseTrainingResult {
            try Task.checkCancellation()
            guard samples.isEmpty == false else {
                throw ML5Error.invalidTrainingSamples
            }
            _ = try samples.map {
                try DenseCPUTrainer.prepare($0, configuration: configuration)
            }
            let layers = try DenseCPUTrainer.initialize(configuration: configuration)
            var indices = Array(samples.indices)
            DenseCPUTrainer.shuffle(&indices, seed: configuration.seed)
            let validationCount = Int(Double(indices.count) * configuration.validationFraction)
            let trainingCount = indices.count - validationCount
            let trainingIndices = Array(indices[..<trainingCount])
            let validationIndices = Array(indices[trainingCount...])
            let checkpoint = try DenseTrainingCheckpoint(
                configuration: configuration,
                options: options,
                completedEpochs: 0,
                history: [],
                sampleCount: samples.count,
                sampleFingerprint: try DenseCPUTrainer.fingerprint(samples),
                backend: .metal(deviceName: deviceName),
                optimizerStep: 0,
                trainingIndices: trainingIndices,
                validationIndices: validationIndices,
                layers: layers.map(DenseLayerTrainingState.init(layer:)),
                bestLoss: nil,
                bestLayers: nil,
                staleEpochCount: 0
            )
            return try await run(samples, checkpoint: checkpoint, progress: progress)
        }

        /// Continues a checkpoint made on this named Metal device.
        ///
        /// - Throws: ``ML5Error/invalidTrainingCheckpoint(reason:)`` when the checkpoint uses a
        ///   different backend or device; otherwise the errors documented by
        ///   ``train(_:configuration:options:progress:)``.
        public func resume(
            _ checkpoint: DenseTrainingCheckpoint,
            samples: [DenseTrainingSample],
            progress: DenseTrainingProgressHandler? = nil
        ) async throws -> DenseTrainingResult {
            guard checkpoint.backend == .metal(deviceName: deviceName) else {
                throw ML5Error.invalidTrainingCheckpoint(
                    reason: "A Metal trainer can resume only a checkpoint from the same device."
                )
            }
            return try await run(samples, checkpoint: checkpoint, progress: progress)
        }

        private func run(
            _ samples: [DenseTrainingSample],
            checkpoint: DenseTrainingCheckpoint,
            progress: DenseTrainingProgressHandler?
        ) async throws -> DenseTrainingResult {
            let selectedDevice = device
            let selectedCommandQueue = commandQueue
            return try await DenseCPUTrainer.run(
                samples,
                checkpoint: checkpoint,
                progress: progress,
                gradientProvider: { batch, prepared, layers, configuration in
                    (
                        try MPSGraphDenseBatch.gradients(
                            sampleIndices: batch,
                            samples: prepared,
                            layers: layers,
                            configuration: configuration,
                            device: selectedDevice,
                            commandQueue: selectedCommandQueue
                        ),
                        1
                    )
                }
            )
        }
    }

    enum MPSGraphDenseBatch {
        fileprivate static func gradients(
            sampleIndices: [Int],
            samples: [PreparedSample],
            layers: [TrainableLayer],
            configuration: DenseNetworkConfiguration,
            device: any MTLDevice,
            commandQueue: any MTLCommandQueue
        ) throws -> [LayerGradients] {
            let graph = MPSGraph()
            let graphDevice = MPSGraphDevice(mtlDevice: device)
            let inputTensor = graph.placeholder(
                shape: Self.shape([sampleIndices.count, configuration.inputFeatures.count]),
                dataType: .float32,
                name: "inputs"
            )
            let targetTensor = graph.placeholder(
                shape: Self.shape([sampleIndices.count, configuration.outputNames.count]),
                dataType: .float32,
                name: "targets"
            )
            var weightTensors: [MPSGraphTensor] = []
            var biasTensors: [MPSGraphTensor] = []
            var activation = inputTensor
            for (index, layer) in layers.enumerated() {
                let weight = graph.placeholder(
                    shape: Self.shape([layer.outputCount, layer.inputCount]),
                    dataType: .float32,
                    name: "weights_\(index)"
                )
                let bias = graph.placeholder(
                    shape: Self.shape([1, layer.outputCount]),
                    dataType: .float32,
                    name: "biases_\(index)"
                )
                weightTensors.append(weight)
                biasTensors.append(bias)
                let transposedWeight = graph.transposeTensor(
                    weight,
                    dimension: 0,
                    withDimension: 1,
                    name: nil
                )
                activation = graph.addition(
                    graph.matrixMultiplication(
                        primary: activation,
                        secondary: transposedWeight,
                        name: nil
                    ),
                    bias,
                    name: nil
                )
                let function =
                    index < configuration.hiddenLayers.count
                    ? configuration.hiddenLayers[index].activation
                    : configuration.outputActivation
                activation = Self.activate(activation, function: function, graph: graph)
            }

            let loss = Self.loss(
                prediction: activation,
                target: targetTensor,
                function: configuration.loss,
                graph: graph
            )
            let parameterTensors = weightTensors + biasTensors
            let gradientMap = graph.gradients(of: loss, with: parameterTensors, name: nil)
            let gradientTensors = try Self.requiredGradients(
                for: parameterTensors,
                from: gradientMap
            )

            let inputs = sampleIndices.flatMap { samples[$0].input }.map(Float.init)
            let targets = sampleIndices.flatMap { samples[$0].target }.map(Float.init)
            var feeds: [MPSGraphTensor: MPSGraphTensorData] = [
                inputTensor: Self.tensorData(
                    inputs,
                    shape: Self.shape([
                        sampleIndices.count, configuration.inputFeatures.count,
                    ]),
                    device: graphDevice
                ),
                targetTensor: Self.tensorData(
                    targets,
                    shape: Self.shape([sampleIndices.count, configuration.outputNames.count]),
                    device: graphDevice
                ),
            ]
            for (index, layer) in layers.enumerated() {
                feeds[weightTensors[index]] = Self.tensorData(
                    layer.weights.map(Float.init),
                    shape: Self.shape([layer.outputCount, layer.inputCount]),
                    device: graphDevice
                )
                feeds[biasTensors[index]] = Self.tensorData(
                    layer.biases.map(Float.init),
                    shape: Self.shape([1, layer.outputCount]),
                    device: graphDevice
                )
            }

            let results = graph.run(
                with: commandQueue,
                feeds: feeds,
                targetTensors: gradientTensors,
                targetOperations: nil
            )
            var gradients: [LayerGradients] = []
            gradients.reserveCapacity(layers.count)
            for index in layers.indices {
                gradients.append(
                    LayerGradients(
                        weights: try Self.values(
                            for: gradientTensors[index],
                            count: layers[index].weights.count,
                            results: results
                        ),
                        biases: try Self.values(
                            for: gradientTensors[layers.count + index],
                            count: layers[index].biases.count,
                            results: results
                        )
                    )
                )
            }
            return gradients
        }

        private static func activate(
            _ tensor: MPSGraphTensor,
            function: ActivationFunction,
            graph: MPSGraph
        ) -> MPSGraphTensor {
            switch function {
            case .linear:
                tensor
            case .rectifiedLinear:
                graph.reLU(with: tensor, name: nil)
            case .sigmoid:
                graph.sigmoid(with: tensor, name: nil)
            case .hyperbolicTangent:
                graph.tanh(with: tensor, name: nil)
            case .softmax:
                graph.softMax(with: tensor, axis: 1, name: nil)
            }
        }

        private static func loss(
            prediction: MPSGraphTensor,
            target: MPSGraphTensor,
            function: TrainingLoss,
            graph: MPSGraph
        ) -> MPSGraphTensor {
            switch function {
            case .meanSquaredError:
                return graph.mean(
                    of: graph.square(
                        with: graph.subtraction(prediction, target, name: nil),
                        name: nil
                    ),
                    axes: [0, 1],
                    name: nil
                )
            case .categoricalCrossEntropy:
                let clipped = Self.clamp(prediction, graph: graph)
                let terms = graph.multiplication(
                    target,
                    graph.logarithm(with: clipped, name: nil),
                    name: nil
                )
                let sampleLosses = graph.negative(
                    with: graph.reductionSum(with: terms, axes: [1], name: nil),
                    name: nil
                )
                return graph.mean(of: sampleLosses, axes: [0], name: nil)
            case .binaryCrossEntropy:
                let clipped = Self.clamp(prediction, graph: graph)
                let one = graph.constant(1, dataType: .float32)
                let positive = graph.multiplication(
                    target,
                    graph.logarithm(with: clipped, name: nil),
                    name: nil
                )
                let negative = graph.multiplication(
                    graph.subtraction(one, target, name: nil),
                    graph.logarithm(
                        with: graph.subtraction(one, clipped, name: nil),
                        name: nil
                    ),
                    name: nil
                )
                return graph.negative(
                    with: graph.mean(
                        of: graph.addition(positive, negative, name: nil),
                        axes: [0, 1],
                        name: nil
                    ),
                    name: nil
                )
            }
        }

        private static func clamp(_ tensor: MPSGraphTensor, graph: MPSGraph) -> MPSGraphTensor {
            graph.clamp(
                tensor,
                min: graph.constant(1e-7, dataType: .float32),
                max: graph.constant(1 - 1e-7, dataType: .float32),
                name: nil
            )
        }

        private static func tensorData(
            _ values: [Float],
            shape: [NSNumber],
            device: MPSGraphDevice
        ) -> MPSGraphTensorData {
            values.withUnsafeBytes {
                MPSGraphTensorData(
                    device: device,
                    data: Data($0),
                    shape: shape,
                    dataType: .float32
                )
            }
        }

        private static func shape(_ dimensions: [Int]) -> [NSNumber] {
            dimensions.map(NSNumber.init(value:))
        }

        static func requiredGradients(
            for parameters: [MPSGraphTensor],
            from gradients: [MPSGraphTensor: MPSGraphTensor]
        ) throws -> [MPSGraphTensor] {
            try parameters.map { parameter in
                guard let gradient = gradients[parameter] else {
                    throw ML5Error.trainingAcceleratorUnavailable(
                        reason: "MPSGraph did not produce every parameter gradient."
                    )
                }
                return gradient
            }
        }

        static func values(
            for tensor: MPSGraphTensor,
            count: Int,
            results: [MPSGraphTensor: MPSGraphTensorData]
        ) throws -> [Double] {
            guard let data = results[tensor] else {
                throw ML5Error.trainingAcceleratorUnavailable(
                    reason: "MPSGraph did not return a requested gradient."
                )
            }
            var values = Array(repeating: Float.zero, count: count)
            data.mpsndarray().readBytes(&values, strideBytes: nil)
            return values.map(Double.init)
        }
    }
#endif

/// Preferred compute device for high-level dense training.
public enum DenseTrainingDevicePreference: String, Sendable, Hashable, Codable {
    /// Select Metal when available and otherwise apply the configured fallback.
    case automatic
    /// Always use deterministic scalar CPU training.
    case cpu
    /// Request the system's default Metal device.
    case metal
}

/// Behavior when a requested automatic or Metal training device cannot be constructed.
public enum DenseTrainingFallback: String, Sendable, Hashable, Codable {
    /// Report accelerator construction failure to the caller.
    case none
    /// Continue with deterministic CPU training.
    case cpu
}

/// Explicit device-selection and fallback policy for ``DenseTrainer``.
public struct DenseTrainingExecutionPolicy: Sendable, Hashable, Codable {
    /// Preferred compute device.
    public let preference: DenseTrainingDevicePreference
    /// Behavior used only when Metal is unavailable before training begins.
    public let fallback: DenseTrainingFallback

    /// Creates an explicit device-selection policy.
    public init(
        preference: DenseTrainingDevicePreference = .automatic,
        fallback: DenseTrainingFallback = .cpu
    ) {
        self.preference = preference
        self.fallback = fallback
    }
}

/// A stateless high-level trainer that applies an explicit CPU/Metal selection policy.
///
/// CPU fallback occurs only when a Metal trainer cannot be constructed. Errors after training
/// begins are returned rather than replaying updates on a numerically different backend.
public struct DenseTrainer: Sendable {
    /// Device and fallback behavior used for new runs.
    public let executionPolicy: DenseTrainingExecutionPolicy

    #if canImport(Metal) && canImport(MetalPerformanceShadersGraph)
        private let makeMetalTrainer: @Sendable () throws -> DenseMPSGraphTrainer
    #endif

    /// Creates a high-level trainer with explicit execution behavior.
    public init(executionPolicy: DenseTrainingExecutionPolicy = .init()) {
        self.executionPolicy = executionPolicy
        #if canImport(Metal) && canImport(MetalPerformanceShadersGraph)
            makeMetalTrainer = { try DenseMPSGraphTrainer() }
        #endif
    }

    #if canImport(Metal) && canImport(MetalPerformanceShadersGraph)
        init(
            executionPolicy: DenseTrainingExecutionPolicy,
            makeMetalTrainer: @escaping @Sendable () throws -> DenseMPSGraphTrainer
        ) {
            self.executionPolicy = executionPolicy
            self.makeMetalTrainer = makeMetalTrainer
        }
    #endif

    /// Fits a model on the selected backend with progress and lifecycle controls.
    ///
    /// - Throws: Accelerator construction errors when fallback is disabled, or the errors
    ///   documented by the selected concrete trainer.
    public func train(
        _ samples: [DenseTrainingSample],
        configuration: DenseNetworkConfiguration,
        options: DenseTrainingOptions = .standard,
        progress: DenseTrainingProgressHandler? = nil
    ) async throws -> DenseTrainingResult {
        switch executionPolicy.preference {
        case .cpu:
            return try await DenseCPUTrainer().train(
                samples,
                configuration: configuration,
                options: options,
                progress: progress
            )
        case .automatic, .metal:
            #if canImport(Metal) && canImport(MetalPerformanceShadersGraph)
                let trainer: DenseMPSGraphTrainer
                do {
                    trainer = try makeMetalTrainer()
                } catch {
                    guard executionPolicy.fallback == .cpu else { throw error }
                    return try await DenseCPUTrainer().train(
                        samples,
                        configuration: configuration,
                        options: options,
                        progress: progress
                    )
                }
                return try await trainer.train(
                    samples,
                    configuration: configuration,
                    options: options,
                    progress: progress
                )
            #else
                guard executionPolicy.fallback == .cpu else {
                    throw ML5Error.trainingAcceleratorUnavailable(
                        reason: "Metal Performance Shaders Graph is unavailable on this platform."
                    )
                }
                return try await DenseCPUTrainer().train(
                    samples,
                    configuration: configuration,
                    options: options,
                    progress: progress
                )
            #endif
        }
    }

    /// Resumes on the backend recorded in a checkpoint without numerical fallback.
    ///
    /// - Throws: An accelerator or checkpoint error if the recorded backend is unavailable or
    ///   differs from the selected Metal device.
    public func resume(
        _ checkpoint: DenseTrainingCheckpoint,
        samples: [DenseTrainingSample],
        progress: DenseTrainingProgressHandler? = nil
    ) async throws -> DenseTrainingResult {
        switch checkpoint.backend {
        case .cpu:
            return try await DenseCPUTrainer().resume(
                checkpoint,
                samples: samples,
                progress: progress
            )
        case .metal:
            #if canImport(Metal) && canImport(MetalPerformanceShadersGraph)
                return try await makeMetalTrainer().resume(
                    checkpoint,
                    samples: samples,
                    progress: progress
                )
            #else
                throw ML5Error.trainingAcceleratorUnavailable(
                    reason: "Metal Performance Shaders Graph is unavailable on this platform."
                )
            #endif
        }
    }
}

extension DenseNetworkMath {
    static func derivative(
        preactivation: Double,
        activation: Double,
        using function: ActivationFunction
    ) -> Double {
        switch function {
        case .linear:
            1
        case .rectifiedLinear:
            preactivation > 0 ? 1 : 0
        case .sigmoid:
            activation * (1 - activation)
        case .hyperbolicTangent:
            1 - activation * activation
        case .softmax:
            activation * (1 - activation)
        }
    }
}

private struct PreparedSample: Sendable {
    let input: [Double]
    let target: [Double]
}

private struct ForwardPass: Sendable {
    let activations: [[Double]]
    let preactivations: [[Double]]
}

private struct LayerGradients: Sendable {
    var weights: [Double]
    var biases: [Double]

    init(layer: TrainableLayer) {
        weights = Array(repeating: 0, count: layer.weights.count)
        biases = Array(repeating: 0, count: layer.biases.count)
    }

    init(weights: [Double], biases: [Double]) {
        self.weights = weights
        self.biases = biases
    }
}

struct DenseLayerTrainingState: Sendable, Hashable, Codable {
    let parameters: DenseLayerParameters
    let weightVelocity: [Double]
    let biasVelocity: [Double]
    let weightFirstMoment: [Double]
    let biasFirstMoment: [Double]
    let weightSecondMoment: [Double]
    let biasSecondMoment: [Double]

    init(layer: TrainableLayer) {
        parameters = DenseLayerParameters(
            validatedInputCount: layer.inputCount,
            validatedOutputCount: layer.outputCount,
            weights: layer.weights,
            biases: layer.biases
        )
        weightVelocity = layer.weightVelocity
        biasVelocity = layer.biasVelocity
        weightFirstMoment = layer.weightFirstMoment
        biasFirstMoment = layer.biasFirstMoment
        weightSecondMoment = layer.weightSecondMoment
        biasSecondMoment = layer.biasSecondMoment
    }

    init(
        parameters: DenseLayerParameters,
        weightVelocity: [Double],
        biasVelocity: [Double],
        weightFirstMoment: [Double],
        biasFirstMoment: [Double],
        weightSecondMoment: [Double],
        biasSecondMoment: [Double]
    ) throws {
        let weightCollections = [
            weightVelocity,
            weightFirstMoment,
            weightSecondMoment,
        ]
        let biasCollections = [
            biasVelocity,
            biasFirstMoment,
            biasSecondMoment,
        ]
        guard weightCollections.allSatisfy({ $0.count == parameters.weights.count }),
            biasCollections.allSatisfy({ $0.count == parameters.biases.count })
        else {
            throw ML5Error.invalidTrainingCheckpoint(
                reason: "Optimizer storage must match its layer parameter counts."
            )
        }
        guard weightCollections.joined().allSatisfy(\.isFinite),
            biasCollections.joined().allSatisfy(\.isFinite)
        else {
            throw ML5Error.invalidTrainingCheckpoint(
                reason: "Optimizer storage must contain only finite values."
            )
        }
        self.parameters = parameters
        self.weightVelocity = weightVelocity
        self.biasVelocity = biasVelocity
        self.weightFirstMoment = weightFirstMoment
        self.biasFirstMoment = biasFirstMoment
        self.weightSecondMoment = weightSecondMoment
        self.biasSecondMoment = biasSecondMoment
    }

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            parameters: container.decode(DenseLayerParameters.self, forKey: .parameters),
            weightVelocity: container.decode([Double].self, forKey: .weightVelocity),
            biasVelocity: container.decode([Double].self, forKey: .biasVelocity),
            weightFirstMoment: container.decode([Double].self, forKey: .weightFirstMoment),
            biasFirstMoment: container.decode([Double].self, forKey: .biasFirstMoment),
            weightSecondMoment: container.decode([Double].self, forKey: .weightSecondMoment),
            biasSecondMoment: container.decode([Double].self, forKey: .biasSecondMoment)
        )
    }

    var trainableLayer: TrainableLayer {
        TrainableLayer(
            parameters: parameters,
            weightVelocity: weightVelocity,
            biasVelocity: biasVelocity,
            weightFirstMoment: weightFirstMoment,
            biasFirstMoment: biasFirstMoment,
            weightSecondMoment: weightSecondMoment,
            biasSecondMoment: biasSecondMoment
        )
    }
}

struct TrainableLayer: Sendable {
    let inputCount: Int
    let outputCount: Int
    var weights: [Double]
    var biases: [Double]
    var weightVelocity: [Double]
    var biasVelocity: [Double]
    var weightFirstMoment: [Double]
    var biasFirstMoment: [Double]
    var weightSecondMoment: [Double]
    var biasSecondMoment: [Double]

    init(inputCount: Int, outputCount: Int, weights: [Double], biases: [Double]) {
        self.inputCount = inputCount
        self.outputCount = outputCount
        self.weights = weights
        self.biases = biases
        weightVelocity = Array(repeating: 0, count: weights.count)
        biasVelocity = Array(repeating: 0, count: biases.count)
        weightFirstMoment = Array(repeating: 0, count: weights.count)
        biasFirstMoment = Array(repeating: 0, count: biases.count)
        weightSecondMoment = Array(repeating: 0, count: weights.count)
        biasSecondMoment = Array(repeating: 0, count: biases.count)
    }

    init(
        parameters: DenseLayerParameters,
        weightVelocity: [Double],
        biasVelocity: [Double],
        weightFirstMoment: [Double],
        biasFirstMoment: [Double],
        weightSecondMoment: [Double],
        biasSecondMoment: [Double]
    ) {
        inputCount = parameters.inputCount
        outputCount = parameters.outputCount
        weights = parameters.weights
        biases = parameters.biases
        self.weightVelocity = weightVelocity
        self.biasVelocity = biasVelocity
        self.weightFirstMoment = weightFirstMoment
        self.biasFirstMoment = biasFirstMoment
        self.weightSecondMoment = weightSecondMoment
        self.biasSecondMoment = biasSecondMoment
    }

    func affine(_ input: [Double]) -> [Double] {
        (0..<outputCount).map { outputIndex in
            let rowStart = outputIndex * inputCount
            return (0..<inputCount).reduce(biases[outputIndex]) {
                $0 + weights[rowStart + $1] * input[$1]
            }
        }
    }
}

private struct DenseSeededGenerator: RandomNumberGenerator {
    var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }

    mutating func uniform(in range: ClosedRange<Double>) -> Double {
        let unit = (Double(next() >> 11) + 0.5) / 9_007_199_254_740_992
        return range.lowerBound + (range.upperBound - range.lowerBound) * unit
    }

    mutating func normal() -> Double {
        let first = (Double(next() >> 11) + 0.5) / 9_007_199_254_740_992
        let second = (Double(next() >> 11) + 0.5) / 9_007_199_254_740_992
        return sqrt(-2 * log(first)) * cos(2 * .pi * second)
    }
}
