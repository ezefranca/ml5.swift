import Foundation

/// An actor-isolated facade for a typed prediction task and its backend.
///
/// The actor serializes access to the backend while keeping all values passed to
/// callers `Sendable`. Core ML-backed instances can be created with
/// `init(task:modelAt:configuration:)`.
public actor NeuralNetwork<Definition: NeuralNetworkTask> {
    private let task: Definition
    private let predictor: any ModelPredicting

    /// Creates a network using an injectable prediction backend.
    ///
    /// This initializer makes deterministic tests and future non-Core-ML backends
    /// possible without weakening the public type model.
    public init(task: Definition, predictor: some ModelPredicting) {
        self.task = task
        self.predictor = predictor
    }

    /// Loads a compiled Core ML model and creates a network for `task`.
    public init(
        task: Definition,
        modelAt modelURL: URL,
        configuration: CoreMLModelConfiguration = .init()
    ) throws {
        self.task = task
        self.predictor = try CoreMLModelPredictor(
            contentsOf: modelURL,
            configuration: configuration
        )
    }

    /// Produces a typed prediction.
    ///
    /// Cancellation is checked before invoking the backend and before decoding its
    /// result. A backend that has already begun evaluation may complete first, but
    /// its result will not be decoded after cancellation.
    public func predict(_ features: FeatureVector) async throws -> Definition.Prediction {
        try Task.checkCancellation()
        let output = try await predictor.predict(features)
        try Task.checkCancellation()
        return try task.decode(output)
    }

    /// Produces typed predictions for a batch in exactly the supplied order.
    ///
    /// Cancellation is checked before backend execution and before each output is
    /// decoded. A backend returning the wrong output count is rejected.
    public func predict(_ batch: [FeatureVector]) async throws -> [Definition.Prediction] {
        try Task.checkCancellation()
        let outputs = try await predictor.predict(batch)
        try Task.checkCancellation()
        guard outputs.count == batch.count else {
            throw ML5Error.batchPredictionCountMismatch(
                expected: batch.count,
                actual: outputs.count
            )
        }
        return try outputs.map { output in
            try Task.checkCancellation()
            return try task.decode(output)
        }
    }

    /// Captures a typed synchronous snapshot for latency-sensitive inference.
    ///
    /// - Throws: ``ML5Error/unsupportedOperation(_:)`` when the backend cannot vend
    ///   a synchronous snapshot, or a backend-specific snapshot error.
    public func makeInferenceSnapshot() async throws -> NeuralNetworkInferenceSnapshot<Definition> {
        guard let provider = predictor as? any ModelInferenceSnapshotProviding else {
            throw ML5Error.unsupportedOperation(.synchronousInferenceSnapshot)
        }
        return NeuralNetworkInferenceSnapshot(
            task: task,
            modelSnapshot: try await provider.makeInferenceSnapshot()
        )
    }

    /// Explicitly reports that arbitrary-model, on-device training is unavailable.
    ///
    /// Use `NeuralNetworkTrainingAdapter` in a separate integration module when
    /// Create ML or another training technology can produce a `ModelPredicting`
    /// backend for the task.
    public func train<Sample: Sendable>(_ samples: [Sample]) async throws {
        try Task.checkCancellation()
        guard samples.isEmpty == false else {
            throw ML5Error.invalidTrainingSamples
        }
        throw ML5Error.unsupportedOperation(.onDeviceTraining)
    }
}

/// An immutable typed synchronous snapshot suitable for a draw loop.
public struct NeuralNetworkInferenceSnapshot<Definition: NeuralNetworkTask>: Sendable {
    /// Task definition used to decode raw model output.
    public let task: Definition
    /// Synchronous framework-independent model operation.
    public let modelSnapshot: ModelInferenceSnapshot

    /// Creates a typed snapshot from a task and raw-model snapshot.
    public init(task: Definition, modelSnapshot: ModelInferenceSnapshot) {
        self.task = task
        self.modelSnapshot = modelSnapshot
    }

    /// Produces and decodes one typed prediction synchronously.
    ///
    /// - Throws: A backend prediction or task-decoding error.
    public func predict(_ features: FeatureVector) throws -> Definition.Prediction {
        try task.decode(modelSnapshot.predict(features))
    }

    /// Produces and decodes a batch synchronously in input order.
    ///
    /// - Throws: The first backend prediction or task-decoding error.
    public func predict(_ batch: [FeatureVector]) throws -> [Definition.Prediction] {
        try modelSnapshot.predict(batch).map(task.decode)
    }
}
