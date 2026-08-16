/// A model-output provider that can safely cross concurrency boundaries.
///
/// Implement this protocol to provide a deterministic test double or a custom
/// runtime backend. Implementations must not expose non-Sendable framework values.
public protocol ModelPredicting: Sendable {
    /// Produces framework-independent outputs for a validated feature vector.
    func predict(_ features: FeatureVector) async throws -> ModelOutput

    /// Produces outputs for a batch in exactly the supplied order.
    func predict(_ batch: [FeatureVector]) async throws -> [ModelOutput]
}

extension ModelPredicting {
    /// Sequential default batch implementation for custom prediction backends.
    public func predict(_ batch: [FeatureVector]) async throws -> [ModelOutput] {
        var outputs: [ModelOutput] = []
        outputs.reserveCapacity(batch.count)
        for features in batch {
            try Task.checkCancellation()
            outputs.append(try await predict(features))
        }
        try Task.checkCancellation()
        return outputs
    }
}

/// An immutable, synchronous raw-model prediction operation.
///
/// Snapshot prediction performs no actor hop and is suitable for latency-sensitive
/// render loops when the backend can safely provide a synchronous implementation.
public struct ModelInferenceSnapshot: Sendable {
    private let operation: @Sendable (FeatureVector) throws -> ModelOutput

    /// Creates a snapshot from a thread-safe synchronous operation.
    public init(
        operation: @escaping @Sendable (FeatureVector) throws -> ModelOutput
    ) {
        self.operation = operation
    }

    /// Produces one framework-independent output synchronously.
    ///
    /// - Throws: The backend's prediction or value-conversion error.
    public func predict(_ features: FeatureVector) throws -> ModelOutput {
        try operation(features)
    }

    /// Produces synchronous outputs in exactly the supplied order.
    ///
    /// - Throws: The first backend prediction or value-conversion error.
    public func predict(_ batch: [FeatureVector]) throws -> [ModelOutput] {
        try batch.map(operation)
    }
}

/// A backend that can vend an immutable synchronous inference snapshot.
public protocol ModelInferenceSnapshotProviding: ModelPredicting {
    /// Captures a synchronous operation without sharing mutable backend state.
    func makeInferenceSnapshot() async throws -> ModelInferenceSnapshot
}

/// The contract a future Create ML integration can adopt to produce a predictor.
///
/// ML5 does not currently provide an arbitrary-model training implementation.
/// An adapter can train in its own module, then return a value-safe prediction backend
/// to a `NeuralNetwork`.
public protocol NeuralNetworkTrainingAdapter: Sendable {
    /// The typed task definition the trained predictor supports.
    associatedtype Definition: NeuralNetworkTask
    /// The value-safe sample type accepted by the adapter.
    associatedtype Sample: Sendable

    /// Trains a backend for a task and returns its concurrency-safe predictor.
    func makePredictor(
        training samples: [Sample],
        for task: Definition
    ) async throws -> any ModelPredicting
}
