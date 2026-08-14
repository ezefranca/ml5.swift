/// A model-output provider that can safely cross concurrency boundaries.
///
/// Implement this protocol to provide a deterministic test double or a custom
/// runtime backend. Implementations must not expose non-Sendable framework values.
public protocol ModelPredicting: Sendable {
    func predict(_ features: FeatureVector) async throws -> ModelOutput
}

/// The contract a future Create ML integration can adopt to produce a predictor.
///
/// ML5 does not currently provide an arbitrary-model training implementation.
/// An adapter can train in its own module, then return a value-safe prediction backend
/// to a `NeuralNetwork`.
public protocol NeuralNetworkTrainingAdapter: Sendable {
    associatedtype Definition: NeuralNetworkTask
    associatedtype Sample: Sendable

    func makePredictor(
        training samples: [Sample],
        for task: Definition
    ) async throws -> any ModelPredicting
}
