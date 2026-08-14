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
