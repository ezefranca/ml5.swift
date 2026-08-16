import Foundation
import Testing

@testable import ML5

private actor DefaultBatchPredictor: ModelPredicting {
    func predict(_ features: FeatureVector) async throws -> ModelOutput {
        guard let value = features["x"]?.numericValue else {
            throw ML5Error.missingFeature("x")
        }
        return try ModelOutput(["estimate": .number(value * 2)])
    }
}

private actor WrongCountPredictor: ModelPredicting {
    func predict(_: FeatureVector) async throws -> ModelOutput {
        try ModelOutput(["estimate": .number(0)])
    }

    func predict(_: [FeatureVector]) async throws -> [ModelOutput] {
        []
    }
}

private actor SnapshotPredictor: ModelInferenceSnapshotProviding {
    func predict(_ features: FeatureVector) async throws -> ModelOutput {
        try Self.output(for: features)
    }

    func makeInferenceSnapshot() async throws -> ModelInferenceSnapshot {
        ModelInferenceSnapshot(operation: Self.output)
    }

    private nonisolated static func output(for features: FeatureVector) throws -> ModelOutput {
        guard let value = features["x"]?.numericValue else {
            throw ML5Error.missingFeature("x")
        }
        return try ModelOutput(["estimate": .number(value + 1)])
    }
}

@Suite("ML5 batch and snapshot inference")
struct ML5InferenceTests {
    @Test("The default backend batch preserves order and handles empty input")
    func defaultBatch() async throws {
        let predictor = DefaultBatchPredictor()
        let features = try [1.0, 2.0, 3.0].map {
            try FeatureVector(["x": .number($0)])
        }

        let output = try await predictor.predict(features)
        #expect(output.map { $0["estimate"] } == [.number(2), .number(4), .number(6)])
        #expect(try await predictor.predict([]).isEmpty)

        let cancelled = Task { try await predictor.predict(features) }
        cancelled.cancel()
        await #expect(throws: CancellationError.self) {
            try await cancelled.value
        }
    }

    @Test("Neural networks decode ordered batches and reject count mismatches")
    func typedBatch() async throws {
        let task = RegressionTask(
            configuration: RegressionConfiguration(valueOutput: "estimate")
        )
        let features = try [1.0, 2.0].map {
            try FeatureVector(["x": .number($0)])
        }
        let network = NeuralNetwork(task: task, predictor: DefaultBatchPredictor())

        #expect(try await network.predict(features).map(\.value) == [2, 4])
        #expect(try await network.predict([FeatureVector]()).isEmpty)

        let malformed = NeuralNetwork(task: task, predictor: WrongCountPredictor())
        await #expect(
            throws: ML5Error.batchPredictionCountMismatch(expected: 2, actual: 0)
        ) {
            try await malformed.predict(features)
        }
    }

    @Test("Raw snapshots predict synchronously in scalar and batch forms")
    func rawSnapshot() throws {
        let snapshot = ModelInferenceSnapshot { features in
            guard let value = features["x"]?.numericValue else {
                throw ML5Error.missingFeature("x")
            }
            return try ModelOutput(["estimate": .number(value - 1)])
        }
        let features = try [2.0, 4.0].map {
            try FeatureVector(["x": .number($0)])
        }

        #expect(try snapshot.predict(features[0])["estimate"] == .number(1))
        #expect(
            try snapshot.predict(features).map { $0["estimate"] } == [.number(1), .number(3)]
        )
        #expect(try snapshot.predict([FeatureVector]()).isEmpty)
        #expect(throws: ML5Error.missingFeature("x")) {
            try snapshot.predict(FeatureVector(["other": .number(1)]))
        }
    }

    @Test("Typed snapshots decode without actor hops")
    func typedSnapshot() async throws {
        let task = RegressionTask(
            configuration: RegressionConfiguration(valueOutput: "estimate")
        )
        let network = NeuralNetwork(task: task, predictor: SnapshotPredictor())
        let snapshot = try await network.makeInferenceSnapshot()
        let features = try [1.0, 3.0].map {
            try FeatureVector(["x": .number($0)])
        }

        #expect(snapshot.task.kind == .regression)
        #expect(try snapshot.predict(features[0]).value == 2)
        #expect(try snapshot.predict(features).map(\.value) == [2, 4])
    }

    @Test("Networks report backends without synchronous snapshots")
    func unsupportedSnapshot() async {
        let task = RegressionTask(
            configuration: RegressionConfiguration(valueOutput: "estimate")
        )
        let network = NeuralNetwork(task: task, predictor: DefaultBatchPredictor())

        await #expect(
            throws: ML5Error.unsupportedOperation(.synchronousInferenceSnapshot)
        ) {
            try await network.makeInferenceSnapshot()
        }
    }
}
