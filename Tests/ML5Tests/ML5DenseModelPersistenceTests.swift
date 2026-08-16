@preconcurrency import CoreML
import Foundation
import Testing

@testable import ML5

private enum InjectedCoreMLFailure: Error {
    case compilation
}

@Suite("ML5 dense-model persistence and Core ML export", .serialized)
struct ML5DenseModelPersistenceTests {
    @Test("Model metadata validates and round-trips")
    func metadata() throws {
        for (name, version) in [("", "1"), (" Model", "1"), ("Model", "")] {
            #expect(throws: ML5Error.self) {
                _ = try ML5ModelMetadata(name: name, version: version)
            }
        }
        for value in ["", " Author", "License "] {
            #expect(throws: ML5Error.self) {
                _ = try ML5ModelMetadata(name: "Model", version: "1", author: value)
            }
        }
        #expect(throws: ML5Error.self) {
            _ = try ML5ModelMetadata(
                name: "Model",
                version: "1",
                additional: [" bad": "value"]
            )
        }

        let value = try completeMetadata()
        #expect(
            try JSONDecoder().decode(
                ML5ModelMetadata.self,
                from: JSONEncoder().encode(value)
            ) == value
        )
        #expect(value.name == "Canonical Dense Model")
        #expect(value.version == "1.2.3")
        #expect(value.author == "ML5 Contributors")
        #expect(value.license == "MIT")
        #expect(value.summary == "A deterministic test model.")
        #expect(value.source?.host == "example.com")
        #expect(value.additional["dataset"] == "canonical-v1")
    }

    @Test("Archives detect modification and persist atomically")
    func archives() async throws {
        let model = try makeModel(activation: .hyperbolicTangent)
        let archive = try model.archived(metadata: completeMetadata())
        #expect(archive.formatVersion == DenseModelArchive.currentFormatVersion)
        #expect(archive.integrityDigest.count == 64)
        #expect(archive.integrityDigest.allSatisfy { $0.isHexDigit && !$0.isUppercase })
        #expect(try DenseModelArchive(data: archive.encodedData()) == archive)
        #expect(
            try JSONDecoder().decode(
                DenseModelArchive.self,
                from: JSONEncoder().encode(archive)
            ) == archive
        )

        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "ml5-archive-tests-\(UUID().uuidString)",
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("model.ml5model")
        try archive.write(to: file)
        let loaded = try DenseModelArchive.load(contentsOf: file)
        #expect(loaded == archive)
        #expect(
            try await loaded.model.predict(FeatureVector(["x": .number(1), "y": .number(2)]))
                == model.predict(FeatureVector(["x": .number(1), "y": .number(2)]))
        )

        var object = try #require(
            JSONSerialization.jsonObject(with: archive.encodedData()) as? [String: Any]
        )
        object["integrityDigest"] = String(repeating: "0", count: 64)
        let modified = try JSONSerialization.data(withJSONObject: object)
        #expect(throws: ML5Error.self) { _ = try DenseModelArchive(data: modified) }
        #expect(throws: ML5Error.self) { _ = try DenseModelArchive(data: Data("bad".utf8)) }
        #expect(throws: ML5Error.self) {
            _ = try DenseModelArchive(
                formatVersion: 2,
                model: archive.model,
                metadata: archive.metadata,
                integrityDigest: archive.integrityDigest
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseModelArchive(
                formatVersion: 1,
                model: archive.model,
                metadata: archive.metadata,
                integrityDigest: "wrong"
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseModelArchive.load(
                contentsOf: directory.appendingPathComponent("missing.ml5model")
            )
        }
        #expect(throws: ML5Error.self) { try archive.write(to: directory) }
    }

    @Test("Core ML export validates names and float32 storage")
    func exportValidation() throws {
        let metadata = try completeMetadata()
        for names in [("", "output"), (" input", "output"), ("same", "same")] {
            #expect(throws: ML5Error.self) {
                _ = try DenseCoreMLExportConfiguration(
                    inputName: names.0,
                    outputName: names.1,
                    metadata: metadata
                )
            }
        }
        let configuration = try DenseNetworkConfiguration(
            inputFeatures: ["x"],
            outputNames: ["y"]
        )
        let parameters = try DenseLayerParameters(
            inputCount: 1,
            outputCount: 1,
            weights: [.greatestFiniteMagnitude],
            biases: [0]
        )
        let model = try DenseNetworkModel(configuration: configuration, layers: [parameters])
        let export = try DenseCoreMLExportConfiguration(metadata: metadata)
        #expect(throws: ML5Error.self) {
            _ = try model.coreMLModelData(configuration: export)
        }
        #expect(throws: ML5Error.self) {
            try model.writeCoreMLModel(
                to: URL(fileURLWithPath: "/unreachable/overflow.mlmodel"),
                configuration: export
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try model.compileCoreMLModel(configuration: export)
        }

        let valid = try makeModel(activation: .linear)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "ml5-coreml-write-tests-\(UUID().uuidString)",
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(throws: ML5Error.self) {
            try valid.writeCoreMLModel(to: directory, configuration: export)
        }

        var operations = DenseCoreMLCompilationOperations.system
        operations.compile = { _ in throw InjectedCoreMLFailure.compilation }
        #expect(throws: ML5Error.self) {
            _ = try valid.compileCoreMLModel(configuration: export, operations: operations)
        }

        #expect(
            ML5Error.coreMLExportFailed(reason: "Failure.").errorDescription
                == "Core ML export failed: Failure."
        )
        #expect(
            ML5Error.invalidModelArchive(reason: "Failure.").errorDescription
                == "Invalid model archive: Failure."
        )
        #expect(
            ML5Error.modelPersistenceFailed(path: "/tmp/model", message: "Failure.")
                .errorDescription
                == "Model persistence failed at \"/tmp/model\": Failure."
        )
    }

    @Test("Every dense activation compiles and matches Core ML inference")
    func coreMLInference() async throws {
        let input = [1.25, -0.75]
        for activation in ActivationFunction.allCasesForPersistenceTesting {
            let model = try makeModel(activation: activation)
            let metadata = try ML5ModelMetadata(
                name: "Dense-\(activation.rawValue)",
                version: "1",
                summary: nil
            )
            let export = try DenseCoreMLExportConfiguration(
                inputName: "features",
                outputName: "predictions",
                metadata: metadata
            )
            #expect(try model.coreMLModelData(configuration: export).isEmpty == false)
            let compiled = try model.compileCoreMLModel(configuration: export)
            defer { try? FileManager.default.removeItem(at: compiled) }
            let predictor = try CoreMLModelPredictor(
                contentsOf: compiled,
                configuration: CoreMLModelConfiguration(computeUnits: .cpuOnly)
            )
            let output = try await predictor.predict(
                FeatureVector(["features": .array(input)])
            )
            let tensor = try #require(output["predictions"]?.tensorValueForPersistenceTesting)
            let expected = try model.values(
                for: FeatureVector(["x": .number(input[0]), "y": .number(input[1])])
            )
            #expect(tensor.shape.dimensions == [2])
            #expect(zip(tensor.values, expected).allSatisfy { abs($0 - $1) < 1e-5 })
        }
    }

    @Test("Multi-layer Core ML export preserves metadata and topology")
    func multilayerCoreML() async throws {
        let configuration = try DenseNetworkConfiguration(
            inputFeatures: ["x", "y"],
            outputNames: ["value"],
            hiddenLayers: [
                try DenseLayerConfiguration(neuronCount: 2, activation: .rectifiedLinear)
            ],
            outputActivation: .linear
        )
        let model = try DenseNetworkModel(
            configuration: configuration,
            layers: [
                try DenseLayerParameters(
                    inputCount: 2,
                    outputCount: 2,
                    weights: [1, -1, 0.5, 0.25],
                    biases: [0.1, -0.2]
                ),
                try DenseLayerParameters(
                    inputCount: 2,
                    outputCount: 1,
                    weights: [0.75, -0.5],
                    biases: [0.3]
                ),
            ]
        )
        let export = try DenseCoreMLExportConfiguration(
            inputName: "orderedInput",
            outputName: "orderedOutput",
            metadata: completeMetadata()
        )
        let compiled = try model.compileCoreMLModel(configuration: export)
        defer { try? FileManager.default.removeItem(at: compiled) }
        let native = try MLModel(
            contentsOf: compiled,
            configuration: CoreMLModelConfiguration(computeUnits: .cpuOnly)
                .makeCoreMLConfiguration()
        )
        #expect(
            native.modelDescription.metadata[MLModelMetadataKey.author] as? String
                == "ML5 Contributors")
        #expect(native.modelDescription.metadata[MLModelMetadataKey.license] as? String == "MIT")
        let predictor = try CoreMLModelPredictor(contentsOf: compiled)
        let output = try await predictor.predict(
            FeatureVector(["orderedInput": .array([2, 1])])
        )
        let value = try #require(output["orderedOutput"]?.tensorValueForPersistenceTesting)
        let expected = try model.values(
            for: FeatureVector(["x": .number(2), "y": .number(1)])
        )
        #expect(abs(value.values[0] - expected[0]) < 1e-5)
    }

    private func completeMetadata() throws -> ML5ModelMetadata {
        try ML5ModelMetadata(
            name: "Canonical Dense Model",
            version: "1.2.3",
            author: "ML5 Contributors",
            license: "MIT",
            summary: "A deterministic test model.",
            source: URL(string: "https://example.com/models/canonical"),
            additional: ["dataset": "canonical-v1"]
        )
    }

    private func makeModel(activation: ActivationFunction) throws -> DenseNetworkModel {
        let configuration = try DenseNetworkConfiguration(
            inputFeatures: ["x", "y"],
            outputNames: ["first", "second"],
            outputActivation: activation,
            loss: activation == .softmax ? .categoricalCrossEntropy : .meanSquaredError
        )
        return try DenseNetworkModel(
            configuration: configuration,
            layers: [
                try DenseLayerParameters(
                    inputCount: 2,
                    outputCount: 2,
                    weights: [0.5, -0.25, -0.75, 0.4],
                    biases: [0.1, -0.2]
                )
            ]
        )
    }
}

extension ActivationFunction {
    fileprivate static var allCasesForPersistenceTesting: [Self] {
        [.linear, .rectifiedLinear, .sigmoid, .hyperbolicTangent, .softmax]
    }
}

extension FeatureValue {
    fileprivate var tensorValueForPersistenceTesting: Tensor? {
        guard case let .tensor(value) = self else { return nil }
        return value
    }
}
