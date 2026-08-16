@preconcurrency import CoreML
import CryptoKit
import Foundation

/// Human-readable ownership, license, version, and provenance for a persisted model.
public struct ML5ModelMetadata: Sendable, Hashable, Codable {
    /// Display name of the model.
    public let name: String
    /// Application-defined semantic or dataset version.
    public let version: String
    /// Model creator or organization, when known.
    public let author: String?
    /// SPDX identifier or human-readable license, when known.
    public let license: String?
    /// Concise purpose and expected-input description, when supplied.
    public let summary: String?
    /// Source, model card, or training provenance URL, when supplied.
    public let source: URL?
    /// Additional stable metadata with nonempty trimmed keys.
    public let additional: [String: String]

    /// Creates validated model metadata.
    ///
    /// - Throws: ``ML5Error/invalidModelArchive(reason:)`` when required text is empty or
    ///   untrimmed, optional text is empty or untrimmed, or an additional key is invalid.
    public init(
        name: String,
        version: String,
        author: String? = nil,
        license: String? = nil,
        summary: String? = nil,
        source: URL? = nil,
        additional: [String: String] = [:]
    ) throws {
        guard Self.isValidRequiredText(name), Self.isValidRequiredText(version) else {
            throw ML5Error.invalidModelArchive(
                reason: "Model name and version must be nonempty and trimmed."
            )
        }
        guard [author, license, summary].allSatisfy(Self.isValidOptionalText) else {
            throw ML5Error.invalidModelArchive(
                reason: "Optional model metadata must be nonempty and trimmed when present."
            )
        }
        guard additional.keys.allSatisfy(Self.isValidRequiredText) else {
            throw ML5Error.invalidModelArchive(
                reason: "Additional metadata keys must be nonempty and trimmed."
            )
        }
        self.name = name
        self.version = version
        self.author = author
        self.license = license
        self.summary = summary
        self.source = source
        self.additional = additional
    }

    /// Decodes and revalidates persisted model metadata.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            name: container.decode(String.self, forKey: .name),
            version: container.decode(String.self, forKey: .version),
            author: container.decodeIfPresent(String.self, forKey: .author),
            license: container.decodeIfPresent(String.self, forKey: .license),
            summary: container.decodeIfPresent(String.self, forKey: .summary),
            source: container.decodeIfPresent(URL.self, forKey: .source),
            additional: container.decode([String: String].self, forKey: .additional)
        )
    }

    /// Encodes all metadata fields using stable property names.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        try container.encode(version, forKey: .version)
        try container.encodeIfPresent(author, forKey: .author)
        try container.encodeIfPresent(license, forKey: .license)
        try container.encodeIfPresent(summary, forKey: .summary)
        try container.encodeIfPresent(source, forKey: .source)
        try container.encode(additional, forKey: .additional)
    }

    private static func isValidRequiredText(_ value: String) -> Bool {
        !value.isEmpty && value == value.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func isValidOptionalText(_ value: String?) -> Bool {
        value.map(isValidRequiredText) ?? true
    }

    private enum CodingKeys: String, CodingKey {
        case name
        case version
        case author
        case license
        case summary
        case source
        case additional
    }
}

/// A versioned, integrity-checked archive containing an immutable dense model and its metadata.
public struct DenseModelArchive: Sendable, Hashable, Codable {
    /// Archive schema written by this release.
    public static let currentFormatVersion = 1

    /// Persisted archive schema version.
    public let formatVersion: Int
    /// Immutable validated dense model.
    public let model: DenseNetworkModel
    /// Ownership, license, version, and provenance metadata.
    public let metadata: ML5ModelMetadata
    /// Lowercase SHA-256 digest of the canonical archive payload.
    public let integrityDigest: String

    /// Creates an archive and computes its integrity digest.
    public init(model: DenseNetworkModel, metadata: ML5ModelMetadata) throws {
        formatVersion = Self.currentFormatVersion
        self.model = model
        self.metadata = metadata
        integrityDigest = try Self.digest(
            formatVersion: formatVersion,
            model: model,
            metadata: metadata
        )
    }

    init(
        formatVersion: Int,
        model: DenseNetworkModel,
        metadata: ML5ModelMetadata,
        integrityDigest: String
    ) throws {
        guard formatVersion == Self.currentFormatVersion else {
            throw ML5Error.invalidModelArchive(
                reason: "Unsupported archive format version \(formatVersion)."
            )
        }
        let expected = try Self.digest(
            formatVersion: formatVersion,
            model: model,
            metadata: metadata
        )
        guard integrityDigest == expected else {
            throw ML5Error.invalidModelArchive(
                reason: "The archive integrity digest does not match its payload."
            )
        }
        self.formatVersion = formatVersion
        self.model = model
        self.metadata = metadata
        self.integrityDigest = integrityDigest
    }

    /// Decodes and verifies a model archive.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            formatVersion: container.decode(Int.self, forKey: .formatVersion),
            model: container.decode(DenseNetworkModel.self, forKey: .model),
            metadata: container.decode(ML5ModelMetadata.self, forKey: .metadata),
            integrityDigest: container.decode(String.self, forKey: .integrityDigest)
        )
    }

    /// Encodes the model, metadata, format version, and verified digest.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(formatVersion, forKey: .formatVersion)
        try container.encode(model, forKey: .model)
        try container.encode(metadata, forKey: .metadata)
        try container.encode(integrityDigest, forKey: .integrityDigest)
    }

    /// Produces deterministic sorted-key JSON suitable for files or network transport.
    public func encodedData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(self)
    }

    /// Decodes and verifies an archive from in-memory JSON.
    ///
    /// - Throws: ``ML5Error/invalidModelArchive(reason:)`` for malformed or modified data.
    public init(data: Data) throws {
        do {
            self = try JSONDecoder().decode(Self.self, from: data)
        } catch let error as ML5Error {
            throw error
        } catch {
            throw ML5Error.invalidModelArchive(reason: error.localizedDescription)
        }
    }

    /// Atomically writes the archive to a file URL.
    ///
    /// - Throws: ``ML5Error/modelPersistenceFailed(path:message:)`` when encoding or writing
    ///   fails.
    public func write(to fileURL: URL) throws {
        do {
            try encodedData().write(to: fileURL, options: .atomic)
        } catch {
            throw ML5Error.modelPersistenceFailed(
                path: fileURL.path,
                message: error.localizedDescription
            )
        }
    }

    /// Reads and verifies an archive from a file URL.
    ///
    /// - Throws: ``ML5Error/modelPersistenceFailed(path:message:)`` for file access failures, or
    ///   ``ML5Error/invalidModelArchive(reason:)`` for malformed or modified content.
    public static func load(contentsOf fileURL: URL) throws -> Self {
        let data: Data
        do {
            data = try Data(contentsOf: fileURL, options: .mappedIfSafe)
        } catch {
            throw ML5Error.modelPersistenceFailed(
                path: fileURL.path,
                message: error.localizedDescription
            )
        }
        return try Self(data: data)
    }

    private static func digest(
        formatVersion: Int,
        model: DenseNetworkModel,
        metadata: ML5ModelMetadata
    ) throws -> String {
        let payload = IntegrityPayload(
            formatVersion: formatVersion,
            model: model,
            metadata: metadata
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let digest = SHA256.hash(data: try encoder.encode(payload))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private struct IntegrityPayload: Encodable {
        let formatVersion: Int
        let model: DenseNetworkModel
        let metadata: ML5ModelMetadata
    }

    private enum CodingKeys: String, CodingKey {
        case formatVersion
        case model
        case metadata
        case integrityDigest
    }
}

extension DenseNetworkModel {
    /// Wraps this model in a versioned, integrity-checked archive.
    public func archived(metadata: ML5ModelMetadata) throws -> DenseModelArchive {
        try DenseModelArchive(model: self, metadata: metadata)
    }
}

/// Naming and metadata applied when exporting a dense model to Core ML.
public struct DenseCoreMLExportConfiguration: Sendable, Hashable {
    /// Single float32 multi-array input name, in configured feature order.
    public let inputName: String
    /// Single float32 multi-array output name, in configured output order.
    public let outputName: String
    /// Metadata embedded into the Core ML model description.
    public let metadata: ML5ModelMetadata

    /// Creates a Core ML export configuration with distinct valid feature names.
    ///
    /// - Throws: ``ML5Error/coreMLExportFailed(reason:)`` for empty, untrimmed, or equal names.
    public init(
        inputName: String = "features",
        outputName: String = "predictions",
        metadata: ML5ModelMetadata
    ) throws {
        let values = [inputName, outputName]
        guard
            values.allSatisfy({
                !$0.isEmpty && $0 == $0.trimmingCharacters(in: .whitespacesAndNewlines)
            })
        else {
            throw ML5Error.coreMLExportFailed(
                reason: "Core ML input and output names must be nonempty and trimmed."
            )
        }
        guard inputName != outputName else {
            throw ML5Error.coreMLExportFailed(
                reason: "Core ML input and output names must be distinct."
            )
        }
        self.inputName = inputName
        self.outputName = outputName
        self.metadata = metadata
    }
}

extension DenseNetworkModel {
    /// Serializes this dense network as an Apple Core ML neural-network model specification.
    ///
    /// The exported model accepts one float32 multi-array in this model's
    /// `configuration.inputFeatures` order and returns one float32 multi-array in its
    /// `configuration.outputNames` order.
    ///
    /// - Throws: ``ML5Error/coreMLExportFailed(reason:)`` when a double-precision parameter is
    ///   outside Core ML neural-network float32 storage.
    public func coreMLModelData(
        configuration export: DenseCoreMLExportConfiguration
    ) throws -> Data {
        try DenseCoreMLModelEncoder.encode(model: self, configuration: export)
    }

    /// Atomically writes an uncompiled `.mlmodel` specification.
    ///
    /// - Throws: ``ML5Error/coreMLExportFailed(reason:)`` when serialization or writing fails.
    public func writeCoreMLModel(
        to fileURL: URL,
        configuration export: DenseCoreMLExportConfiguration
    ) throws {
        do {
            try coreMLModelData(configuration: export).write(to: fileURL, options: .atomic)
        } catch let error as ML5Error {
            throw error
        } catch {
            throw ML5Error.coreMLExportFailed(reason: error.localizedDescription)
        }
    }

    /// Writes and asks Core ML to compile a temporary `.mlmodelc` directory.
    ///
    /// The caller owns the returned temporary directory and should move or remove it when done.
    ///
    /// - Throws: ``ML5Error/coreMLExportFailed(reason:)`` when writing or compilation fails.
    public func compileCoreMLModel(
        configuration export: DenseCoreMLExportConfiguration
    ) throws -> URL {
        try compileCoreMLModel(configuration: export, operations: .system)
    }

    func compileCoreMLModel(
        configuration export: DenseCoreMLExportConfiguration,
        operations: DenseCoreMLCompilationOperations
    ) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "ml5-coreml-\(UUID().uuidString)",
            isDirectory: true
        )
        let source = directory.appendingPathComponent("\(export.metadata.name).mlmodel")
        do {
            try operations.createDirectory(directory)
            try writeCoreMLModel(to: source, configuration: export)
            let compiled = try operations.compile(source)
            operations.remove(directory)
            return compiled
        } catch let error as ML5Error {
            operations.remove(directory)
            throw error
        } catch {
            operations.remove(directory)
            throw ML5Error.coreMLExportFailed(reason: error.localizedDescription)
        }
    }
}

private enum DenseCoreMLModelEncoder {
    static func encode(
        model: DenseNetworkModel,
        configuration export: DenseCoreMLExportConfiguration
    ) throws -> Data {
        let floatLayers = try model.layers.enumerated().map { index, layer in
            try FloatLayer(
                index: index,
                parameters: layer,
                activation: index < model.configuration.hiddenLayers.count
                    ? model.configuration.hiddenLayers[index].activation
                    : model.configuration.outputActivation
            )
        }
        let description = modelDescription(model: model, export: export)
        let network = neuralNetwork(layers: floatLayers, export: export)
        var output = ProtobufWriter()
        output.varint(field: 1, value: 4)
        output.message(field: 2, value: description)
        output.message(field: 500, value: network)
        return output.data
    }

    private static func modelDescription(
        model: DenseNetworkModel,
        export: DenseCoreMLExportConfiguration
    ) -> Data {
        var description = ProtobufWriter()
        description.message(
            field: 1,
            value: featureDescription(
                name: export.inputName,
                count: model.configuration.inputFeatures.count
            )
        )
        description.message(
            field: 10,
            value: featureDescription(
                name: export.outputName,
                count: model.configuration.outputNames.count
            )
        )
        description.message(field: 100, value: metadata(model: model, export: export))
        return description.data
    }

    private static func featureDescription(name: String, count: Int) -> Data {
        var array = ProtobufWriter()
        array.varint(field: 1, value: UInt64(count))
        array.varint(field: 2, value: 65_568)
        var type = ProtobufWriter()
        type.message(field: 5, value: array.data)
        var feature = ProtobufWriter()
        feature.string(field: 1, value: name)
        feature.message(field: 3, value: type.data)
        return feature.data
    }

    private static func metadata(
        model: DenseNetworkModel,
        export: DenseCoreMLExportConfiguration
    ) -> Data {
        let metadata = export.metadata
        var output = ProtobufWriter()
        output.string(
            field: 1,
            value: metadata.summary ?? "ML5 dense network exported for Apple Core ML."
        )
        output.string(field: 2, value: metadata.version)
        if let author = metadata.author {
            output.string(field: 3, value: author)
        }
        if let license = metadata.license {
            output.string(field: 4, value: license)
        }
        var values = metadata.additional
        values["com.ml5.swift.inputFeatures"] = model.configuration.inputFeatures
            .map(\.rawValue).joined(separator: ",")
        values["com.ml5.swift.outputNames"] = model.configuration.outputNames
            .map(\.rawValue).joined(separator: ",")
        if let source = metadata.source {
            values["com.ml5.swift.source"] = source.absoluteString
        }
        for (key, value) in values.sorted(by: { $0.key < $1.key }) {
            var entry = ProtobufWriter()
            entry.string(field: 1, value: key)
            entry.string(field: 2, value: value)
            output.message(field: 100, value: entry.data)
        }
        return output.data
    }

    private static func neuralNetwork(
        layers: [FloatLayer],
        export: DenseCoreMLExportConfiguration
    ) -> Data {
        var network = ProtobufWriter()
        var input = export.inputName
        for layer in layers {
            let affineOutput = "ml5_affine_\(layer.index)"
            let isFinal = layer.index == layers.count - 1
            let activationOutput = isFinal ? export.outputName : "ml5_activation_\(layer.index)"
            let innerOutput = layer.activation == .linear ? activationOutput : affineOutput
            network.message(
                field: 1,
                value: innerProductLayer(layer, input: input, output: innerOutput)
            )
            if let encodedActivation = activationLayer(
                layer.activation,
                index: layer.index,
                input: affineOutput,
                output: activationOutput
            ) {
                network.message(
                    field: 1,
                    value: encodedActivation
                )
            }
            input = activationOutput
        }
        network.varint(field: 5, value: 1)
        return network.data
    }

    private static func innerProductLayer(
        _ layer: FloatLayer,
        input: String,
        output: String
    ) -> Data {
        var weights = ProtobufWriter()
        weights.packedFloat(field: 1, values: layer.weights)
        var biases = ProtobufWriter()
        biases.packedFloat(field: 1, values: layer.biases)
        var parameters = ProtobufWriter()
        parameters.varint(field: 1, value: UInt64(layer.inputCount))
        parameters.varint(field: 2, value: UInt64(layer.outputCount))
        parameters.varint(field: 10, value: 1)
        parameters.message(field: 20, value: weights.data)
        parameters.message(field: 21, value: biases.data)

        var outputLayer = ProtobufWriter()
        outputLayer.string(field: 1, value: "dense_\(layer.index)")
        outputLayer.string(field: 2, value: input)
        outputLayer.string(field: 3, value: output)
        outputLayer.message(field: 140, value: parameters.data)
        return outputLayer.data
    }

    private static func activationLayer(
        _ activation: ActivationFunction,
        index: Int,
        input: String,
        output: String
    ) -> Data? {
        var layer = ProtobufWriter()
        layer.string(field: 1, value: "activation_\(index)")
        layer.string(field: 2, value: input)
        layer.string(field: 3, value: output)
        switch activation {
        case .linear:
            return nil
        case .rectifiedLinear:
            var parameters = ProtobufWriter()
            parameters.message(field: 10, value: Data())
            layer.message(field: 130, value: parameters.data)
        case .sigmoid:
            var parameters = ProtobufWriter()
            parameters.message(field: 40, value: Data())
            layer.message(field: 130, value: parameters.data)
        case .hyperbolicTangent:
            var parameters = ProtobufWriter()
            parameters.message(field: 30, value: Data())
            layer.message(field: 130, value: parameters.data)
        case .softmax:
            var parameters = ProtobufWriter()
            parameters.varint(field: 1, value: UInt64(bitPattern: Int64(-1)))
            layer.message(field: 950, value: parameters.data)
        }
        return layer.data
    }

    private struct FloatLayer {
        let index: Int
        let inputCount: Int
        let outputCount: Int
        let weights: [Float]
        let biases: [Float]
        let activation: ActivationFunction

        init(
            index: Int,
            parameters: DenseLayerParameters,
            activation: ActivationFunction
        ) throws {
            let weights = parameters.weights.map(Float.init)
            let biases = parameters.biases.map(Float.init)
            guard weights.allSatisfy(\.isFinite), biases.allSatisfy(\.isFinite) else {
                throw ML5Error.coreMLExportFailed(
                    reason: "Dense parameters must fit finite Core ML float32 storage."
                )
            }
            self.index = index
            inputCount = parameters.inputCount
            outputCount = parameters.outputCount
            self.weights = weights
            self.biases = biases
            self.activation = activation
        }
    }
}

struct DenseCoreMLCompilationOperations: @unchecked Sendable {
    var createDirectory: (URL) throws -> Void
    var compile: (URL) throws -> URL
    var remove: (URL) -> Void

    static let system = Self(
        createDirectory: { url in
            try FileManager.default.createDirectory(
                at: url,
                withIntermediateDirectories: true
            )
        },
        compile: MLModel.compileModel(at:),
        remove: { try? FileManager.default.removeItem(at: $0) }
    )
}

private struct ProtobufWriter {
    private(set) var data = Data()

    mutating func varint(field: Int, value: UInt64) {
        appendVarint(UInt64(field << 3))
        appendVarint(value)
    }

    mutating func string(field: Int, value: String) {
        message(field: field, value: Data(value.utf8))
    }

    mutating func message(field: Int, value: Data) {
        appendVarint(UInt64((field << 3) | 2))
        appendVarint(UInt64(value.count))
        data.append(value)
    }

    mutating func packedFloat(field: Int, values: [Float]) {
        var bytes = Data()
        bytes.reserveCapacity(values.count * MemoryLayout<UInt32>.size)
        for value in values {
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { bytes.append(contentsOf: $0) }
        }
        message(field: field, value: bytes)
    }

    private mutating func appendVarint(_ input: UInt64) {
        var value = input
        while value >= 0x80 {
            data.append(UInt8(value & 0x7f) | 0x80)
            value >>= 7
        }
        data.append(UInt8(value))
    }
}
