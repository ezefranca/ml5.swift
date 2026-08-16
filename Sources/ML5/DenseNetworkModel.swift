import Foundation

/// Validated row-major weights and biases for one dense layer.
public struct DenseLayerParameters: Sendable, Hashable, Codable {
    /// Number of values consumed by each sample.
    public let inputCount: Int
    /// Number of values produced by each sample.
    public let outputCount: Int
    /// Weights in output-major row order: `outputIndex * inputCount + inputIndex`.
    public let weights: [Double]
    /// One bias per output value.
    public let biases: [Double]

    /// Creates shape-checked finite dense-layer parameters.
    ///
    /// - Throws: ``ML5Error/invalidModel(reason:)`` when dimensions are nonpositive or overflow,
    ///   storage counts disagree with dimensions, or any parameter is nonfinite.
    public init(
        inputCount: Int,
        outputCount: Int,
        weights: [Double],
        biases: [Double]
    ) throws {
        guard inputCount > 0 else {
            throw ML5Error.invalidModel(reason: "A layer input count must be positive.")
        }
        guard outputCount > 0 else {
            throw ML5Error.invalidModel(reason: "A layer output count must be positive.")
        }
        guard inputCount <= Int.max / outputCount else {
            throw ML5Error.invalidModel(reason: "Layer dimensions overflow their weight count.")
        }
        let expectedWeightCount = inputCount * outputCount
        guard weights.count == expectedWeightCount else {
            throw ML5Error.invalidModel(
                reason:
                    "A layer requires \(expectedWeightCount) weights, but received \(weights.count)."
            )
        }
        guard biases.count == outputCount else {
            throw ML5Error.invalidModel(
                reason: "A layer requires \(outputCount) biases, but received \(biases.count)."
            )
        }
        guard weights.allSatisfy(\.isFinite) else {
            throw ML5Error.invalidModel(reason: "Every layer weight must be finite.")
        }
        guard biases.allSatisfy(\.isFinite) else {
            throw ML5Error.invalidModel(reason: "Every layer bias must be finite.")
        }
        self.inputCount = inputCount
        self.outputCount = outputCount
        self.weights = weights
        self.biases = biases
    }

    /// Decodes and revalidates persisted dense-layer parameters.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            inputCount: container.decode(Int.self, forKey: .inputCount),
            outputCount: container.decode(Int.self, forKey: .outputCount),
            weights: container.decode([Double].self, forKey: .weights),
            biases: container.decode([Double].self, forKey: .biases)
        )
    }

    /// Encodes dimensions and row-major parameter storage.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(inputCount, forKey: .inputCount)
        try container.encode(outputCount, forKey: .outputCount)
        try container.encode(weights, forKey: .weights)
        try container.encode(biases, forKey: .biases)
    }

    private enum CodingKeys: String, CodingKey {
        case inputCount
        case outputCount
        case weights
        case biases
    }
}

/// An immutable dense network that supports async, batch, and synchronous snapshot inference.
public struct DenseNetworkModel: ModelInferenceSnapshotProviding, Sendable, Hashable, Codable {
    /// Architecture, ordered fields, and training hyperparameters associated with the model.
    public let configuration: DenseNetworkConfiguration
    /// Hidden-layer parameters followed by output-layer parameters.
    public let layers: [DenseLayerParameters]

    /// Creates a model whose parameter shapes exactly match its configuration.
    ///
    /// - Throws: ``ML5Error/invalidModel(reason:)`` for an incorrect layer count or any
    ///   disconnected layer dimensions.
    public init(
        configuration: DenseNetworkConfiguration,
        layers: [DenseLayerParameters]
    ) throws {
        let expectedLayerCount = configuration.hiddenLayers.count + 1
        guard layers.count == expectedLayerCount else {
            throw ML5Error.invalidModel(
                reason: "The architecture requires \(expectedLayerCount) parameter layers."
            )
        }

        var expectedInputCount = configuration.inputFeatures.count
        for (index, layer) in layers.enumerated() {
            guard layer.inputCount == expectedInputCount else {
                throw ML5Error.invalidModel(
                    reason: "Layer \(index) does not consume the preceding layer width."
                )
            }
            let expectedOutputCount =
                index < configuration.hiddenLayers.count
                ? configuration.hiddenLayers[index].neuronCount
                : configuration.outputNames.count
            guard layer.outputCount == expectedOutputCount else {
                throw ML5Error.invalidModel(
                    reason: "Layer \(index) does not produce its configured width."
                )
            }
            expectedInputCount = layer.outputCount
        }

        self.configuration = configuration
        self.layers = layers
    }

    /// Decodes and revalidates an immutable model.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            configuration: container.decode(
                DenseNetworkConfiguration.self,
                forKey: .configuration
            ),
            layers: container.decode([DenseLayerParameters].self, forKey: .layers)
        )
    }

    /// Encodes the configuration and every parameter layer.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(configuration, forKey: .configuration)
        try container.encode(layers, forKey: .layers)
    }

    /// Produces framework-independent numeric outputs in configured order.
    public func predict(_ features: FeatureVector) async throws -> ModelOutput {
        try Task.checkCancellation()
        let output = try modelOutput(for: features)
        try Task.checkCancellation()
        return output
    }

    /// Produces an immutable synchronous snapshot without sharing mutable state.
    public func makeInferenceSnapshot() async throws -> ModelInferenceSnapshot {
        try Task.checkCancellation()
        return ModelInferenceSnapshot { features in
            try modelOutput(for: features)
        }
    }

    func values(for features: FeatureVector) throws -> [Double] {
        var values = try configuration.inputFeatures.map { name in
            guard let value = features[name] else {
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

        for (index, layer) in layers.enumerated() {
            values = DenseNetworkMath.affine(values, layer: layer)
            let activation =
                index < configuration.hiddenLayers.count
                ? configuration.hiddenLayers[index].activation
                : configuration.outputActivation
            values = DenseNetworkMath.activate(values, using: activation)
        }
        return values
    }

    private func modelOutput(for features: FeatureVector) throws -> ModelOutput {
        let values = try values(for: features)
        var output: [OutputName: FeatureValue] = [:]
        for (name, value) in zip(configuration.outputNames, values) {
            output[name] = .number(value)
        }
        return try ModelOutput(output)
    }

    private enum CodingKeys: String, CodingKey {
        case configuration
        case layers
    }
}

enum DenseNetworkMath {
    static func affine(_ input: [Double], layer: DenseLayerParameters) -> [Double] {
        (0..<layer.outputCount).map { outputIndex in
            let rowStart = outputIndex * layer.inputCount
            return (0..<layer.inputCount).reduce(layer.biases[outputIndex]) {
                $0 + layer.weights[rowStart + $1] * input[$1]
            }
        }
    }

    static func activate(_ input: [Double], using function: ActivationFunction) -> [Double] {
        switch function {
        case .linear:
            return input
        case .rectifiedLinear:
            return input.map { max(0, $0) }
        case .sigmoid:
            return input.map { value in
                if value >= 0 {
                    return 1 / (1 + exp(-value))
                }
                let exponential = exp(value)
                return exponential / (1 + exponential)
            }
        case .hyperbolicTangent:
            return input.map(Foundation.tanh)
        case .softmax:
            let maximum = input.reduce(-Double.infinity, max)
            let exponentials = input.map { exp($0 - maximum) }
            let sum = exponentials.reduce(0, +)
            return exponentials.map { $0 / sum }
        }
    }
}
