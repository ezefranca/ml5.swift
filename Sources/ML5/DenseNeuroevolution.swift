import Foundation

/// The architecture and ordered feature semantics inherited by an evolved dense brain.
public struct DenseBrainTopology: Sendable, Hashable, Codable {
    /// Scalar inputs in network-column order.
    public let inputFeatures: [FeatureName]
    /// Numeric outputs in network-column order.
    public let outputNames: [OutputName]
    /// Hidden widths and element-wise activations in execution order.
    public let hiddenLayers: [DenseLayerConfiguration]
    /// Activation applied to the final layer.
    public let outputActivation: ActivationFunction

    /// Input, hidden, and output widths in graph order.
    public var layerWidths: [Int] {
        [inputFeatures.count] + hiddenLayers.map(\.neuronCount) + [outputNames.count]
    }

    /// Creates a validated topology snapshot.
    ///
    /// - Throws: ``ML5Error/invalidNeuroevolutionConfiguration(reason:)`` when input or output
    ///   names are empty or duplicated.
    public init(
        inputFeatures: [FeatureName],
        outputNames: [OutputName],
        hiddenLayers: [DenseLayerConfiguration],
        outputActivation: ActivationFunction
    ) throws {
        guard inputFeatures.isEmpty == false, outputNames.isEmpty == false else {
            throw ML5Error.invalidNeuroevolutionConfiguration(
                reason: "A brain topology requires at least one input and one output."
            )
        }
        guard Set(inputFeatures).count == inputFeatures.count else {
            throw ML5Error.invalidNeuroevolutionConfiguration(
                reason: "Brain topology input names must be unique."
            )
        }
        guard Set(outputNames).count == outputNames.count else {
            throw ML5Error.invalidNeuroevolutionConfiguration(
                reason: "Brain topology output names must be unique."
            )
        }
        self.inputFeatures = inputFeatures
        self.outputNames = outputNames
        self.hiddenLayers = hiddenLayers
        self.outputActivation = outputActivation
    }

    /// Captures the topology of a validated dense-network configuration.
    public init(configuration: DenseNetworkConfiguration) {
        inputFeatures = configuration.inputFeatures
        outputNames = configuration.outputNames
        hiddenLayers = configuration.hiddenLayers
        outputActivation = configuration.outputActivation
    }

    /// Decodes and revalidates a topology snapshot.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            inputFeatures: container.decode([FeatureName].self, forKey: .inputFeatures),
            outputNames: container.decode([OutputName].self, forKey: .outputNames),
            hiddenLayers: container.decode(
                [DenseLayerConfiguration].self,
                forKey: .hiddenLayers
            ),
            outputActivation: container.decode(
                ActivationFunction.self,
                forKey: .outputActivation
            )
        )
    }

    /// Encodes every topology field using stable property names.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(inputFeatures, forKey: .inputFeatures)
        try container.encode(outputNames, forKey: .outputNames)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(outputActivation, forKey: .outputActivation)
    }

    private enum CodingKeys: String, CodingKey {
        case inputFeatures
        case outputNames
        case hiddenLayers
        case outputActivation
    }
}

/// A synchronous softmax classification produced by an evolved dense brain.
public struct DenseBrainClassification: Sendable, Hashable, Codable {
    /// Highest-scoring output, resolving ties in configured output order.
    public let label: OutputName
    /// Probability assigned to ``label``.
    public let confidence: Double
    /// Complete probability distribution keyed by output name.
    public let scores: [OutputName: Double]

    /// Creates and validates a complete probability result.
    ///
    /// - Throws: ``ML5Error/invalidNeuroevolutionConfiguration(reason:)`` when scores are empty,
    ///   nonfinite, outside `0...1`, do not contain the label, or do not sum to one.
    public init(label: OutputName, scores: [OutputName: Double]) throws {
        guard scores.isEmpty == false else {
            throw ML5Error.invalidNeuroevolutionConfiguration(
                reason: "A brain classification requires at least one score."
            )
        }
        guard scores.values.allSatisfy({ $0.isFinite && (0...1).contains($0) }) else {
            throw ML5Error.invalidNeuroevolutionConfiguration(
                reason: "Brain classification scores must be finite probabilities."
            )
        }
        guard let confidence = scores[label] else {
            throw ML5Error.invalidNeuroevolutionConfiguration(
                reason: "A brain classification must contain its selected label."
            )
        }
        guard abs(scores.values.reduce(0, +) - 1) <= 1e-9 else {
            throw ML5Error.invalidNeuroevolutionConfiguration(
                reason: "Brain classification probabilities must sum to one."
            )
        }
        self.label = label
        self.confidence = confidence
        self.scores = scores
    }

    /// Decodes and revalidates a classification.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            label: container.decode(OutputName.self, forKey: .label),
            scores: container.decode([OutputName: Double].self, forKey: .scores)
        )
    }

    /// Encodes the label and complete probability distribution.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(label, forKey: .label)
        try container.encode(scores, forKey: .scores)
    }

    private enum CodingKeys: String, CodingKey {
        case label
        case scores
    }
}

/// How selected dense-brain parameters change during mutation.
public enum DenseMutationStrategy: String, Sendable, Hashable, Codable, CaseIterable {
    /// Adds a normally distributed delta whose standard deviation is the configured magnitude.
    case gaussian
    /// Adds a uniformly distributed delta in `-magnitude...magnitude`.
    case uniform
    /// Replaces the selected parameter with a value in `-magnitude...magnitude`.
    case reset
}

/// Validated, deterministic dense-brain mutation behavior.
public struct DenseMutationConfiguration: Sendable, Hashable, Codable {
    /// Distribution or replacement rule applied to selected parameters.
    public let strategy: DenseMutationStrategy
    /// Independent probability that each eligible parameter changes.
    public let probability: Double
    /// Nonnegative finite distribution scale or reset half-range.
    public let magnitude: Double
    /// Whether bias values are eligible in addition to weights.
    public let mutatesBiases: Bool

    /// Creates a mutation policy.
    ///
    /// - Throws: ``ML5Error/invalidNeuroevolutionConfiguration(reason:)`` when probability is
    ///   outside `0...1` or magnitude is negative or nonfinite.
    public init(
        strategy: DenseMutationStrategy = .gaussian,
        probability: Double = 0.1,
        magnitude: Double = 0.1,
        mutatesBiases: Bool = true
    ) throws {
        guard probability.isFinite, (0...1).contains(probability) else {
            throw ML5Error.invalidNeuroevolutionConfiguration(
                reason: "Mutation probability must be finite and in the range 0...1."
            )
        }
        guard magnitude.isFinite, magnitude >= 0 else {
            throw ML5Error.invalidNeuroevolutionConfiguration(
                reason: "Mutation magnitude must be finite and nonnegative."
            )
        }
        self.strategy = strategy
        self.probability = probability
        self.magnitude = magnitude
        self.mutatesBiases = mutatesBiases
    }

    /// Decodes and revalidates a mutation policy.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            strategy: container.decode(DenseMutationStrategy.self, forKey: .strategy),
            probability: container.decode(Double.self, forKey: .probability),
            magnitude: container.decode(Double.self, forKey: .magnitude),
            mutatesBiases: container.decode(Bool.self, forKey: .mutatesBiases)
        )
    }

    /// Encodes every mutation option.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(strategy, forKey: .strategy)
        try container.encode(probability, forKey: .probability)
        try container.encode(magnitude, forKey: .magnitude)
        try container.encode(mutatesBiases, forKey: .mutatesBiases)
    }

    private enum CodingKeys: String, CodingKey {
        case strategy
        case probability
        case magnitude
        case mutatesBiases
    }
}

/// How compatible parent parameters are combined into an offspring brain.
public enum DenseCrossoverStrategy: String, Sendable, Hashable, Codable, CaseIterable {
    /// Selects every parameter independently from either parent.
    case uniform
    /// Selects a deterministic seeded prefix from the first parent and the suffix from the second.
    case singlePoint
    /// Computes a convex weighted average of both parents.
    case blend
}

/// Validated, deterministic dense-brain crossover behavior.
public struct DenseCrossoverConfiguration: Sendable, Hashable, Codable {
    /// Parent-combination rule.
    public let strategy: DenseCrossoverStrategy
    /// First-parent selection probability for uniform crossover.
    public let firstParentProbability: Double
    /// First-parent coefficient for blend crossover.
    public let blendFactor: Double

    /// Creates a crossover policy.
    ///
    /// - Throws: ``ML5Error/invalidNeuroevolutionConfiguration(reason:)`` when either
    ///   probability or coefficient is nonfinite or outside `0...1`.
    public init(
        strategy: DenseCrossoverStrategy = .uniform,
        firstParentProbability: Double = 0.5,
        blendFactor: Double = 0.5
    ) throws {
        guard
            firstParentProbability.isFinite,
            (0...1).contains(firstParentProbability)
        else {
            throw ML5Error.invalidNeuroevolutionConfiguration(
                reason: "First-parent probability must be finite and in the range 0...1."
            )
        }
        guard blendFactor.isFinite, (0...1).contains(blendFactor) else {
            throw ML5Error.invalidNeuroevolutionConfiguration(
                reason: "Crossover blend factor must be finite and in the range 0...1."
            )
        }
        self.strategy = strategy
        self.firstParentProbability = firstParentProbability
        self.blendFactor = blendFactor
    }

    /// Decodes and revalidates a crossover policy.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            strategy: container.decode(DenseCrossoverStrategy.self, forKey: .strategy),
            firstParentProbability: container.decode(
                Double.self,
                forKey: .firstParentProbability
            ),
            blendFactor: container.decode(Double.self, forKey: .blendFactor)
        )
    }

    /// Encodes every crossover option.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(strategy, forKey: .strategy)
        try container.encode(firstParentProbability, forKey: .firstParentProbability)
        try container.encode(blendFactor, forKey: .blendFactor)
    }

    private enum CodingKeys: String, CodingKey {
        case strategy
        case firstParentProbability
        case blendFactor
    }
}

/// An immutable, independently evolvable dense-network value for synchronous agent loops.
public struct DenseBrain: Sendable, Hashable, Codable {
    /// Immutable dense network containing the brain's topology and parameters.
    public let model: DenseNetworkModel
    /// Number of mutation or crossover generations applied to this lineage.
    public let generation: Int

    /// Topology inherited independently from the model's training hyperparameters.
    public var topology: DenseBrainTopology {
        DenseBrainTopology(configuration: model.configuration)
    }

    /// Creates a brain around a validated model.
    ///
    /// - Throws: ``ML5Error/invalidNeuroevolutionConfiguration(reason:)`` when generation is
    ///   negative.
    public init(model: DenseNetworkModel, generation: Int = 0) throws {
        guard generation >= 0 else {
            throw ML5Error.invalidNeuroevolutionConfiguration(
                reason: "A brain generation cannot be negative."
            )
        }
        self.model = model
        self.generation = generation
    }

    /// Decodes and revalidates a brain.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            model: container.decode(DenseNetworkModel.self, forKey: .model),
            generation: container.decode(Int.self, forKey: .generation)
        )
    }

    /// Encodes the immutable model and lineage generation.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(model, forKey: .model)
        try container.encode(generation, forKey: .generation)
    }

    /// Returns an independent value copy suitable for a separate agent lineage.
    public func copied() -> Self {
        self
    }

    /// Produces numeric model output synchronously without actor or task suspension.
    public func predict(_ features: FeatureVector) throws -> ModelOutput {
        let values = try model.values(for: features)
        return try ModelOutput(
            Dictionary(
                uniqueKeysWithValues: zip(model.configuration.outputNames, values).map {
                    ($0, FeatureValue.number($1))
                }
            )
        )
    }

    /// Produces a synchronous, ordered softmax classification.
    ///
    /// - Throws: ``ML5Error/invalidNeuroevolutionConfiguration(reason:)`` when the brain does
    ///   not have a softmax output layer.
    public func classify(_ features: FeatureVector) throws -> DenseBrainClassification {
        guard model.configuration.outputActivation == .softmax else {
            throw ML5Error.invalidNeuroevolutionConfiguration(
                reason: "Synchronous brain classification requires a softmax output."
            )
        }
        let values = try model.values(for: features)
        var bestIndex = 0
        for index in values.indices.dropFirst() where values[index] > values[bestIndex] {
            bestIndex = index
        }
        return try DenseBrainClassification(
            label: model.configuration.outputNames[bestIndex],
            scores: Dictionary(
                uniqueKeysWithValues: zip(model.configuration.outputNames, values)
            )
        )
    }

    /// Returns an independently parameterized child using a deterministic mutation seed.
    public func mutated(
        using configuration: DenseMutationConfiguration,
        seed: UInt64
    ) throws -> Self {
        guard generation < Int.max else {
            throw ML5Error.invalidNeuroevolutionConfiguration(
                reason: "A brain generation cannot overflow."
            )
        }
        var generator = DenseEvolutionGenerator(state: seed)
        let layers = try model.layers.map { layer in
            let weights = try layer.weights.map {
                try Self.mutatedValue(
                    $0,
                    configuration: configuration,
                    generator: &generator
                )
            }
            let biases =
                if configuration.mutatesBiases {
                    try layer.biases.map {
                        try Self.mutatedValue(
                            $0,
                            configuration: configuration,
                            generator: &generator
                        )
                    }
                } else {
                    layer.biases
                }
            return try DenseLayerParameters(
                inputCount: layer.inputCount,
                outputCount: layer.outputCount,
                weights: weights,
                biases: biases
            )
        }
        return try Self(
            model: DenseNetworkModel(configuration: model.configuration, layers: layers),
            generation: generation + 1
        )
    }

    /// Returns a deterministic child of this brain and a topology-compatible parent.
    public func crossed(
        with other: Self,
        using configuration: DenseCrossoverConfiguration,
        seed: UInt64
    ) throws -> Self {
        guard topology == other.topology else {
            throw ML5Error.incompatibleBrainTopologies
        }
        let parentGeneration = max(generation, other.generation)
        guard parentGeneration < Int.max else {
            throw ML5Error.invalidNeuroevolutionConfiguration(
                reason: "A brain generation cannot overflow."
            )
        }
        var generator = DenseEvolutionGenerator(state: seed)
        let parameterCount = model.layers.reduce(0) {
            $0 + $1.weights.count + $1.biases.count
        }
        let crossoverPoint = Int(generator.next() % UInt64(parameterCount + 1))
        var parameterIndex = 0
        let layers = try zip(model.layers, other.model.layers).map { first, second in
            let weights = zip(first.weights, second.weights).map {
                let value = Self.crossedValue(
                    first: $0,
                    second: $1,
                    parameterIndex: parameterIndex,
                    crossoverPoint: crossoverPoint,
                    configuration: configuration,
                    generator: &generator
                )
                parameterIndex += 1
                return value
            }
            let biases = zip(first.biases, second.biases).map {
                let value = Self.crossedValue(
                    first: $0,
                    second: $1,
                    parameterIndex: parameterIndex,
                    crossoverPoint: crossoverPoint,
                    configuration: configuration,
                    generator: &generator
                )
                parameterIndex += 1
                return value
            }
            return try DenseLayerParameters(
                inputCount: first.inputCount,
                outputCount: first.outputCount,
                weights: weights,
                biases: biases
            )
        }
        return try Self(
            model: DenseNetworkModel(configuration: model.configuration, layers: layers),
            generation: parentGeneration + 1
        )
    }

    /// Captures a versioned, validated brain snapshot for persistence or transfer.
    public func snapshot() -> DenseBrainSnapshot {
        DenseBrainSnapshot(brain: self)
    }

    private static func mutatedValue(
        _ value: Double,
        configuration: DenseMutationConfiguration,
        generator: inout DenseEvolutionGenerator
    ) throws -> Double {
        guard generator.unitInterval() < configuration.probability else { return value }
        let mutation =
            switch configuration.strategy {
            case .gaussian:
                value + generator.normal() * configuration.magnitude
            case .uniform:
                value + generator.uniformDelta(magnitude: configuration.magnitude)
            case .reset:
                generator.uniformDelta(magnitude: configuration.magnitude)
            }
        guard mutation.isFinite else {
            throw ML5Error.invalidNeuroevolutionConfiguration(
                reason: "Mutation produced a nonfinite parameter."
            )
        }
        return mutation
    }

    private static func crossedValue(
        first: Double,
        second: Double,
        parameterIndex: Int,
        crossoverPoint: Int,
        configuration: DenseCrossoverConfiguration,
        generator: inout DenseEvolutionGenerator
    ) -> Double {
        switch configuration.strategy {
        case .uniform:
            generator.unitInterval() < configuration.firstParentProbability
                ? first : second
        case .singlePoint:
            parameterIndex < crossoverPoint ? first : second
        case .blend:
            first * configuration.blendFactor
                + second * (1 - configuration.blendFactor)
        }
    }

    private enum CodingKeys: String, CodingKey {
        case model
        case generation
    }
}

/// A versioned, validated serialization envelope for one dense brain.
public struct DenseBrainSnapshot: Sendable, Hashable, Codable {
    /// Snapshot schema written by this release.
    public static let currentFormatVersion = 1

    /// Persisted snapshot schema version.
    public let formatVersion: Int
    /// Immutable evolved brain.
    public let brain: DenseBrain

    /// Captures a brain using the current snapshot schema.
    public init(brain: DenseBrain) {
        formatVersion = Self.currentFormatVersion
        self.brain = brain
    }

    init(formatVersion: Int, brain: DenseBrain) throws {
        guard formatVersion == Self.currentFormatVersion else {
            throw ML5Error.invalidNeuroevolutionConfiguration(
                reason: "Unsupported brain snapshot format version \(formatVersion)."
            )
        }
        self.formatVersion = formatVersion
        self.brain = brain
    }

    /// Decodes and validates a brain snapshot schema.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            formatVersion: container.decode(Int.self, forKey: .formatVersion),
            brain: container.decode(DenseBrain.self, forKey: .brain)
        )
    }

    /// Encodes the schema version and brain.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(formatVersion, forKey: .formatVersion)
        try container.encode(brain, forKey: .brain)
    }

    private enum CodingKeys: String, CodingKey {
        case formatVersion
        case brain
    }
}

/// A versioned homogeneous brain population with deterministic mutation resume state.
public struct DenseBrainPopulation: Sendable, Hashable, Codable {
    /// Population schema written by this release.
    public static let currentFormatVersion = 1

    /// Persisted population schema version.
    public let formatVersion: Int
    /// Shared lineage generation for every brain.
    public let generation: Int
    /// Nonempty topology-compatible brain values.
    public let brains: [DenseBrain]
    /// SplitMix64 state consumed when deriving the next generation's per-brain seeds.
    public let randomState: UInt64

    /// Creates a generation from topology-compatible brains at the same generation.
    ///
    /// - Throws: ``ML5Error/invalidNeuroevolutionConfiguration(reason:)`` for an empty,
    ///   mixed-generation, or mixed-topology population.
    public init(brains: [DenseBrain], seed: UInt64) throws {
        try self.init(
            formatVersion: Self.currentFormatVersion,
            brains: brains,
            randomState: seed
        )
    }

    init(formatVersion: Int, brains: [DenseBrain], randomState: UInt64) throws {
        guard formatVersion == Self.currentFormatVersion else {
            throw ML5Error.invalidNeuroevolutionConfiguration(
                reason: "Unsupported brain population format version \(formatVersion)."
            )
        }
        guard let first = brains.first else {
            throw ML5Error.invalidNeuroevolutionConfiguration(
                reason: "A brain population cannot be empty."
            )
        }
        guard brains.allSatisfy({ $0.generation == first.generation }) else {
            throw ML5Error.invalidNeuroevolutionConfiguration(
                reason: "Every population brain must belong to the same generation."
            )
        }
        guard brains.allSatisfy({ $0.topology == first.topology }) else {
            throw ML5Error.incompatibleBrainTopologies
        }
        self.formatVersion = formatVersion
        generation = first.generation
        self.brains = brains
        self.randomState = randomState
    }

    /// Decodes and revalidates the population schema, generations, and topologies.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let decodedGeneration = try container.decode(Int.self, forKey: .generation)
        try self.init(
            formatVersion: container.decode(Int.self, forKey: .formatVersion),
            brains: container.decode([DenseBrain].self, forKey: .brains),
            randomState: container.decode(UInt64.self, forKey: .randomState)
        )
        guard generation == decodedGeneration else {
            throw ML5Error.invalidNeuroevolutionConfiguration(
                reason: "A population generation does not match its brains."
            )
        }
    }

    /// Encodes schema, generation, brains, and deterministic resume state.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(formatVersion, forKey: .formatVersion)
        try container.encode(generation, forKey: .generation)
        try container.encode(brains, forKey: .brains)
        try container.encode(randomState, forKey: .randomState)
    }

    /// Mutates every brain with a derived seed and returns resumable next-generation state.
    public func mutated(using configuration: DenseMutationConfiguration) throws -> Self {
        var generator = DenseEvolutionGenerator(state: randomState)
        let nextBrains = try brains.map { brain in
            try brain.mutated(using: configuration, seed: generator.next())
        }
        return try Self(
            formatVersion: Self.currentFormatVersion,
            brains: nextBrains,
            randomState: generator.state
        )
    }

    private enum CodingKeys: String, CodingKey {
        case formatVersion
        case generation
        case brains
        case randomState
    }
}

private struct DenseEvolutionGenerator: RandomNumberGenerator {
    private(set) var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }

    mutating func unitInterval() -> Double {
        Double(next() >> 11) * 0x1.0p-53
    }

    mutating func uniformDelta(magnitude: Double) -> Double {
        (unitInterval() * 2 - 1) * magnitude
    }

    mutating func normal() -> Double {
        let first = (Double(next() >> 11) + 0.5) * 0x1.0p-53
        let second = unitInterval()
        return sqrt(-2 * log(first)) * cos(2 * .pi * second)
    }
}
