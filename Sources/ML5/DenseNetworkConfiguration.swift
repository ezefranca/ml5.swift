import Foundation

/// Activation functions supported by ML5 dense-network backends.
public enum ActivationFunction: String, Sendable, Hashable, Codable, CaseIterable {
    /// Leaves the preactivation value unchanged.
    case linear
    /// Replaces negative values with zero.
    case rectifiedLinear
    /// Maps values to the open interval from zero to one.
    case sigmoid
    /// Maps values to the open interval from negative one to one.
    case hyperbolicTangent
    /// Normalizes an output layer into a probability distribution.
    case softmax
}

/// Strategies for deterministically initializing dense-layer weights.
public enum WeightInitialization: String, Sendable, Hashable, Codable, CaseIterable {
    /// Samples uniformly using the Glorot/Xavier fan-in and fan-out scale.
    case glorotUniform
    /// Samples normally using the He fan-in scale.
    case heNormal
    /// Initializes every weight and bias to zero.
    case zeros
}

/// Loss functions supported by dense-network trainers.
public enum TrainingLoss: String, Sendable, Hashable, Codable, CaseIterable {
    /// Mean squared error for scalar or vector regression.
    case meanSquaredError
    /// Multiclass negative log likelihood for a softmax output.
    case categoricalCrossEntropy
    /// Binary negative log likelihood for one or more sigmoid outputs.
    case binaryCrossEntropy
}

/// Optimization algorithms supported by dense-network trainers.
public enum OptimizerKind: String, Sendable, Hashable, Codable, CaseIterable {
    /// Stochastic gradient descent with optional momentum.
    case stochasticGradientDescent
    /// Adaptive moment estimation.
    case adam
}

/// Validated hyperparameters for a dense-network optimizer.
public struct OptimizerConfiguration: Sendable, Hashable, Codable {
    /// Standard Adam hyperparameters.
    public static let adam = OptimizerConfiguration(
        uncheckedKind: .adam,
        momentum: 0,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8
    )

    /// Stochastic gradient descent without momentum.
    public static let stochasticGradientDescent = OptimizerConfiguration(
        uncheckedKind: .stochasticGradientDescent,
        momentum: 0,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8
    )

    /// The update algorithm.
    public let kind: OptimizerKind
    /// Momentum used by stochastic gradient descent, in the half-open range zero to one.
    public let momentum: Double
    /// Adam's first-moment decay, in the half-open range zero to one.
    public let beta1: Double
    /// Adam's second-moment decay, in the half-open range zero to one.
    public let beta2: Double
    /// Positive finite denominator stabilizer used by Adam.
    public let epsilon: Double

    /// Creates validated optimizer hyperparameters.
    ///
    /// - Throws: ``ML5Error/invalidTrainingConfiguration(reason:)`` when a value is nonfinite or
    ///   outside its mathematical domain.
    public init(
        kind: OptimizerKind = .adam,
        momentum: Double = 0,
        beta1: Double = 0.9,
        beta2: Double = 0.999,
        epsilon: Double = 1e-8
    ) throws {
        guard momentum.isFinite, (0..<1).contains(momentum) else {
            throw ML5Error.invalidTrainingConfiguration(
                reason: "Optimizer momentum must be finite and in the range 0..<1."
            )
        }
        guard beta1.isFinite, (0..<1).contains(beta1) else {
            throw ML5Error.invalidTrainingConfiguration(
                reason: "Adam beta1 must be finite and in the range 0..<1."
            )
        }
        guard beta2.isFinite, (0..<1).contains(beta2) else {
            throw ML5Error.invalidTrainingConfiguration(
                reason: "Adam beta2 must be finite and in the range 0..<1."
            )
        }
        guard epsilon.isFinite, epsilon > 0 else {
            throw ML5Error.invalidTrainingConfiguration(
                reason: "Adam epsilon must be finite and greater than zero."
            )
        }
        self.kind = kind
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
    }

    /// Decodes and revalidates persisted optimizer hyperparameters.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            kind: container.decode(OptimizerKind.self, forKey: .kind),
            momentum: container.decode(Double.self, forKey: .momentum),
            beta1: container.decode(Double.self, forKey: .beta1),
            beta2: container.decode(Double.self, forKey: .beta2),
            epsilon: container.decode(Double.self, forKey: .epsilon)
        )
    }

    /// Encodes every optimizer hyperparameter.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(kind, forKey: .kind)
        try container.encode(momentum, forKey: .momentum)
        try container.encode(beta1, forKey: .beta1)
        try container.encode(beta2, forKey: .beta2)
        try container.encode(epsilon, forKey: .epsilon)
    }

    private enum CodingKeys: String, CodingKey {
        case kind
        case momentum
        case beta1
        case beta2
        case epsilon
    }

    private init(
        uncheckedKind kind: OptimizerKind,
        momentum: Double,
        beta1: Double,
        beta2: Double,
        epsilon: Double
    ) {
        self.kind = kind
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
    }
}

/// The neuron count and activation used by one hidden dense layer.
public struct DenseLayerConfiguration: Sendable, Hashable, Codable {
    /// Number of neurons in the layer.
    public let neuronCount: Int
    /// Element-wise activation applied after weights and biases.
    public let activation: ActivationFunction

    /// Creates a hidden-layer definition.
    ///
    /// - Throws: ``ML5Error/invalidTrainingConfiguration(reason:)`` when `neuronCount` is not
    ///   positive or when the layer uses softmax, which is reserved for output layers.
    public init(neuronCount: Int, activation: ActivationFunction = .rectifiedLinear) throws {
        guard neuronCount > 0 else {
            throw ML5Error.invalidTrainingConfiguration(
                reason: "A dense layer must contain at least one neuron."
            )
        }
        guard activation != .softmax else {
            throw ML5Error.invalidTrainingConfiguration(
                reason: "Softmax is supported only as an output activation."
            )
        }
        self.neuronCount = neuronCount
        self.activation = activation
    }

    /// Decodes and revalidates a persisted hidden-layer definition.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            neuronCount: container.decode(Int.self, forKey: .neuronCount),
            activation: container.decode(ActivationFunction.self, forKey: .activation)
        )
    }

    /// Encodes the neuron count and activation.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(neuronCount, forKey: .neuronCount)
        try container.encode(activation, forKey: .activation)
    }

    private enum CodingKeys: String, CodingKey {
        case neuronCount
        case activation
    }
}

/// A complete, deterministic dense-network architecture and training configuration.
public struct DenseNetworkConfiguration: Sendable, Hashable, Codable {
    /// Scalar numeric input features in matrix-column order.
    public let inputFeatures: [FeatureName]
    /// Numeric outputs in matrix-column order.
    public let outputNames: [OutputName]
    /// Zero or more hidden dense layers.
    public let hiddenLayers: [DenseLayerConfiguration]
    /// Activation applied by the output layer.
    public let outputActivation: ActivationFunction
    /// Initial weight distribution.
    public let weightInitialization: WeightInitialization
    /// Objective minimized during training.
    public let loss: TrainingLoss
    /// Optimizer and its validated hyperparameters.
    public let optimizer: OptimizerConfiguration
    /// Positive finite gradient-step scale.
    public let learningRate: Double
    /// Positive number of samples in each training update.
    public let batchSize: Int
    /// Positive maximum number of complete training passes.
    public let epochs: Int
    /// Fraction of supplied training samples reserved for validation, in `0..<1`.
    public let validationFraction: Double
    /// Seed controlling initialization, splitting, and sample ordering.
    public let seed: UInt64

    /// Creates and cross-validates a dense-network configuration.
    ///
    /// - Throws: ``ML5Error/invalidTrainingConfiguration(reason:)`` for empty or duplicate
    ///   feature/output lists, invalid numeric options, or an incompatible loss and output
    ///   activation.
    public init(
        inputFeatures: [FeatureName],
        outputNames: [OutputName],
        hiddenLayers: [DenseLayerConfiguration] = [],
        outputActivation: ActivationFunction = .linear,
        weightInitialization: WeightInitialization = .glorotUniform,
        loss: TrainingLoss = .meanSquaredError,
        optimizer: OptimizerConfiguration = .adam,
        learningRate: Double = 0.001,
        batchSize: Int = 32,
        epochs: Int = 100,
        validationFraction: Double = 0.2,
        seed: UInt64 = 0
    ) throws {
        guard inputFeatures.isEmpty == false else {
            throw ML5Error.invalidTrainingConfiguration(
                reason: "At least one input feature is required."
            )
        }
        guard Set(inputFeatures).count == inputFeatures.count else {
            throw ML5Error.invalidTrainingConfiguration(
                reason: "Input feature names must be unique."
            )
        }
        guard outputNames.isEmpty == false else {
            throw ML5Error.invalidTrainingConfiguration(
                reason: "At least one output name is required."
            )
        }
        guard Set(outputNames).count == outputNames.count else {
            throw ML5Error.invalidTrainingConfiguration(
                reason: "Output names must be unique."
            )
        }
        guard learningRate.isFinite, learningRate > 0 else {
            throw ML5Error.invalidTrainingConfiguration(
                reason: "Learning rate must be finite and greater than zero."
            )
        }
        guard batchSize > 0 else {
            throw ML5Error.invalidTrainingConfiguration(
                reason: "Batch size must be greater than zero."
            )
        }
        guard epochs > 0 else {
            throw ML5Error.invalidTrainingConfiguration(
                reason: "Epoch count must be greater than zero."
            )
        }
        guard validationFraction.isFinite, (0..<1).contains(validationFraction) else {
            throw ML5Error.invalidTrainingConfiguration(
                reason: "Validation fraction must be finite and in the range 0..<1."
            )
        }
        switch loss {
        case .meanSquaredError:
            guard outputActivation != .softmax else {
                throw ML5Error.invalidTrainingConfiguration(
                    reason: "Mean squared error does not support a softmax output."
                )
            }
        case .categoricalCrossEntropy:
            guard outputActivation == .softmax, outputNames.count > 1 else {
                throw ML5Error.invalidTrainingConfiguration(
                    reason: "Categorical cross entropy requires multiple softmax outputs."
                )
            }
        case .binaryCrossEntropy:
            guard outputActivation == .sigmoid else {
                throw ML5Error.invalidTrainingConfiguration(
                    reason: "Binary cross entropy requires a sigmoid output."
                )
            }
        }

        self.inputFeatures = inputFeatures
        self.outputNames = outputNames
        self.hiddenLayers = hiddenLayers
        self.outputActivation = outputActivation
        self.weightInitialization = weightInitialization
        self.loss = loss
        self.optimizer = optimizer
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.epochs = epochs
        self.validationFraction = validationFraction
        self.seed = seed
    }

    /// Decodes and revalidates a persisted dense-network configuration.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            inputFeatures: container.decode([FeatureName].self, forKey: .inputFeatures),
            outputNames: container.decode([OutputName].self, forKey: .outputNames),
            hiddenLayers: container.decode([DenseLayerConfiguration].self, forKey: .hiddenLayers),
            outputActivation: container.decode(ActivationFunction.self, forKey: .outputActivation),
            weightInitialization: container.decode(
                WeightInitialization.self,
                forKey: .weightInitialization
            ),
            loss: container.decode(TrainingLoss.self, forKey: .loss),
            optimizer: container.decode(OptimizerConfiguration.self, forKey: .optimizer),
            learningRate: container.decode(Double.self, forKey: .learningRate),
            batchSize: container.decode(Int.self, forKey: .batchSize),
            epochs: container.decode(Int.self, forKey: .epochs),
            validationFraction: container.decode(Double.self, forKey: .validationFraction),
            seed: container.decode(UInt64.self, forKey: .seed)
        )
    }

    /// Encodes the complete architecture and training configuration.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(inputFeatures, forKey: .inputFeatures)
        try container.encode(outputNames, forKey: .outputNames)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(outputActivation, forKey: .outputActivation)
        try container.encode(weightInitialization, forKey: .weightInitialization)
        try container.encode(loss, forKey: .loss)
        try container.encode(optimizer, forKey: .optimizer)
        try container.encode(learningRate, forKey: .learningRate)
        try container.encode(batchSize, forKey: .batchSize)
        try container.encode(epochs, forKey: .epochs)
        try container.encode(validationFraction, forKey: .validationFraction)
        try container.encode(seed, forKey: .seed)
    }

    private enum CodingKeys: String, CodingKey {
        case inputFeatures
        case outputNames
        case hiddenLayers
        case outputActivation
        case weightInitialization
        case loss
        case optimizer
        case learningRate
        case batchSize
        case epochs
        case validationFraction
        case seed
    }
}
