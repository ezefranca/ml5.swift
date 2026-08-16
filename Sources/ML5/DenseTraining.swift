import Foundation

/// A feature vector paired with finite numeric targets for dense-network training.
public struct DenseTrainingSample: Sendable, Hashable, Codable {
    /// Model input features.
    public let features: FeatureVector
    /// Regression values, one-hot classes, or independent binary targets in output order.
    public let targets: [Double]

    /// Creates a training sample with at least one finite target.
    ///
    /// - Throws: ``ML5Error/invalidTrainingSample(reason:)`` when targets are empty or nonfinite.
    public init(features: FeatureVector, targets: [Double]) throws {
        guard targets.isEmpty == false else {
            throw ML5Error.invalidTrainingSample(reason: "Targets cannot be empty.")
        }
        guard targets.allSatisfy(\.isFinite) else {
            throw ML5Error.invalidTrainingSample(reason: "Every target must be finite.")
        }
        self.features = features
        self.targets = targets
    }

    /// Creates a scalar dense sample from an existing regression sample.
    public init(regression sample: RegressionSample) {
        features = sample.features
        targets = [sample.target]
    }

    /// Creates a one-hot sample for an ordered list of classification labels.
    ///
    /// - Throws: ``ML5Error/invalidTrainingSample(reason:)`` when labels are empty, duplicated, or
    ///   do not contain the sample's expected label.
    public static func classification<Label: ClassificationLabel>(
        features: FeatureVector,
        label: Label,
        labels: [Label]
    ) throws -> Self {
        guard labels.isEmpty == false else {
            throw ML5Error.invalidTrainingSample(
                reason: "Classification labels cannot be empty."
            )
        }
        guard Set(labels).count == labels.count else {
            throw ML5Error.invalidTrainingSample(
                reason: "Classification labels must be unique."
            )
        }
        guard let selectedIndex = labels.firstIndex(of: label) else {
            throw ML5Error.invalidTrainingSample(
                reason: "The expected classification label is not configured."
            )
        }
        var targets = Array(repeating: 0.0, count: labels.count)
        targets[selectedIndex] = 1
        return try Self(features: features, targets: targets)
    }

    /// Decodes and revalidates a persisted dense training sample.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            features: container.decode(FeatureVector.self, forKey: .features),
            targets: container.decode([Double].self, forKey: .targets)
        )
    }

    /// Encodes features and ordered numeric targets.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(features, forKey: .features)
        try container.encode(targets, forKey: .targets)
    }

    private enum CodingKeys: String, CodingKey {
        case features
        case targets
    }
}

/// Loss measurements captured after one complete training epoch.
public struct DenseEpochMetrics: Sendable, Hashable {
    /// One-based epoch number.
    public let epoch: Int
    /// Mean loss across the epoch's training partition after updates.
    public let trainingLoss: Double
    /// Mean loss across the held-out validation partition, or `nil` when it is empty.
    public let validationLoss: Double?
}

/// The immutable model and complete loss history produced by dense training.
public struct DenseTrainingResult: Sendable, Hashable {
    /// Trained immutable model.
    public let model: DenseNetworkModel
    /// One metrics value for every completed epoch.
    public let history: [DenseEpochMetrics]
}

/// A deterministic reference trainer for small dense classification and regression models.
///
/// This implementation prioritizes readable numerical semantics and reproducible tests. Use an
/// accelerated training device for large datasets after selecting a backend that supports it.
public struct DenseCPUTrainer: Sendable {
    /// Creates a stateless CPU reference trainer.
    public init() {}

    /// Fits an immutable dense model using deterministic mini-batch backpropagation.
    ///
    /// Training validates every sample against the configuration before allocating parameters.
    /// Cancellation is checked between samples, batches, and epochs.
    ///
    /// - Throws: ``ML5Error`` for invalid samples, incompatible targets, numerical instability,
    ///   or model construction failures; `CancellationError` when cancelled.
    public func train(
        _ samples: [DenseTrainingSample],
        configuration: DenseNetworkConfiguration
    ) async throws -> DenseTrainingResult {
        try Task.checkCancellation()
        guard samples.isEmpty == false else {
            throw ML5Error.invalidTrainingSamples
        }
        let prepared = try samples.map { try Self.prepare($0, configuration: configuration) }
        var layers = try Self.initialize(configuration: configuration)
        var indices = Array(prepared.indices)
        Self.shuffle(&indices, seed: configuration.seed)
        let validationCount = Int(Double(indices.count) * configuration.validationFraction)
        let trainingCount = indices.count - validationCount
        let trainingIndices = Array(indices[..<trainingCount])
        let validationIndices = Array(indices[trainingCount...])
        var history: [DenseEpochMetrics] = []
        history.reserveCapacity(configuration.epochs)
        var optimizerStep = 0

        for epochIndex in 0..<configuration.epochs {
            try Task.checkCancellation()
            var epochIndices = trainingIndices
            Self.shuffle(&epochIndices, seed: configuration.seed &+ UInt64(epochIndex) &+ 1)
            for batchStart in stride(
                from: 0,
                to: epochIndices.count,
                by: configuration.batchSize
            ) {
                try Task.checkCancellation()
                let batchEnd = min(batchStart + configuration.batchSize, epochIndices.count)
                let batch = epochIndices[batchStart..<batchEnd]
                var gradients = layers.map(LayerGradients.init(layer:))
                for sampleIndex in batch {
                    try Task.checkCancellation()
                    let pass = Self.forward(
                        prepared[sampleIndex].input, layers: layers, configuration: configuration)
                    try Self.accumulateGradients(
                        pass: pass,
                        target: prepared[sampleIndex].target,
                        layers: layers,
                        configuration: configuration,
                        gradients: &gradients
                    )
                }
                optimizerStep += 1
                try Self.apply(
                    gradients: gradients,
                    sampleCount: batch.count,
                    step: optimizerStep,
                    configuration: configuration,
                    layers: &layers
                )
            }

            let trainingLoss = try Self.meanLoss(
                indices: trainingIndices,
                samples: prepared,
                layers: layers,
                configuration: configuration
            )
            let validationLoss =
                validationIndices.isEmpty
                ? nil
                : try Self.meanLoss(
                    indices: validationIndices,
                    samples: prepared,
                    layers: layers,
                    configuration: configuration
                )
            history.append(
                DenseEpochMetrics(
                    epoch: epochIndex + 1,
                    trainingLoss: trainingLoss,
                    validationLoss: validationLoss
                )
            )
            await Task.yield()
        }
        try Task.checkCancellation()
        return DenseTrainingResult(
            model: try Self.model(configuration: configuration, layers: layers),
            history: history
        )
    }

    private static func prepare(
        _ sample: DenseTrainingSample,
        configuration: DenseNetworkConfiguration
    ) throws -> PreparedSample {
        guard sample.targets.count == configuration.outputNames.count else {
            throw ML5Error.invalidTrainingSample(
                reason:
                    "Expected \(configuration.outputNames.count) targets, but received \(sample.targets.count)."
            )
        }
        let input = try configuration.inputFeatures.map { name in
            guard let value = sample.features[name] else {
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
        switch configuration.loss {
        case .meanSquaredError:
            break
        case .categoricalCrossEntropy:
            guard
                sample.targets.allSatisfy({ (0...1).contains($0) }),
                abs(sample.targets.reduce(0, +) - 1) <= 1e-9
            else {
                throw ML5Error.invalidTrainingSample(
                    reason: "Categorical targets must form a probability distribution."
                )
            }
        case .binaryCrossEntropy:
            guard sample.targets.allSatisfy({ (0...1).contains($0) }) else {
                throw ML5Error.invalidTrainingSample(
                    reason: "Binary targets must be in the closed range from zero to one."
                )
            }
        }
        return PreparedSample(input: input, target: sample.targets)
    }

    private static func initialize(configuration: DenseNetworkConfiguration) throws
        -> [TrainableLayer]
    {
        let widths =
            configuration.hiddenLayers.map(\.neuronCount) + [configuration.outputNames.count]
        var inputCount = configuration.inputFeatures.count
        var generator = DenseSeededGenerator(state: configuration.seed)
        var layers: [TrainableLayer] = []
        layers.reserveCapacity(widths.count)
        for outputCount in widths {
            guard inputCount <= Int.max / outputCount else {
                throw ML5Error.invalidModel(reason: "Layer dimensions overflow their weight count.")
            }
            let weightCount = inputCount * outputCount
            let weights: [Double]
            switch configuration.weightInitialization {
            case .glorotUniform:
                let limit = sqrt(6 / Double(inputCount + outputCount))
                weights = (0..<weightCount).map { _ in generator.uniform(in: -limit...limit) }
            case .heNormal:
                let standardDeviation = sqrt(2 / Double(inputCount))
                weights = (0..<weightCount).map { _ in
                    generator.normal() * standardDeviation
                }
            case .zeros:
                weights = Array(repeating: 0, count: weightCount)
            }
            layers.append(
                TrainableLayer(
                    inputCount: inputCount,
                    outputCount: outputCount,
                    weights: weights,
                    biases: Array(repeating: 0, count: outputCount)
                )
            )
            inputCount = outputCount
        }
        return layers
    }

    private static func forward(
        _ input: [Double],
        layers: [TrainableLayer],
        configuration: DenseNetworkConfiguration
    ) -> ForwardPass {
        var activations = [input]
        var preactivations: [[Double]] = []
        preactivations.reserveCapacity(layers.count)
        for (index, layer) in layers.enumerated() {
            let values = layer.affine(activations[index])
            preactivations.append(values)
            let activation =
                index < configuration.hiddenLayers.count
                ? configuration.hiddenLayers[index].activation
                : configuration.outputActivation
            activations.append(DenseNetworkMath.activate(values, using: activation))
        }
        return ForwardPass(activations: activations, preactivations: preactivations)
    }

    private static func accumulateGradients(
        pass: ForwardPass,
        target: [Double],
        layers: [TrainableLayer],
        configuration: DenseNetworkConfiguration,
        gradients: inout [LayerGradients]
    ) throws {
        let prediction = pass.activations[pass.activations.count - 1]
        var delta: [Double]
        switch configuration.loss {
        case .meanSquaredError:
            delta = zip(prediction, target).map { 2 * ($0 - $1) / Double(target.count) }
            delta = zip(delta, prediction).enumerated().map { index, pair in
                pair.0
                    * DenseNetworkMath.derivative(
                        preactivation: pass.preactivations[pass.preactivations.count - 1][index],
                        activation: pair.1,
                        using: configuration.outputActivation
                    )
            }
        case .categoricalCrossEntropy, .binaryCrossEntropy:
            delta = zip(prediction, target).map { ($0 - $1) / Double(target.count) }
        }

        for layerIndex in layers.indices.reversed() {
            let previousActivation = pass.activations[layerIndex]
            for outputIndex in 0..<layers[layerIndex].outputCount {
                gradients[layerIndex].biases[outputIndex] += delta[outputIndex]
                let rowStart = outputIndex * layers[layerIndex].inputCount
                for inputIndex in 0..<layers[layerIndex].inputCount {
                    gradients[layerIndex].weights[rowStart + inputIndex] +=
                        delta[outputIndex] * previousActivation[inputIndex]
                }
            }

            if layerIndex > 0 {
                var previousDelta = Array(repeating: 0.0, count: layers[layerIndex].inputCount)
                for inputIndex in previousDelta.indices {
                    for outputIndex in 0..<layers[layerIndex].outputCount {
                        previousDelta[inputIndex] +=
                            layers[layerIndex].weights[
                                outputIndex * layers[layerIndex].inputCount + inputIndex
                            ] * delta[outputIndex]
                    }
                    let function = configuration.hiddenLayers[layerIndex - 1].activation
                    previousDelta[inputIndex] *= DenseNetworkMath.derivative(
                        preactivation: pass.preactivations[layerIndex - 1][inputIndex],
                        activation: pass.activations[layerIndex][inputIndex],
                        using: function
                    )
                }
                delta = previousDelta
            }
        }
    }

    private static func apply(
        gradients: [LayerGradients],
        sampleCount: Int,
        step: Int,
        configuration: DenseNetworkConfiguration,
        layers: inout [TrainableLayer]
    ) throws {
        let scale = 1 / Double(sampleCount)
        for layerIndex in layers.indices {
            switch configuration.optimizer.kind {
            case .stochasticGradientDescent:
                for index in layers[layerIndex].weights.indices {
                    let gradient = gradients[layerIndex].weights[index] * scale
                    layers[layerIndex].weightVelocity[index] =
                        configuration.optimizer.momentum
                        * layers[layerIndex].weightVelocity[index]
                        - configuration.learningRate * gradient
                    layers[layerIndex].weights[index] +=
                        layers[layerIndex].weightVelocity[index]
                }
                for index in layers[layerIndex].biases.indices {
                    let gradient = gradients[layerIndex].biases[index] * scale
                    layers[layerIndex].biasVelocity[index] =
                        configuration.optimizer.momentum * layers[layerIndex].biasVelocity[index]
                        - configuration.learningRate * gradient
                    layers[layerIndex].biases[index] += layers[layerIndex].biasVelocity[index]
                }
            case .adam:
                for index in layers[layerIndex].weights.indices {
                    let gradient = gradients[layerIndex].weights[index] * scale
                    layers[layerIndex].weightFirstMoment[index] =
                        configuration.optimizer.beta1
                        * layers[layerIndex].weightFirstMoment[index]
                        + (1 - configuration.optimizer.beta1) * gradient
                    layers[layerIndex].weightSecondMoment[index] =
                        configuration.optimizer.beta2
                        * layers[layerIndex].weightSecondMoment[index]
                        + (1 - configuration.optimizer.beta2) * gradient * gradient
                    layers[layerIndex].weights[index] -= Self.adamUpdate(
                        firstMoment: layers[layerIndex].weightFirstMoment[index],
                        secondMoment: layers[layerIndex].weightSecondMoment[index],
                        step: step,
                        configuration: configuration
                    )
                }
                for index in layers[layerIndex].biases.indices {
                    let gradient = gradients[layerIndex].biases[index] * scale
                    layers[layerIndex].biasFirstMoment[index] =
                        configuration.optimizer.beta1 * layers[layerIndex].biasFirstMoment[index]
                        + (1 - configuration.optimizer.beta1) * gradient
                    layers[layerIndex].biasSecondMoment[index] =
                        configuration.optimizer.beta2 * layers[layerIndex].biasSecondMoment[index]
                        + (1 - configuration.optimizer.beta2) * gradient * gradient
                    layers[layerIndex].biases[index] -= Self.adamUpdate(
                        firstMoment: layers[layerIndex].biasFirstMoment[index],
                        secondMoment: layers[layerIndex].biasSecondMoment[index],
                        step: step,
                        configuration: configuration
                    )
                }
            }
            guard
                layers[layerIndex].weights.allSatisfy(\.isFinite),
                layers[layerIndex].biases.allSatisfy(\.isFinite)
            else {
                throw ML5Error.invalidModel(
                    reason: "Training produced nonfinite model parameters."
                )
            }
        }
    }

    private static func adamUpdate(
        firstMoment: Double,
        secondMoment: Double,
        step: Int,
        configuration: DenseNetworkConfiguration
    ) -> Double {
        let correctedFirst =
            firstMoment / (1 - pow(configuration.optimizer.beta1, Double(step)))
        let correctedSecond =
            secondMoment / (1 - pow(configuration.optimizer.beta2, Double(step)))
        return configuration.learningRate * correctedFirst
            / (sqrt(correctedSecond) + configuration.optimizer.epsilon)
    }

    private static func meanLoss(
        indices: [Int],
        samples: [PreparedSample],
        layers: [TrainableLayer],
        configuration: DenseNetworkConfiguration
    ) throws -> Double {
        var total = 0.0
        for index in indices {
            try Task.checkCancellation()
            let pass = Self.forward(
                samples[index].input,
                layers: layers,
                configuration: configuration
            )
            let prediction = pass.activations[pass.activations.count - 1]
            total += Self.loss(
                prediction: prediction,
                target: samples[index].target,
                function: configuration.loss
            )
        }
        let result = total / Double(indices.count)
        guard result.isFinite else {
            throw ML5Error.invalidModel(reason: "Training produced nonfinite loss.")
        }
        return result
    }

    private static func loss(
        prediction: [Double],
        target: [Double],
        function: TrainingLoss
    ) -> Double {
        switch function {
        case .meanSquaredError:
            return zip(prediction, target).reduce(0) { $0 + ($1.0 - $1.1) * ($1.0 - $1.1) }
                / Double(target.count)
        case .categoricalCrossEntropy:
            return -zip(prediction, target).reduce(0) {
                $0 + $1.1 * log(max($1.0, Double.leastNonzeroMagnitude))
            }
        case .binaryCrossEntropy:
            return -zip(prediction, target).reduce(0) {
                let probability = min(max($1.0, Double.leastNonzeroMagnitude), 1 - Double.ulpOfOne)
                return $0 + $1.1 * log(probability) + (1 - $1.1) * log(1 - probability)
            } / Double(target.count)
        }
    }

    private static func model(
        configuration: DenseNetworkConfiguration,
        layers: [TrainableLayer]
    ) throws -> DenseNetworkModel {
        try DenseNetworkModel(
            configuration: configuration,
            layers: layers.map {
                try DenseLayerParameters(
                    inputCount: $0.inputCount,
                    outputCount: $0.outputCount,
                    weights: $0.weights,
                    biases: $0.biases
                )
            }
        )
    }

    private static func shuffle(_ values: inout [Int], seed: UInt64) {
        guard values.count > 1 else { return }
        var generator = DenseSeededGenerator(state: seed)
        for index in stride(from: values.count - 1, through: 1, by: -1) {
            let destination = Int(generator.next() % UInt64(index + 1))
            if destination != index {
                values.swapAt(index, destination)
            }
        }
    }
}

extension DenseNetworkMath {
    static func derivative(
        preactivation: Double,
        activation: Double,
        using function: ActivationFunction
    ) -> Double {
        switch function {
        case .linear:
            1
        case .rectifiedLinear:
            preactivation > 0 ? 1 : 0
        case .sigmoid:
            activation * (1 - activation)
        case .hyperbolicTangent:
            1 - activation * activation
        case .softmax:
            activation * (1 - activation)
        }
    }
}

private struct PreparedSample {
    let input: [Double]
    let target: [Double]
}

private struct ForwardPass {
    let activations: [[Double]]
    let preactivations: [[Double]]
}

private struct LayerGradients {
    var weights: [Double]
    var biases: [Double]

    init(layer: TrainableLayer) {
        weights = Array(repeating: 0, count: layer.weights.count)
        biases = Array(repeating: 0, count: layer.biases.count)
    }
}

private struct TrainableLayer {
    let inputCount: Int
    let outputCount: Int
    var weights: [Double]
    var biases: [Double]
    var weightVelocity: [Double]
    var biasVelocity: [Double]
    var weightFirstMoment: [Double]
    var biasFirstMoment: [Double]
    var weightSecondMoment: [Double]
    var biasSecondMoment: [Double]

    init(inputCount: Int, outputCount: Int, weights: [Double], biases: [Double]) {
        self.inputCount = inputCount
        self.outputCount = outputCount
        self.weights = weights
        self.biases = biases
        weightVelocity = Array(repeating: 0, count: weights.count)
        biasVelocity = Array(repeating: 0, count: biases.count)
        weightFirstMoment = Array(repeating: 0, count: weights.count)
        biasFirstMoment = Array(repeating: 0, count: biases.count)
        weightSecondMoment = Array(repeating: 0, count: weights.count)
        biasSecondMoment = Array(repeating: 0, count: biases.count)
    }

    func affine(_ input: [Double]) -> [Double] {
        (0..<outputCount).map { outputIndex in
            let rowStart = outputIndex * inputCount
            return (0..<inputCount).reduce(biases[outputIndex]) {
                $0 + weights[rowStart + $1] * input[$1]
            }
        }
    }
}

private struct DenseSeededGenerator: RandomNumberGenerator {
    var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }

    mutating func uniform(in range: ClosedRange<Double>) -> Double {
        let unit = (Double(next() >> 11) + 0.5) / 9_007_199_254_740_992
        return range.lowerBound + (range.upperBound - range.lowerBound) * unit
    }

    mutating func normal() -> Double {
        let first = (Double(next() >> 11) + 0.5) / 9_007_199_254_740_992
        let second = (Double(next() >> 11) + 0.5) / 9_007_199_254_740_992
        return sqrt(-2 * log(first)) * cos(2 * .pi * second)
    }
}
