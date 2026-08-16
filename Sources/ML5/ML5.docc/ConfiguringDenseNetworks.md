# Configuring Dense Networks

Describe a reproducible dense-network architecture before selecting a training
backend.

## Declare ordered inputs and outputs

``DenseNetworkConfiguration`` treats feature and output order as model data. Names
must be nonempty and unique, and every loop-control value is checked before training:

```swift
let configuration = try DenseNetworkConfiguration(
    inputFeatures: ["x", "y"],
    outputNames: ["classA", "classB"],
    hiddenLayers: [
        try DenseLayerConfiguration(neuronCount: 16),
        try DenseLayerConfiguration(
            neuronCount: 8,
            activation: .hyperbolicTangent
        ),
    ],
    outputActivation: .softmax,
    weightInitialization: .glorotUniform,
    loss: .categoricalCrossEntropy,
    optimizer: .adam,
    learningRate: 0.001,
    batchSize: 32,
    epochs: 100,
    validationFraction: 0.2,
    seed: 42
)
```

Hidden layers support linear, rectified-linear, sigmoid, and hyperbolic-tangent
activation. Softmax is an output-only operation because its values depend on every
neuron in the layer.

## Match loss and output semantics

The configuration rejects incompatible combinations:

- Mean squared error accepts linear, rectified-linear, sigmoid, or
  hyperbolic-tangent output.
- Categorical cross entropy requires at least two softmax outputs.
- Binary cross entropy requires sigmoid output and supports one or more independent
  binary targets.

``OptimizerConfiguration`` supports stochastic gradient descent with momentum and
Adam. It validates momentum, moment-decay values, and the Adam denominator
stabilizer even when the selected optimizer does not consume every field. This keeps
persisted configurations portable when an optimizer is changed.

All configuration types are `Sendable`, `Hashable`, and `Codable`. Decoding invokes
the same validating initializers, so edited or corrupt configuration files cannot
bypass architecture, loss, or numeric invariants.
