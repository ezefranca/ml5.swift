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

## Run an immutable model

``DenseNetworkModel`` pairs a configuration with ``DenseLayerParameters``. Parameters
use documented output-major row order and are validated for dimensions, storage
counts, and finite values. Model construction also verifies every connection against
the configured hidden and output widths.

```swift
let model = try DenseNetworkModel(
    configuration: configuration,
    layers: restoredParameters
)
let output = try await model.predict(features)
let snapshot = try await model.makeInferenceSnapshot()
```

Dense models conform to ``ModelInferenceSnapshotProviding``. Because their parameter
arrays are immutable values, snapshot prediction requires no actor hop or lock. Model
and parameter decoding repeats every shape and finiteness check before accepting an
archive.
