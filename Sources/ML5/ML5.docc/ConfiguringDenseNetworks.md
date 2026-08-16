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

## Train with the CPU reference backend

``DenseCPUTrainer`` accepts ``DenseTrainingSample`` values containing ordered numeric
targets. It supports mini-batch SGD with momentum and Adam across mean-squared,
categorical-cross-entropy, and binary-cross-entropy objectives:

```swift
let samples = try observations.map { observation in
    try DenseTrainingSample(
        features: observation.features,
        targets: [observation.expectedValue]
    )
}
let result = try await DenseCPUTrainer().train(
    samples,
    configuration: configuration
)
```

Use ``DenseTrainingSample/classification(features:label:labels:)`` to create one-hot
targets from an ordered typed label list. The trainer validates target dimensions and
loss-specific probability domains before allocating parameters.

Initialization, validation splitting, and per-epoch shuffling are derived from the
configuration seed. Identical samples and configuration therefore produce identical
``DenseTrainingResult/model`` and ``DenseTrainingResult/history`` values. The CPU
backend is a readable numerical reference and a practical choice for small models;
larger workloads should select an accelerated backend when one is available.

## Observe, stop, and resume training

Both dense trainers accept ``DenseTrainingOptions`` and a serial asynchronous
``DenseTrainingProgressHandler``. Progress is delivered after an epoch becomes durable, so a
checkpoint supplied by ``DenseTrainingProgress/checkpoint`` contains the matching parameters,
optimizer moments, partitions, loss history, and next epoch:

```swift
let options = try DenseTrainingOptions(
    earlyStopping: DenseEarlyStoppingConfiguration(
        patience: 8,
        minimumImprovement: 0.0001
    ),
    checkpointInterval: 5
)
let result = try await DenseCPUTrainer().train(
    samples,
    configuration: configuration,
    options: options
) { update in
    if let checkpoint = update.checkpoint {
        try await checkpointStore.save(checkpoint)
    }
}
```

The callback is awaited before training continues. It can update an actor-isolated UI or throw
to stop the run; ordinary task cancellation is also checked between samples, batches, graph
executions, metrics passes, and epochs. ``DenseTrainingCheckpoint`` is `Codable` and binds its
schema to ``DenseTrainingCheckpoint/currentFormatVersion`` and its state to the ordered sample
content using ``DenseTrainingCheckpoint/sampleFingerprint``. Resume
with the same concrete trainer or let ``DenseTrainer/resume(_:samples:progress:)`` select the
backend recorded by ``DenseTrainingCheckpoint/backend``:

```swift
let checkpoint = try JSONDecoder().decode(
    DenseTrainingCheckpoint.self,
    from: savedData
)
let result = try await DenseTrainer().resume(checkpoint, samples: samples)
```

Resume rejects reordered or changed samples and refuses CPU/Metal or Metal-device changes.
This preserves optimizer and numerical semantics instead of treating a saved model as a full
training checkpoint. ``DenseTrainingResult/stopReason`` distinguishes the configured epoch limit
from early stopping. When requested, an early-stopped result restores the best model while its
checkpoint retains the exact final optimizer state.

## Train with Metal Performance Shaders Graph

``DenseMPSGraphTrainer`` executes batched forward evaluation and automatic
differentiation through MPSGraph on a Metal command queue. Select the system device or
provide an explicit `MTLDevice`:

```swift
let trainer = try DenseMPSGraphTrainer()
let result = try await trainer.train(samples, configuration: configuration)
```

The actor exposes `deviceName` for diagnostics. Construction reports
``ML5Error/trainingAcceleratorUnavailable(reason:)`` when Metal or a command queue is
unavailable; it never silently changes the requested backend. MPSGraph computes
float32 batch gradients, while parameter updates, validation metrics, cancellation
boundaries, and the final double-precision ``DenseNetworkModel`` use the shared
reference semantics.

Use ``DenseCPUTrainer`` when exact cross-machine seed reproducibility is required.
GPU scheduling and float32 arithmetic can introduce small backend-dependent numeric
differences even though ordering, shapes, objectives, and update equations match.

For policy-based selection, use ``DenseTrainer`` with a
``DenseTrainingExecutionPolicy``. CPU fallback is explicit and applies only when Metal cannot be
constructed before a run starts; a graph or numerical failure never causes the operation to be
replayed silently on another backend:

```swift
let trainer = DenseTrainer(
    executionPolicy: DenseTrainingExecutionPolicy(
        preference: .automatic,
        fallback: .cpu
    )
)
let result = try await trainer.train(samples, configuration: configuration)
```
