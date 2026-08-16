# Your First ML5 Model

Train a small native dense model, then perform typed inference.

```swift
import ML5

let samples = try (-2...2).map { value in
    try DenseTrainingSample(
        features: FeatureVector(["x": .number(Double(value))]),
        targets: [(2 * Double(value)) + 1]
    )
}
let configuration = try DenseNetworkConfiguration(
    inputFeatures: ["x"],
    outputNames: ["y"],
    learningRate: 0.05,
    batchSize: samples.count,
    epochs: 40,
    validationFraction: 0,
    seed: 7
)
let result = try await DenseCPUTrainer().train(samples, configuration: configuration)
let output = try await result.model.predict(
    FeatureVector(["x": .number(3)])
)
```

Use ``DenseTrainer`` with an explicit ``DenseTrainingExecutionPolicy`` to select
MPSGraph/Metal acceleration and its fallback policy. A failure after accelerated
training begins is surfaced and never replayed silently on CPU.

Loaded Core ML models use ``NeuralNetwork`` and typed tasks. Construction,
prediction, training, downloads, and model compilation can fail; catch ``ML5Error``
at the feature boundary and check task cancellation before updating UI state.

Continue with <doc:ManagingDatasets>, <doc:ConfiguringDenseNetworks>, and
<doc:ML5Compatibility>.
