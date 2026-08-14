# ``ML5``

Typed, on-device classification and regression for Apple platforms.

ML5 is an independent native Swift implementation inspired by the approachable
conceptual model of [ml5.js](https://ml5js.org/). It provides typed task decoding,
actor-isolated Core ML prediction, and value-safe interfaces designed for Swift
concurrency.

## Overview

Create a task that describes which Core ML outputs carry its result, load a compiled
model, and await a typed prediction:

```swift
let task = ClassificationTask<String>(
    configuration: try ClassificationConfiguration(
        labelOutput: "label",
        confidenceOutput: "confidence"
    )
)
let network = try NeuralNetwork(
    task: task,
    modelAt: modelURL
)
let result = try await network.predict(
    try FeatureVector(["feature": .number(0.5)])
)
```

The first release intentionally supports scalar feature values and scalar model
outputs. Its explicit boundaries leave room for future image, tensor, and Create ML
training adapters without exposing non-Sendable Core ML objects.

## Topics

### Tasks and values

- ``ClassificationTask``
- ``RegressionTask``
- ``FeatureVector``
- ``FeatureValue``
- ``ClassificationSample``
- ``RegressionSample``

### Prediction

- ``NeuralNetwork``
- ``CoreMLModelPredictor``
- ``ModelPredicting``
- ``CoreMLModelConfiguration``

### Extensibility

- ``NeuralNetworkTrainingAdapter``
- ``UnsupportedOperation``
