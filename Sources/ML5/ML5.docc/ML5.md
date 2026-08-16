# ``ML5``

Typed, on-device machine-learning foundations for Apple platforms.

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

The Core ML boundary accepts validated scalar, numeric-array, dictionary, tensor,
sequence, and image values without exposing non-`Sendable` framework objects. Use a
``FeatureSchema`` when model inputs need a stable order, exact tensor dimensions,
defaults, or explicit missing and unknown-field policies.

## Topics

### Tasks and values

- ``ClassificationTask``
- ``RegressionTask``
- ``FeatureVector``
- ``FeatureValue``
- ``FeatureValueKind``
- ``FeatureSchema``
- ``FeatureField``
- ``Tensor``
- ``TensorShape``
- ``FeatureSequence``
- ``ML5Image``
- ``ML5ImagePixelFormat``
- ``ClassificationSample``
- ``RegressionSample``

### Data guides

- <doc:StructuredModelData>
- <doc:ManagingDatasets>
- <doc:PreprocessingData>
- <doc:InferenceModes>
- <doc:EvaluatingModels>

### Datasets

- ``ML5Dataset``
- ``DatasetSampleID``
- ``DatasetEntry``
- ``DatasetSnapshot``
- ``DatasetSplitConfiguration``
- ``DatasetSplit``

### Preprocessing

- ``FeaturePreprocessingPipeline``
- ``FeatureNormalizationRule``
- ``FittedFeatureNormalization``
- ``NormalizationStrategy``
- ``NormalizationRange``
- ``NumericFeatureStatistics``

### Prediction

- ``NeuralNetwork``
- ``CoreMLModelPredictor``
- ``ModelPredicting``
- ``CoreMLModelConfiguration``
- ``ModelInferenceSnapshot``
- ``NeuralNetworkInferenceSnapshot``
- ``ModelInferenceSnapshotProviding``
- ``ClassificationPrediction``
- ``RankedClassificationPrediction``
- ``RankedClassificationTask``
- ``RankedClassificationConfiguration``
- ``ClassificationScoreInterpretation``
- ``TemperatureScaling``
- ``RegressionPrediction``
- ``RegressionVectorPrediction``
- ``RegressionVectorTask``
- ``RegressionVectorConfiguration``

### Evaluation

- ``ClassificationEvaluation``
- ``ClassificationLabelMetrics``
- ``RegressionEvaluation``
- ``RegressionComponentMetrics``

### Extensibility

- ``NeuralNetworkTrainingAdapter``
- ``UnsupportedOperation``
