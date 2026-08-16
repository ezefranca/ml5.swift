# Evaluating Classification and Regression

Turn raw model scores into deterministic typed rankings and measure predictions
without importing Core ML.

## Rank classification results

Use ``RankedClassificationTask`` when a model emits a numeric dictionary whose keys
are raw labels. Probability-like weights are normalized automatically:

```swift
let task = RankedClassificationTask<String>(
    configuration: RankedClassificationConfiguration(scoresOutput: "classProbability")
)
let ranking = try task.decode(modelOutput)

let winner = ranking.best
let alternatives = try ranking.top(3)
```

The result always contains a complete distribution. Its confidence values sum to
one and are ordered from highest to lowest. Equal confidence uses the label's stable
raw representation as a deterministic tie breaker.

If a model emits logits, supply a validated ``TemperatureScaling`` value. A
temperature below one sharpens the result; a temperature above one softens it.
Stable softmax subtraction prevents large finite logits from overflowing:

```swift
let task = RankedClassificationTask<MyLabel>(
    configuration: RankedClassificationConfiguration(
        scoresOutput: "logits",
        interpretation: .logits(try TemperatureScaling(temperature: 1.4))
    )
)
```

Temperature should be selected using held-out validation data. It changes reported
confidence, not which logit is greatest.

## Decode vector regression

``RegressionVectorTask`` preserves the output order declared by
``RegressionVectorConfiguration``. Each component accepts a model number or integer:

```swift
let task = RegressionVectorTask(
    configuration: try RegressionVectorConfiguration(
        valueOutputs: ["positionX", "positionY", "speed"]
    )
)
let prediction = try task.decode(modelOutput)
```

The configuration and ``RegressionVectorPrediction`` are validated, `Sendable`, and
`Codable`, making the output order explicit across checkpoints and process boundaries.

## Evaluate held-out data

``ClassificationEvaluation`` reports top-one and top-k accuracy, typed confusion
counts, and per-label precision, recall, F1, and support. Macro metrics give each
observed label equal weight, including labels present only in expected or predicted
values.

```swift
let metrics = try ClassificationEvaluation(
    expected: validationLabels,
    predictions: validationRankings
)
print(metrics.accuracy)
print(try metrics.topKAccuracy(3))
```

``RegressionEvaluation`` accepts scalar or vector predictions and reports mean
absolute error, mean squared error, root mean squared error, and R² both in aggregate
and per component. R² is `nil` for a component whose expected values are constant,
because its total variance is zero.
