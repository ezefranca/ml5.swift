# Preprocessing Data

Fit reversible numerical transforms from training data and reuse them at inference.

## Fit only on training data

Build rules for the numeric features a model consumes, then fit after creating a
dataset split. This avoids leaking validation or test information into the model:

```swift
let rules = [
    FeatureNormalizationRule(
        feature: "position",
        strategy: .minMax(
            try NormalizationRange(lowerBound: -1, upperBound: 1)
        )
    ),
    FeatureNormalizationRule(
        feature: "velocity",
        strategy: .standardScore
    ),
]

let pipeline = try FeaturePreprocessingPipeline.fit(
    samples: trainingFeatures,
    rules: rules
)
let normalized = try pipeline.normalize(input)
```

``NormalizationStrategy/standardScore`` uses the population mean and standard
deviation. ``NormalizationStrategy/minMax(_:)`` maps the fitted extrema into a
validated output interval. Constant standard-score features become zero; constant
min-max features become the interval midpoint. Both cases denormalize to the fitted
constant.

## Supported feature values

Pipelines preserve number, numeric-array, numeric-dictionary, and tensor structure.
A statistic is fitted across all scalar elements for each named feature. Dictionary
keys are sorted during fitting so their iteration order cannot change floating-point
results. Tensor shapes and dictionary keys remain unchanged during transformation.

Integer, string, Boolean, sequence, and image features are intentionally rejected.
Convert categories separately and choose an explicit pixel preprocessing adapter for
images rather than silently applying an unsuitable scalar rule.

## Reverse and persist transforms

Call ``FeaturePreprocessingPipeline/denormalize(_:)`` to map data back to its original
scale. This is useful for interpreting regression output and debugging model inputs.
Unconfigured features pass through unchanged in either direction.

The fitted pipeline, strategies, intervals, and ``NumericFeatureStatistics`` are all
immutable, `Sendable`, `Hashable`, and `Codable`. Decoding revalidates numerical and
uniqueness invariants. Save the pipeline beside a model checkpoint and apply the same
artifact at inference instead of refitting from live input.
