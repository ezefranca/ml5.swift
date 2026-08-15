# Structured Model Data

Represent model inputs without leaking Core ML reference types into application state.

## Define an ordered schema

A ``FeatureSchema`` records model order separately from dictionary storage and checks
each value before prediction:

```swift
let imageShape = try TensorShape([1, 3, 224, 224])
let schema = try FeatureSchema([
    try FeatureField(
        name: "pixels",
        kind: .tensor,
        tensorShape: imageShape
    ),
    try FeatureField(
        name: "temperature",
        kind: .number,
        defaultValue: .number(1)
    ),
    try FeatureField(
        name: "note",
        kind: .string,
        isRequired: false
    ),
])

let tensor = try Tensor(
    shape: imageShape,
    values: Array(repeating: 0, count: imageShape.elementCount)
)
let supplied = try FeatureVector(["pixels": .tensor(tensor)])
let resolved = try schema.resolve(supplied, missing: .useDefaults)
```

The default policies reject missing and unknown fields. Choose
``MissingFeaturePolicy/useDefaults`` to insert declared defaults and omit fields
marked optional. Choose ``UnknownFeaturePolicy/preserve`` only when the downstream
model intentionally accepts additional inputs.

## Choose a value representation

Use ``FeatureValue/array(_:)`` for nonempty one-dimensional numeric data and
``FeatureValue/tensor(_:)`` when dimensions are semantically significant. Core ML
represents both with `MLMultiArray`; ML5 decodes multi-array outputs as tensors so
their shape is never discarded.

``FeatureValue/dictionary(_:)`` accepts nonempty, trimmed string keys and finite
numeric values. ``FeatureSequence`` preserves either strings or signed 64-bit
integers, including empty sequences when the model permits them.

## Bridge image pixels

``ML5Image`` owns an immutable copy of grayscale, RGBA, or BGRA bytes. Its initializer
validates positive dimensions, row stride, integer overflow, and exact storage size.
Create one from a `CVPixelBuffer` when capturing native media, or call
``ML5Image/makePixelBuffer()`` when an adapter needs Core Video storage.

Core Video commonly supports BGRA rather than RGBA allocation. ML5 therefore
channel-swizzles RGBA input into BGRA while preserving its visible color. Conversions
always copy storage, which makes image values safe to serialize and send across actor
boundaries.

## Persist values safely

Schemas, tensors, images, feature vectors, and model outputs conform to `Codable`.
Decoding re-runs all public validation, so corrupt or hand-edited data fails instead
of constructing an invalid value. Treat the encoded form as model data and avoid
storing secrets or user imagery without the same protection applied elsewhere in the
application.
