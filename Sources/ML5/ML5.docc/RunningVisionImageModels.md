# Running Vision Image Models

Classify images, extract model-defined features, or create native Vision feature prints
without allowing framework objects to cross a concurrency boundary.

## Choose the matching adapter

Use ``VisionCoreMLImageModel`` for a Core ML model that declares an image input. Its
result method must match the model type:

- ``VisionCoreMLImageModel/classify(_:orientation:cropAndScale:maximumResults:)``
  accepts classifier models and returns ``VisionClassification`` values.
- ``VisionCoreMLImageModel/extractFeatures(_:orientation:cropAndScale:)`` accepts a
  general model whose outputs Vision represents as Core ML feature-value observations.

Use ``VisionImageFeatureExtractor`` when the app needs Apple's built-in image embedding
rather than a separately distributed Core ML model.

The adapters are actors. Core ML and Vision objects remain isolated inside them, while
inputs and results use `Sendable`, serializable ML5 values.

## Classify an image

Load a trusted resource through ``ML5ModelCache`` and preserve its model card and
integrity digest:

```swift
let model = try await VisionCoreMLImageModel.load(
    from: source,
    using: modelCache,
    configuration: CoreMLModelConfiguration(computeUnits: .all)
)

let predictions = try await model.classify(
    image,
    orientation: .right,
    cropAndScale: .centerCrop,
    maximumResults: 5
)
```

Results sort by descending score, with the identifier as a deterministic tie breaker.
Vision forwards model scores, so ML5 rejects nonfinite values but does not normalize or
clamp them. Apply a model-specific calibration step before describing a score as a
probability.

``VisionImageOrientation`` uses EXIF orientation semantics. Supply the orientation of
the stored pixels rather than rotating bytes first. ``VisionImageCropAndScale`` makes
the aspect-ratio decision explicit; changing it can materially change classifications.

## Extract model-defined features

Feature-extractor Core ML models can return tensors, scalars, dictionaries, sequences,
or images through ``VisionCoreMLFeature``:

```swift
let features = try await model.extractFeatures(
    image,
    orientation: .up,
    cropAndScale: .scaleFit
)

for feature in features {
    print(feature.name.rawValue, feature.value.kind)
}
```

Outputs sort by their declared model name. If a classifier is sent to `extractFeatures`,
or a general feature model is sent to `classify`, the call fails with
``ML5Error/unsupportedVisionResult(reason:)`` instead of silently discarding data.

## Compare native feature prints

``VisionImageFeatureExtractor`` wraps `VNGenerateImageFeaturePrintRequest`:

```swift
let extractor = VisionImageFeatureExtractor(
    configuration: VisionFeaturePrintConfiguration(
        revision: .revision2,
        cropAndScale: .centerCrop
    )
)

let first = try await extractor.extract(firstImage)
let second = try await extractor.extract(secondImage)
let distance = try first.distance(to: second)
```

Smaller Euclidean distances mean more similar images. Persist the
``VisionFeaturePrint/revision`` and ``VisionFeaturePrint/elementType`` with the values.
Only prints with matching revision, storage type, and element count are comparable.

## Handle cancellation and failures

The adapters check cooperative task cancellation before pixel conversion, before the
framework request, and before publishing a result. Vision's synchronous request itself
cannot be interrupted once submitted. Avoid launching obsolete work, cancel queued
tasks promptly, and bound image dimensions before creating ``ML5Image`` values.

Invalid options, incompatible observations, and framework failures use distinct
``ML5Error`` cases. Keep the localized description for diagnostics while presenting
model-appropriate recovery in the app.

## See Also

- <doc:LoadingAndCachingModels>
- <doc:StructuredModelData>
- <doc:DeployingModelsResponsibly>
- [VNCoreMLRequest](https://developer.apple.com/documentation/vision/vncoremlrequest)
- [VNGenerateImageFeaturePrintRequest](https://developer.apple.com/documentation/vision/vngenerateimagefeatureprintrequest)
