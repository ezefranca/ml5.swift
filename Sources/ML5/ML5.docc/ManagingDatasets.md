# Managing Datasets

Accumulate samples safely and create reproducible training partitions.

## Add and remove samples

``ML5Dataset`` is an actor, so multiple tasks can add data without racing its order
or identifier generator. Each insertion returns a stable ``DatasetSampleID``:

```swift
let dataset = ML5Dataset<RegressionSample>()
let sample = try RegressionSample(
    features: FeatureVector(["x": .number(0.5)]),
    target: 1
)

let id = try await dataset.add(sample)
let stored = await dataset.sample(for: id)
let removed = await dataset.remove(id)
```

Batch insertion is atomic. If its identifiers would overflow, the operation throws
without storing a partial batch. Removing a sample does not reuse its identifier,
which keeps logs and checkpoints unambiguous.

## Shuffle and split reproducibly

Call ``ML5Dataset/shuffle(seed:)`` to change dataset order with ML5's stable
SplitMix64-based shuffle. The same seed and starting order produce the same result.

``ML5Dataset/split(using:seed:)`` partitions a copy and leaves stored order unchanged:

```swift
let configuration = try DatasetSplitConfiguration(
    validationFraction: 0.15,
    testFraction: 0.15
)
let split = await dataset.split(using: configuration, seed: 2026)
```

Validation and test counts round down. Training receives the remainder so every
sample appears exactly once. Omitting the seed preserves current order; omitting the
configuration uses ``DatasetSplitConfiguration/standard`` (80% training and 20%
validation).

## Checkpoint dataset state

Use ``ML5Dataset/snapshot()`` to capture order, stable identifiers, and the next
reserved identifier. ``DatasetSnapshot`` conditionally conforms to `Codable` when
the sample type does, and decoding revalidates uniqueness and identifier bounds.

```swift
let snapshot = try await dataset.snapshot()
let data = try JSONEncoder().encode(snapshot)
let restoredSnapshot = try JSONDecoder().decode(
    DatasetSnapshot<RegressionSample>.self,
    from: data
)
let restored = ML5Dataset(snapshot: restoredSnapshot)
```

``ClassificationSample`` serializes labels through
``ClassificationLabel/ml5RawValue`` and reconstructs them with
``ClassificationLabel/init(ml5RawValue:)``. This keeps dataset files tied to the
model's stable label text instead of an enum's implementation-specific encoding.
