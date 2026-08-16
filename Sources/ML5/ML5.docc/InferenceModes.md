# Inference Modes

Choose actor-isolated async, ordered batch, or immutable synchronous prediction.

## Predict asynchronously

The asynchronous `predict(_:)` method is the general-purpose path. The network
actor serializes access to its backend, checks cooperative cancellation before and
after evaluation, and decodes framework-independent output through its typed task.

Use the batch overload to preserve input order while reducing backend overhead:

```swift
let results = try await network.predict([firstFeatures, secondFeatures])
```

Custom ``ModelPredicting`` backends inherit a sequential, cancellation-aware batch
implementation. ``CoreMLModelPredictor`` uses Core ML's native batch provider when
available and verifies that output count exactly matches input count.

## Capture a draw-loop snapshot

When every actor hop matters, ask a snapshot-capable backend for an immutable typed
operation before entering the render loop:

```swift
let snapshot = try await network.makeInferenceSnapshot()

func draw() throws {
    let prediction = try snapshot.predict(currentFeatures)
    // Update the simulation from prediction without awaiting the network actor.
}
```

``NeuralNetworkInferenceSnapshot`` stores a task value and a
``ModelInferenceSnapshot``. Prediction is synchronous, retains typed decoding, and
supports scalar or ordered batch input. It does not start detached work or block on
the network actor.

A custom backend opts in through ``ModelInferenceSnapshotProviding``. Networks using
other backends throw
``UnsupportedOperation/synchronousInferenceSnapshot`` explicitly. Snapshot creation
does not make a slow model fast; measure inference time against the frame budget and
prefer async prediction when work can span frames.
