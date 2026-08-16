# ml5.js Compatibility and Native Differences

Use approachable model workflows without importing browser semantics.

| ml5.js concept | ML5 equivalent | Native difference |
| --- | --- | --- |
| neural network options | ``DenseNetworkConfiguration`` | Typed, validated, immutable configuration. |
| `addData`, `normalizeData` | ``ML5Dataset``, ``FeaturePreprocessingPipeline`` | Dataset mutation is actor-isolated; fitted preprocessing is a value. |
| `train` | ``DenseTrainer`` | Async progress, cancellation, checkpoints, explicit CPU/Metal policy. |
| `classify`, `predict` | ``NeuralNetwork`` and tasks | Typed results and throwing async calls replace callbacks. |
| save/load | ``DenseModelArchive``, ``ML5ModelCache`` | Versioned integrity checks and model cards are mandatory. |
| feature extraction | ``VisionImageFeatureExtractor`` | Vision/Core ML replace browser image tensors. |
| neuroevolution helpers | ``DenseBrain`` | Copies, mutation, and crossover have deterministic value semantics. |

See the authoritative [ml5.js documentation](https://docs.ml5js.org/) for
upstream browser behavior. ML5 does not execute TensorFlow.js, accept JavaScript
model objects, emulate callback timing, or claim that arbitrary Core ML models
can be retrained. Native dense networks are trainable; loaded Core ML models are
inference-only unless an explicit application-supplied adapter supports training.

Core ML and Vision operations are actor-isolated. Synchronous inference snapshots
are immutable and explicitly opt out of the actor hop for latency-sensitive loops.
MPSGraph availability depends on a usable Metal device. Remote models require HTTPS,
a SHA-256 digest, provenance, ownership, and license metadata.
