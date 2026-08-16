# Architecture

## Overview

ML5 separates task semantics from model execution:

1. ``FeatureVector`` and ``ModelOutput`` are validated, `Sendable` value types.
2. A ``ModelPredicting`` backend transforms a feature vector into raw output.
3. A ``NeuralNetworkTask`` decodes that output as a classification or regression
   prediction.
4. ``NeuralNetwork`` owns the task and backend behind an actor boundary.

## Core ML boundary

``CoreMLModelPredictor`` owns normal async and batch execution inside an actor, and no
detached task captures its model operation. Input and output conversion happens at
the execution boundary, where ``FeatureValue`` cases become `MLFeatureValue`
instances only for the duration of the Core ML call. Numeric arrays and tensors use dense
double-precision `MLMultiArray` storage, dictionaries retain string keys, sequences
retain their homogeneous element type, and images cross the boundary as copied
`CVPixelBuffer` storage. RGBA input is channel-swizzled to Core Video's supported
BGRA layout.

All serializable boundary types decode through the same validating initializers as
ordinary callers. A malformed archive therefore cannot bypass shape, finiteness,
field-name, image-stride, schema-uniqueness, or default-value invariants.

Prediction APIs are asynchronous and cooperatively check cancellation before and
after Core ML evaluation. Evaluation already underway may complete first, but a
cancelled caller will not receive decoded output.

Snapshot creation is an explicit exception to actor-isolated execution.
``ModelInferenceSnapshotProviding`` vends an immutable synchronous closure backed by
the loaded model. This value contains no mutable ML5 state and performs conversion
and prediction directly on the caller. It exists for latency-sensitive loops; callers
remain responsible for choosing an appropriate thread. Core ML snapshot calls are
serialized with a private lock; custom backend implementers must make their snapshot
closures safe for the concurrency promised by `Sendable`.

## Dataset isolation

``ML5Dataset`` owns mutable sample order and identifier allocation inside an actor.
Its ``DatasetSnapshot`` and ``DatasetSplit`` results are immutable values, so training
adapters can consume them without sharing actor state. Seeded shuffling uses a fixed
SplitMix64 sequence and an implementation-owned Fisher-Yates pass for reproducibility
across supported platforms.

``FeaturePreprocessingPipeline`` follows the same separation: fitting is an explicit
operation over training values, while normalization and denormalization use an
immutable snapshot of population statistics. A fitted pipeline contains no mutable
state and can be shared across actors or persisted beside a model checkpoint.

## Prediction semantics and evaluation

Task decoding owns model-output semantics. ``RankedClassificationTask`` converts a
string-keyed score dictionary into a complete, deterministic distribution using
normalization or temperature-scaled stable softmax. ``RegressionVectorTask`` maps
independently named numeric outputs into an explicit order. Neither operation depends
on Core ML, so injected, trained, and loaded backends share identical behavior.

Evaluation remains a pure value operation. ``ClassificationEvaluation`` retains the
typed expected labels and rankings needed for top-k accuracy while publishing
confusion and macro summaries. ``RegressionEvaluation`` computes aggregate and
per-component error metrics; undefined R² for constant targets is represented by
`nil` rather than a sentinel or nonfinite number.

## Training

Core ML model loading does not provide general-purpose, arbitrary-model on-device
training. Accordingly, ``NeuralNetwork/train(_:)`` throws
``UnsupportedOperation/onDeviceTraining`` rather than pretending to train.

``NeuralNetworkTrainingAdapter`` is the extension point for a separate Create ML
integration. An adapter can train a task-specific model and return a
``ModelPredicting`` backend, preserving the same value-safe execution boundary.

ML5-owned dense networks are trainable independently of Core ML model loading.
``DenseCPUTrainer`` is the deterministic reference implementation for initialization,
forward evaluation, backpropagation, SGD/Adam updates, loss calculation, and validation
history. It returns an immutable ``DenseNetworkModel`` that can be passed directly to
``NeuralNetwork/init(task:predictor:)``. Accelerated trainers must preserve the same
configuration, output ordering, and model representation.

``DenseMPSGraphTrainer`` is the Apple-accelerated implementation. It builds float32
batch graphs for affine layers, activations, configured loss, and automatic
differentiation, then executes them on an explicitly owned Metal command queue.
Optimizer updates and metric calculation share the reference code, preventing the
CPU and Metal paths from drifting into different checkpoint formats or loss
definitions. ``DenseTrainingCheckpoint`` captures parameters, SGD velocity or Adam
moments, optimizer step, deterministic partitions, early-stopping state, and ordered
sample identity. Resume requires the recorded backend and, for Metal, the same device
name.

``DenseTrainer`` adds policy-based selection through ``DenseTrainingExecutionPolicy``.
Its fallback is explicit and is considered only when Metal construction fails before
the first update. Runtime graph, cancellation, and numerical failures remain visible
to the caller and are never replayed on CPU.

## Persistence and deployment

``DenseModelArchive`` persists the validated model and ``ML5ModelMetadata`` as
deterministic JSON. A format version makes incompatible schema changes explicit, while
a SHA-256 digest covers the model and metadata payload so accidental edits and partial
transfers are rejected during loading. The archive is a durable ML5 training artifact;
it retains double-precision parameters and can be used for further ML5 inference or
export.

Core ML deployment is a separate, explicit operation. ML5 writes Apple's version 4
neural-network protobuf representation with float32 parameters, exact-rank multi-array
mapping, and an N-dimensional softmax. Core ML compilation then validates the native
artifact. This self-contained representation supports the complete ML5 dense-layer
surface without requiring a Python conversion tool or an external weight bundle.
See <doc:PersistingAndExportingDenseModels> for ownership and ordering details.

## Neuroevolution values

``DenseBrain`` wraps the same immutable ``DenseNetworkModel`` representation used by
training and export. Synchronous prediction is therefore a pure value operation without
an actor hop, lock, or shared mutable parameter storage. ``DenseBrain/copied()`` retains
value semantics; subsequent mutation or crossover builds fresh parameter arrays and
cannot change the parent.

Mutation and crossover consume a private, stable SplitMix64 sequence from an explicit
seed. ``DenseBrainPopulation`` persists the sequence state alongside a homogeneous
generation so decoding and applying the same policy produces the same next population.
Fitness evaluation, selection, elitism, and population sizing remain application-owned:
they depend on the simulation rather than on the neural-network representation. See
<doc:EvolvingDenseBrains> for the complete lifecycle.
