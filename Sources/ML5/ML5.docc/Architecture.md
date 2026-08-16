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
