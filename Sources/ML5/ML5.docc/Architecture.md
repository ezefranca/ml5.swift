# Architecture

## Overview

ML5 separates task semantics from model execution:

1. ``FeatureVector`` and ``ModelOutput`` are validated, `Sendable` value types.
2. A ``ModelPredicting`` backend transforms a feature vector into raw output.
3. A ``NeuralNetworkTask`` decodes that output as a classification or regression
   prediction.
4. ``NeuralNetwork`` owns the task and backend behind an actor boundary.

## Core ML boundary

``CoreMLModelPredictor`` owns `MLModel` inside an actor. The model does not leave that
actor, and no detached task captures it. Input and output conversion happens at the
actor boundary, where ``FeatureValue`` cases become `MLFeatureValue` instances only
for the duration of the Core ML call. Numeric arrays and tensors use dense
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

## Training

Core ML model loading does not provide general-purpose, arbitrary-model on-device
training. Accordingly, ``NeuralNetwork/train(_:)`` throws
``UnsupportedOperation/onDeviceTraining`` rather than pretending to train.

``NeuralNetworkTrainingAdapter`` is the extension point for a separate Create ML
integration. An adapter can train a task-specific model and return a
``ModelPredicting`` backend, preserving the same value-safe execution boundary.
