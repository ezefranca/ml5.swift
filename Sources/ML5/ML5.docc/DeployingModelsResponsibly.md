# Deploying Models Responsibly

Separate model creation from on-device inference, disclose data use, and test resource
behavior on every supported device class.

## Keep Create ML in a macOS tool

ML5's inference, dense reference training, and Vision APIs run on the package's iOS and
macOS baseline. Apple's Create ML framework is a separate macOS authoring workflow. Do
not import Create ML into an iOS application target or make it an ML5 runtime dependency.

A production model workflow can use these stages:

1. Prepare consented, licensed training and validation data outside the application.
2. Train and evaluate with Create ML or another audited tool on macOS.
3. Export `.mlmodel` or `.mlpackage`, record the tool and dataset provenance, and review
   the model metadata and license.
4. Compile during the application build, or let ``ML5ModelCache`` compile a trusted
   digest-pinned source.
5. Re-run app-level accuracy, fairness, privacy, memory, and energy gates on physical
   devices before release.

``DenseCPUTrainer`` and ``DenseMPSGraphTrainer`` serve small, explicitly configured
dense networks. They are not replacements for Create ML's task-specific data ingestion,
augmentation, evaluation, or model families.

## Declare data access at the application boundary

ML5 does not open the camera, microphone, Photos library, contacts, or network merely by
being linked. ``ML5Image`` owns bytes already supplied by the caller. The optional remote
model source performs an HTTPS download only when the app asks its cache to resolve that
source.

The containing app remains responsible for:

- purpose strings and authorization before accessing camera, microphone, or Photos data;
- a privacy manifest that accurately declares the app's and all dependencies' accessed
  APIs, tracking behavior, and collected data;
- consent, retention, deletion, redaction, logging, and backup policies for inputs,
  labels, predictions, checkpoints, and downloaded models;
- a network policy for model hosts, certificate validation, redirects, telemetry, and
  offline behavior; and
- model-card and license review before redistributing third-party weights.

Never place secrets or personal data in ``ML5ModelMetadata``. Cache manifests and dense
archives are plain serializable files, not encrypted vaults. Use platform data protection
and an application-owned protected location when persisted artifacts are sensitive.

## Bound memory deliberately

An ``ML5Image`` owns its entire pixel payload and Vision creates a pixel buffer for each
request. A 4-channel image requires at least `width * height * 4` bytes for each live
copy, before model activations and framework caches. Downsample to the model's useful
resolution, avoid retaining duplicate frames, serialize live camera inference, and place
an explicit limit on remote-model cache entries.

Dense training also retains parameters, gradients, optimizer state, batches, validation
samples, and optional checkpoints. Measure peak resident memory with representative
topologies and stop training before the operating system has to terminate the app.

## Design for heat and energy

Continuous camera inference and training can create sustained CPU, GPU, and Neural Engine
load. Treat a per-frame model request as an opt-in performance decision, not a default.
Throttle inference independently of display refresh, discard superseded frames, suspend
when the scene is inactive, and react to serious or critical thermal state by reducing
frequency or stopping nonessential work.

Low Power Mode and background execution should also reduce or pause optional training and
inference. Validate energy use with Instruments on physical devices; simulator timing and
compute placement do not represent shipping hardware.

## Treat compute units as preferences

``CoreMLModelConfiguration`` exposes `.all`, `.cpuOnly`, `.cpuAndGPU`, and
`.cpuAndNeuralEngine`. Selecting a set permits Core ML to use those processors; it does
not promise that a particular layer or request executes on the Neural Engine. Model
structure, operating-system version, hardware, current resources, and Core ML policy all
affect placement.

Prefer `.all` unless the app has a measured reason to restrict execution. Test the exact
model on the oldest and newest supported device classes. Provide a functional CPU path
for correctness tests and devices where the preferred accelerator cannot serve the
operation, and report ``ML5Error/trainingAcceleratorUnavailable(reason:)`` when an
explicit training policy forbids fallback.

## Release checklist

Before shipping a model version, verify its digest, model card, attribution, license,
input orientation, crop policy, preprocessing, output interpretation, supported devices,
peak memory, latency distribution, sustained thermal behavior, cancellation, offline
behavior, corrupted-cache recovery, and deletion path. Version these decisions beside
the model so a weight update cannot silently change the application contract.

## See Also

- <doc:LoadingAndCachingModels>
- <doc:RunningVisionImageModels>
- <doc:PersistingAndExportingDenseModels>
- [Create ML](https://developer.apple.com/documentation/createml)
- [Core ML model configuration](https://developer.apple.com/documentation/coreml/mlmodelconfiguration)
