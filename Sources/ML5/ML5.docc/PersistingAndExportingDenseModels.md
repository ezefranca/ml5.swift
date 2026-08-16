# Persisting and Exporting Dense Models

Preserve an ML5 training artifact, record its provenance, and deploy the same network
through Core ML.

## Archive a validated model

``ML5ModelMetadata`` keeps the information needed to identify and redistribute a
trained model. Name and version are required; author, license, summary, source URL,
and application-specific key-value metadata are optional and validated.

```swift
let metadata = try ML5ModelMetadata(
    name: "NatureClassifier",
    version: "1.0.0",
    author: "Example Team",
    license: "MIT",
    summary: "Classifies three simulated agent behaviors.",
    source: URL(string: "https://example.com/models/nature-classifier")
)

let archive = try trainedModel.archived(metadata: metadata)
try archive.write(to: archiveURL)
```

``DenseModelArchive`` writes deterministic, sorted-key JSON atomically. Its
``DenseModelArchive/formatVersion`` identifies the schema and its
``DenseModelArchive/integrityDigest`` is a SHA-256 digest of the model and metadata
payload. Loading repeats every dense-model validation and refuses an unsupported
format or changed payload:

```swift
let restored = try DenseModelArchive.load(contentsOf: archiveURL)
let prediction = try await restored.model.predict(features)
```

The digest detects accidental corruption and modification; it is not a signature.
Authenticate models from an untrusted distribution channel with a platform code
signature or a separately verified cryptographic signature.

## Export to Core ML

Create a ``DenseCoreMLExportConfiguration`` with distinct Core ML feature names. The
exporter stores all current ML5 dense activations, topology, float32 weights, and model
metadata in Apple's self-contained neural-network model format:

```swift
let configuration = try DenseCoreMLExportConfiguration(
    inputName: "features",
    outputName: "predictions",
    metadata: metadata
)

try restored.model.writeCoreMLModel(
    to: modelURL,
    configuration: configuration
)
```

Core ML exposes one float32 multi-array input ordered exactly like
``DenseNetworkConfiguration/inputFeatures`` and one float32 multi-array output ordered
like ``DenseNetworkConfiguration/outputNames``. Preserve those configured names when
building arrays at the boundary:

```swift
let input = try FeatureVector([
    "features": .array([positionX, positionY, velocityX, velocityY])
])
```

ML5 trains and archives parameters as `Double`. Export converts them to the native
neural-network format's float32 storage and rejects values that cannot be represented
as finite floats. Keep the ML5 archive when exact continuation or checkpoint resume is
required; use the Core ML artifact for Apple-framework deployment.

## Compile with Apple's runtime

``DenseNetworkModel/compileCoreMLModel(configuration:)`` serializes a temporary
`.mlmodel`, asks Core ML to compile it, removes the uncompiled temporary source, and
returns the compiled `.mlmodelc` directory. The caller owns that returned directory:

```swift
let compiledURL = try restored.model.compileCoreMLModel(
    configuration: configuration
)
defer { try? FileManager.default.removeItem(at: compiledURL) }

let predictor = try CoreMLModelPredictor(contentsOf: compiledURL)
let result = try await predictor.predict(input)
```

For an app target, writing `.mlmodel` is usually the better integration path because
Xcode compiles it as a build resource. Runtime compilation is useful for imported or
newly trained models. Compilation and prediction remain local to the Apple device.

## Handle failures deliberately

Archive validation reports ``ML5Error/invalidModelArchive(reason:)``. File access uses
``ML5Error/modelPersistenceFailed(path:message:)``, and serialization or native Core ML
compilation uses ``ML5Error/coreMLExportFailed(reason:)``. Keep these cases distinct in
diagnostics: invalid model data should not be retried as though it were a temporary file
or accelerator problem.

## See Also

- <doc:ConfiguringDenseNetworks>
- <doc:Architecture>
- <doc:InferenceModes>
