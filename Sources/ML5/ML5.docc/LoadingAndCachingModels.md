# Loading and Caching Models

Attach a model card and SHA-256 digest to every external Core ML asset, then resolve it
through an actor-isolated compiled-model cache.

## Describe trusted bytes

``ML5ModelSource`` requires an ``ML5ModelDigest`` and ``ML5ModelMetadata`` for every
location. This keeps version, author, license, purpose, and provenance beside the exact
bytes an application intends to execute:

```swift
let digest = try ML5ModelDigest(
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
let metadata = try ML5ModelMetadata(
    name: "Agent Classifier",
    version: "2.1.0",
    author: "Example Team",
    license: "MIT",
    source: URL(string: "https://example.com/models/agent-classifier")
)
let source = try ML5ModelSource(
    fileURL: modelURL,
    integrityDigest: digest,
    metadata: metadata
)
```

Create a digest during a trusted build or release process with
``ML5ModelDigest/sha256(contentsOf:)``. A regular file is hashed by content. A directory
such as `.mlmodelc` or `.mlpackage` is traversed recursively in sorted relative-path
order; relative names, lengths, and bytes all affect the digest. Symbolic links are
rejected so a verified tree cannot redirect loading outside its root.

SHA-256 detects corruption and binds a source to a published checksum. It does not by
itself identify who published that checksum. Ship expected digests in signed application
code or retrieve them through another authenticated, version-pinned channel.

## Resolve a bundled resource

Use ``ML5ModelSource/bundledResource(named:withExtension:subdirectory:in:integrityDigest:metadata:)``
for a model copied into an app or Swift package bundle:

```swift
let source = try ML5ModelSource.bundledResource(
    named: "AgentClassifier",
    withExtension: "mlmodelc",
    in: .main,
    integrityDigest: digest,
    metadata: metadata
)
```

The source supports uncompiled `.mlmodel`, `.mlpackage`, and compiled `.mlmodelc`
resources. The first two are compiled through Core ML and cached. A compiled directory
is verified and loaded in place.

## Download only authenticated models

Remote sources require HTTPS, a `.mlmodel` file name, and a digest known before the
request:

```swift
let source = try ML5ModelSource(
    remoteURL: URL(string: "https://cdn.example.com/agent-v2.mlmodel")!,
    integrityDigest: digest,
    metadata: metadata
)
```

ML5 validates a successful HTTP status, checks cancellation, hashes the complete
download, and refuses a mismatch before handing bytes to the Core ML compiler. Remote
`.mlpackage`, `.mlmodelc`, and arbitrary archives are deliberately excluded because a
multi-file transport needs its own authenticated archive and extraction policy.

## Own a bounded cache

Choose an application-owned cache directory and a positive entry limit:

```swift
let cache = try ML5ModelCache(
    configuration: ML5ModelCacheConfiguration(
        directory: applicationSupportURL.appending(path: "ML5Models"),
        maximumEntryCount: 8
    )
)

let predictor = try await CoreMLModelPredictor.load(
    from: source,
    using: cache,
    configuration: CoreMLModelConfiguration(computeUnits: .all)
)
```

``ML5ModelCache`` is an actor, so downloads, compilation, manifest updates, and eviction
cannot race inside one cache instance. Its directory contains one entry per source
digest. Each manifest stores the source digest, a separately computed digest for the
compiled directory, the model card, byte count, schema version, and last verified access.

A cache hit rehashes the compiled directory. Missing, malformed, or changed entries are
removed and rebuilt from the declared source. Least-recently accessed entries are evicted
after insertion when the configured count is exceeded.

## Inspect and clear storage

Present storage diagnostics with ``ML5ModelCache/entries()``. Entries include the source
and compiled digests, model metadata, allocated byte count, and access date:

```swift
let entries = try await cache.entries()
for entry in entries {
    print(entry.metadata.name, entry.byteCount)
}

_ = try await cache.removeModel(withSourceDigest: digest)
try await cache.removeAll()
```

Removal is explicit; ML5 never touches files outside the configured directory. Use a
cache or Application Support location according to whether your app can redownload the
model. Do not place user-created training archives in this compiled-resource cache.

## See Also

- <doc:PersistingAndExportingDenseModels>
- <doc:Architecture>
- <doc:InferenceModes>
