@preconcurrency import CoreML
import CryptoKit
import Foundation

/// A validated lowercase SHA-256 digest for model resource integrity checks.
public struct ML5ModelDigest: RawRepresentable, Sendable, Hashable, Codable,
    CustomStringConvertible
{
    /// Lowercase 64-character hexadecimal SHA-256 value.
    public let rawValue: String

    /// The hexadecimal digest for diagnostics and manifests.
    public var description: String { rawValue }

    /// Creates and validates a lowercase SHA-256 digest.
    ///
    /// - Throws: ``ML5Error/invalidModelDigest(_:)`` when the value is not exactly 64 lowercase
    ///   hexadecimal characters.
    public init(_ rawValue: String) throws {
        guard
            rawValue.count == 64,
            rawValue.utf8.allSatisfy({
                ($0 >= 0x30 && $0 <= 0x39) || ($0 >= 0x61 && $0 <= 0x66)
            })
        else {
            throw ML5Error.invalidModelDigest(rawValue)
        }
        self.rawValue = rawValue
    }

    /// Creates a digest for `RawRepresentable` conformance.
    ///
    /// - Precondition: `rawValue` is a valid lowercase SHA-256 hexadecimal value.
    public init(rawValue: String) {
        precondition((try? Self(rawValue)) != nil)
        self.rawValue = rawValue
    }

    /// Hashes in-memory model bytes.
    public static func sha256(data: Data) -> Self {
        Self(uncheckedBytes: SHA256.hash(data: data))
    }

    /// Hashes a model file or a directory tree in stable relative-path order.
    ///
    /// Directory hashing rejects symbolic links and incorporates each relative path and file
    /// length before its bytes, so names and boundaries cannot collide.
    ///
    /// - Throws: ``ML5Error/modelResourceFailed(location:message:)`` when the URL is absent,
    ///   unsupported, contains a symbolic link, or cannot be read.
    public static func sha256(contentsOf url: URL) throws -> Self {
        do {
            let values = try url.resourceValues(forKeys: [
                .isRegularFileKey,
                .isDirectoryKey,
                .isSymbolicLinkKey,
            ])
            guard values.isSymbolicLink != true else {
                throw ML5Error.invalidModelSource(
                    reason: "Symbolic links are not accepted for integrity-checked models."
                )
            }
            if values.isRegularFile == true {
                return sha256(data: try Data(contentsOf: url, options: .mappedIfSafe))
            }
            guard values.isDirectory == true else {
                throw ML5Error.invalidModelSource(
                    reason: "A model resource must be a regular file or directory."
                )
            }
            return try directoryDigest(url)
        } catch let error as ML5Error {
            throw error
        } catch {
            throw ML5Error.modelResourceFailed(
                location: url.path,
                message: error.localizedDescription
            )
        }
    }

    /// Decodes and validates a hexadecimal digest.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        try self.init(container.decode(String.self))
    }

    /// Encodes the lowercase hexadecimal value.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }

    private static func directoryDigest(_ directory: URL) throws -> Self {
        var files: [(path: String, url: URL)] = []
        try collectRegularFiles(in: directory, relativePath: "", result: &files)

        var hasher = SHA256()
        for file in files.sorted(by: { $0.path < $1.path }) {
            let pathData = Data(file.path.utf8)
            update(&hasher, length: pathData.count)
            hasher.update(data: pathData)
            let fileData = try Data(contentsOf: file.url, options: .mappedIfSafe)
            update(&hasher, length: fileData.count)
            hasher.update(data: fileData)
        }
        return Self(uncheckedBytes: hasher.finalize())
    }

    private static func collectRegularFiles(
        in directory: URL,
        relativePath: String,
        result: inout [(path: String, url: URL)]
    ) throws {
        let children = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [
                .isRegularFileKey,
                .isDirectoryKey,
                .isSymbolicLinkKey,
            ]
        )
        for child in children {
            let values = try child.resourceValues(forKeys: [
                .isRegularFileKey,
                .isDirectoryKey,
                .isSymbolicLinkKey,
            ])
            guard values.isSymbolicLink != true else {
                throw ML5Error.invalidModelSource(
                    reason: "Symbolic links are not accepted for integrity-checked models."
                )
            }
            let childPath =
                relativePath.isEmpty
                ? child.lastPathComponent
                : "\(relativePath)/\(child.lastPathComponent)"
            if values.isDirectory == true {
                try collectRegularFiles(in: child, relativePath: childPath, result: &result)
            } else if values.isRegularFile == true {
                result.append((childPath, child))
            }
        }
    }

    private static func update(_ hasher: inout SHA256, length: Int) {
        var value = UInt64(length).bigEndian
        withUnsafeBytes(of: &value) { hasher.update(data: Data($0)) }
    }

    private init<Digest: Sequence>(uncheckedBytes digest: Digest) where Digest.Element == UInt8 {
        rawValue = digest.map { String(format: "%02x", $0) }.joined()
    }
}

/// A trusted location, digest, and model card for a compiled or compilable Core ML resource.
public struct ML5ModelSource: Sendable, Hashable, Codable {
    /// Storage location of a model resource.
    public enum Location: Sendable, Hashable, Codable {
        /// Local `.mlmodel`, `.mlpackage`, or compiled `.mlmodelc` URL.
        case file(URL)
        /// HTTPS `.mlmodel` URL downloaded and compiled into an ``ML5ModelCache``.
        case remote(URL)
    }

    /// Validated file or remote location.
    public let location: Location
    /// Required SHA-256 digest of source bytes or the compiled source directory.
    public let integrityDigest: ML5ModelDigest
    /// Ownership, license, version, and provenance retained with cache entries.
    public let metadata: ML5ModelMetadata

    /// Creates an integrity-checked local model source.
    ///
    /// - Throws: ``ML5Error/invalidModelSource(reason:)`` unless `fileURL` is a file URL.
    public init(
        fileURL: URL,
        integrityDigest: ML5ModelDigest,
        metadata: ML5ModelMetadata
    ) throws {
        try self.init(
            location: .file(fileURL),
            integrityDigest: integrityDigest,
            metadata: metadata
        )
    }

    /// Creates an integrity-checked remote model source.
    ///
    /// Only HTTPS `.mlmodel` files are accepted because remote model packages and compiled
    /// directories require an authenticated archive format outside this API.
    ///
    /// - Throws: ``ML5Error/invalidModelSource(reason:)`` for a non-HTTPS or non-`.mlmodel` URL.
    public init(
        remoteURL: URL,
        integrityDigest: ML5ModelDigest,
        metadata: ML5ModelMetadata
    ) throws {
        try self.init(
            location: .remote(remoteURL),
            integrityDigest: integrityDigest,
            metadata: metadata
        )
    }

    /// Resolves an integrity-checked resource from a bundle and creates a local source.
    ///
    /// - Throws: ``ML5Error/invalidModelSource(reason:)`` when the resource is absent.
    public static func bundledResource(
        named name: String,
        withExtension extensionName: String,
        subdirectory: String? = nil,
        in bundle: Bundle = .main,
        integrityDigest: ML5ModelDigest,
        metadata: ML5ModelMetadata
    ) throws -> Self {
        guard
            let url = bundle.url(
                forResource: name,
                withExtension: extensionName,
                subdirectory: subdirectory
            )
        else {
            throw ML5Error.invalidModelSource(
                reason: "Bundle resource \(name).\(extensionName) was not found."
            )
        }
        return try Self(
            fileURL: url,
            integrityDigest: integrityDigest,
            metadata: metadata
        )
    }

    /// Decodes and revalidates a model source.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            location: container.decode(Location.self, forKey: .location),
            integrityDigest: container.decode(ML5ModelDigest.self, forKey: .integrityDigest),
            metadata: container.decode(ML5ModelMetadata.self, forKey: .metadata)
        )
    }

    /// Encodes location, digest, and model card.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(location, forKey: .location)
        try container.encode(integrityDigest, forKey: .integrityDigest)
        try container.encode(metadata, forKey: .metadata)
    }

    private init(
        location: Location,
        integrityDigest: ML5ModelDigest,
        metadata: ML5ModelMetadata
    ) throws {
        switch location {
        case let .file(url):
            guard url.isFileURL else {
                throw ML5Error.invalidModelSource(
                    reason: "Local model URLs must use the file scheme."
                )
            }
        case let .remote(url):
            guard url.scheme?.lowercased() == "https", url.lastPathComponent.isEmpty == false else {
                throw ML5Error.invalidModelSource(
                    reason: "Remote model URLs must use HTTPS and include a file name."
                )
            }
            guard url.pathExtension.lowercased() == "mlmodel" else {
                throw ML5Error.invalidModelSource(
                    reason: "Remote model URLs must identify a .mlmodel file."
                )
            }
        }
        self.location = location
        self.integrityDigest = integrityDigest
        self.metadata = metadata
    }

    private enum CodingKeys: String, CodingKey {
        case location
        case integrityDigest
        case metadata
    }
}

/// Disk limits and location for compiled Core ML model caching.
public struct ML5ModelCacheConfiguration: Sendable, Hashable {
    /// Directory owned by the cache actor.
    public let directory: URL
    /// Maximum number of compiled entries retained after each insertion.
    public let maximumEntryCount: Int

    /// Creates a cache configuration.
    ///
    /// - Throws: ``ML5Error/invalidModelSource(reason:)`` unless the directory is a file URL and
    ///   maximum entry count is positive.
    public init(directory: URL, maximumEntryCount: Int = 8) throws {
        guard directory.isFileURL else {
            throw ML5Error.invalidModelSource(reason: "A model cache must use a file URL.")
        }
        guard maximumEntryCount > 0 else {
            throw ML5Error.invalidModelSource(
                reason: "A model cache must retain at least one entry."
            )
        }
        self.directory = directory
        self.maximumEntryCount = maximumEntryCount
    }
}

/// A prepared compiled model and the verified provenance used to obtain it.
public struct ML5CachedModel: Sendable, Hashable {
    /// Compiled `.mlmodelc` directory ready for `MLModel` loading.
    public let modelURL: URL
    /// Verified digest of the original file bytes or compiled source directory.
    public let sourceDigest: ML5ModelDigest
    /// Verified digest of the returned compiled directory.
    public let compiledDigest: ML5ModelDigest
    /// Model card supplied with the source.
    public let metadata: ML5ModelMetadata
    /// Whether an already compiled and verified cache entry satisfied the request.
    public let isCacheHit: Bool
}

/// Read-only cache inventory for diagnostics, storage UI, and explicit eviction.
public struct ML5ModelCacheEntry: Sendable, Hashable, Codable {
    /// Digest used as the cache key.
    public let sourceDigest: ML5ModelDigest
    /// Digest protecting the compiled `.mlmodelc` directory.
    public let compiledDigest: ML5ModelDigest
    /// Model card stored when the entry was compiled.
    public let metadata: ML5ModelMetadata
    /// Most recent successful verified access.
    public let lastAccessDate: Date
    /// Recursive allocated byte count of the compiled directory.
    public let byteCount: UInt64
}

/// An actor that downloads, verifies, compiles, inventories, and evicts Core ML models.
public actor ML5ModelCache {
    private let configuration: ML5ModelCacheConfiguration
    private let operations: ML5ModelCacheOperations

    /// Creates the configured cache directory when needed.
    ///
    /// - Throws: ``ML5Error/modelCacheFailed(path:message:)`` when the directory cannot be
    ///   created.
    public init(configuration: ML5ModelCacheConfiguration) throws {
        try self.init(configuration: configuration, operations: .system)
    }

    init(
        configuration: ML5ModelCacheConfiguration,
        operations: ML5ModelCacheOperations
    ) throws {
        self.configuration = configuration
        self.operations = operations
        do {
            try FileManager.default.createDirectory(
                at: configuration.directory,
                withIntermediateDirectories: true
            )
        } catch {
            throw ML5Error.modelCacheFailed(
                path: configuration.directory.path,
                message: error.localizedDescription
            )
        }
    }

    /// Resolves, verifies, and when necessary compiles a model source.
    ///
    /// Valid verified cache entries avoid file access or a remote request. Invalid entries are
    /// removed and rebuilt from the declared source.
    public func model(for source: ML5ModelSource) async throws -> ML5CachedModel {
        try Task.checkCancellation()
        if let cached = try cachedModel(for: source) {
            return cached
        }

        switch source.location {
        case let .file(url):
            let actual = try ML5ModelDigest.sha256(contentsOf: url)
            try Self.verify(expected: source.integrityDigest, actual: actual)
            if url.pathExtension.lowercased() == "mlmodelc" {
                return ML5CachedModel(
                    modelURL: url,
                    sourceDigest: actual,
                    compiledDigest: actual,
                    metadata: source.metadata,
                    isCacheHit: false
                )
            }
            guard ["mlmodel", "mlpackage"].contains(url.pathExtension.lowercased()) else {
                throw ML5Error.invalidModelSource(
                    reason: "Local models must use .mlmodel, .mlpackage, or .mlmodelc."
                )
            }
            return try compileAndCache(sourceURL: url, source: source)
        case let .remote(url):
            let download: ML5ModelDownload
            do {
                download = try await operations.download(url)
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                throw ML5Error.modelResourceFailed(
                    location: url.absoluteString,
                    message: error.localizedDescription
                )
            }
            try Task.checkCancellation()
            guard let statusCode = download.statusCode, (200..<300).contains(statusCode) else {
                throw ML5Error.modelResourceFailed(
                    location: url.absoluteString,
                    message:
                        "The server returned HTTP status \(download.statusCode.map(String.init) ?? "unknown")."
                )
            }
            let actual = ML5ModelDigest.sha256(data: download.data)
            try Self.verify(expected: source.integrityDigest, actual: actual)
            return try withTemporarySource(data: download.data, source: source)
        }
    }

    /// Returns verified entries ordered from most to least recently accessed.
    public func entries() throws -> [ML5ModelCacheEntry] {
        try manifests().map(\.entry).sorted {
            if $0.lastAccessDate == $1.lastAccessDate {
                return $0.sourceDigest.rawValue < $1.sourceDigest.rawValue
            }
            return $0.lastAccessDate > $1.lastAccessDate
        }
    }

    /// Removes one compiled entry, returning whether it existed.
    public func removeModel(withSourceDigest digest: ML5ModelDigest) throws -> Bool {
        let url = entryURL(for: digest)
        guard FileManager.default.fileExists(atPath: url.path) else { return false }
        do {
            try FileManager.default.removeItem(at: url)
            return true
        } catch {
            throw ML5Error.modelCacheFailed(path: url.path, message: error.localizedDescription)
        }
    }

    /// Removes all entries and recreates the owned cache directory.
    public func removeAll() throws {
        do {
            if FileManager.default.fileExists(atPath: configuration.directory.path) {
                try FileManager.default.removeItem(at: configuration.directory)
            }
            try FileManager.default.createDirectory(
                at: configuration.directory,
                withIntermediateDirectories: true
            )
        } catch {
            throw ML5Error.modelCacheFailed(
                path: configuration.directory.path,
                message: error.localizedDescription
            )
        }
    }

    private func cachedModel(for source: ML5ModelSource) throws -> ML5CachedModel? {
        let directory = entryURL(for: source.integrityDigest)
        guard FileManager.default.fileExists(atPath: directory.path) else { return nil }
        do {
            var manifest = try readManifest(at: directory)
            guard manifest.sourceDigest == source.integrityDigest else {
                try FileManager.default.removeItem(at: directory)
                return nil
            }
            let modelURL = directory.appendingPathComponent("model.mlmodelc", isDirectory: true)
            let actual = try ML5ModelDigest.sha256(contentsOf: modelURL)
            guard actual == manifest.compiledDigest else {
                try FileManager.default.removeItem(at: directory)
                return nil
            }
            manifest = try manifest.accessed(at: operations.now())
            try writeManifest(manifest, to: directory)
            return ML5CachedModel(
                modelURL: modelURL,
                sourceDigest: source.integrityDigest,
                compiledDigest: actual,
                metadata: source.metadata,
                isCacheHit: true
            )
        } catch {
            try? FileManager.default.removeItem(at: directory)
            return nil
        }
    }

    private func withTemporarySource(
        data: Data,
        source: ML5ModelSource
    ) throws -> ML5CachedModel {
        let directory = configuration.directory.appendingPathComponent(
            ".download-\(UUID().uuidString)",
            isDirectory: true
        )
        let modelURL = directory.appendingPathComponent("source.mlmodel")
        do {
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            try data.write(to: modelURL, options: .atomic)
            defer { try? FileManager.default.removeItem(at: directory) }
            return try compileAndCache(sourceURL: modelURL, source: source)
        } catch let error as ML5Error {
            throw error
        } catch {
            throw ML5Error.modelResourceFailed(
                location: sourceLocation(source),
                message: error.localizedDescription
            )
        }
    }

    private func compileAndCache(
        sourceURL: URL,
        source: ML5ModelSource
    ) throws -> ML5CachedModel {
        let compiledURL: URL
        do {
            compiledURL = try operations.compile(sourceURL)
        } catch {
            throw ML5Error.modelResourceFailed(
                location: sourceLocation(source),
                message: error.localizedDescription
            )
        }
        defer { try? FileManager.default.removeItem(at: compiledURL) }

        let target = entryURL(for: source.integrityDigest)
        let staging = configuration.directory.appendingPathComponent(
            ".staging-\(UUID().uuidString)",
            isDirectory: true
        )
        do {
            try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
            let stagedModel = staging.appendingPathComponent("model.mlmodelc", isDirectory: true)
            try FileManager.default.copyItem(at: compiledURL, to: stagedModel)
            let compiledDigest = try ML5ModelDigest.sha256(contentsOf: stagedModel)
            let manifest = try ML5ModelCacheManifest(
                sourceDigest: source.integrityDigest,
                compiledDigest: compiledDigest,
                metadata: source.metadata,
                lastAccessDate: operations.now(),
                byteCount: Self.byteCount(of: stagedModel)
            )
            try writeManifest(manifest, to: staging)
            try FileManager.default.moveItem(at: staging, to: target)
            try enforceLimit(preserving: source.integrityDigest)
            return ML5CachedModel(
                modelURL: target.appendingPathComponent("model.mlmodelc", isDirectory: true),
                sourceDigest: source.integrityDigest,
                compiledDigest: compiledDigest,
                metadata: source.metadata,
                isCacheHit: false
            )
        } catch let error as ML5Error {
            try? FileManager.default.removeItem(at: staging)
            throw error
        } catch {
            try? FileManager.default.removeItem(at: staging)
            throw ML5Error.modelCacheFailed(path: target.path, message: error.localizedDescription)
        }
    }

    private func manifests() throws -> [(url: URL, entry: ML5ModelCacheEntry)] {
        do {
            let urls = try FileManager.default.contentsOfDirectory(
                at: configuration.directory,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            )
            return urls.compactMap { url in
                guard
                    (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true,
                    let manifest = try? readManifest(at: url)
                else { return nil }
                return (url, manifest.entry)
            }
        } catch {
            throw ML5Error.modelCacheFailed(
                path: configuration.directory.path,
                message: error.localizedDescription
            )
        }
    }

    private func enforceLimit(preserving digest: ML5ModelDigest) throws {
        var values = try manifests()
        guard values.count > configuration.maximumEntryCount else { return }
        values.removeAll { $0.entry.sourceDigest == digest }
        values.sort {
            if $0.entry.lastAccessDate == $1.entry.lastAccessDate {
                return $0.entry.sourceDigest.rawValue < $1.entry.sourceDigest.rawValue
            }
            return $0.entry.lastAccessDate < $1.entry.lastAccessDate
        }
        let removalCount = max(0, values.count + 1 - configuration.maximumEntryCount)
        for value in values.prefix(removalCount) {
            try FileManager.default.removeItem(at: value.url)
        }
    }

    private func entryURL(for digest: ML5ModelDigest) -> URL {
        configuration.directory.appendingPathComponent(digest.rawValue, isDirectory: true)
    }

    private func readManifest(at directory: URL) throws -> ML5ModelCacheManifest {
        let url = directory.appendingPathComponent("manifest.json")
        do {
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            return try decoder.decode(
                ML5ModelCacheManifest.self,
                from: Data(contentsOf: url, options: .mappedIfSafe)
            )
        } catch let error as ML5Error {
            throw error
        } catch {
            throw ML5Error.modelCacheFailed(path: url.path, message: error.localizedDescription)
        }
    }

    private func writeManifest(_ manifest: ML5ModelCacheManifest, to directory: URL) throws {
        let url = directory.appendingPathComponent("manifest.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(manifest).write(to: url, options: .atomic)
    }

    private static func verify(expected: ML5ModelDigest, actual: ML5ModelDigest) throws {
        guard expected == actual else {
            throw ML5Error.modelIntegrityMismatch(
                expected: expected.rawValue,
                actual: actual.rawValue
            )
        }
    }

    private static func byteCount(of directory: URL) throws -> UInt64 {
        let children = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey]
        )
        var total: UInt64 = 0
        for url in children {
            let values = try url.resourceValues(forKeys: [
                .isRegularFileKey,
                .isDirectoryKey,
            ])
            if values.isDirectory == true {
                total += try byteCount(of: url)
            } else if values.isRegularFile == true {
                total += UInt64(try Data(contentsOf: url, options: .mappedIfSafe).count)
            }
        }
        return total
    }

    private func sourceLocation(_ source: ML5ModelSource) -> String {
        switch source.location {
        case let .file(url):
            url.path
        case let .remote(url):
            url.absoluteString
        }
    }
}

extension CoreMLModelPredictor {
    /// Resolves an integrity-checked model through a cache and loads it for prediction.
    public static func load(
        from source: ML5ModelSource,
        using cache: ML5ModelCache,
        configuration: CoreMLModelConfiguration = .init()
    ) async throws -> CoreMLModelPredictor {
        let cached = try await cache.model(for: source)
        return try CoreMLModelPredictor(
            contentsOf: cached.modelURL,
            configuration: configuration
        )
    }
}

private struct ML5ModelCacheManifest: Sendable, Hashable, Codable {
    static let currentFormatVersion = 1

    let formatVersion: Int
    let sourceDigest: ML5ModelDigest
    let compiledDigest: ML5ModelDigest
    let metadata: ML5ModelMetadata
    let lastAccessDate: Date
    let byteCount: UInt64

    var entry: ML5ModelCacheEntry {
        ML5ModelCacheEntry(
            sourceDigest: sourceDigest,
            compiledDigest: compiledDigest,
            metadata: metadata,
            lastAccessDate: lastAccessDate,
            byteCount: byteCount
        )
    }

    init(
        sourceDigest: ML5ModelDigest,
        compiledDigest: ML5ModelDigest,
        metadata: ML5ModelMetadata,
        lastAccessDate: Date,
        byteCount: UInt64
    ) throws {
        try self.init(
            formatVersion: Self.currentFormatVersion,
            sourceDigest: sourceDigest,
            compiledDigest: compiledDigest,
            metadata: metadata,
            lastAccessDate: lastAccessDate,
            byteCount: byteCount
        )
    }

    init(
        formatVersion: Int,
        sourceDigest: ML5ModelDigest,
        compiledDigest: ML5ModelDigest,
        metadata: ML5ModelMetadata,
        lastAccessDate: Date,
        byteCount: UInt64
    ) throws {
        guard formatVersion == Self.currentFormatVersion else {
            throw ML5Error.invalidModelSource(
                reason: "Unsupported model cache manifest version \(formatVersion)."
            )
        }
        guard byteCount > 0 else {
            throw ML5Error.invalidModelSource(
                reason: "A cached compiled model must contain at least one byte."
            )
        }
        self.formatVersion = formatVersion
        self.sourceDigest = sourceDigest
        self.compiledDigest = compiledDigest
        self.metadata = metadata
        self.lastAccessDate = lastAccessDate
        self.byteCount = byteCount
    }

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            formatVersion: container.decode(Int.self, forKey: .formatVersion),
            sourceDigest: container.decode(ML5ModelDigest.self, forKey: .sourceDigest),
            compiledDigest: container.decode(ML5ModelDigest.self, forKey: .compiledDigest),
            metadata: container.decode(ML5ModelMetadata.self, forKey: .metadata),
            lastAccessDate: container.decode(Date.self, forKey: .lastAccessDate),
            byteCount: container.decode(UInt64.self, forKey: .byteCount)
        )
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(formatVersion, forKey: .formatVersion)
        try container.encode(sourceDigest, forKey: .sourceDigest)
        try container.encode(compiledDigest, forKey: .compiledDigest)
        try container.encode(metadata, forKey: .metadata)
        try container.encode(lastAccessDate, forKey: .lastAccessDate)
        try container.encode(byteCount, forKey: .byteCount)
    }

    func accessed(at date: Date) throws -> Self {
        try Self(
            sourceDigest: sourceDigest,
            compiledDigest: compiledDigest,
            metadata: metadata,
            lastAccessDate: date,
            byteCount: byteCount
        )
    }

    private enum CodingKeys: String, CodingKey {
        case formatVersion
        case sourceDigest
        case compiledDigest
        case metadata
        case lastAccessDate
        case byteCount
    }
}

struct ML5ModelDownload: Sendable {
    let data: Data
    let statusCode: Int?
}

struct ML5ModelCacheOperations: @unchecked Sendable {
    var download: @Sendable (URL) async throws -> ML5ModelDownload
    var compile: @Sendable (URL) throws -> URL
    var now: @Sendable () -> Date

    static let system = Self(
        download: { url in
            let (data, response) = try await URLSession.shared.data(from: url)
            return ML5ModelDownload(
                data: data,
                statusCode: (response as? HTTPURLResponse)?.statusCode
            )
        },
        compile: MLModel.compileModel(at:),
        now: Date.init
    )
}
