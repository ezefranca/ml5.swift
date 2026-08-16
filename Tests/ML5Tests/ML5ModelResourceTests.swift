@preconcurrency import CoreML
import Foundation
import Testing

@testable import ML5

private enum ResourceFixtureFailure: Error {
    case download
    case compile
}

private final class LockedCallCount: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0

    func increment() {
        lock.lock()
        value += 1
        lock.unlock()
    }

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
}

private final class LockedDateSequence: @unchecked Sendable {
    private let lock = NSLock()
    private var offset: TimeInterval = 0

    func next() -> Date {
        lock.lock()
        defer {
            offset += 1
            lock.unlock()
        }
        return Date(timeIntervalSince1970: 1_700_000_000 + offset)
    }
}

private final class ModelResourceURLProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.host == "ml5-system.test"
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        guard let url = request.url else { return }
        let response: URLResponse =
            if url.path == "/http" {
                HTTPURLResponse(
                    url: url,
                    statusCode: 200,
                    httpVersion: "HTTP/1.1",
                    headerFields: nil
                )
                    ?? URLResponse(
                        url: url,
                        mimeType: "application/octet-stream",
                        expectedContentLength: 3,
                        textEncodingName: nil
                    )
            } else {
                URLResponse(
                    url: url,
                    mimeType: "application/octet-stream",
                    expectedContentLength: 3,
                    textEncodingName: nil
                )
            }
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data("url".utf8))
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

@Suite("ML5 model resources and cache", .serialized)
struct ML5ModelResourceTests {
    @Test("SHA-256 values validate files, trees, names, and symbolic links")
    func digests() throws {
        let known = try ML5ModelDigest(
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )
        #expect(ML5ModelDigest.sha256(data: Data("abc".utf8)) == known)
        #expect(known.description == known.rawValue)
        #expect(ML5ModelDigest(rawValue: known.rawValue) == known)
        #expect(
            try JSONDecoder().decode(
                ML5ModelDigest.self,
                from: JSONEncoder().encode(known)
            ) == known
        )
        for value in ["", String(repeating: "0", count: 63), String(repeating: "G", count: 64)] {
            #expect(throws: ML5Error.self) { _ = try ML5ModelDigest(value) }
        }

        let root = temporaryDirectory(named: "digest")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("model.mlmodel")
        try Data("abc".utf8).write(to: file)
        #expect(try ML5ModelDigest.sha256(contentsOf: file) == known)

        let firstTree = root.appendingPathComponent("first", isDirectory: true)
        let secondTree = root.appendingPathComponent("second", isDirectory: true)
        try makeTree(at: firstTree, reverseCreationOrder: false)
        try makeTree(at: secondTree, reverseCreationOrder: true)
        let firstDigest = try ML5ModelDigest.sha256(contentsOf: firstTree)
        #expect(try ML5ModelDigest.sha256(contentsOf: secondTree) == firstDigest)
        try Data("changed".utf8).write(to: secondTree.appendingPathComponent("a.txt"))
        #expect(try ML5ModelDigest.sha256(contentsOf: secondTree) != firstDigest)

        let empty = root.appendingPathComponent("empty", isDirectory: true)
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        #expect(try ML5ModelDigest.sha256(contentsOf: empty).rawValue.count == 64)

        let nestedLink = firstTree.appendingPathComponent("link")
        try FileManager.default.createSymbolicLink(
            at: nestedLink,
            withDestinationURL: firstTree.appendingPathComponent("a.txt")
        )
        #expect(throws: ML5Error.self) {
            _ = try ML5ModelDigest.sha256(contentsOf: firstTree)
        }
        let rootLink = root.appendingPathComponent("root-link")
        try FileManager.default.createSymbolicLink(at: rootLink, withDestinationURL: file)
        #expect(throws: ML5Error.self) {
            _ = try ML5ModelDigest.sha256(contentsOf: rootLink)
        }
        #expect(throws: ML5Error.self) {
            _ = try ML5ModelDigest.sha256(contentsOf: root.appendingPathComponent("missing"))
        }
        #expect(throws: ML5Error.self) {
            _ = try ML5ModelDigest.sha256(contentsOf: URL(fileURLWithPath: "/dev/null"))
        }
    }

    @Test("File, remote, and bundled sources retain integrity and model cards")
    func sources() throws {
        let metadata = try modelMetadata(name: "Source Model")
        let bundledURL = try #require(
            Bundle.module.url(forResource: "BundledModel", withExtension: "model-fixture")
        )
        let bundledDigest = try ML5ModelDigest.sha256(contentsOf: bundledURL)
        let bundled = try ML5ModelSource.bundledResource(
            named: "BundledModel",
            withExtension: "model-fixture",
            in: .module,
            integrityDigest: bundledDigest,
            metadata: metadata
        )
        #expect(bundled.location == .file(bundledURL))
        #expect(bundled.integrityDigest == bundledDigest)
        #expect(bundled.metadata == metadata)
        #expect(
            try JSONDecoder().decode(
                ML5ModelSource.self,
                from: JSONEncoder().encode(bundled)
            ) == bundled
        )
        #expect(throws: ML5Error.self) {
            _ = try ML5ModelSource.bundledResource(
                named: "Missing",
                withExtension: "mlmodel",
                in: .module,
                integrityDigest: bundledDigest,
                metadata: metadata
            )
        }

        #expect(throws: ML5Error.self) {
            _ = try ML5ModelSource(
                fileURL: URL(string: "https://example.com/model.mlmodel")!,
                integrityDigest: bundledDigest,
                metadata: metadata
            )
        }
        for url in [
            URL(string: "http://example.com/model.mlmodel")!,
            URL(string: "https://example.com")!,
            URL(string: "https://example.com/model.mlpackage")!,
        ] {
            #expect(throws: ML5Error.self) {
                _ = try ML5ModelSource(
                    remoteURL: url,
                    integrityDigest: bundledDigest,
                    metadata: metadata
                )
            }
        }
        let remote = try ML5ModelSource(
            remoteURL: URL(string: "https://example.com/models/model.mlmodel")!,
            integrityDigest: bundledDigest,
            metadata: metadata
        )
        #expect(
            try JSONDecoder().decode(
                ML5ModelSource.self,
                from: JSONEncoder().encode(remote)
            ) == remote
        )

        #expect(throws: ML5Error.self) {
            _ = try ML5ModelCacheConfiguration(
                directory: URL(string: "https://example.com/cache")!
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try ML5ModelCacheConfiguration(
                directory: temporaryDirectory(named: "invalid-count"),
                maximumEntryCount: 0
            )
        }
        let configuration = try ML5ModelCacheConfiguration(
            directory: temporaryDirectory(named: "valid-cache"),
            maximumEntryCount: 2
        )
        #expect(configuration.maximumEntryCount == 2)
    }

    @Test("Local source compilation, loading, inventory, and explicit eviction work")
    func localCache() async throws {
        let fixture = try makeModelFixture(name: "Local", weight: 2)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let cacheURL = fixture.directory.appendingPathComponent("cache", isDirectory: true)
        let cache = try ML5ModelCache(
            configuration: ML5ModelCacheConfiguration(directory: cacheURL, maximumEntryCount: 2)
        )
        let source = try fixture.source()

        let first = try await cache.model(for: source)
        #expect(first.isCacheHit == false)
        #expect(first.sourceDigest == fixture.digest)
        #expect(first.compiledDigest.rawValue.count == 64)
        #expect(first.metadata == fixture.metadata)
        #expect(FileManager.default.fileExists(atPath: first.modelURL.path))
        let second = try await cache.model(for: source)
        #expect(second.isCacheHit)
        #expect(second.modelURL == first.modelURL)

        let entries = try await cache.entries()
        #expect(entries.count == 1)
        #expect(entries[0].sourceDigest == fixture.digest)
        #expect(entries[0].compiledDigest == first.compiledDigest)
        #expect(entries[0].metadata == fixture.metadata)
        #expect(entries[0].byteCount > 0)
        #expect(
            try JSONDecoder().decode(
                ML5ModelCacheEntry.self,
                from: JSONEncoder().encode(entries[0])
            ) == entries[0]
        )

        let predictor = try await CoreMLModelPredictor.load(from: source, using: cache)
        let prediction = try await predictor.predict(
            FeatureVector(["features": .array([3])])
        )
        guard case let .tensor(tensor) = prediction["predictions"] else {
            Issue.record("Expected a tensor prediction")
            return
        }
        #expect(abs(tensor.values[0] - 6) < 1e-5)

        #expect(try await cache.removeModel(withSourceDigest: fixture.digest))
        #expect(try await cache.removeModel(withSourceDigest: fixture.digest) == false)
        #expect(try await cache.entries().isEmpty)

        let compiledURL = try await MLModel.compileModel(at: fixture.modelURL)
        defer { try? FileManager.default.removeItem(at: compiledURL) }
        let compiledDigest = try ML5ModelDigest.sha256(contentsOf: compiledURL)
        let compiledSource = try ML5ModelSource(
            fileURL: compiledURL,
            integrityDigest: compiledDigest,
            metadata: fixture.metadata
        )
        let direct = try await cache.model(for: compiledSource)
        #expect(direct.modelURL == compiledURL)
        #expect(direct.sourceDigest == compiledDigest)
        #expect(direct.compiledDigest == compiledDigest)
        #expect(direct.isCacheHit == false)
        #expect(try await cache.entries().isEmpty)

        try await cache.removeAll()
        #expect(FileManager.default.fileExists(atPath: cacheURL.path))
        try FileManager.default.removeItem(at: cacheURL)
        try await cache.removeAll()
        #expect(FileManager.default.fileExists(atPath: cacheURL.path))
    }

    @Test("Remote downloads verify status and bytes, then rebuild tampered cache entries")
    func remoteCache() async throws {
        let fixture = try makeModelFixture(name: "Remote", weight: 1.5)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let data = try Data(contentsOf: fixture.modelURL)
        let calls = LockedCallCount()
        let dates = LockedDateSequence()
        var operations = ML5ModelCacheOperations.system
        operations.download = { _ in
            calls.increment()
            return ML5ModelDownload(data: data, statusCode: 200)
        }
        operations.now = dates.next
        let cache = try ML5ModelCache(
            configuration: ML5ModelCacheConfiguration(
                directory: fixture.directory.appendingPathComponent("remote-cache"),
                maximumEntryCount: 2
            ),
            operations: operations
        )
        let source = try ML5ModelSource(
            remoteURL: URL(string: "https://example.com/remote.mlmodel")!,
            integrityDigest: fixture.digest,
            metadata: fixture.metadata
        )

        let first = try await cache.model(for: source)
        #expect(calls.count == 1)
        #expect(first.isCacheHit == false)
        #expect(try await cache.model(for: source).isCacheHit)
        #expect(calls.count == 1)

        let tamperedFile = try #require(firstRegularFile(in: first.modelURL))
        var tampered = try Data(contentsOf: tamperedFile)
        tampered.append(0)
        try tampered.write(to: tamperedFile)
        let rebuilt = try await cache.model(for: source)
        #expect(rebuilt.isCacheHit == false)
        #expect(calls.count == 2)
        #expect(try ML5ModelDigest.sha256(contentsOf: rebuilt.modelURL) == rebuilt.compiledDigest)
    }

    @Test("Cache manifests rebuild safely and entry limits evict old models")
    func manifestsAndLimits() async throws {
        let firstFixture = try makeModelFixture(name: "First", weight: 1)
        let secondFixture = try makeModelFixture(name: "Second", weight: 2)
        let thirdFixture = try makeModelFixture(name: "Third", weight: 3)
        defer {
            try? FileManager.default.removeItem(at: firstFixture.directory)
            try? FileManager.default.removeItem(at: secondFixture.directory)
            try? FileManager.default.removeItem(at: thirdFixture.directory)
        }
        let root = temporaryDirectory(named: "manifest-cache")
        defer { try? FileManager.default.removeItem(at: root) }
        let dates = LockedDateSequence()
        var operations = ML5ModelCacheOperations.system
        operations.now = dates.next
        let cache = try ML5ModelCache(
            configuration: ML5ModelCacheConfiguration(directory: root, maximumEntryCount: 2),
            operations: operations
        )
        let firstSource = try firstFixture.source()
        let secondSource = try secondFixture.source()
        let thirdSource = try thirdFixture.source()
        _ = try await cache.model(for: firstSource)
        _ = try await cache.model(for: secondSource)
        #expect(try await cache.entries().count == 2)
        _ = try await cache.model(for: firstSource)
        #expect(try await cache.entries()[0].sourceDigest == firstFixture.digest)

        let firstEntry = root.appendingPathComponent(firstFixture.digest.rawValue)
        let manifestURL = firstEntry.appendingPathComponent("manifest.json")
        try mutateManifest(at: manifestURL) { $0["sourceDigest"] = secondFixture.digest.rawValue }
        #expect(try await cache.model(for: firstSource).isCacheHit == false)
        try mutateManifest(at: manifestURL) { $0["formatVersion"] = 2 }
        #expect(try await cache.model(for: firstSource).isCacheHit == false)
        try mutateManifest(at: manifestURL) { $0["byteCount"] = 0 }
        #expect(try await cache.model(for: firstSource).isCacheHit == false)
        try Data("bad manifest".utf8).write(to: manifestURL)
        #expect(try await cache.model(for: firstSource).isCacheHit == false)

        let junk = root.appendingPathComponent("junk", isDirectory: true)
        try FileManager.default.createDirectory(at: junk, withIntermediateDirectories: true)
        try Data("bad".utf8).write(to: junk.appendingPathComponent("manifest.json"))
        try Data("stray".utf8).write(to: root.appendingPathComponent("stray-file"))
        #expect(try await cache.entries().count == 2)
        _ = try await cache.model(for: thirdSource)
        #expect(try await cache.entries().count == 2)

        let limitedRoot = temporaryDirectory(named: "limited-cache")
        defer { try? FileManager.default.removeItem(at: limitedRoot) }
        let limited = try ML5ModelCache(
            configuration: ML5ModelCacheConfiguration(
                directory: limitedRoot,
                maximumEntryCount: 1
            ),
            operations: operations
        )
        _ = try await limited.model(for: firstSource)
        _ = try await limited.model(for: secondSource)
        _ = try await limited.model(for: thirdSource)
        let retained = try await limited.entries()
        #expect(retained.count == 1)
        #expect(retained[0].sourceDigest == thirdFixture.digest)

        var constantOperations = ML5ModelCacheOperations.system
        constantOperations.now = { Date(timeIntervalSince1970: 1_700_000_000) }
        let tiedRoot = temporaryDirectory(named: "tied-cache")
        defer { try? FileManager.default.removeItem(at: tiedRoot) }
        let tied = try ML5ModelCache(
            configuration: ML5ModelCacheConfiguration(
                directory: tiedRoot,
                maximumEntryCount: 2
            ),
            operations: constantOperations
        )
        _ = try await tied.model(for: firstSource)
        _ = try await tied.model(for: secondSource)
        _ = try await tied.model(for: thirdSource)
        let tiedEntries = try await tied.entries()
        #expect(tiedEntries.count == 2)
        #expect(tiedEntries[0].sourceDigest.rawValue < tiedEntries[1].sourceDigest.rawValue)

        let tiedLimitedRoot = temporaryDirectory(named: "tied-limited-cache")
        defer { try? FileManager.default.removeItem(at: tiedLimitedRoot) }
        let tiedLimited = try ML5ModelCache(
            configuration: ML5ModelCacheConfiguration(
                directory: tiedLimitedRoot,
                maximumEntryCount: 1
            ),
            operations: constantOperations
        )
        _ = try await tiedLimited.model(for: firstSource)
        _ = try await tiedLimited.model(for: secondSource)
        _ = try await tiedLimited.model(for: thirdSource)
        #expect(try await tiedLimited.entries().count == 1)
    }

    @Test("Invalid sources, responses, cancellation, and infrastructure failures are explicit")
    func failures() async throws {
        let fixture = try makeModelFixture(name: "Failures", weight: 1)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let wrongDigest = ML5ModelDigest.sha256(data: Data("wrong".utf8))
        let wrongSource = try ML5ModelSource(
            fileURL: fixture.modelURL,
            integrityDigest: wrongDigest,
            metadata: fixture.metadata
        )
        let cache = try ML5ModelCache(
            configuration: ML5ModelCacheConfiguration(
                directory: fixture.directory.appendingPathComponent("failures-cache")
            )
        )
        await #expect(throws: ML5Error.self) {
            _ = try await cache.model(for: wrongSource)
        }

        let textURL = fixture.directory.appendingPathComponent("model.txt")
        let modelData = try Data(contentsOf: fixture.modelURL)
        try modelData.write(to: textURL)
        let textSource = try ML5ModelSource(
            fileURL: textURL,
            integrityDigest: ML5ModelDigest.sha256(data: modelData),
            metadata: fixture.metadata
        )
        await #expect(throws: ML5Error.self) {
            _ = try await cache.model(for: textSource)
        }

        let cancelled = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await cache.model(for: try fixture.source())
        }
        await #expect(throws: CancellationError.self) { _ = try await cancelled.value }

        for statusCode in [404, nil] as [Int?] {
            var operations = ML5ModelCacheOperations.system
            operations.download = { _ in
                ML5ModelDownload(data: modelData, statusCode: statusCode)
            }
            let statusCache = try makeInjectedCache(
                in: fixture.directory,
                name: "status-\(statusCode.map(String.init) ?? "nil")",
                operations: operations
            )
            await #expect(throws: ML5Error.self) {
                _ = try await statusCache.model(for: try remoteSource(for: fixture))
            }
        }

        var downloadFailure = ML5ModelCacheOperations.system
        downloadFailure.download = { _ in throw ResourceFixtureFailure.download }
        let downloadCache = try makeInjectedCache(
            in: fixture.directory,
            name: "download-failure",
            operations: downloadFailure
        )
        await #expect(throws: ML5Error.self) {
            _ = try await downloadCache.model(for: try remoteSource(for: fixture))
        }
        downloadFailure.download = { _ in throw CancellationError() }
        let cancellationCache = try makeInjectedCache(
            in: fixture.directory,
            name: "download-cancellation",
            operations: downloadFailure
        )
        await #expect(throws: CancellationError.self) {
            _ = try await cancellationCache.model(for: try remoteSource(for: fixture))
        }

        var compileFailure = ML5ModelCacheOperations.system
        compileFailure.compile = { _ in throw ResourceFixtureFailure.compile }
        let compileCache = try makeInjectedCache(
            in: fixture.directory,
            name: "compile-failure",
            operations: compileFailure
        )
        await #expect(throws: ML5Error.self) {
            _ = try await compileCache.model(for: try fixture.source())
        }
        compileFailure.download = { _ in
            ML5ModelDownload(data: modelData, statusCode: 200)
        }
        let remoteCompileCache = try makeInjectedCache(
            in: fixture.directory,
            name: "remote-compile-failure",
            operations: compileFailure
        )
        await #expect(throws: ML5Error.self) {
            _ = try await remoteCompileCache.model(for: try remoteSource(for: fixture))
        }

        var integrityFailure = ML5ModelCacheOperations.system
        integrityFailure.download = { _ in
            ML5ModelDownload(data: Data("wrong".utf8), statusCode: 200)
        }
        let integrityCache = try makeInjectedCache(
            in: fixture.directory,
            name: "remote-integrity-failure",
            operations: integrityFailure
        )
        await #expect(throws: ML5Error.self) {
            _ = try await integrityCache.model(for: try remoteSource(for: fixture))
        }

        var emptyCompilation = ML5ModelCacheOperations.system
        emptyCompilation.download = { _ in
            ML5ModelDownload(data: modelData, statusCode: 200)
        }
        emptyCompilation.compile = { _ in
            let url = fixture.directory.appendingPathComponent(
                "empty-\(UUID().uuidString).mlmodelc",
                isDirectory: true
            )
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            return url
        }
        let emptyCache = try makeInjectedCache(
            in: fixture.directory,
            name: "empty-compilation",
            operations: emptyCompilation
        )
        await #expect(throws: ML5Error.self) {
            _ = try await emptyCache.model(for: try remoteSource(for: fixture))
        }

        var postDownloadCancellation = ML5ModelCacheOperations.system
        postDownloadCancellation.download = { _ in
            withUnsafeCurrentTask { $0?.cancel() }
            return ML5ModelDownload(data: modelData, statusCode: 200)
        }
        let postDownloadCache = try makeInjectedCache(
            in: fixture.directory,
            name: "post-download-cancellation",
            operations: postDownloadCancellation
        )
        let postDownloadTask = Task {
            try await postDownloadCache.model(for: try remoteSource(for: fixture))
        }
        await #expect(throws: CancellationError.self) {
            _ = try await postDownloadTask.value
        }

        let brokenCompileRoot = fixture.directory.appendingPathComponent(
            "broken-compile-root",
            isDirectory: true
        )
        let brokenCompileCache = try ML5ModelCache(
            configuration: ML5ModelCacheConfiguration(directory: brokenCompileRoot)
        )
        try FileManager.default.removeItem(at: brokenCompileRoot)
        try Data("block".utf8).write(to: brokenCompileRoot)
        await #expect(throws: ML5Error.self) {
            _ = try await brokenCompileCache.model(for: try fixture.source())
        }

        var validDownload = ML5ModelCacheOperations.system
        validDownload.download = { _ in
            ML5ModelDownload(data: modelData, statusCode: 200)
        }
        let brokenDownloadRoot = fixture.directory.appendingPathComponent(
            "broken-download-root",
            isDirectory: true
        )
        let brokenDownloadCache = try ML5ModelCache(
            configuration: ML5ModelCacheConfiguration(directory: brokenDownloadRoot),
            operations: validDownload
        )
        try FileManager.default.removeItem(at: brokenDownloadRoot)
        try Data("block".utf8).write(to: brokenDownloadRoot)
        await #expect(throws: ML5Error.self) {
            _ = try await brokenDownloadCache.model(for: try remoteSource(for: fixture))
        }

        let brokenInventoryRoot = fixture.directory.appendingPathComponent(
            "broken-inventory-root",
            isDirectory: true
        )
        let brokenInventory = try ML5ModelCache(
            configuration: ML5ModelCacheConfiguration(directory: brokenInventoryRoot)
        )
        try FileManager.default.removeItem(at: brokenInventoryRoot)
        try Data("block".utf8).write(to: brokenInventoryRoot)
        await #expect(throws: ML5Error.self) {
            _ = try await brokenInventory.entries()
        }

        #expect(URLProtocol.registerClass(ModelResourceURLProtocol.self))
        defer { URLProtocol.unregisterClass(ModelResourceURLProtocol.self) }
        let systemHTTP = try await ML5ModelCacheOperations.system.download(
            URL(string: "https://ml5-system.test/http")!
        )
        #expect(systemHTTP.data == Data("url".utf8))
        #expect(systemHTTP.statusCode == 200)
        let systemPlain = try await ML5ModelCacheOperations.system.download(
            URL(string: "https://ml5-system.test/plain")!
        )
        #expect(systemPlain.statusCode == nil)

        let lockedParent = fixture.directory.appendingPathComponent(
            "locked-parent",
            isDirectory: true
        )
        let lockedRoot = lockedParent.appendingPathComponent("cache", isDirectory: true)
        let lockedCache = try ML5ModelCache(
            configuration: ML5ModelCacheConfiguration(directory: lockedRoot)
        )
        _ = try await lockedCache.model(for: try fixture.source())
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o500],
            ofItemAtPath: lockedRoot.path
        )
        await #expect(throws: ML5Error.self) {
            _ = try await lockedCache.removeModel(withSourceDigest: fixture.digest)
        }
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o700],
            ofItemAtPath: lockedRoot.path
        )
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o500],
            ofItemAtPath: lockedParent.path
        )
        await #expect(throws: ML5Error.self) {
            try await lockedCache.removeAll()
        }
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o700],
            ofItemAtPath: lockedParent.path
        )

        let blockingFile = fixture.directory.appendingPathComponent("blocking-file")
        try Data("block".utf8).write(to: blockingFile)
        #expect(throws: ML5Error.self) {
            _ = try ML5ModelCache(
                configuration: ML5ModelCacheConfiguration(
                    directory: blockingFile.appendingPathComponent("cache")
                )
            )
        }

        #expect(
            ML5Error.invalidModelDigest("bad").errorDescription?.contains("SHA-256") == true
        )
        #expect(
            ML5Error.modelIntegrityMismatch(expected: "a", actual: "b").errorDescription
                == "Model integrity check failed: expected a, but computed b."
        )
        #expect(
            ML5Error.invalidModelSource(reason: "Failure.").errorDescription
                == "Invalid model source: Failure."
        )
        #expect(
            ML5Error.modelResourceFailed(location: "remote", message: "Failure.")
                .errorDescription
                == "Model resource failed at \"remote\": Failure."
        )
        #expect(
            ML5Error.modelCacheFailed(path: "/cache", message: "Failure.").errorDescription
                == "Model cache failed at \"/cache\": Failure."
        )
    }

    private func makeTree(at directory: URL, reverseCreationOrder: Bool) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let nested = directory.appendingPathComponent("nested", isDirectory: true)
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        let values = [
            (directory.appendingPathComponent("a.txt"), Data("alpha".utf8)),
            (nested.appendingPathComponent("b.txt"), Data("beta".utf8)),
        ]
        for (url, data) in reverseCreationOrder ? values.reversed() : values {
            try data.write(to: url)
        }
    }

    private func modelMetadata(name: String) throws -> ML5ModelMetadata {
        try ML5ModelMetadata(
            name: name,
            version: "1.0.0",
            author: "ML5 Tests",
            license: "MIT",
            summary: "Integrity-checked test model.",
            source: URL(string: "https://example.com/models/\(name)")
        )
    }

    private func makeModelFixture(name: String, weight: Double) throws -> ModelResourceFixture {
        let directory = temporaryDirectory(named: "resource-\(name)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let configuration = try DenseNetworkConfiguration(
            inputFeatures: ["x"],
            outputNames: ["y"],
            validationFraction: 0
        )
        let model = try DenseNetworkModel(
            configuration: configuration,
            layers: [
                try DenseLayerParameters(
                    inputCount: 1,
                    outputCount: 1,
                    weights: [weight],
                    biases: [0]
                )
            ]
        )
        let metadata = try modelMetadata(name: name)
        let export = try DenseCoreMLExportConfiguration(metadata: metadata)
        let modelURL = directory.appendingPathComponent("\(name).mlmodel")
        try model.writeCoreMLModel(to: modelURL, configuration: export)
        return ModelResourceFixture(
            directory: directory,
            modelURL: modelURL,
            digest: try ML5ModelDigest.sha256(contentsOf: modelURL),
            metadata: metadata
        )
    }

    private func remoteSource(for fixture: ModelResourceFixture) throws -> ML5ModelSource {
        try ML5ModelSource(
            remoteURL: #require(
                URL(string: "https://example.com/\(fixture.metadata.name).mlmodel")
            ),
            integrityDigest: fixture.digest,
            metadata: fixture.metadata
        )
    }

    private func makeInjectedCache(
        in parent: URL,
        name: String,
        operations: ML5ModelCacheOperations
    ) throws -> ML5ModelCache {
        try ML5ModelCache(
            configuration: ML5ModelCacheConfiguration(
                directory: parent.appendingPathComponent(name, isDirectory: true)
            ),
            operations: operations
        )
    }

    private func firstRegularFile(in directory: URL) -> URL? {
        guard
            let enumerator = FileManager.default.enumerator(
                at: directory,
                includingPropertiesForKeys: [.isRegularFileKey]
            )
        else { return nil }
        for case let url as URL in enumerator
        where (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true {
            return url
        }
        return nil
    }

    private func mutateManifest(
        at url: URL,
        mutation: (inout [String: Any]) -> Void
    ) throws {
        var object = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]
        )
        mutation(&object)
        try JSONSerialization.data(withJSONObject: object).write(to: url)
    }

    private func temporaryDirectory(named name: String) -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent(
            "ml5-\(name)-\(UUID().uuidString)",
            isDirectory: true
        )
    }
}

private struct ModelResourceFixture {
    let directory: URL
    let modelURL: URL
    let digest: ML5ModelDigest
    let metadata: ML5ModelMetadata

    func source() throws -> ML5ModelSource {
        try ML5ModelSource(
            fileURL: modelURL,
            integrityDigest: digest,
            metadata: metadata
        )
    }
}
