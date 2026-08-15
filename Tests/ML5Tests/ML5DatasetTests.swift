import Foundation
import Testing

@testable import ML5

private enum DatasetLabel: String, ClassificationLabel {
    case cat

    init?(ml5RawValue: String) {
        self.init(rawValue: ml5RawValue)
    }

    var ml5RawValue: String { rawValue }
}

@Suite("ML5 datasets")
struct ML5DatasetTests {
    @Test("Sample identifiers are positive, ordered, hashable, and serializable")
    func identifiers() throws {
        let first = try DatasetSampleID(rawValue: 1)
        let second = try DatasetSampleID(rawValue: 2)

        #expect(first < second)
        #expect(Set([first, first, second]).count == 2)
        #expect(
            try JSONDecoder().decode(
                DatasetSampleID.self,
                from: JSONEncoder().encode(second)
            ) == second
        )
        #expect(throws: ML5Error.invalidDatasetSampleID(0)) {
            _ = try DatasetSampleID(rawValue: 0)
        }
        #expect(throws: ML5Error.invalidDatasetSampleID(0)) {
            _ = try JSONDecoder().decode(DatasetSampleID.self, from: Data("0".utf8))
        }
    }

    @Test("Classification and regression samples preserve validated Codable values")
    func sampleCodable() throws {
        let features = try FeatureVector(["x": .number(1)])
        let classification = ClassificationSample(features: features, label: DatasetLabel.cat)
        let regression = try RegressionSample(features: features, target: 2.5)

        #expect(Set([classification, classification]).count == 1)
        #expect(Set([regression, regression]).count == 1)
        #expect(
            try JSONDecoder().decode(
                ClassificationSample<DatasetLabel>.self,
                from: JSONEncoder().encode(classification)
            ) == classification
        )
        #expect(
            try JSONDecoder().decode(
                RegressionSample.self,
                from: JSONEncoder().encode(regression)
            ) == regression
        )
        #expect(throws: ML5Error.invalidClassLabel("dog")) {
            _ = try JSONDecoder().decode(
                ClassificationSample<DatasetLabel>.self,
                from: Data(
                    #"{"features":{"x":{"kind":"number","number":1}},"label":"dog"}"#
                        .utf8
                )
            )
        }
    }

    @Test("Split configuration derives training fraction and rejects malformed values")
    func splitConfiguration() throws {
        let standard = try DatasetSplitConfiguration()
        let thirds = try DatasetSplitConfiguration(validationFraction: 0.25, testFraction: 0.25)

        #expect(standard == .standard)
        #expect(standard.trainingFraction == 0.8)
        #expect(standard.validationFraction == 0.2)
        #expect(standard.testFraction == 0)
        #expect(thirds.trainingFraction == 0.5)
        #expect(
            try JSONDecoder().decode(
                DatasetSplitConfiguration.self,
                from: JSONEncoder().encode(thirds)
            ) == thirds
        )

        let invalidFractions: [(Double, Double)] = [
            (.nan, 0),
            (0, .infinity),
            (-0.1, 0),
            (0, -0.1),
            (0.6, 0.5),
        ]
        for (validation, test) in invalidFractions {
            #expect(throws: ML5Error.self) {
                _ = try DatasetSplitConfiguration(
                    validationFraction: validation,
                    testFraction: test
                )
            }
        }
    }

    @Test("Datasets accumulate, look up, remove, checkpoint, and restore samples")
    func lifecycle() async throws {
        let dataset = ML5Dataset<String>()
        #expect(await dataset.isEmpty)
        #expect(await dataset.count == 0)
        #expect(try await dataset.add(contentsOf: []).isEmpty)

        let first = try await dataset.add("first")
        let added = try await dataset.add(contentsOf: ["second", "third"])
        #expect(first.rawValue == 1)
        #expect(added.map(\.rawValue) == [2, 3])
        #expect(await !dataset.isEmpty)
        #expect(await dataset.count == 3)
        #expect(await dataset.entries.map(\.sample) == ["first", "second", "third"])
        #expect(await dataset.sample(for: added[0]) == "second")
        #expect(await dataset.sample(for: try DatasetSampleID(rawValue: 99)) == nil)
        #expect(await dataset.remove(first) == "first")
        #expect(await dataset.remove(first) == nil)

        let snapshot = try await dataset.snapshot()
        let decoded = try JSONDecoder().decode(
            DatasetSnapshot<String>.self,
            from: JSONEncoder().encode(snapshot)
        )
        #expect(decoded == snapshot)
        #expect(Set([snapshot, decoded]).count == 1)

        let restored = ML5Dataset(snapshot: decoded)
        #expect(await restored.entries == decoded.entries)
        #expect(try await restored.add("fourth").rawValue == 4)
        #expect(await restored.removeAll() == 3)
        #expect(await restored.isEmpty)
    }

    @Test("Snapshots reject duplicate and out-of-range identifiers")
    func invalidSnapshots() throws {
        let first = try DatasetSampleID(rawValue: 1)
        let second = try DatasetSampleID(rawValue: 2)
        let duplicate = DatasetEntry(id: first, sample: "duplicate")

        #expect(throws: ML5Error.self) {
            _ = try DatasetSnapshot(
                entries: [DatasetEntry(id: first, sample: "first"), duplicate],
                nextIdentifier: second
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try DatasetSnapshot(
                entries: [DatasetEntry(id: second, sample: "second")],
                nextIdentifier: second
            )
        }
    }

    @Test("Batch insertion is atomic at identifier exhaustion")
    func identifierExhaustion() async throws {
        let almostExhausted = try DatasetSnapshot<String>(
            entries: [],
            nextIdentifier: DatasetSampleID(rawValue: UInt64.max - 1)
        )
        let dataset = ML5Dataset(snapshot: almostExhausted)

        await #expect(throws: ML5Error.datasetIdentifierExhausted) {
            try await dataset.add(contentsOf: ["one", "two"])
        }
        #expect(await dataset.isEmpty)
        #expect(try await dataset.add("last").rawValue == UInt64.max - 1)
        #expect(try await dataset.snapshot().nextIdentifier.rawValue == UInt64.max)
        await #expect(throws: ML5Error.datasetIdentifierExhausted) {
            try await dataset.add("overflow")
        }
        #expect(await dataset.count == 1)
    }

    @Test("Seeded shuffle is reproducible and handles empty or singleton datasets")
    func shuffle() async throws {
        let first = ML5Dataset<Int>()
        let second = ML5Dataset<Int>()
        _ = try await first.add(contentsOf: Array(0..<10))
        _ = try await second.add(contentsOf: Array(0..<10))

        await first.shuffle(seed: 42)
        await second.shuffle(seed: 42)
        let firstOrder = await first.entries
        let secondOrder = await second.entries
        #expect(firstOrder == secondOrder)
        #expect(firstOrder.map(\.sample) == [0, 9, 5, 8, 6, 4, 7, 2, 1, 3])
        #expect(Set(firstOrder.map(\.sample)) == Set(0..<10))

        let empty = ML5Dataset<Int>()
        await empty.shuffle(seed: 1)
        _ = try await empty.add(7)
        await empty.shuffle(seed: 1)
        #expect(await empty.entries.map(\.sample) == [7])
    }

    @Test("Splits preserve current order or reproducibly shuffle a copy")
    func splits() async throws {
        let dataset = ML5Dataset<Int>()
        _ = try await dataset.add(contentsOf: Array(0..<10))

        let standard = await dataset.split()
        #expect(standard.training.map(\.sample) == Array(0..<8))
        #expect(standard.validation.map(\.sample) == [8, 9])
        #expect(standard.test.isEmpty)

        let configuration = try DatasetSplitConfiguration(
            validationFraction: 0.2,
            testFraction: 0.3
        )
        let shuffled = await dataset.split(using: configuration, seed: 7)
        let repeated = await dataset.split(using: configuration, seed: 7)
        #expect(shuffled == repeated)
        #expect(shuffled.training.count == 5)
        #expect(shuffled.validation.count == 2)
        #expect(shuffled.test.count == 3)
        #expect(
            shuffled.training.count + shuffled.validation.count + shuffled.test.count == 10
        )
        #expect(
            try JSONDecoder().decode(
                DatasetSplit<Int>.self,
                from: JSONEncoder().encode(shuffled)
            ) == shuffled
        )

        let empty = await ML5Dataset<Int>().split()
        #expect(empty.training.isEmpty)
        #expect(empty.validation.isEmpty)
        #expect(empty.test.isEmpty)
    }
}
