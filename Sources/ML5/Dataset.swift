import Foundation

/// A stable identifier assigned to one sample in an ``ML5Dataset``.
@frozen
public struct DatasetSampleID: Sendable, Hashable, Comparable, Codable {
    /// The positive integer representation persisted in dataset snapshots.
    public let rawValue: UInt64

    /// Reconstructs a positive sample identifier.
    ///
    /// - Throws: ``ML5Error/invalidDatasetSampleID(_:)`` when `rawValue` is zero.
    public init(rawValue: UInt64) throws {
        guard rawValue > 0 else {
            throw ML5Error.invalidDatasetSampleID(rawValue)
        }
        self.rawValue = rawValue
    }

    /// Orders identifiers by their integer representation.
    public static func < (lhs: Self, rhs: Self) -> Bool {
        lhs.rawValue < rhs.rawValue
    }

    /// Decodes and validates the single integer representation.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        try self.init(rawValue: container.decode(UInt64.self))
    }

    /// Encodes the identifier as a single integer.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }
}

/// One identified value in a dataset snapshot or split.
@frozen
public struct DatasetEntry<Sample: Sendable>: Sendable {
    /// The stable identifier assigned when the sample was added.
    public let id: DatasetSampleID
    /// The application-defined sample value.
    public let sample: Sample

    /// Creates an identified sample value.
    public init(id: DatasetSampleID, sample: Sample) {
        self.id = id
        self.sample = sample
    }
}

extension DatasetEntry: Equatable where Sample: Equatable {}
extension DatasetEntry: Hashable where Sample: Hashable {}
extension DatasetEntry: Codable where Sample: Codable {}

/// An immutable, serializable dataset state suitable for checkpoints.
@frozen
public struct DatasetSnapshot<Sample: Sendable>: Sendable {
    /// Entries in their current deterministic order.
    public let entries: [DatasetEntry<Sample>]
    /// The identifier reserved for the next inserted sample.
    public let nextIdentifier: DatasetSampleID

    /// Creates a snapshot with unique identifiers below the next reserved value.
    ///
    /// - Throws: ``ML5Error/invalidDatasetSnapshot(reason:)`` for duplicate or
    ///   out-of-range identifier metadata.
    public init(
        entries: [DatasetEntry<Sample>],
        nextIdentifier: DatasetSampleID
    ) throws {
        var identifiers: Set<DatasetSampleID> = []
        for entry in entries {
            guard identifiers.insert(entry.id).inserted else {
                throw ML5Error.invalidDatasetSnapshot(
                    reason: "Sample identifier \(entry.id.rawValue) appears more than once."
                )
            }
            guard entry.id < nextIdentifier else {
                throw ML5Error.invalidDatasetSnapshot(
                    reason:
                        "Sample identifier \(entry.id.rawValue) must be below the next identifier."
                )
            }
        }
        self.entries = entries
        self.nextIdentifier = nextIdentifier
    }
}

extension DatasetSnapshot: Equatable where Sample: Equatable {}
extension DatasetSnapshot: Hashable where Sample: Hashable {}

extension DatasetSnapshot: Codable where Sample: Codable {
    /// Decodes entries and revalidates all snapshot identity invariants.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            entries: container.decode([DatasetEntry<Sample>].self, forKey: .entries),
            nextIdentifier: container.decode(DatasetSampleID.self, forKey: .nextIdentifier)
        )
    }

    /// Encodes ordered entries and the next reserved identifier.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(entries, forKey: .entries)
        try container.encode(nextIdentifier, forKey: .nextIdentifier)
    }

    private enum CodingKeys: String, CodingKey {
        case entries
        case nextIdentifier
    }
}

/// Fractions used to divide a dataset into training, validation, and test entries.
@frozen
public struct DatasetSplitConfiguration: Sendable, Hashable, Codable {
    /// The conventional 80% training, 20% validation, and 0% test split.
    public static let standard = Self(
        trainingFraction: 0.8,
        validationFraction: 0.2,
        testFraction: 0
    )

    /// Fraction assigned to training after validation and test allocation.
    public let trainingFraction: Double
    /// Fraction assigned to validation.
    public let validationFraction: Double
    /// Fraction assigned to final testing.
    public let testFraction: Double

    /// Creates a split whose validation and test fractions are finite, nonnegative,
    /// and sum to no more than one.
    ///
    /// The training fraction is the remaining proportion.
    ///
    /// - Throws: ``ML5Error/invalidDatasetSplit(reason:)`` for invalid fractions.
    public init(validationFraction: Double = 0.2, testFraction: Double = 0) throws {
        guard
            validationFraction.isFinite,
            testFraction.isFinite,
            validationFraction >= 0,
            testFraction >= 0,
            validationFraction + testFraction <= 1
        else {
            throw ML5Error.invalidDatasetSplit(
                reason:
                    "Validation and test fractions must be finite, nonnegative, and sum to at most one."
            )
        }
        self.init(
            trainingFraction: 1 - validationFraction - testFraction,
            validationFraction: validationFraction,
            testFraction: testFraction
        )
    }

    /// Decodes fractions and revalidates their range and sum.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            validationFraction: container.decode(Double.self, forKey: .validationFraction),
            testFraction: container.decode(Double.self, forKey: .testFraction)
        )
    }

    /// Encodes the independently configured validation and test fractions.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(validationFraction, forKey: .validationFraction)
        try container.encode(testFraction, forKey: .testFraction)
    }

    private enum CodingKeys: String, CodingKey {
        case validationFraction
        case testFraction
    }

    private init(
        trainingFraction: Double,
        validationFraction: Double,
        testFraction: Double
    ) {
        self.trainingFraction = trainingFraction
        self.validationFraction = validationFraction
        self.testFraction = testFraction
    }
}

/// Immutable partitions produced by an ``ML5Dataset`` split.
@frozen
public struct DatasetSplit<Sample: Sendable>: Sendable {
    /// Entries used to fit model parameters.
    public let training: [DatasetEntry<Sample>]
    /// Entries used for tuning and early-stopping decisions.
    public let validation: [DatasetEntry<Sample>]
    /// Entries held back for final evaluation.
    public let test: [DatasetEntry<Sample>]

    /// Creates an immutable three-way split.
    public init(
        training: [DatasetEntry<Sample>],
        validation: [DatasetEntry<Sample>],
        test: [DatasetEntry<Sample>]
    ) {
        self.training = training
        self.validation = validation
        self.test = test
    }
}

extension DatasetSplit: Equatable where Sample: Equatable {}
extension DatasetSplit: Hashable where Sample: Hashable {}
extension DatasetSplit: Codable where Sample: Codable {}

/// Actor-isolated accumulation, removal, ordering, and splitting of training samples.
public actor ML5Dataset<Sample: Sendable> {
    private var storage: [DatasetEntry<Sample>]
    private var nextIdentifierRawValue: UInt64

    /// Creates an empty dataset.
    public init() {
        storage = []
        nextIdentifierRawValue = 1
    }

    /// Restores a dataset from an already validated snapshot.
    public init(snapshot: DatasetSnapshot<Sample>) {
        storage = snapshot.entries
        nextIdentifierRawValue = snapshot.nextIdentifier.rawValue
    }

    /// Number of currently stored samples.
    public var count: Int {
        storage.count
    }

    /// Whether the dataset currently contains no samples.
    public var isEmpty: Bool {
        storage.isEmpty
    }

    /// Entries in their current deterministic order.
    public var entries: [DatasetEntry<Sample>] {
        storage
    }

    /// Returns the sample for an identifier, or `nil` after it is removed.
    public func sample(for id: DatasetSampleID) -> Sample? {
        storage.first { $0.id == id }?.sample
    }

    /// Adds one sample and returns its stable identifier.
    ///
    /// - Throws: ``ML5Error/datasetIdentifierExhausted`` when no identifier remains.
    @discardableResult
    public func add(_ sample: Sample) throws -> DatasetSampleID {
        try add(contentsOf: [sample])[0]
    }

    /// Adds a batch atomically and returns identifiers in input order.
    ///
    /// An empty batch is a no-op.
    ///
    /// - Throws: ``ML5Error/datasetIdentifierExhausted`` when the entire batch cannot
    ///   receive identifiers without overflow.
    @discardableResult
    public func add(contentsOf samples: [Sample]) throws -> [DatasetSampleID] {
        let requestedCount = UInt64(samples.count)
        guard requestedCount <= UInt64.max - nextIdentifierRawValue else {
            throw ML5Error.datasetIdentifierExhausted
        }
        let identifiers = try samples.indices.map { offset in
            try DatasetSampleID(rawValue: nextIdentifierRawValue + UInt64(offset))
        }
        storage.append(
            contentsOf: zip(identifiers, samples).map { id, sample in
                DatasetEntry(id: id, sample: sample)
            }
        )
        nextIdentifierRawValue += requestedCount
        return identifiers
    }

    /// Removes and returns one sample, or returns `nil` for an unknown identifier.
    @discardableResult
    public func remove(_ id: DatasetSampleID) -> Sample? {
        guard let index = storage.firstIndex(where: { $0.id == id }) else {
            return nil
        }
        return storage.remove(at: index).sample
    }

    /// Removes every sample and returns the number removed.
    @discardableResult
    public func removeAll() -> Int {
        let removedCount = storage.count
        storage.removeAll(keepingCapacity: true)
        return removedCount
    }

    /// Reorders entries reproducibly using a stable SplitMix64 generator.
    public func shuffle(seed: UInt64) {
        storage.ml5Shuffle(seed: seed)
    }

    /// Returns train, validation, and test partitions without mutating the dataset.
    ///
    /// Passing a seed shuffles a copy reproducibly before partitioning. Omitting the
    /// seed preserves the dataset's current order. Fractional counts round down, and
    /// the training partition receives all remaining entries.
    public func split(
        using configuration: DatasetSplitConfiguration = .standard,
        seed: UInt64? = nil
    ) -> DatasetSplit<Sample> {
        var selected = storage
        if let seed {
            selected.ml5Shuffle(seed: seed)
        }
        let validationCount = Int(Double(selected.count) * configuration.validationFraction)
        let testCount = Int(Double(selected.count) * configuration.testFraction)
        let trainingCount = selected.count - validationCount - testCount
        let validationEnd = trainingCount + validationCount
        return DatasetSplit(
            training: Array(selected[..<trainingCount]),
            validation: Array(selected[trainingCount..<validationEnd]),
            test: Array(selected[validationEnd..<selected.endIndex])
        )
    }

    /// Captures entries and identifier state for persistence or reproducible resume.
    public func snapshot() throws -> DatasetSnapshot<Sample> {
        try DatasetSnapshot(
            entries: storage,
            nextIdentifier: DatasetSampleID(rawValue: nextIdentifierRawValue)
        )
    }
}

private struct ML5SeededRandomNumberGenerator: RandomNumberGenerator {
    var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }
}

extension Array {
    fileprivate mutating func ml5Shuffle(seed: UInt64) {
        guard count > 1 else { return }
        var generator = ML5SeededRandomNumberGenerator(state: seed)
        for index in stride(from: count - 1, through: 1, by: -1) {
            let destination = Int(generator.next() % UInt64(index + 1))
            if destination != index {
                swapAt(index, destination)
            }
        }
    }
}
