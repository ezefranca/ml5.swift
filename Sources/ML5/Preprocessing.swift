import Foundation

/// A finite, increasing output interval for min-max normalization.
@frozen
public struct NormalizationRange: Sendable, Hashable, Codable {
    /// The normalized lower bound.
    public let lowerBound: Double
    /// The normalized upper bound.
    public let upperBound: Double

    /// Creates a finite interval whose lower bound is strictly below its upper bound.
    ///
    /// - Throws: ``ML5Error/invalidNormalization(reason:)`` for an invalid interval.
    public init(lowerBound: Double = 0, upperBound: Double = 1) throws {
        guard lowerBound.isFinite, upperBound.isFinite, lowerBound < upperBound else {
            throw ML5Error.invalidNormalization(
                reason: "A normalization range must have finite, increasing bounds."
            )
        }
        self.lowerBound = lowerBound
        self.upperBound = upperBound
    }

    /// Decodes bounds and revalidates the interval.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            lowerBound: container.decode(Double.self, forKey: .lowerBound),
            upperBound: container.decode(Double.self, forKey: .upperBound)
        )
    }

    /// Encodes both interval bounds.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(lowerBound, forKey: .lowerBound)
        try container.encode(upperBound, forKey: .upperBound)
    }

    private enum CodingKeys: String, CodingKey {
        case lowerBound
        case upperBound
    }
}

/// A reversible numerical normalization strategy.
@frozen
public enum NormalizationStrategy: Sendable, Hashable, Codable {
    /// Subtract the fitted mean and divide by population standard deviation.
    case standardScore
    /// Map the fitted minimum and maximum into a validated output interval.
    case minMax(NormalizationRange)
}

/// A named feature and the strategy used to fit it.
@frozen
public struct FeatureNormalizationRule: Sendable, Hashable, Codable {
    /// The feature transformed by this rule.
    public let feature: FeatureName
    /// The reversible transformation to fit.
    public let strategy: NormalizationStrategy

    /// Creates a named normalization rule.
    public init(feature: FeatureName, strategy: NormalizationStrategy) {
        self.feature = feature
        self.strategy = strategy
    }
}

/// Fitted population statistics for one feature's numeric elements.
@frozen
public struct NumericFeatureStatistics: Sendable, Hashable, Codable {
    /// Number of scalar elements observed while fitting.
    public let count: Int
    /// Smallest observed value.
    public let minimum: Double
    /// Largest observed value.
    public let maximum: Double
    /// Arithmetic population mean.
    public let mean: Double
    /// Population variance.
    public let variance: Double

    /// Population standard deviation.
    public var standardDeviation: Double {
        variance.squareRoot()
    }

    /// Creates validated fitted statistics.
    ///
    /// - Throws: ``ML5Error/invalidNormalization(reason:)`` when the count, finite
    ///   values, ordering, or variance is invalid.
    public init(
        count: Int,
        minimum: Double,
        maximum: Double,
        mean: Double,
        variance: Double
    ) throws {
        guard
            count > 0,
            minimum.isFinite,
            maximum.isFinite,
            mean.isFinite,
            variance.isFinite,
            minimum <= mean,
            mean <= maximum,
            variance >= 0
        else {
            throw ML5Error.invalidNormalization(
                reason: "Fitted statistics must be nonempty, finite, ordered, and nonnegative."
            )
        }
        self.count = count
        self.minimum = minimum
        self.maximum = maximum
        self.mean = mean
        self.variance = variance
    }

    /// Decodes statistics and revalidates every invariant.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            count: container.decode(Int.self, forKey: .count),
            minimum: container.decode(Double.self, forKey: .minimum),
            maximum: container.decode(Double.self, forKey: .maximum),
            mean: container.decode(Double.self, forKey: .mean),
            variance: container.decode(Double.self, forKey: .variance)
        )
    }

    /// Encodes the complete fitted population summary.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(count, forKey: .count)
        try container.encode(minimum, forKey: .minimum)
        try container.encode(maximum, forKey: .maximum)
        try container.encode(mean, forKey: .mean)
        try container.encode(variance, forKey: .variance)
    }

    static func fit(_ values: [Double], feature: FeatureName) throws -> Self {
        guard let first = values.first else {
            throw ML5Error.invalidNormalization(
                reason: "Feature \(feature.rawValue.debugDescription) contains no numeric values."
            )
        }
        var minimum = first
        var maximum = first
        var mean = 0.0
        var squaredDeviation = 0.0
        for (offset, value) in values.enumerated() {
            minimum = Swift.min(minimum, value)
            maximum = Swift.max(maximum, value)
            let count = Double(offset + 1)
            let delta = value - mean
            mean += delta / count
            squaredDeviation += delta * (value - mean)
            guard mean.isFinite, squaredDeviation.isFinite else {
                throw ML5Error.invalidNormalization(
                    reason: "Statistics for \(feature.rawValue.debugDescription) overflowed."
                )
            }
        }
        return try Self(
            count: values.count,
            minimum: minimum,
            maximum: maximum,
            mean: mean,
            variance: squaredDeviation / Double(values.count)
        )
    }

    private enum CodingKeys: String, CodingKey {
        case count
        case minimum
        case maximum
        case mean
        case variance
    }
}

/// One fitted, reversible feature transformation.
@frozen
public struct FittedFeatureNormalization: Sendable, Hashable, Codable {
    /// The feature transformed by this stage.
    public let feature: FeatureName
    /// Value kind observed while fitting.
    public let valueKind: FeatureValueKind
    /// Strategy applied to each numeric element.
    public let strategy: NormalizationStrategy
    /// Population statistics fitted from training data.
    public let statistics: NumericFeatureStatistics

    /// Creates a fitted stage for a supported numeric value kind.
    ///
    /// Supported kinds are number, array, dictionary, and tensor.
    ///
    /// - Throws: ``ML5Error/invalidNormalization(reason:)`` for an unsupported kind.
    public init(
        feature: FeatureName,
        valueKind: FeatureValueKind,
        strategy: NormalizationStrategy,
        statistics: NumericFeatureStatistics
    ) throws {
        guard Self.supportedKinds.contains(valueKind) else {
            throw ML5Error.invalidNormalization(
                reason:
                    "Feature \(feature.rawValue.debugDescription) has unsupported \(valueKind.rawValue) values."
            )
        }
        self.feature = feature
        self.valueKind = valueKind
        self.strategy = strategy
        self.statistics = statistics
    }

    /// Decodes a fitted stage and revalidates its supported value kind.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            feature: container.decode(FeatureName.self, forKey: .feature),
            valueKind: container.decode(FeatureValueKind.self, forKey: .valueKind),
            strategy: container.decode(NormalizationStrategy.self, forKey: .strategy),
            statistics: container.decode(NumericFeatureStatistics.self, forKey: .statistics)
        )
    }

    /// Encodes the feature identity, kind, strategy, and fitted statistics.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(feature, forKey: .feature)
        try container.encode(valueKind, forKey: .valueKind)
        try container.encode(strategy, forKey: .strategy)
        try container.encode(statistics, forKey: .statistics)
    }

    fileprivate static let supportedKinds: Set<FeatureValueKind> = [
        .number, .array, .dictionary, .tensor,
    ]

    private enum CodingKeys: String, CodingKey {
        case feature
        case valueKind
        case strategy
        case statistics
    }
}

/// An immutable sequence of fitted, serializable feature transformations.
@frozen
public struct FeaturePreprocessingPipeline: Sendable, Hashable, Codable {
    /// Fitted stages in deterministic application order.
    public let normalizations: [FittedFeatureNormalization]

    /// Creates a nonempty pipeline with at most one stage per feature.
    ///
    /// - Throws: ``ML5Error/invalidNormalization(reason:)`` for an empty pipeline or
    ///   duplicate feature.
    public init(normalizations: [FittedFeatureNormalization]) throws {
        guard !normalizations.isEmpty else {
            throw ML5Error.invalidNormalization(
                reason: "A preprocessing pipeline requires at least one normalization."
            )
        }
        var names: Set<FeatureName> = []
        for normalization in normalizations where !names.insert(normalization.feature).inserted {
            throw ML5Error.invalidNormalization(
                reason:
                    "Feature \(normalization.feature.rawValue.debugDescription) is normalized more than once."
            )
        }
        self.normalizations = normalizations
    }

    /// Fits ordered normalization rules from feature vectors.
    ///
    /// - Parameters:
    ///   - samples: Nonempty training feature vectors.
    ///   - rules: Nonempty, uniquely named normalization rules.
    /// - Returns: An immutable pipeline containing one fitted stage per rule.
    /// - Throws: ``ML5Error`` when data is missing, inconsistent, unsupported, or
    ///   numerically unstable.
    public static func fit(
        samples: [FeatureVector],
        rules: [FeatureNormalizationRule]
    ) throws -> Self {
        guard !samples.isEmpty else {
            throw ML5Error.invalidTrainingSamples
        }
        guard !rules.isEmpty else {
            throw ML5Error.invalidNormalization(
                reason: "Fitting requires at least one normalization rule."
            )
        }
        var ruleNames: Set<FeatureName> = []
        for rule in rules where !ruleNames.insert(rule.feature).inserted {
            throw ML5Error.invalidNormalization(
                reason: "Feature \(rule.feature.rawValue.debugDescription) has duplicate rules."
            )
        }

        let normalizations = try rules.map { rule in
            guard let firstValue = samples[0][rule.feature] else {
                throw ML5Error.missingFeature(rule.feature.rawValue)
            }
            let kind = firstValue.kind
            var values = try components(of: firstValue, feature: rule.feature)
            for sample in samples.dropFirst() {
                guard let value = sample[rule.feature] else {
                    throw ML5Error.missingFeature(rule.feature.rawValue)
                }
                guard value.kind == kind else {
                    throw ML5Error.featureKindMismatch(
                        name: rule.feature.rawValue,
                        expected: kind,
                        actual: value.kind
                    )
                }
                values.append(contentsOf: try components(of: value, feature: rule.feature))
            }
            return try FittedFeatureNormalization(
                feature: rule.feature,
                valueKind: kind,
                strategy: rule.strategy,
                statistics: NumericFeatureStatistics.fit(values, feature: rule.feature)
            )
        }
        return try Self(normalizations: normalizations)
    }

    /// Fits rules from classification samples without changing their labels.
    ///
    /// - Returns: An immutable pipeline fitted from the samples' feature vectors.
    /// - Throws: The same validation failures as ``fit(samples:rules:)``.
    public static func fit<Label: ClassificationLabel>(
        samples: [ClassificationSample<Label>],
        rules: [FeatureNormalizationRule]
    ) throws -> Self {
        try fit(samples: samples.map(\.features), rules: rules)
    }

    /// Fits rules from scalar-regression samples without using their targets.
    ///
    /// - Returns: An immutable pipeline fitted from the samples' feature vectors.
    /// - Throws: The same validation failures as ``fit(samples:rules:)``.
    public static func fit(
        samples: [RegressionSample],
        rules: [FeatureNormalizationRule]
    ) throws -> Self {
        try fit(samples: samples.map(\.features), rules: rules)
    }

    /// Applies every fitted normalization while preserving unconfigured features.
    ///
    /// - Throws: ``ML5Error`` for missing, mismatched, unsupported, or overflowing values.
    public func normalize(_ features: FeatureVector) throws -> FeatureVector {
        try applying(features, direction: .normalize)
    }

    /// Reverses every fitted normalization while preserving unconfigured features.
    ///
    /// - Throws: ``ML5Error`` for missing, mismatched, unsupported, or overflowing values.
    public func denormalize(_ features: FeatureVector) throws -> FeatureVector {
        try applying(features, direction: .denormalize)
    }

    /// Normalizes a classification sample while preserving its label.
    ///
    /// - Throws: The same validation failures as ``normalize(_:)``.
    public func normalize<Label: ClassificationLabel>(
        _ sample: ClassificationSample<Label>
    ) throws -> ClassificationSample<Label> {
        ClassificationSample(features: try normalize(sample.features), label: sample.label)
    }

    /// Normalizes a regression sample while preserving its target.
    ///
    /// - Throws: The same validation failures as ``normalize(_:)``.
    public func normalize(_ sample: RegressionSample) throws -> RegressionSample {
        try RegressionSample(features: normalize(sample.features), target: sample.target)
    }

    /// Decodes stages and revalidates pipeline uniqueness and nonemptiness.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        try self.init(
            normalizations: container.decode([FittedFeatureNormalization].self)
        )
    }

    /// Encodes stages in deterministic application order.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(normalizations)
    }

    enum Direction {
        case normalize
        case denormalize
    }

    private func applying(
        _ features: FeatureVector,
        direction: Direction
    ) throws -> FeatureVector {
        var values = features.values
        for normalization in normalizations {
            guard let value = values[normalization.feature] else {
                throw ML5Error.missingFeature(normalization.feature.rawValue)
            }
            guard value.kind == normalization.valueKind else {
                throw ML5Error.featureKindMismatch(
                    name: normalization.feature.rawValue,
                    expected: normalization.valueKind,
                    actual: value.kind
                )
            }
            values[normalization.feature] = try Self.transform(
                value,
                using: normalization,
                direction: direction
            )
        }
        return try FeatureVector(values)
    }

    static func components(
        of value: FeatureValue,
        feature: FeatureName
    ) throws -> [Double] {
        switch value {
        case let .number(value):
            [value]
        case let .array(values):
            values
        case let .dictionary(values):
            values.sorted { $0.key < $1.key }.map(\.value)
        case let .tensor(tensor):
            tensor.values
        case .integer, .string, .boolean, .sequence, .image:
            throw ML5Error.invalidNormalization(
                reason:
                    "Feature \(feature.rawValue.debugDescription) has unsupported \(value.kind.rawValue) values."
            )
        }
    }

    static func transform(
        _ value: FeatureValue,
        using normalization: FittedFeatureNormalization,
        direction: Direction
    ) throws -> FeatureValue {
        let transformValue: (Double) throws -> Double = { value in
            let result = try Self.transformed(
                value,
                normalization: normalization,
                direction: direction
            )
            guard result.isFinite else {
                throw ML5Error.invalidNormalization(
                    reason:
                        "Transforming \(normalization.feature.rawValue.debugDescription) overflowed."
                )
            }
            return result
        }

        switch value {
        case let .number(value):
            return .number(try transformValue(value))
        case let .array(values):
            return .array(try values.map(transformValue))
        case let .dictionary(values):
            return .dictionary(try values.mapValues(transformValue))
        case let .tensor(tensor):
            return .tensor(
                try Tensor(shape: tensor.shape, values: tensor.values.map(transformValue))
            )
        case .integer, .string, .boolean, .sequence, .image:
            throw ML5Error.invalidNormalization(
                reason:
                    "Feature \(normalization.feature.rawValue.debugDescription) has unsupported values."
            )
        }
    }

    private static func transformed(
        _ value: Double,
        normalization: FittedFeatureNormalization,
        direction: Direction
    ) throws -> Double {
        let statistics = normalization.statistics
        switch (normalization.strategy, direction) {
        case (.standardScore, .normalize):
            let deviation = statistics.standardDeviation
            return deviation == 0 ? 0 : (value - statistics.mean) / deviation
        case (.standardScore, .denormalize):
            return value * statistics.standardDeviation + statistics.mean
        case let (.minMax(range), .normalize):
            let inputWidth = statistics.maximum - statistics.minimum
            guard inputWidth != 0 else {
                return (range.lowerBound + range.upperBound) / 2
            }
            let proportion = (value - statistics.minimum) / inputWidth
            return range.lowerBound + proportion * (range.upperBound - range.lowerBound)
        case let (.minMax(range), .denormalize):
            let inputWidth = statistics.maximum - statistics.minimum
            guard inputWidth != 0 else { return statistics.minimum }
            let proportion =
                (value - range.lowerBound) / (range.upperBound - range.lowerBound)
            return statistics.minimum + proportion * inputWidth
        }
    }
}
