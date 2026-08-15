import Foundation

/// A validated name for a Core ML model input.
public struct FeatureName: RawRepresentable, Hashable, Codable, Sendable, ExpressibleByStringLiteral
{
    /// The nonempty, whitespace-trimmed Core ML input name.
    public let rawValue: String

    /// Creates and validates an input name from runtime text.
    ///
    /// - Throws: ``ML5Error/invalidFieldName(_:)`` for empty or untrimmed text.
    public init(_ rawValue: String) throws {
        try Self.validate(rawValue)
        self.rawValue = rawValue
    }

    /// Creates an input name for `RawRepresentable` conformance.
    ///
    /// - Precondition: `rawValue` is nonempty and contains no surrounding whitespace.
    public init(rawValue: String) {
        precondition((try? Self.validate(rawValue)) != nil)
        self.rawValue = rawValue
    }

    /// Creates an input name from a valid string literal.
    ///
    /// - Precondition: `value` is nonempty and contains no surrounding whitespace.
    public init(stringLiteral value: String) {
        self.init(rawValue: value)
    }

    /// Decodes and validates a name from its string representation.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        try self.init(container.decode(String.self))
    }

    /// Encodes the validated raw string.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }

    private static func validate(_ value: String) throws {
        guard value.isEmpty == false, value == value.trimmingCharacters(in: .whitespacesAndNewlines)
        else {
            throw ML5Error.invalidFieldName(value)
        }
    }
}

/// A validated name for a Core ML model output.
public struct OutputName: RawRepresentable, Hashable, Codable, Sendable, ExpressibleByStringLiteral
{
    /// The nonempty, whitespace-trimmed Core ML output name.
    public let rawValue: String

    /// Creates and validates an output name from runtime text.
    ///
    /// - Throws: ``ML5Error/invalidFieldName(_:)`` for empty or untrimmed text.
    public init(_ rawValue: String) throws {
        try Self.validate(rawValue)
        self.rawValue = rawValue
    }

    /// Creates an output name for `RawRepresentable` conformance.
    ///
    /// - Precondition: `rawValue` is nonempty and contains no surrounding whitespace.
    public init(rawValue: String) {
        precondition((try? Self.validate(rawValue)) != nil)
        self.rawValue = rawValue
    }

    /// Creates an output name from a valid string literal.
    ///
    /// - Precondition: `value` is nonempty and contains no surrounding whitespace.
    public init(stringLiteral value: String) {
        self.init(rawValue: value)
    }

    /// Decodes and validates a name from its string representation.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        try self.init(container.decode(String.self))
    }

    /// Encodes the validated raw string.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }

    private static func validate(_ value: String) throws {
        guard value.isEmpty == false, value == value.trimmingCharacters(in: .whitespacesAndNewlines)
        else {
            throw ML5Error.invalidFieldName(value)
        }
    }
}

/// Value kinds supported by ML5's framework-independent model boundary.
public enum FeatureValueKind: String, Codable, Sendable, Hashable {
    /// A double-precision floating-point scalar.
    case number
    /// A signed 64-bit integer scalar.
    case integer
    /// A Unicode string scalar.
    case string
    /// A Boolean represented as zero or one at the Core ML boundary.
    case boolean
    /// A finite one-dimensional numeric array.
    case array
    /// A finite string-keyed numeric dictionary.
    case dictionary
    /// A dense finite tensor with explicit dimensions.
    case tensor
    /// A homogeneous string or integer sequence.
    case sequence
    /// Immutable grayscale, RGBA, or BGRA image pixels.
    case image
}

/// A transport-safe feature or model-output value.
public enum FeatureValue: Codable, Sendable, Hashable {
    /// A finite double-precision value when used in validated vectors and outputs.
    case number(Double)
    /// A signed 64-bit integer.
    case integer(Int64)
    /// A Unicode string.
    case string(String)
    /// A Boolean value converted to a Core ML integer at prediction time.
    case boolean(Bool)
    /// A finite nonempty one-dimensional numeric array.
    case array([Double])
    /// A finite nonempty string-keyed numeric dictionary.
    case dictionary([String: Double])
    /// A dense finite tensor.
    case tensor(Tensor)
    /// A homogeneous string or integer sequence.
    case sequence(FeatureSequence)
    /// Immutable image pixels.
    case image(ML5Image)

    /// The scalar kind represented by this value.
    public var kind: FeatureValueKind {
        switch self {
        case .number:
            .number
        case .integer:
            .integer
        case .string:
            .string
        case .boolean:
            .boolean
        case .array:
            .array
        case .dictionary:
            .dictionary
        case .tensor:
            .tensor
        case .sequence:
            .sequence
        case .image:
            .image
        }
    }

    /// Returns a numeric representation for number and integer values.
    public var numericValue: Double? {
        switch self {
        case let .number(value):
            value
        case let .integer(value):
            Double(value)
        case .string, .boolean, .array, .dictionary, .tensor, .sequence, .image:
            nil
        }
    }

    /// Decodes a tagged value and revalidates its collection invariants.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let kind = try container.decode(FeatureValueKind.self, forKey: .kind)
        self =
            switch kind {
            case .number:
                .number(try container.decode(Double.self, forKey: .number))
            case .integer:
                .integer(try container.decode(Int64.self, forKey: .integer))
            case .string:
                .string(try container.decode(String.self, forKey: .string))
            case .boolean:
                .boolean(try container.decode(Bool.self, forKey: .boolean))
            case .array:
                .array(try container.decode([Double].self, forKey: .array))
            case .dictionary:
                .dictionary(try container.decode([String: Double].self, forKey: .dictionary))
            case .tensor:
                .tensor(try container.decode(Tensor.self, forKey: .tensor))
            case .sequence:
                .sequence(try container.decode(FeatureSequence.self, forKey: .sequence))
            case .image:
                .image(try container.decode(ML5Image.self, forKey: .image))
            }
        try validate(field: "value")
    }

    /// Encodes a stable kind tag and its associated value.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(kind, forKey: .kind)
        switch self {
        case let .number(value):
            try container.encode(value, forKey: .number)
        case let .integer(value):
            try container.encode(value, forKey: .integer)
        case let .string(value):
            try container.encode(value, forKey: .string)
        case let .boolean(value):
            try container.encode(value, forKey: .boolean)
        case let .array(value):
            try container.encode(value, forKey: .array)
        case let .dictionary(value):
            try container.encode(value, forKey: .dictionary)
        case let .tensor(value):
            try container.encode(value, forKey: .tensor)
        case let .sequence(value):
            try container.encode(value, forKey: .sequence)
        case let .image(value):
            try container.encode(value, forKey: .image)
        }
    }

    func validate(field: String) throws {
        switch self {
        case let .number(value):
            guard value.isFinite else { throw ML5Error.invalidNumericValue(field: field) }
        case let .array(values):
            guard !values.isEmpty else { throw ML5Error.emptyCollection(field: field) }
            guard values.allSatisfy(\.isFinite) else {
                throw ML5Error.invalidNumericValue(field: field)
            }
        case let .dictionary(values):
            guard !values.isEmpty else { throw ML5Error.emptyCollection(field: field) }
            guard
                values.keys.allSatisfy({
                    !$0.isEmpty && $0 == $0.trimmingCharacters(in: .whitespacesAndNewlines)
                })
            else {
                throw ML5Error.invalidDictionaryKey(field: field)
            }
            guard values.values.allSatisfy(\.isFinite) else {
                throw ML5Error.invalidNumericValue(field: field)
            }
        case .integer, .string, .boolean, .tensor, .sequence, .image:
            break
        }
    }

    private enum CodingKeys: String, CodingKey {
        case kind
        case number
        case integer
        case string
        case boolean
        case array
        case dictionary
        case tensor
        case sequence
        case image
    }
}

/// A non-empty, validated feature vector.
public struct FeatureVector: Sendable, Hashable, Codable {
    /// Validated feature values keyed by stable model input names.
    public let values: [FeatureName: FeatureValue]

    /// Creates a nonempty vector and rejects nonfinite numeric values.
    ///
    /// - Throws: ``ML5Error/emptyFeatureVector`` or
    ///   ``ML5Error/invalidNumericValue(field:)``.
    public init(_ values: [FeatureName: FeatureValue]) throws {
        guard values.isEmpty == false else {
            throw ML5Error.emptyFeatureVector
        }

        for (name, value) in values {
            try value.validate(field: name.rawValue)
        }
        self.values = values
    }

    /// Decodes string-keyed values and revalidates every name and value.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        let decoded = try container.decode([String: FeatureValue].self)
        try self.init(
            Dictionary(
                uniqueKeysWithValues: decoded.map { key, value in
                    (try FeatureName(key), value)
                })
        )
    }

    /// Encodes values as a human-readable string-keyed dictionary.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(
            Dictionary(uniqueKeysWithValues: values.map { ($0.key.rawValue, $0.value) })
        )
    }

    /// Returns the value for an input name, or `nil` when it is absent.
    public subscript(_ name: FeatureName) -> FeatureValue? {
        values[name]
    }
}

/// A non-empty, validated set of scalar model outputs.
public struct ModelOutput: Sendable, Hashable, Codable {
    /// Validated scalar values keyed by stable model output names.
    public let values: [OutputName: FeatureValue]

    /// Creates a nonempty output and rejects nonfinite numeric values.
    ///
    /// - Throws: ``ML5Error/emptyModelOutput`` or
    ///   ``ML5Error/invalidNumericValue(field:)``.
    public init(_ values: [OutputName: FeatureValue]) throws {
        guard values.isEmpty == false else {
            throw ML5Error.emptyModelOutput
        }

        for (name, value) in values {
            try value.validate(field: name.rawValue)
        }
        self.values = values
    }

    /// Decodes string-keyed values and revalidates every name and value.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        let decoded = try container.decode([String: FeatureValue].self)
        try self.init(
            Dictionary(
                uniqueKeysWithValues: decoded.map { key, value in
                    (try OutputName(key), value)
                })
        )
    }

    /// Encodes values as a human-readable string-keyed dictionary.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(
            Dictionary(uniqueKeysWithValues: values.map { ($0.key.rawValue, $0.value) })
        )
    }

    /// Returns the value for an output name, or `nil` when it is absent.
    public subscript(_ name: OutputName) -> FeatureValue? {
        values[name]
    }
}
