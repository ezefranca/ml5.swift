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

    private static func validate(_ value: String) throws {
        guard value.isEmpty == false, value == value.trimmingCharacters(in: .whitespacesAndNewlines)
        else {
            throw ML5Error.invalidFieldName(value)
        }
    }
}

/// The scalar value kinds currently supported by ML5's Core ML bridge.
public enum FeatureValueKind: String, Codable, Sendable, Equatable {
    /// A double-precision floating-point scalar.
    case number
    /// A signed 64-bit integer scalar.
    case integer
    /// A Unicode string scalar.
    case string
    /// A Boolean represented as zero or one at the Core ML boundary.
    case boolean
}

/// A transport-safe scalar feature or model-output value.
///
/// Image, tensor, sequence, and dictionary features are intentionally outside this
/// first vertical slice. Keeping this type value-based makes it safe to pass through
/// actors without retaining Core ML framework objects.
public enum FeatureValue: Codable, Sendable, Equatable {
    /// A finite double-precision value when used in validated vectors and outputs.
    case number(Double)
    /// A signed 64-bit integer.
    case integer(Int64)
    /// A Unicode string.
    case string(String)
    /// A Boolean value converted to a Core ML integer at prediction time.
    case boolean(Bool)

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
        }
    }

    /// Returns a numeric representation for number and integer values.
    public var numericValue: Double? {
        switch self {
        case let .number(value):
            value
        case let .integer(value):
            Double(value)
        case .string, .boolean:
            nil
        }
    }

    func validate(field: String) throws {
        if case let .number(value) = self, value.isFinite == false {
            throw ML5Error.invalidNumericValue(field: field)
        }
    }
}

/// A non-empty, validated feature vector.
public struct FeatureVector: Sendable, Equatable {
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

    /// Returns the value for an input name, or `nil` when it is absent.
    public subscript(_ name: FeatureName) -> FeatureValue? {
        values[name]
    }
}

/// A non-empty, validated set of scalar model outputs.
public struct ModelOutput: Sendable, Equatable {
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

    /// Returns the value for an output name, or `nil` when it is absent.
    public subscript(_ name: OutputName) -> FeatureValue? {
        values[name]
    }
}
