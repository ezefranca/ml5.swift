import Foundation

/// A validated name for a Core ML model input.
public struct FeatureName: RawRepresentable, Hashable, Codable, Sendable, ExpressibleByStringLiteral {
    public let rawValue: String

    public init(_ rawValue: String) throws {
        try Self.validate(rawValue)
        self.rawValue = rawValue
    }

    public init(rawValue: String) {
        precondition((try? Self.validate(rawValue)) != nil, "FeatureName must be non-empty and trimmed.")
        self.rawValue = rawValue
    }

    public init(stringLiteral value: String) {
        self.init(rawValue: value)
    }

    private static func validate(_ value: String) throws {
        guard value.isEmpty == false, value == value.trimmingCharacters(in: .whitespacesAndNewlines) else {
            throw ML5Error.invalidFieldName(value)
        }
    }
}

/// A validated name for a Core ML model output.
public struct OutputName: RawRepresentable, Hashable, Codable, Sendable, ExpressibleByStringLiteral {
    public let rawValue: String

    public init(_ rawValue: String) throws {
        try Self.validate(rawValue)
        self.rawValue = rawValue
    }

    public init(rawValue: String) {
        precondition((try? Self.validate(rawValue)) != nil, "OutputName must be non-empty and trimmed.")
        self.rawValue = rawValue
    }

    public init(stringLiteral value: String) {
        self.init(rawValue: value)
    }

    private static func validate(_ value: String) throws {
        guard value.isEmpty == false, value == value.trimmingCharacters(in: .whitespacesAndNewlines) else {
            throw ML5Error.invalidFieldName(value)
        }
    }
}

/// The scalar value kinds currently supported by ML5's Core ML bridge.
public enum FeatureValueKind: String, Codable, Sendable, Equatable {
    case number
    case integer
    case string
    case boolean
}

/// A transport-safe scalar feature or model-output value.
///
/// Image, tensor, sequence, and dictionary features are intentionally outside this
/// first vertical slice. Keeping this type value-based makes it safe to pass through
/// actors without retaining Core ML framework objects.
public enum FeatureValue: Codable, Sendable, Equatable {
    case number(Double)
    case integer(Int64)
    case string(String)
    case boolean(Bool)

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
    public let values: [FeatureName: FeatureValue]

    public init(_ values: [FeatureName: FeatureValue]) throws {
        guard values.isEmpty == false else {
            throw ML5Error.emptyFeatureVector
        }

        for (name, value) in values {
            try value.validate(field: name.rawValue)
        }
        self.values = values
    }

    public subscript(_ name: FeatureName) -> FeatureValue? {
        values[name]
    }
}

/// A non-empty, validated set of scalar model outputs.
public struct ModelOutput: Sendable, Equatable {
    public let values: [OutputName: FeatureValue]

    public init(_ values: [OutputName: FeatureValue]) throws {
        guard values.isEmpty == false else {
            throw ML5Error.emptyModelOutput
        }

        for (name, value) in values {
            try value.validate(field: name.rawValue)
        }
        self.values = values
    }

    public subscript(_ name: OutputName) -> FeatureValue? {
        values[name]
    }
}
