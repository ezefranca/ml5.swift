/// One ordered input declaration in a ``FeatureSchema``.
@frozen
public struct FeatureField: Sendable, Hashable, Codable {
    /// The stable model input name.
    public let name: FeatureName
    /// The required value kind.
    public let kind: FeatureValueKind
    /// The exact required shape for tensor values.
    public let tensorShape: TensorShape?
    /// Whether absence remains an error when no default is available.
    public let isRequired: Bool
    /// A validated value inserted by default-resolving policies.
    public let defaultValue: FeatureValue?

    /// Creates one validated schema field.
    ///
    /// - Throws: ``ML5Error/invalidConfiguration(reason:)`` when shape or default
    ///   metadata contradicts the declared kind.
    public init(
        name: FeatureName,
        kind: FeatureValueKind,
        tensorShape: TensorShape? = nil,
        isRequired: Bool = true,
        defaultValue: FeatureValue? = nil
    ) throws {
        guard (kind == .tensor) == (tensorShape != nil) else {
            throw ML5Error.invalidConfiguration(
                reason: "Only tensor fields require an exact tensor shape."
            )
        }
        if let defaultValue {
            guard defaultValue.kind == kind else {
                throw ML5Error.featureKindMismatch(
                    name: name.rawValue,
                    expected: kind,
                    actual: defaultValue.kind
                )
            }
            try defaultValue.validate(field: name.rawValue)
            if case let .tensor(tensor) = defaultValue,
                let tensorShape,
                tensor.shape != tensorShape
            {
                throw ML5Error.tensorShapeMismatch(
                    name: name.rawValue,
                    expected: tensorShape.dimensions,
                    actual: tensor.shape.dimensions
                )
            }
        }
        self.name = name
        self.kind = kind
        self.tensorShape = tensorShape
        self.isRequired = isRequired
        self.defaultValue = defaultValue
    }

    /// Decodes field metadata and revalidates its kind, shape, and default.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            name: container.decode(FeatureName.self, forKey: .name),
            kind: container.decode(FeatureValueKind.self, forKey: .kind),
            tensorShape: container.decodeIfPresent(TensorShape.self, forKey: .tensorShape),
            isRequired: container.decode(Bool.self, forKey: .isRequired),
            defaultValue: container.decodeIfPresent(FeatureValue.self, forKey: .defaultValue)
        )
    }

    /// Encodes the complete field declaration.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        try container.encode(kind, forKey: .kind)
        try container.encodeIfPresent(tensorShape, forKey: .tensorShape)
        try container.encode(isRequired, forKey: .isRequired)
        try container.encodeIfPresent(defaultValue, forKey: .defaultValue)
    }

    private enum CodingKeys: String, CodingKey {
        case name
        case kind
        case tensorShape
        case isRequired
        case defaultValue
    }
}

/// How absent schema fields are handled during feature resolution.
@frozen
public enum MissingFeaturePolicy: String, Sendable, Hashable, Codable {
    /// Every schema field must be present in the supplied vector.
    case reject
    /// Insert declared defaults, omit optional fields, and reject unresolved required fields.
    case useDefaults
}

/// How names absent from an ordered schema are handled.
@frozen
public enum UnknownFeaturePolicy: String, Sendable, Hashable, Codable {
    /// Reject the first unexpected name in stable lexical order.
    case reject
    /// Preserve additional validated values in the resolved vector.
    case preserve
}

/// Ordered model-input declarations with deterministic resolution policies.
@frozen
public struct FeatureSchema: Sendable, Hashable, Codable {
    /// Fields in model input order.
    public let fields: [FeatureField]

    /// Creates a nonempty schema with unique names.
    ///
    /// - Throws: ``ML5Error/invalidConfiguration(reason:)`` or
    ///   ``ML5Error/duplicateFeatureName(_:)``.
    public init(_ fields: [FeatureField]) throws {
        guard !fields.isEmpty else {
            throw ML5Error.invalidConfiguration(reason: "A feature schema cannot be empty.")
        }
        var names: Set<FeatureName> = []
        for field in fields where !names.insert(field.name).inserted {
            throw ML5Error.duplicateFeatureName(field.name.rawValue)
        }
        self.fields = fields
    }

    /// Decodes ordered fields and revalidates uniqueness and nonemptiness.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        try self.init(container.decode([FeatureField].self))
    }

    /// Encodes fields in declared model order.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(fields)
    }

    /// Input names in declared model order.
    public var names: [FeatureName] {
        fields.map(\.name)
    }

    /// Validates, defaults, and optionally preserves a supplied feature vector.
    ///
    /// - Parameters:
    ///   - features: The values to resolve.
    ///   - missing: How absent schema fields are handled.
    ///   - unknown: How values not declared by the schema are handled.
    /// - Returns: A new immutable feature vector after policy application.
    /// - Throws: A schema validation error when the supplied values do not satisfy
    ///   the selected policies.
    public func resolve(
        _ features: FeatureVector,
        missing: MissingFeaturePolicy = .reject,
        unknown: UnknownFeaturePolicy = .reject
    ) throws -> FeatureVector {
        let declaredNames = Set(names)
        let unknownNames = features.values.keys
            .filter { !declaredNames.contains($0) }
            .sorted { $0.rawValue < $1.rawValue }
        if unknown == .reject, let first = unknownNames.first {
            throw ML5Error.unexpectedFeature(first.rawValue)
        }

        var resolved: [FeatureName: FeatureValue] = [:]
        for field in fields {
            if let value = features[field.name] {
                try validate(value, for: field)
                resolved[field.name] = value
            } else {
                switch missing {
                case .reject:
                    throw ML5Error.missingFeature(field.name.rawValue)
                case .useDefaults:
                    if let defaultValue = field.defaultValue {
                        resolved[field.name] = defaultValue
                    } else if field.isRequired {
                        throw ML5Error.missingFeature(field.name.rawValue)
                    }
                }
            }
        }
        if unknown == .preserve {
            for name in unknownNames {
                resolved[name] = features[name]
            }
        }
        return try FeatureVector(resolved)
    }

    /// Returns resolved values in declared model order.
    ///
    /// - Throws: A schema validation error when the supplied values do not satisfy
    ///   the selected policies.
    public func orderedValues(
        in features: FeatureVector,
        missing: MissingFeaturePolicy = .reject,
        unknown: UnknownFeaturePolicy = .reject
    ) throws -> [FeatureValue] {
        let resolved = try resolve(features, missing: missing, unknown: unknown)
        return fields.compactMap { resolved[$0.name] }
    }

    private func validate(_ value: FeatureValue, for field: FeatureField) throws {
        guard value.kind == field.kind else {
            throw ML5Error.featureKindMismatch(
                name: field.name.rawValue,
                expected: field.kind,
                actual: value.kind
            )
        }
        try value.validate(field: field.name.rawValue)
        if case let .tensor(tensor) = value,
            let tensorShape = field.tensorShape,
            tensor.shape != tensorShape
        {
            throw ML5Error.tensorShapeMismatch(
                name: field.name.rawValue,
                expected: tensorShape.dimensions,
                actual: tensor.shape.dimensions
            )
        }
    }
}
