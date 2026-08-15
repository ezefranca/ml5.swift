import Foundation

/// Errors produced by ML5 validation, decoding, and Core ML integration.
public enum ML5Error: Error, Equatable, Sendable, LocalizedError {
    /// A feature or output name was empty or contained surrounding whitespace.
    case invalidFieldName(String)
    /// A feature vector contained no values.
    case emptyFeatureVector
    /// A model returned no supported scalar values.
    case emptyModelOutput
    /// A numeric feature or output was NaN or infinite.
    case invalidNumericValue(field: String)
    /// A task configuration contained contradictory settings.
    case invalidConfiguration(reason: String)
    /// A training request contained no samples.
    case invalidTrainingSamples
    /// A task's required output was absent from the model result.
    case missingOutput(name: String)
    /// A model output's scalar kind did not match the task configuration.
    case unexpectedOutputType(name: String, expected: FeatureValueKind, actual: FeatureValueKind)
    /// A classification label could not be constructed from the model string.
    case invalidClassLabel(String)
    /// A confidence value was nonfinite or outside the closed range from zero to one.
    case invalidConfidence(Double)
    /// A regression target or prediction was nonfinite.
    case invalidRegressionValue(Double)
    /// The selected backend deliberately does not provide an operation.
    case unsupportedOperation(UnsupportedOperation)
    /// Core ML could not load the compiled model at the supplied path.
    case modelLoadingFailed(path: String, message: String)
    /// Core ML could not prepare inputs, execute, or convert the model result.
    case predictionFailed(message: String)

    /// A localized explanation suitable for logs and user-facing error presentation.
    public var errorDescription: String? {
        switch self {
        case let .invalidFieldName(name):
            "Field names must be non-empty and contain no leading or trailing whitespace: \(name.debugDescription)."
        case .emptyFeatureVector:
            "A feature vector must contain at least one feature."
        case .emptyModelOutput:
            "A model output must contain at least one value."
        case let .invalidNumericValue(field):
            "The numeric value for \(field.debugDescription) must be finite."
        case let .invalidConfiguration(reason):
            "Invalid configuration: \(reason)"
        case .invalidTrainingSamples:
            "Training requires at least one sample."
        case let .missingOutput(name):
            "The model did not provide the required output \(name.debugDescription)."
        case let .unexpectedOutputType(name, expected, actual):
            "Output \(name.debugDescription) must be \(expected.rawValue), but was \(actual.rawValue)."
        case let .invalidClassLabel(label):
            "The model label \(label.debugDescription) cannot be decoded by this classification task."
        case let .invalidConfidence(value):
            "Classification confidence must be finite and between 0 and 1, but was \(value)."
        case let .invalidRegressionValue(value):
            "A regression prediction must be finite, but was \(value)."
        case let .unsupportedOperation(operation):
            operation.description
        case let .modelLoadingFailed(path, message):
            "Unable to load the Core ML model at \(path.debugDescription): \(message)"
        case let .predictionFailed(message):
            "Core ML prediction failed: \(message)"
        }
    }
}

/// Operations intentionally unavailable in the current Core ML-backed foundation.
public enum UnsupportedOperation: String, Sendable, Equatable {
    /// Training arbitrary neural networks on device requires a separate training adapter.
    case onDeviceTraining

    /// A description of why the operation is unavailable and which extension point to use.
    public var description: String {
        switch self {
        case .onDeviceTraining:
            "On-device training for arbitrary models is not supported by Core ML model loading. Supply a future Create ML-compatible training adapter instead."
        }
    }
}
