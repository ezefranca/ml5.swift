import Foundation

/// Errors produced by ML5 validation, decoding, and Core ML integration.
public enum ML5Error: Error, Equatable, Sendable, LocalizedError {
    case invalidFieldName(String)
    case emptyFeatureVector
    case emptyModelOutput
    case invalidNumericValue(field: String)
    case invalidConfiguration(reason: String)
    case invalidTrainingSamples
    case missingOutput(name: String)
    case unexpectedOutputType(name: String, expected: FeatureValueKind, actual: FeatureValueKind)
    case invalidClassLabel(String)
    case invalidConfidence(Double)
    case invalidRegressionValue(Double)
    case unsupportedOperation(UnsupportedOperation)
    case modelLoadingFailed(path: String, message: String)
    case predictionFailed(message: String)

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

    public var description: String {
        switch self {
        case .onDeviceTraining:
            "On-device training for arbitrary models is not supported by Core ML model loading. Supply a future Create ML-compatible training adapter instead."
        }
    }
}
