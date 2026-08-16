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
    /// A tensor shape was empty, nonpositive, or overflowed its element count.
    case invalidTensorShape([Int])
    /// Tensor storage did not match the declared shape.
    case invalidTensorElementCount(expected: Int, actual: Int)
    /// An array or dictionary feature was empty.
    case emptyCollection(field: String)
    /// A numeric dictionary contained an empty or untrimmed key.
    case invalidDictionaryKey(field: String)
    /// Image dimensions, row storage, format, or data size was unsupported.
    case invalidImage(reason: String)
    /// An ordered schema declared the same feature name more than once.
    case duplicateFeatureName(String)
    /// A required schema feature was absent.
    case missingFeature(String)
    /// A vector contained a feature absent from its strict schema.
    case unexpectedFeature(String)
    /// A value kind did not match its schema field.
    case featureKindMismatch(
        name: String,
        expected: FeatureValueKind,
        actual: FeatureValueKind
    )
    /// A tensor's dimensions did not match its schema field.
    case tensorShapeMismatch(name: String, expected: [Int], actual: [Int])
    /// A dataset sample identifier was zero.
    case invalidDatasetSampleID(UInt64)
    /// A persisted dataset contained contradictory identity metadata.
    case invalidDatasetSnapshot(reason: String)
    /// A dataset could not allocate another stable sample identifier.
    case datasetIdentifierExhausted
    /// Dataset split fractions were nonfinite, negative, or totaled more than one.
    case invalidDatasetSplit(reason: String)
    /// A preprocessing configuration, fitted statistic, or transformation was invalid.
    case invalidNormalization(reason: String)
    /// A batch backend returned a different number of outputs than inputs.
    case batchPredictionCountMismatch(expected: Int, actual: Int)
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
        case let .invalidTensorShape(dimensions):
            "Tensor dimensions must be nonempty, positive, and representable: \(dimensions)."
        case let .invalidTensorElementCount(expected, actual):
            "Tensor storage requires \(expected) elements, but received \(actual)."
        case let .emptyCollection(field):
            "The collection feature \(field.debugDescription) cannot be empty."
        case let .invalidDictionaryKey(field):
            "Dictionary keys for \(field.debugDescription) must be nonempty and trimmed."
        case let .invalidImage(reason):
            "Invalid image feature: \(reason)"
        case let .duplicateFeatureName(name):
            "The feature schema declares \(name.debugDescription) more than once."
        case let .missingFeature(name):
            "The required feature \(name.debugDescription) is missing."
        case let .unexpectedFeature(name):
            "The feature \(name.debugDescription) is not declared by the schema."
        case let .featureKindMismatch(name, expected, actual):
            "Feature \(name.debugDescription) must be \(expected.rawValue), but was \(actual.rawValue)."
        case let .tensorShapeMismatch(name, expected, actual):
            "Tensor feature \(name.debugDescription) requires shape \(expected), but received \(actual)."
        case let .invalidDatasetSampleID(rawValue):
            "Dataset sample identifiers must be positive, but received \(rawValue)."
        case let .invalidDatasetSnapshot(reason):
            "Invalid dataset snapshot: \(reason)"
        case .datasetIdentifierExhausted:
            "The dataset exhausted its stable sample identifier space."
        case let .invalidDatasetSplit(reason):
            "Invalid dataset split: \(reason)"
        case let .invalidNormalization(reason):
            "Invalid normalization: \(reason)"
        case let .batchPredictionCountMismatch(expected, actual):
            "Batch prediction requires \(expected) outputs, but received \(actual)."
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
    /// The selected backend cannot provide immutable synchronous inference.
    case synchronousInferenceSnapshot

    /// A description of why the operation is unavailable and which extension point to use.
    public var description: String {
        switch self {
        case .onDeviceTraining:
            "On-device training for arbitrary models is not supported by Core ML model loading. Supply a future Create ML-compatible training adapter instead."
        case .synchronousInferenceSnapshot:
            "This prediction backend does not provide an immutable synchronous inference snapshot. Use async prediction or select a snapshot-capable backend."
        }
    }
}
