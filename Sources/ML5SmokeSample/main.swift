import Foundation
import ML5

@main
struct ML5SmokeSample {
    static func main() async throws {
        let samples = try (-2...2).map { value in
            try DenseTrainingSample(
                features: FeatureVector(["x": .number(Double(value))]),
                targets: [(2 * Double(value)) + 1]
            )
        }
        let configuration = try DenseNetworkConfiguration(
            inputFeatures: ["x"],
            outputNames: ["y"],
            learningRate: 0.05,
            batchSize: 5,
            epochs: 40,
            validationFraction: 0,
            seed: 7
        )
        let result = try await DenseCPUTrainer().train(
            samples,
            configuration: configuration
        )
        let prediction = try await result.model.predict(
            FeatureVector(["x": .number(3)])
        )
        guard let value = prediction["y"]?.numericValue else {
            throw SmokeSampleError.missingPrediction
        }
        print("ML5 predicted \(value) after \(result.history.count) epochs.")
        if ProcessInfo.processInfo.environment["SWIFT_PACKAGE_INSTRUMENTS_HOLD"] == "1" {
            try await Task.sleep(for: .seconds(20))
        }
    }
}

private enum SmokeSampleError: Error {
    case missingPrediction
}
