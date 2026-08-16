import Foundation
import Testing

@testable import ML5

@Suite("ML5 dense neuroevolution")
struct ML5DenseNeuroevolutionTests {
    @Test("Topology, synchronous prediction, classification, and copies are value safe")
    func brainValues() async throws {
        let brain = try makeBrain()
        #expect(brain.generation == 0)
        #expect(brain.topology.inputFeatures == ["x", "y"])
        #expect(brain.topology.outputNames == ["negative", "positive"])
        #expect(brain.topology.hiddenLayers.isEmpty)
        #expect(brain.topology.outputActivation == .softmax)
        #expect(brain.topology.layerWidths == [2, 2])
        #expect(DenseBrainTopology(configuration: brain.model.configuration) == brain.topology)
        #expect(brain.copied() == brain)

        let features = try FeatureVector(["x": .number(1), "y": .number(1)])
        let synchronous = try brain.predict(features)
        let asynchronous = try await brain.model.predict(features)
        #expect(synchronous == asynchronous)
        let classification = try brain.classify(features)
        #expect(classification.label == "positive")
        #expect(classification.confidence > 0.98)
        #expect(abs(classification.scores.values.reduce(0, +) - 1) < 1e-12)

        let tied = try makeBrain(parameter: 0)
        #expect(try tied.classify(features).label == "negative")
        let linear = try makeBrain(outputActivation: .linear)
        #expect(throws: ML5Error.self) { _ = try linear.classify(features) }

        #expect(
            try JSONDecoder().decode(
                DenseBrain.self,
                from: JSONEncoder().encode(brain)
            ) == brain
        )
        #expect(
            try JSONDecoder().decode(
                DenseBrainTopology.self,
                from: JSONEncoder().encode(brain.topology)
            ) == brain.topology
        )
        #expect(throws: ML5Error.self) {
            _ = try DenseBrain(model: brain.model, generation: -1)
        }
        var object = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(brain)) as? [String: Any]
        )
        object["generation"] = -1
        #expect(throws: ML5Error.self) {
            _ = try JSONDecoder().decode(
                DenseBrain.self,
                from: JSONSerialization.data(withJSONObject: object)
            )
        }
    }

    @Test("Topology and classification values reject malformed state")
    func valueValidation() throws {
        let hidden = [
            try DenseLayerConfiguration(neuronCount: 3, activation: .hyperbolicTangent)
        ]
        #expect(throws: ML5Error.self) {
            _ = try DenseBrainTopology(
                inputFeatures: [],
                outputNames: ["value"],
                hiddenLayers: hidden,
                outputActivation: .linear
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseBrainTopology(
                inputFeatures: ["x", "x"],
                outputNames: ["value"],
                hiddenLayers: hidden,
                outputActivation: .linear
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseBrainTopology(
                inputFeatures: ["x"],
                outputNames: ["value", "value"],
                hiddenLayers: hidden,
                outputActivation: .linear
            )
        }
        let validTopology = try DenseBrainTopology(
            inputFeatures: ["x"],
            outputNames: ["value"],
            hiddenLayers: hidden,
            outputActivation: .linear
        )
        #expect(validTopology.layerWidths == [1, 3, 1])

        #expect(throws: ML5Error.self) {
            _ = try DenseBrainClassification(label: "a", scores: [:])
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseBrainClassification(label: "a", scores: ["a": -.infinity])
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseBrainClassification(label: "a", scores: ["a": -0.1, "b": 1.1])
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseBrainClassification(label: "missing", scores: ["a": 1])
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseBrainClassification(label: "a", scores: ["a": 0.2, "b": 0.2])
        }
        let classification = try DenseBrainClassification(
            label: "b",
            scores: ["a": 0.25, "b": 0.75]
        )
        #expect(classification.confidence == 0.75)
        #expect(
            try JSONDecoder().decode(
                DenseBrainClassification.self,
                from: JSONEncoder().encode(classification)
            ) == classification
        )
        #expect(
            ML5Error.invalidNeuroevolutionConfiguration(reason: "Failure.")
                .errorDescription
                == "Invalid neuroevolution configuration: Failure."
        )
        #expect(
            ML5Error.incompatibleBrainTopologies.errorDescription
                == "Dense brain topologies are incompatible."
        )
    }

    @Test("Mutation strategies are deterministic, bounded, and independently copied")
    func mutation() throws {
        for probability in [-0.1, 1.1, .infinity] {
            #expect(throws: ML5Error.self) {
                _ = try DenseMutationConfiguration(probability: probability)
            }
        }
        for magnitude in [-0.1, .infinity] {
            #expect(throws: ML5Error.self) {
                _ = try DenseMutationConfiguration(magnitude: magnitude)
            }
        }

        let parent = try makeBrain()
        let unchanged = try parent.mutated(
            using: DenseMutationConfiguration(probability: 0),
            seed: 1
        )
        #expect(unchanged.model == parent.model)
        #expect(unchanged.generation == 1)

        for strategy in DenseMutationStrategy.allCases {
            let configuration = try DenseMutationConfiguration(
                strategy: strategy,
                probability: 1,
                magnitude: 0.2,
                mutatesBiases: true
            )
            #expect(
                try JSONDecoder().decode(
                    DenseMutationConfiguration.self,
                    from: JSONEncoder().encode(configuration)
                ) == configuration
            )
            let first = try parent.mutated(using: configuration, seed: 123)
            let second = try parent.mutated(using: configuration, seed: 123)
            #expect(first == second)
            #expect(first.model != parent.model)
            #expect(first.topology == parent.topology)
            #expect(first.generation == parent.generation + 1)

            for (oldLayer, newLayer) in zip(parent.model.layers, first.model.layers) {
                switch strategy {
                case .gaussian:
                    #expect(newLayer.weights.allSatisfy { $0.isFinite })
                case .uniform:
                    #expect(
                        zip(oldLayer.weights, newLayer.weights).allSatisfy {
                            abs($0 - $1) <= 0.2
                        }
                    )
                case .reset:
                    #expect(newLayer.weights.allSatisfy { (-0.2...0.2).contains($0) })
                    #expect(newLayer.biases.allSatisfy { (-0.2...0.2).contains($0) })
                }
            }
        }

        let weightsOnly = try parent.mutated(
            using: DenseMutationConfiguration(
                strategy: .uniform,
                probability: 1,
                magnitude: 0.5,
                mutatesBiases: false
            ),
            seed: 4
        )
        #expect(weightsOnly.model.layers[0].biases == parent.model.layers[0].biases)
        let zeroMagnitude = try parent.mutated(
            using: DenseMutationConfiguration(probability: 1, magnitude: 0),
            seed: 5
        )
        #expect(zeroMagnitude.model == parent.model)

        let terminal = try DenseBrain(model: parent.model, generation: .max)
        #expect(throws: ML5Error.self) {
            _ = try terminal.mutated(
                using: DenseMutationConfiguration(probability: 1),
                seed: 0
            )
        }

        let overflow = try makeBrain(parameter: .greatestFiniteMagnitude)
        let largeMutation = try DenseMutationConfiguration(
            strategy: .uniform,
            probability: 1,
            magnitude: .greatestFiniteMagnitude
        )
        var observedOverflow = false
        for seed in UInt64(0)..<100 where observedOverflow == false {
            do {
                _ = try overflow.mutated(using: largeMutation, seed: seed)
            } catch ML5Error.invalidNeuroevolutionConfiguration {
                observedOverflow = true
            }
        }
        #expect(observedOverflow)
    }

    @Test("Crossover strategies validate topology and reproduce deterministically")
    func crossover() throws {
        for probability in [-0.1, 1.1, .nan] {
            #expect(throws: ML5Error.self) {
                _ = try DenseCrossoverConfiguration(firstParentProbability: probability)
            }
        }
        for factor in [-0.1, 1.1, .infinity] {
            #expect(throws: ML5Error.self) {
                _ = try DenseCrossoverConfiguration(blendFactor: factor)
            }
        }

        let first = try makeBrain(parameter: 1, generation: 2)
        let second = try makeBrain(parameter: 3, generation: 4)
        for strategy in DenseCrossoverStrategy.allCases {
            let configuration = try DenseCrossoverConfiguration(strategy: strategy)
            #expect(
                try JSONDecoder().decode(
                    DenseCrossoverConfiguration.self,
                    from: JSONEncoder().encode(configuration)
                ) == configuration
            )
            let child = try first.crossed(with: second, using: configuration, seed: 77)
            let repeated = try first.crossed(with: second, using: configuration, seed: 77)
            #expect(child == repeated)
            #expect(child.topology == first.topology)
            #expect(child.generation == 5)
        }

        let allFirst = try first.crossed(
            with: second,
            using: DenseCrossoverConfiguration(
                strategy: .uniform,
                firstParentProbability: 1
            ),
            seed: 0
        )
        #expect(allFirst.model.layers == first.model.layers)
        let allSecond = try first.crossed(
            with: second,
            using: DenseCrossoverConfiguration(
                strategy: .uniform,
                firstParentProbability: 0
            ),
            seed: 0
        )
        #expect(allSecond.model.layers == second.model.layers)
        let blended = try first.crossed(
            with: second,
            using: DenseCrossoverConfiguration(strategy: .blend, blendFactor: 0.25),
            seed: 0
        )
        #expect(blended.model.layers[0].weights.allSatisfy { $0 == 2.5 })
        #expect(blended.model.layers[0].biases.allSatisfy { $0 == 2.5 })

        let singlePoint = try first.crossed(
            with: second,
            using: DenseCrossoverConfiguration(strategy: .singlePoint),
            seed: 12
        )
        let parameters = singlePoint.model.layers[0].weights + singlePoint.model.layers[0].biases
        #expect(parameters.allSatisfy { $0 == 1 || $0 == 3 })

        #expect(throws: ML5Error.incompatibleBrainTopologies) {
            _ = try first.crossed(
                with: makeBrain(extraInput: true),
                using: DenseCrossoverConfiguration(),
                seed: 0
            )
        }
        let terminal = try DenseBrain(model: first.model, generation: .max)
        #expect(throws: ML5Error.self) {
            _ = try terminal.crossed(
                with: second,
                using: DenseCrossoverConfiguration(),
                seed: 0
            )
        }
    }

    @Test("Brain snapshots and populations round-trip with reproducible resume")
    func persistence() throws {
        let first = try makeBrain(parameter: 1)
        let second = try makeBrain(parameter: 2)
        let snapshot = first.snapshot()
        #expect(snapshot.formatVersion == DenseBrainSnapshot.currentFormatVersion)
        #expect(snapshot.brain == first)
        #expect(
            try JSONDecoder().decode(
                DenseBrainSnapshot.self,
                from: JSONEncoder().encode(snapshot)
            ) == snapshot
        )
        #expect(throws: ML5Error.self) {
            _ = try DenseBrainSnapshot(formatVersion: 2, brain: first)
        }

        #expect(throws: ML5Error.self) {
            _ = try DenseBrainPopulation(brains: [], seed: 0)
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseBrainPopulation(
                brains: [first, DenseBrain(model: second.model, generation: 1)],
                seed: 0
            )
        }
        #expect(throws: ML5Error.incompatibleBrainTopologies) {
            _ = try DenseBrainPopulation(
                brains: [first, makeBrain(extraInput: true)],
                seed: 0
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try DenseBrainPopulation(formatVersion: 2, brains: [first], randomState: 0)
        }

        let population = try DenseBrainPopulation(brains: [first, second], seed: 987)
        #expect(population.formatVersion == DenseBrainPopulation.currentFormatVersion)
        #expect(population.generation == 0)
        #expect(population.brains.count == 2)
        #expect(population.randomState == 987)
        let encoded = try JSONEncoder().encode(population)
        let restored = try JSONDecoder().decode(DenseBrainPopulation.self, from: encoded)
        #expect(restored == population)
        let mutation = try DenseMutationConfiguration(
            strategy: .gaussian,
            probability: 0.5,
            magnitude: 0.1
        )
        let next = try population.mutated(using: mutation)
        let resumedNext = try restored.mutated(using: mutation)
        #expect(next == resumedNext)
        #expect(next.generation == 1)
        #expect(next.randomState != population.randomState)
        #expect(next.brains.allSatisfy { $0.generation == 1 })

        var object = try #require(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any]
        )
        object["generation"] = 3
        #expect(throws: ML5Error.self) {
            _ = try JSONDecoder().decode(
                DenseBrainPopulation.self,
                from: JSONSerialization.data(withJSONObject: object)
            )
        }

        let terminal = try DenseBrain(model: first.model, generation: .max)
        let terminalPopulation = try DenseBrainPopulation(brains: [terminal], seed: 0)
        #expect(throws: ML5Error.self) {
            _ = try terminalPopulation.mutated(using: mutation)
        }
    }

    private func makeBrain(
        parameter: Double? = nil,
        generation: Int = 0,
        outputActivation: ActivationFunction = .softmax,
        extraInput: Bool = false
    ) throws -> DenseBrain {
        let inputFeatures: [FeatureName] = extraInput ? ["x", "y", "z"] : ["x", "y"]
        let configuration = try DenseNetworkConfiguration(
            inputFeatures: inputFeatures,
            outputNames: ["negative", "positive"],
            outputActivation: outputActivation,
            loss: outputActivation == .softmax ? .categoricalCrossEntropy : .meanSquaredError,
            validationFraction: 0
        )
        let weights: [Double]
        let biases: [Double]
        if let parameter {
            weights = Array(repeating: parameter, count: inputFeatures.count * 2)
            biases = Array(repeating: parameter, count: 2)
        } else {
            weights = extraInput ? [-1, -1, 0, 1, 1, 0] : [-1, -1, 1, 1]
            biases = [0, 0]
        }
        let layer = try DenseLayerParameters(
            inputCount: inputFeatures.count,
            outputCount: 2,
            weights: weights,
            biases: biases
        )
        return try DenseBrain(
            model: DenseNetworkModel(configuration: configuration, layers: [layer]),
            generation: generation
        )
    }
}
