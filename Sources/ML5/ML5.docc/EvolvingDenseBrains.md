# Evolving Dense Brains

Use trained dense networks in synchronous simulation loops, then derive independent
offspring with reproducible mutation and crossover.

## Create an immutable brain

``DenseBrain`` wraps a validated ``DenseNetworkModel`` and records a lineage generation.
The model may come from CPU or Metal training, a ``DenseModelArchive``, or direct
construction:

```swift
let brain = try DenseBrain(model: trainingResult.model)
let output = try brain.predict(agentFeatures)
```

Prediction is synchronous and framework independent. It preserves the model's configured
input and output order and performs no actor hop, task suspension, or Core ML call. This
makes it appropriate for a P5 draw loop or a Matter simulation step. The model's existing
async prediction remains available when a uniform `ModelPredicting` interface is more
important than synchronous latency.

For a softmax output, classify into a configured output name and a complete probability
distribution:

```swift
let decision = try brain.classify(agentFeatures)
switch decision.label.rawValue {
case "left": steerLeft(confidence: decision.confidence)
case "right": steerRight(confidence: decision.confidence)
default: break
}
```

``DenseBrain/classify(_:)`` requires softmax because independent sigmoid outputs do not
represent one mutually exclusive label. Equal probabilities resolve in
``DenseNetworkConfiguration/outputNames`` order.

## Copy without shared mutation

Brains, models, and parameter layers are immutable Swift values. Calling
``DenseBrain/copied()`` creates a separate lineage value; Swift may share read-only
copy-on-write storage internally, but no public operation can mutate either parent.
Mutation and crossover construct new validated parameter arrays:

```swift
let parent = brain.copied()
let child = try parent.mutated(
    using: DenseMutationConfiguration(
        strategy: .gaussian,
        probability: 0.08,
        magnitude: 0.15
    ),
    seed: generationSeed
)
```

``DenseMutationStrategy/gaussian`` adds a normal delta,
``DenseMutationStrategy/uniform`` adds a bounded uniform delta, and
``DenseMutationStrategy/reset`` replaces the parameter inside the configured symmetric
range. Probability applies independently to each weight and, by default, each bias.
Disable ``DenseMutationConfiguration/mutatesBiases`` when biases must remain fixed.
The same parent, policy, and seed always produce the same child.

## Combine compatible parents

``DenseBrain/crossed(with:using:seed:)`` compares ``DenseBrainTopology`` before reading
parameters. Input/output names and order, hidden widths and activations, and output
activation must match. Training-only settings such as learning rate and epoch limit are
not topology; the child inherits the first parent's complete model configuration.

```swift
let crossover = try DenseCrossoverConfiguration(
    strategy: .uniform,
    firstParentProbability: 0.5
)
let offspring = try firstParent.crossed(
    with: secondParent,
    using: crossover,
    seed: reproductionSeed
)
```

Uniform crossover chooses each weight or bias independently. Single-point crossover
uses one seeded split across the flattened parameter sequence. Blend crossover computes
a convex combination using ``DenseCrossoverConfiguration/blendFactor``. Offspring
generation is one greater than the newer parent generation.

## Persist one brain

``DenseBrain`` is validated and `Codable`. Wrap it in ``DenseBrainSnapshot`` when a
versioned schema boundary is required:

```swift
let data = try JSONEncoder().encode(brain.snapshot())
let restored = try JSONDecoder().decode(DenseBrainSnapshot.self, from: data).brain
```

A snapshot validates its format, brain generation, dense model, topology, and finite
parameters during decoding. It does not cryptographically authenticate an untrusted
file; use a signed distribution channel when provenance matters.

## Resume a population reproducibly

``DenseBrainPopulation`` requires one or more brains with identical topology and
generation. It stores both the generation and SplitMix64 state used to derive the next
per-brain mutation seeds:

```swift
let population = try DenseBrainPopulation(
    brains: initialBrains,
    seed: experimentSeed
)
let next = try population.mutated(using: mutationPolicy)

let checkpoint = try JSONEncoder().encode(next)
let resumed = try JSONDecoder().decode(
    DenseBrainPopulation.self,
    from: checkpoint
)
```

Applying the same mutation policy to `next` and `resumed` produces identical subsequent
populations. ML5 intentionally leaves fitness measurement, parent selection, elitism,
and desired population size to the simulation. After selection, use direct brain
crossover to assemble the next parent set and construct a new population with an
explicit experiment seed.

## See Also

- <doc:ConfiguringDenseNetworks>
- <doc:PersistingAndExportingDenseModels>
- <doc:InferenceModes>
