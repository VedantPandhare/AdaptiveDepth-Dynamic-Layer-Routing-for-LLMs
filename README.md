# AdaptiveDepth: Dynamic Layer Routing for LLMs

AdaptiveDepth is an engineering project that investigated whether a causal language model can execute a carefully selected subset of transformer layers for a complete request while retaining useful answer quality and reducing inference latency.

The project used a compact Qwen2.5-0.5B-based model, fixed reduced-depth routes, route-aware training, latency measurement, response-preserving evaluation, and conservative prompt-level routing experiments.

Its central question was:

> Can a language model use fewer transformer layers for some requests without making the application meaningfully less reliable?

The work was exploratory and evidence-driven. It did not treat a smaller route as successful merely because it was faster; every candidate was evaluated against explicit quality, latency, and full-model-divergence constraints.

## Objective

Autoregressive LLMs normally execute every transformer block for every generated token. AdaptiveDepth explored whether some internal computation could be omitted for selected requests.

The original system was designed around four stages:

1. Measure the effect of skipping transformer layers and compile candidate routes.
2. Train a model to be more robust when selected layers are omitted.
3. Measure when each reduced route preserves the full model's behavior.
4. Use a conservative prompt-level policy to choose the cheapest acceptable route, with full depth as the fallback.

## Methodology

### Fixed layer routes

A route is an ordered subset of the model's transformer blocks. For example, a 20-layer route in a 24-layer model skips four blocks while retaining the original order of the remaining ones.

The runtime applies one fixed route for a complete generation and restores the full layer stack after execution. This made route behavior and cache handling explicit and reproducible.

### Evaluation criteria

Each candidate route was compared with full-depth generation using:

- task quality on a controlled English prompt set;
- fixed-generation-length latency;
- teacher KL divergence from the full model;
- generated-token agreement;
- peak memory where available.

A strict standalone route needed to satisfy all of the following:

| Constraint | Threshold |
|---|---:|
| Maximum quality loss | 0.050 |
| Minimum latency reduction | 10.0% |
| Maximum teacher KL divergence | 0.500 |

### Route-aware training

The model was trained with a frozen full-depth teacher and a trainable student. Training mixed full-depth and reduced-depth routes and used language-model loss plus teacher distillation. The goal was to make the same model more robust to selected layer omissions.

### Prompt-level routing experiments

For the application-oriented part of the work, reduced routes were evaluated on response-preserving prompts. The project then trained conservative classifiers to decide whether a complete prompt could safely use a reduced route or should fall back to the full model.

## What was implemented

- Architecture-aware transformer layer discovery.
- Validated fixed-route execution with full-model restoration.
- Deterministic greedy generation profiling with time-to-first-token and decode timing.
- Layer influence, ablation, candidate-route, Pareto, and static baseline tooling.
- Full-model-to-route teacher KL measurement.
- Resumable static-route experiments and strict go/no-go gates.
- Route-aware teacher/student distillation and resumable checkpoints.
- Response-preserving multi-route evaluation for 20-, 21-, 22-, and 23-layer routes.
- Prompt-level risk routing, calibrated thresholds, semantic label review, and post-route rescue experiments.
- Offline tests using toy transformer models and a real small Qwen2 cache-semantic integration test.

The repository has since been cleaned to retain the reusable inference foundation: model loading, layer discovery, fixed-route execution, deterministic profiling, route definitions, and foundational tests. Large checkpoints and experiment artifacts are intentionally kept outside Git.

## Results

All figures below are measurements from this project's compact model and controlled evaluation setup. They should be interpreted as project evidence, not as universal model-performance claims.

### Full-depth model retention after route-aware training

The 500-step route-aware training run retained the full model's measured quality:

| Measurement | Result |
|---|---:|
| Original full-depth quality | 0.500 |
| Trained full-depth quality | 0.533 |
| Full-depth quality change | +0.033 |
| Full-depth retention gate | Pass |

### Reduced-route trade-off

The stabilized evaluation used fixed generation length. None of the standalone reduced routes met every strict constraint simultaneously.

| Route | Active layers | Latency reduction | Quality change vs. full | Teacher KL | Standalone gate |
|---|---:|---:|---:|---:|---|
| Fast | 20 / 24 | 15.1% | -0.133 | 0.3022 | Fail |
| Target | 21 / 24 | 11.9% | -0.067 | 0.3973 | Fail |
| Balanced | 22 / 24 | 7.8% | -0.067 | 0.2077 | Fail |
| Safe | 23 / 24 | 4.0% | -0.033 | 0.1566 | Fail |

The faster routes delivered the latency target but lost too much quality. The more conservative routes were closer in quality but did not meet the desired latency reduction.

### Response-preserving multi-route dataset

A 400-prompt dataset compared each route against the same full-model response budget.

| Route | Safe prompts | Safe rate | Median latency reduction | Teacher KL | Generation Jaccard |
|---|---:|---:|---:|---:|---:|
| 20-layer | 36 / 400 | 9.0% | 15.6% | 0.4697 | 33.3% |
| 21-layer | 51 / 400 | 12.8% | 11.5% | 0.3547 | 40.0% |
| 22-layer | 78 / 400 | 19.5% | 7.7% | 0.2233 | 43.8% |
| 23-layer | 138 / 400 | 34.5% | 3.6% | 0.1215 | 58.8% |

The collection itself passed its data-readiness checks: all 400 examples were valid, no response-preservation failures occurred, and all four routes had both safe and fallback examples.

### Prompt-routing outcome

Prompt-only routing did not provide enough reliable coverage for the intended application target.

- A semantic review examined 128 ambiguous examples; only 2 were accepted as safe.
- The final two-route policy reached 93.3% observed precision but only 3.8% reduced-route coverage.
- The intended coverage target was 34%, so the combined policy received a `revise_combined_policy` verdict.

## Key findings

1. **Layer skipping creates a real quality/latency frontier.** The project measured routes that were faster and routes that were closer to full-model behavior, but no tested static route satisfied every strict requirement.

2. **Route-aware training preserved the full model.** The trained checkpoint maintained full-depth quality while improving several reduced-route measurements.

3. **Prompt-only signals were not enough.** A safe standalone route for one prompt could not be predicted at useful coverage from prompt text alone under the required precision target.

4. **Fail-closed evaluation mattered.** Failed gates were treated as findings rather than hidden. The system retained full-model fallback rather than presenting reduced routes as generally safe.

5. **Data construction affected conclusions.** The response-preserving 400-prompt evaluation corrected earlier token-starved examples and produced more meaningful route comparisons.

## Relation to prior work

AdaptiveDepth was informed by two relevant papers:

- [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710) studies early exit and self-speculative decoding through layer-dropout training and verification with later layers.
- [ShortGPT: Layers in Large Language Models are More Redundant Than You Expect](https://arxiv.org/abs/2403.03853) studies layer redundancy and direct layer removal using Block Influence.

AdaptiveDepth explored similar motivations?layer redundancy and adaptive computation?but focused on fixed reduced-depth routes, route robustness, and conservative application-level gating in a compact implementation.

## Repository layout

```text
configs/
  model.yaml             # Model and generation defaults
  draft_routes.yaml      # Retained 20- and 23-layer route definitions
src/adaptivedepth/
  model/                 # Model loading, layer discovery, route execution
  runtime/               # Prompt encoding and greedy generation profiling
  routes.py              # Route definition and YAML library support
tests/                    # Offline foundation tests
```

The local `checkpoint-step-000500/` directory is an inference checkpoint of roughly 1.9 GB. It is deliberately ignored by Git. Training-state files and historic experiment outputs are not part of this repository.

## Setup

```powershell
python -m pip install -e ".[dev]"
pytest
```

The tests run without downloading a model. Model inspection requires the configured Hugging Face checkpoint or a compatible local checkpoint:

```powershell
adaptivedepth inspect-model --config configs/model.yaml
```

## License

See [LICENSE](LICENSE). Any distributed checkpoint must also comply with the base model's license.
