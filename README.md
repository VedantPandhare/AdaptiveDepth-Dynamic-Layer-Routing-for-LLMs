# AdaptiveDepth

AdaptiveDepth is a risk-aware adaptive inference framework for causal language
models. It compiles a small library of validated transformer subnetworks
offline, then selects the cheapest route predicted to preserve full-model
behavior for each request.

The implementation follows the
[strengthened design memo](AdaptiveDepth_Strengthened_Revision_Design_Memo.md).
Version 1 uses a fixed route for an entire generation and will use
probe-and-restart fallback rather than unsafe mid-generation layer activation.

## Implemented foundation

- Architecture-aware transformer layer discovery.
- Fixed-route execution with guaranteed full-model restoration.
- Deterministic greedy generation profiling with TTFT and decode timing.
- JSONL task harness and exact-match baseline evaluation.
- ShortGPT-inspired block-influence measurement.
- Single-layer, adjacent-group, and random ablation route generation.
- Constrained, deterministic candidate-route generation.
- Multi-objective Pareto filtering for measured routes.
- Offline unit tests using a toy causal transformer.

See [implementation status](docs/IMPLEMENTATION_STATUS.md) for the phase-level
roadmap.

## Setup

```powershell
python -m pip install -e ".[dev]"
pytest
```

The default experiment configuration targets
`Qwen/Qwen2.5-0.5B-Instruct`. Downloading the checkpoint is only required for
real-model commands; unit tests run fully offline.

## Commands

Inspect the model and discovered layer stack:

```powershell
adaptivedepth inspect-model --config configs/model.yaml
```

Run the full-model baseline:

```powershell
adaptivedepth baseline `
  --config configs/model.yaml `
  --dataset data/smoke_eval.jsonl `
  --output experiments/baseline.json
```

Measure block influence and compile constrained route candidates:

```powershell
adaptivedepth analyze-layers `
  --config configs/model.yaml `
  --dataset data/smoke_eval.jsonl `
  --output experiments/block_influence.json

adaptivedepth compile-candidates `
  --influence-file experiments/block_influence.json `
  --compiler-config configs/compiler.yaml `
  --output experiments/candidate_routes.yaml
```

Generate an ablation suite and benchmark every route:

```powershell
adaptivedepth generate-ablations `
  --strategy single `
  --num-layers 24 `
  --output experiments/single_layer_routes.yaml

adaptivedepth benchmark-routes `
  --routes-file experiments/single_layer_routes.yaml `
  --output experiments/single_layer_benchmarks.json
```

Candidate routes are not considered valid until Phase 4 benchmarks their
quality, latency, divergence, and memory against the full model and static
baselines.
