# AdaptiveDepth

AdaptiveDepth is a risk-aware adaptive inference framework for causal language
models. It compiles a small library of validated transformer subnetworks
offline, then selects the cheapest route predicted to preserve full-model
behavior for each request.

The implementation follows the
[strengthened design memo](AdaptiveDepth_Strengthened_Revision_Design_Memo.md).
Version 1 uses a fixed route for an entire generation and will use
probe-and-restart fallback rather than unsafe mid-generation layer activation.

## Current milestone

The software implementation is at approximately **43%** of the strengthened
design. The Phase 4 static-route experiment is complete on a T4, and the Phase
5 route-aware training pipeline is implemented and ready for its cloud run.

## Implemented foundation

- Architecture-aware transformer layer discovery.
- Fixed-route execution with guaranteed full-model restoration.
- Deterministic greedy generation profiling with TTFT and decode timing.
- JSONL task harness and exact-match baseline evaluation.
- ShortGPT-inspired block-influence measurement.
- Single-layer, adjacent-group, and random ablation route generation.
- Full-model-to-route teacher KL measurement.
- Constrained, deterministic candidate-route generation.
- Fixed-prefix, importance-ranked, and seeded-random static baselines.
- Multi-objective Pareto filtering for measured routes.
- Resumable static-route experiments with environment and dataset validation.
- An automated Phase 4 go/no-go gate comparing compiled routes with baselines.
- A 30-case English evaluation set and a Colab/Kaggle GPU notebook.
- A frozen four-route training library selected from the Phase 4 measurements.
- Frozen-teacher/trainable-student route-aware distillation with deterministic
  sampling and atomic resumable checkpoints.
- Separate English training-data preparation and a Phase 5 T4 notebook.
- Offline unit tests using a toy causal transformer.

See [implementation status](docs/IMPLEMENTATION_STATUS.md) for the phase-level
roadmap and the [Phase 4 runbook](docs/PHASE4_RUNBOOK.md) for the first cloud
experiment.

The first T4 experiment and its `revise_routes` decision are summarized in the
[Phase 4 results](docs/PHASE4_RESULTS.md).
The next execution procedure is documented in the
[Phase 5 runbook](docs/PHASE5_RUNBOOK.md).

## Setup

```powershell
python -m pip install -e ".[dev]"
pytest
```

The default experiment configuration targets
`Qwen/Qwen2.5-0.5B-Instruct`. Downloading the checkpoint is only required for
real-model commands; unit tests run fully offline.

## Commands

Run the complete Phase 4 workflow on a T4 or P100 GPU:

```powershell
adaptivedepth run-phase4 `
  --config configs/cloud_t4.yaml `
  --dataset data/phase4_eval.jsonl `
  --compiler-config configs/compiler_conservative.yaml `
  --workspace experiments/phase4 `
  --resume
```

Before committing to the complete run, execute the smoke configuration:

```powershell
adaptivedepth run-phase4 `
  --config configs/cloud_smoke.yaml `
  --dataset data/smoke_eval.jsonl `
  --compiler-config configs/compiler_conservative.yaml `
  --workspace experiments/phase4_smoke `
  --resume
```

The smoke run uses six prompts, one timed repetition, and eight generated
tokens. Keep its workspace separate because resumable timing artifacts must not
be mixed with the full experiment. Its gate verdict is diagnostic only and
must not be treated as a quality or latency conclusion.

All routes execute the same configured number of profiled tokens for fair
latency comparison. Answer text still stops at the first EOS token.

This loads the model once, measures block influence, compiles candidate and
control routes, benchmarks each route, and writes the gate decision. Progress
is saved after the full baseline and after every route, so the same command can
resume an interrupted run. The runner refuses to combine results from a
different dataset, model, benchmark configuration, code revision, or GPU.

For a hosted notebook, open
[`notebooks/phase4_colab_kaggle.ipynb`](notebooks/phase4_colab_kaggle.ipynb).

## Phase 5 route-aware training

Run the four-step cloud smoke test first:

```powershell
adaptivedepth run-phase5 `
  --config configs/phase5_smoke.yaml `
  --routes-file configs/phase5_routes.yaml `
  --dataset data/phase5_train_smoke.jsonl `
  --output experiments/phase5_smoke `
  --resume
```

For the full run, install the data extra, prepare a separate 2,000-example
English training set, and start the resumable T4 configuration:

```powershell
python -m pip install -e ".[data]"
adaptivedepth prepare-phase5-data --output data/phase5_train.jsonl --max-examples 2000
adaptivedepth run-phase5 `
  --config configs/phase5_t4.yaml `
  --routes-file configs/phase5_routes.yaml `
  --dataset data/phase5_train.jsonl `
  --output experiments/phase5 `
  --resume
```

The hosted workflow is
[`notebooks/phase5_colab_kaggle.ipynb`](notebooks/phase5_colab_kaggle.ipynb).
Training completion is not the quality gate: the resulting student must be
rerun through the Phase 4 static evaluation before routing work continues.

The final roadmap also includes a required
[interactive prompt demonstration](docs/DEMO_APPLICATION_SPEC.md) backed by the
real router and fallback runtime.

### Individual commands

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

Candidate routes are not considered valid until the cloud run benchmarks their
quality, latency, divergence, and memory against the full model and static
baselines. The resulting decision is stored in `experiments/phase4/gate.md`.

Saved predictions can be rescored after a scoring correction without rerunning
GPU inference:

```powershell
adaptivedepth rescore-static-experiment `
  --experiment-file experiments/phase4/experiment.json
```
