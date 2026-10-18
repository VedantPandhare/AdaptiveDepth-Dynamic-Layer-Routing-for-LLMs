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

The software implementation is at approximately **70%** of the strengthened
design. Phase 6.1 produced 400 response-preserving, four-route examples on a
T4, and the first leakage-controlled Phase 7.1 multi-route run completed. Its
conservative verdict is `revise_router`, so adaptive routing remains disabled
until correction.

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
- A resumable compact full-vs-route regret collector with predetermined
  safety labels and a Phase 7 readiness gate.
- A calibrated CPU risk-router trainer with nested out-of-fold threshold
  selection, a locked holdout gate, auditable predictions, and fail-closed
  serialization.
- Response-preserving multi-route regret collection for Routes 20-23 and a
  calibrated fastest-confident Phase 7.1 policy with full-depth fallback.
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

After the 500-step checkpoint is available, run the dedicated post-training
evaluation on a GPU:

```powershell
adaptivedepth run-phase5-evaluation `
  --checkpoint checkpoint-step-000500 `
  --config configs/cloud_t4.yaml `
  --routes-file configs/phase5_routes.yaml `
  --dataset data/phase4_eval.jsonl `
  --workspace experiments/phase5_evaluation `
  --resume
```

The corresponding hosted workflow is
[`notebooks/phase5_post_training_eval.ipynb`](notebooks/phase5_post_training_eval.ipynb).
It checks full-model retention and compares each trained route with its saved
Phase 4 measurement before allowing the compact Phase 6 calibration dataset.

The first post-training gate preserved full-depth quality, while
`route_target_21` passed quality and speed but missed the teacher-KL threshold.
Run the bounded correction from the preserved checkpoint with
[`notebooks/phase5_correction_drive.ipynb`](notebooks/phase5_correction_drive.ipynb).
Its rationale, strict stop rule, and equivalent CLI commands are in the
[`Phase 5 correction runbook`](docs/PHASE5_CORRECTION_RUNBOOK.md).

The stabilized confirmation still rejects every static route. For the
application track, Phase 6 now tests whether the confirmed 21-layer near miss
can be used selectively with full depth as the mandatory fallback. Run it with
[`notebooks/phase6_router_dataset_drive.ipynb`](notebooks/phase6_router_dataset_drive.ipynb)
and see the [`Phase 6 runbook`](docs/PHASE6_RUNBOOK.md).

Equivalent Phase 6 CLI workflow:

```powershell
adaptivedepth prepare-phase6-data `
  --output data/phase6_calibration.jsonl `
  --max-examples 200 `
  --skip-examples 2300

adaptivedepth run-phase6 `
  --checkpoint checkpoint-step-000100 `
  --config configs/phase6_t4.yaml `
  --route-file configs/phase6_route.yaml `
  --dataset data/phase6_calibration.jsonl `
  --workspace experiments/phase6 `
  --resume
```

Phase 6 passed its application-track readiness gate with 200 valid examples,
including 42 safe-route and 158 full-model labels. Train and evaluate the
prompt-only Phase 7 router on CPU:

```powershell
python -m pip install -e ".[router]"
adaptivedepth train-phase7 `
  --dataset phase6/router_dataset.jsonl `
  --phase6-gate phase6/gate.json `
  --config configs/phase7.yaml `
  --output experiments/phase7
```

The first locked run returned `revise_router`: development out-of-fold
predictions did not find a threshold meeting 80% precision at 5% coverage.
The artifact therefore fails closed to full depth. See the
[`Phase 7 runbook`](docs/PHASE7_RUNBOOK.md) and its
[Drive-based CPU notebook](notebooks/phase7_router_training_drive.ipynb).

Inspection then found that 31% of the original Phase 6 examples retained only
one reference token after long-prompt truncation. Phase 6.1 corrects this with
response-preserving encoding and one combined 400-prompt comparison of Routes
20, 21, 22, and 23 against a reused full-model result. Run the corrected
collection with
[`notebooks/phase61_multiroute_drive.ipynb`](notebooks/phase61_multiroute_drive.ipynb)
and see the [`Phase 6.1 runbook`](docs/PHASE61_RUNBOOK.md).

Phase 6.1 passed with 400 valid examples and all four routes eligible. Train
the multi-route prompt router locally on CPU:

```powershell
adaptivedepth train-phase71 `
  --dataset phase61/router_dataset.jsonl `
  --phase61-gate phase61/gate.json `
  --config configs/phase71.yaml `
  --output experiments/phase71
```

The first locked Phase 7.1 run returned `revise_router`: held-out precision was
72.7% against the 80% gate, although coverage and gross latency benefit passed.
The artifact is disabled and always falls back to full depth. See the
[`Phase 7.1 runbook`](docs/PHASE71_RUNBOOK.md) and its
[Drive-based CPU notebook](notebooks/phase71_multiroute_router_drive.ipynb).

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
