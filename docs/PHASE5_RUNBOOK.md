# Phase 5 route-aware training runbook

Phase 5 addresses the Phase 4 `revise_routes` result by training one student
checkpoint to tolerate four measured subnetworks. It does not train the future
risk router and it does not claim that the static quality gate has passed.

## Frozen route library

`configs/phase5_routes.yaml` promotes these Phase 4 routes:

| Training route | Phase 4 source | Active layers | Role |
|---|---|---:|---|
| `route_fast_20` | `candidate_b20_v01` | 20 | aggressive speed endpoint |
| `route_target_21` | `random_b21_v00` | 21 | measured ≥10% latency endpoint |
| `route_balanced_22` | `candidate_b22_v00` | 22 | compromise endpoint |
| `route_safe_23` | `candidate_b23_v00` | 23 | safest reduced-depth endpoint |

The masks are copied exactly from `phase4/experiment_rescored.json`. Training
also samples the full 24-layer student with probability 0.2 to preserve its
full-depth behavior.

## Training objective

Each micro-batch runs through:

1. A frozen, full-depth Qwen teacher.
2. The trainable student using one entire-request route, or full depth.
3. Response-only causal language-model loss.
4. Temperature-scaled KL distillation from teacher to student.

This is route-aware robustness training inspired by LayerSkip-style
self-distillation, while preserving the project's cache-safe whole-request
routing constraint. Teacher and student are separate model instances, so the
target cannot drift as the student updates.

On a T4, the frozen teacher is stored in FP16. The trainable student retains
FP32 master parameters while FP16 autocast handles matrix operations. This is
required for numerically safe gradient scaling; directly training FP16
parameters produces FP16 gradients that PyTorch's scaler refuses to unscale.

## Cloud procedure

1. Push the current repository commit.
2. Open `notebooks/phase5_colab_kaggle.ipynb` and select a T4 or stronger GPU.
3. Run all cells with `RUN_FULL = False`.
4. Confirm the four-step smoke run saves two checkpoints and completes.
5. Set `RUN_FULL = True`, then rerun the last three cells.
6. Download `phase5_student.zip` before discarding the runtime.

The full configuration uses 2,000 English instruction/response examples,
sequence length 128, batch size 1, four-step gradient accumulation, and 500
optimizer steps. It retains only the latest two complete checkpoints to limit
cloud disk use. A checkpoint is considered resumable only after its
`complete.json` marker is written; changed data, routes, or training settings
are rejected on resume.

The downloadable ZIP excludes `training_state.pt` to reduce its size. Preserve
the workspace separately when optimizer-state resume is required.

## Acceptance check after training

The trained checkpoint must be rerun through the same Phase 4 static suite and
gate. The next phase begins only after recording whether quality delta, fixed-
length latency reduction, and teacher KL improved. Do not infer success from a
falling training loss alone.
