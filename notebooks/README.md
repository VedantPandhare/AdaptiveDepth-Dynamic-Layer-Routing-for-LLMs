# Cloud experiment notebooks

`phase4_colab_kaggle.ipynb` runs the first real AdaptiveDepth static-route
experiment on a Colab or Kaggle GPU.

Before running it:

1. Push the desired repository commit to GitHub.
2. If the repository is private, create a fine-grained GitHub token with
   read-only access to this repository's contents, add it to Colab or Kaggle
   Secrets as `GITHUB_TOKEN`, and grant the notebook access to it. Never paste
   the token into a code cell.
3. Enable a GPU runtime.
4. On Kaggle, enable Internet access so GitHub and Hugging Face are reachable.
5. Run the cells from top to bottom.
6. Download `phase4_artifacts.zip` before the runtime is discarded.

The Phase 4 command is resumable. Keep the `experiments/phase4` directory when
reconnecting to the same GPU/software environment. It intentionally rejects a
resume on different hardware because those latency measurements are not
comparable.

Experiment schema version 2 uses fixed-length generation for latency
comparison while preserving EOS-truncated answer text. Delete or separately
archive any schema version 1 smoke workspace instead of resuming it.

`phase5_colab_kaggle.ipynb` performs route-aware teacher/student training after
the Phase 4 `revise_routes` verdict. It first runs four smoke optimizer steps,
then prepares a separate English training set and runs the resumable T4
configuration. See [`../docs/PHASE5_RUNBOOK.md`](../docs/PHASE5_RUNBOOK.md).

`phase5_post_training_eval.ipynb` loads the completed student checkpoint,
benchmarks full-depth and all four trained routes on the original Phase 4
evaluation set, and produces the Phase 5 acceptance gate. It supports a local
checkpoint directory, an uploaded inference ZIP, or a Hugging Face model ID.
See
[`../docs/PHASE5_EVALUATION_RUNBOOK.md`](../docs/PHASE5_EVALUATION_RUNBOOK.md).

`phase5_post_training_eval_drive.ipynb` is the recommended Colab variant when
the trained checkpoint folder is stored in Google Drive. It validates the
folder, copies the checkpoint to Colab local disk before inference, and copies
the final gate artifacts back to `MyDrive/AdaptiveDepth-results`.

`phase5_correction_drive.ipynb` performs the single bounded correction allowed
after the measured `revise_training` verdict. It starts both the teacher and
student from the preserved step-500 checkpoint, targets only
`route_target_21`, saves to a separate workspace, and automatically reruns the
unchanged acceptance gate after the full correction. See
[`../docs/PHASE5_CORRECTION_RUNBOOK.md`](../docs/PHASE5_CORRECTION_RUNBOOK.md).

`phase6_router_dataset_drive.ipynb` builds the compact application-track
route-regret dataset from the corrected checkpoint. It compares the near-miss
21-layer route with full depth, assigns predetermined safe/fallback labels, and
writes a Phase 7 readiness gate. See
[`../docs/PHASE6_RUNBOOK.md`](../docs/PHASE6_RUNBOOK.md).

`phase61_multiroute_drive.ipynb` replaces the token-starved Phase 6 labels with
a response-preserving 400-prompt comparison of Routes 20-23. It rotates timing
order, writes cheapest-safe targets, and stores its resumable workspace in
Drive. See [`../docs/PHASE61_RUNBOOK.md`](../docs/PHASE61_RUNBOOK.md).

`phase81_runtime_signal_pilot_colab.ipynb` runs the bounded Route 20/23
four-token confidence and shared-hidden-state pilot. It reuses the existing
Drive artifacts and writes a resumable signal JSONL for CPU-only evaluation.
See [`../docs/PHASE81_PROBE_PILOT.md`](../docs/PHASE81_PROBE_PILOT.md).

`phase71_multiroute_router_drive.ipynb` trains and evaluates the CPU-only
fastest-confident policy from `phase61_artifacts.zip`. It packages the fitted
router and its fail-closed gate without loading an LLM or requiring a GPU. See
[`../docs/PHASE71_RUNBOOK.md`](../docs/PHASE71_RUNBOOK.md).
