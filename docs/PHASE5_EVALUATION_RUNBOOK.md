# Phase 5 post-training evaluation runbook

The 500-step route-aware student is trained, but Phase 5 is complete only after
the trained checkpoint passes a static quality, latency, and divergence check.
This evaluation uses the same 30 English cases and fixed-generation-length
timing basis as the final Phase 4 run.

## What is evaluated

- The trained checkpoint at all 24 layers.
- The four frozen Phase 5 routes at depths 20, 21, 22, and 23.
- Full-depth quality retention relative to the original Qwen checkpoint.
- Each route's quality drop relative to the trained full model.
- Each route's latency reduction and teacher KL.
- Quality and KL changes relative to the corresponding Phase 4 route.

The committed `data/phase4_reference_metrics.json` contains only the reference
measurements required for this comparison. Its dataset hash prevents comparison
with a different evaluation set.

## Gate

The verdict is `proceed_to_phase6` only when:

1. Trained full-model quality drops by no more than 0.05 from the original
   full-model quality.
2. At least one trained route drops by no more than 0.05 from the trained full
   model.
3. That route reduces fixed-length latency by at least 10%.
4. That route's teacher KL is at most 0.50.

Otherwise the verdict is `revise_training`. This gate deliberately does not
reuse the Phase 4 compiler gate because the Phase 5 routes are already a fixed,
measured library rather than newly compiled candidates.

## Cloud procedure

1. Push the implementation commit to GitHub.
2. Open `notebooks/phase5_post_training_eval.ipynb` on a T4 runtime.
3. Upload `phase5_student.zip` through the Files sidebar, or configure a local
   checkpoint/Google Drive/Hugging Face path in the checkpoint cell.
4. Run every cell in order.
5. Download `phase5_evaluation_artifacts.zip`.
6. Preserve `gate.md`, `gate.json`, and `experiment.json`.

The experiment is resumable after each completed route when the runtime,
checkpoint, dataset, configuration, and repository revision remain unchanged.

## Google Drive variant

When browser upload truncates the approximately 1.9 GB model archive, use
`notebooks/phase5_post_training_eval_drive.ipynb`. Upload the complete
`checkpoint-step-000500` folder to My Drive, edit `DRIVE_CHECKPOINT` if the
folder is nested, and run the notebook. It copies the checkpoint to Colab local
disk for inference and saves the evaluation ZIP and gate reports under
`MyDrive/AdaptiveDepth-results`.
