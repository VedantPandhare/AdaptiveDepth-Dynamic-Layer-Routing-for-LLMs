# Phase 5 targeted correction

## Why this experiment exists

The 500-step student preserved full-depth quality, but no route passed every
gate. `route_target_21` is the only useful correction target: its quality drop
was `0.033`, its latency reduction was `11.2%`, and its teacher KL was `0.6625`.
It therefore passes the quality and speed constraints and misses only the
`0.5000` KL limit. Saved predictions also showed an option-B collapse on many
structured questions.

There is no guarantee that another optimization run will improve held-out
results. This procedure improves the odds and bounds the cost:

- Start the student and frozen full-depth teacher from the preserved step-500
  checkpoint.
- Train only `route_target_21`; do not dilute updates across routes that cannot
  meet the latency gate.
- Use a lower `5e-6` learning rate for only 100 steps.
- Weight teacher distillation at `0.9` and language modeling at `0.1`.
- Include full-depth batches with probability `0.25` to reduce catastrophic
  drift.
- Use 300 held-out Dolly examples after the original 2,000-example slice plus
  200 synthetic multiple-choice examples with exactly balanced A-D labels.
- Write to a new workspace and never modify `checkpoint-step-000500`.

## Colab procedure

1. Push this implementation and open
   [`../notebooks/phase5_correction_drive.ipynb`](../notebooks/phase5_correction_drive.ipynb)
   in a T4 Colab runtime.
2. Put the complete `checkpoint-step-000500` folder in My Drive, or edit
   `DRIVE_CHECKPOINT` in the notebook.
3. Run from the top with `RUN_FULL = False`.
4. After the four-step smoke run completes, set `RUN_FULL = True` and rerun the
   mode, training, and packaging/evaluation cells.
5. Keep `phase5_corrected_student.zip`,
   `phase5_correction_training_summary.json`, and
   `phase5_correction_evaluation_artifacts.zip` from
   `MyDrive/AdaptiveDepth-results`.

The full correction is 20% of the original optimizer-step count. On the same
T4, training should usually take roughly one fifth of the earlier training
time, plus checkpoint and data overhead. The unchanged evaluation is then the
same compute workload as the previous Phase 5 post-training evaluation.

## Acceptance and stop rule

The notebook automatically evaluates the corrected checkpoint using the same
30 cases, four routes, reference metrics, and thresholds as before. Accept it
only when the gate verdict is `proceed_to_phase6`. In particular, at least one
route must simultaneously have:

- quality drop no greater than `0.050`;
- fixed-generation-length latency reduction of at least `10%`; and
- teacher KL no greater than `0.500`.

Full-depth quality retention must also pass. Do not relax a threshold, select
only favorable prompts, or compare against a different baseline after seeing
the result.

If the verdict remains `revise_training`, stop correction training and retain
the original step-500 checkpoint. For the application-focused project, the
honest fallback is to continue with the full model as the safe route and treat
the measured route as experimental, rather than spending unbounded GPU time
chasing a small evaluation set.

## Equivalent CLI commands

Prepare the correction data:

```powershell
adaptivedepth prepare-phase5-correction-data `
  --output data/phase5_correction_train.jsonl `
  --dolly-examples 300 `
  --balanced-examples 200 `
  --skip-dolly-examples 2000
```

Run the bounded correction:

```powershell
adaptivedepth run-phase5-correction `
  --checkpoint checkpoint-step-000500 `
  --config configs/phase5_correction_t4.yaml `
  --routes-file configs/phase5_correction_route.yaml `
  --dataset data/phase5_correction_train.jsonl `
  --output experiments/phase5_correction `
  --resume
```

Then pass the final correction checkpoint to `run-phase5-evaluation` with the
existing `configs/phase5_routes.yaml` and Phase 4 evaluation data.
