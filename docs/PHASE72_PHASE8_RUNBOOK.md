# Phase 7.2 correction and Phase 8 rescue runbook

These stages are CPU-only. They reuse the 400 measured Phase 6.1 prompts; they
do not load the language model or require Colab.

```powershell
adaptivedepth train-phase72 `
  --dataset phase61/router_dataset.jsonl `
  --phase61-gate phase61/gate.json `
  --config configs/phase72.yaml `
  --output experiments/phase72

adaptivedepth train-phase8 `
  --dataset phase61/router_dataset.jsonl `
  --phase72-report experiments/phase72/report.json `
  --phase72-oof experiments/phase72/oof_predictions.jsonl `
  --config configs/phase8.yaml `
  --output experiments/phase8
```

Phase 7.2 uses shared hashed word/character features and jointly selects the
four route thresholds. The locked local result passes the policy checks at
80% precision and 5% coverage, but fails Route 23 calibration. Its artifact is
therefore disabled and may only provide out-of-fold development candidates to
Phase 8.

Phase 8 trains a prompt-grouped rescue predictor over all 1,600 measured
prompt-route outputs. It accepts or restarts a candidate route based on output
diagnostics and the Phase 7.2 uncertainty. The locked development result
accepts 70% of the 20 candidate decisions at 100% observed precision and
rescues every observed unsafe candidate.

These are development results, not deployment claims. Both artifacts fail
closed. A fresh prompt set must confirm the combined router-rescue policy, and
Phase 9 must measure the real cost of short-probe-and-restart execution.

## Two-route application optimization

The application policy is restricted to Route 20, Route 23, and Full. Rebuild
the complete out-of-fold rescue score file, then run the joint optimizer:

```powershell
adaptivedepth train-phase8 `
  --dataset phase61/router_dataset.jsonl `
  --phase72-report experiments/phase72/report.json `
  --phase72-oof experiments/phase72/oof_predictions.jsonl `
  --config configs/phase8.yaml `
  --output experiments/phase8

adaptivedepth optimize-phase72 `
  --dataset phase61/router_dataset.jsonl `
  --router-oof experiments/phase72/oof_predictions.jsonl `
  --rescue-oof experiments/phase8/oof_route_predictions.jsonl `
  --config configs/phase72_application.yaml `
  --output experiments/phase72_application
```

The locked 400-prompt result is `revise_combined_policy`. At 90% required
precision, the best searched policy reaches 4.25% coverage. Forcing 34.25%
coverage yields 47.4% precision and negative projected latency. Threshold
tuning alone therefore cannot meet the application target; the label/model
definition must be corrected before collecting a fresh validation set.

## Application-label correction

Do not remove generation Jaccard globally. The 128 cases that pass KL,
reference-regret, and token-agreement but fail Jaccard contain both harmless
paraphrases and factual errors. Export only those ambiguous Route 20/23 pairs:

```powershell
adaptivedepth prepare-phase72-label-review `
  --dataset phase61/router_dataset.jsonl `
  --output experiments/phase72_label_review.jsonl
```

Upload that JSONL to `MyDrive/AdaptiveDepth-results`, push the current code,
and run
[`phase72_semantic_label_review_colab.ipynb`](../notebooks/phase72_semantic_label_review_colab.ipynb)
on a T4. The conservative judge writes a resumable
`phase72_label_judgments.jsonl` to the same Drive directory.

After downloading the judgments, build a separate application dataset. A
semantic `safe_route` judgment is accepted only at confidence 0.80 or higher:

```powershell
adaptivedepth apply-phase72-label-review `
  --dataset phase61/router_dataset.jsonl `
  --judgments experiments/phase72_label_judgments.jsonl `
  --output experiments/phase72_application_dataset.jsonl `
  --minimum-safe-confidence 0.80
```

The original Phase 6.1 dataset is never overwritten. Retrain Phase 7.2 and
Phase 8 against the application dataset, rerun the joint optimizer, and only
collect fresh prompts if the 34% development gate becomes plausible.
