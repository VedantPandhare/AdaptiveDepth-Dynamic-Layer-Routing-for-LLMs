# Phase 7 calibrated risk-router runbook

## Purpose

Phase 7 trains a small CPU classifier that decides whether a prompt may use
the confirmed 21-layer route or must fall back to all 24 layers. It consumes
the 200 labeled examples produced by Phase 6. It does not run or download the
language model.

The classifier may use only information available before generation:

- prompt text represented by word unigram/bigram TF-IDF;
- prompt length and token count;
- newline, question-mark, digit, code-fence, and math-symbol features.

Reference answers, teacher KL, route regret, generated responses, latency, and
the Phase 6 label are excluded from the classifier input. Those outcome fields
are used only to define or evaluate the target.

## Leakage-controlled evaluation

The split and thresholds in `configs/phase7.yaml` are fixed before evaluation:

1. A deterministic stratified 80/20 split reserves 40 examples as a test set.
2. Five-fold out-of-fold predictions on the 160 development examples choose
   the highest-coverage threshold with at least 80% safe-route precision and
   5% route coverage.
3. The calibrated classifier is fitted on the development set.
4. The held-out 40 examples are evaluated once.
5. A failed artifact is disabled and always chooses full depth.

The gate also requires at least two routed test examples, acceptable
probability calibration, and at least 0.5% gross latency reduction after
fallback decisions. Phase 8 is authorized only by `proceed_to_phase8`.

## Run locally (CPU)

Install the router extra, then point the command at the extracted Phase 6
folder:

```powershell
python -m pip install -e ".[router]"
adaptivedepth train-phase7 `
  --dataset phase6/router_dataset.jsonl `
  --phase6-gate phase6/gate.json `
  --config configs/phase7.yaml `
  --output experiments/phase7
```

This usually takes under two minutes on a laptop CPU. A GPU runtime provides
no meaningful benefit. The equivalent Drive-based Colab workflow is
[`../notebooks/phase7_router_training_drive.ipynb`](../notebooks/phase7_router_training_drive.ipynb).

## Outputs

- `router.joblib`: calibrated classifier plus its manifest;
- `manifest.json`: input hashes, frozen settings, threshold, route, and enabled
  state;
- `report.json` and `gate.md`: machine- and human-readable gate results;
- `development_oof_predictions.jsonl`: threshold-selection audit trail;
- `holdout_predictions.jsonl`: untouched-test audit trail.

Only load `router.joblib` when it was produced by this project; Joblib files
are executable Python artifacts and must not be accepted from untrusted users.

## First 200-example result

The first locked run correctly returned `revise_router`. Development
out-of-fold ranking was weak (ROC AUC `0.5476`, average precision `0.2760`), so
no threshold achieved both 80% precision and 5% coverage. The disabled router
therefore sent every held-out prompt to the full model. Calibration itself was
acceptable, but there was no safe selective speedup.

Do not lower the gate or tune against the already viewed test predictions just
to force a pass. The defensible correction is to gather more independent
Phase 6 examples and/or add a pre-generation signal with genuine predictive
power, then evaluate once on a new final test split.
