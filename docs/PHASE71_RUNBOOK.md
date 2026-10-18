# Phase 7.1 multi-route risk-router runbook

## Purpose

Phase 7.1 replaces the earlier binary Route 21 classifier with four calibrated
prompt-only safety models. Each model estimates whether its corresponding
reduced route is safe for the incoming prompt. The policy checks routes in the
measured cost order and selects the fastest route whose calibrated probability
clears its development threshold. If none qualifies, it uses the full model.

The router uses only prompt text and inexpensive prompt features. Route regret,
generated answers, latency, safety labels, and target routes are excluded from
inference inputs.

## Leakage control and safety

The Phase 6.1 records are divided once into 80% development and 20% holdout,
stratified by the cheapest-safe target. For each route:

1. nested out-of-fold development predictions select a conservative threshold;
2. a calibrated estimator is fitted only on development data;
3. the combined fastest-confident policy is evaluated once on the untouched
   holdout.

The artifact is enabled only when all policy checks pass. Loading a failed
artifact is allowed for inspection, but its inference API always returns
`full`, regardless of the fitted estimators' probabilities.

## Locked application gate

The default configuration requires:

- at least two routes with development thresholds;
- at least 80% safe-route precision;
- at least 5% combined routing coverage;
- at least three routed holdout examples;
- no more than 0.15 expected calibration error and 0.20 Brier score per route;
- at least 0.5% projected gross workload latency reduction.

Thresholds and acceptance criteria must not be weakened after inspecting the
holdout. A failed run motivates better features, more data, or a later rescue
probe?not retrospective gate tuning.

## Run locally

Phase 7.1 is CPU-only and does not load the LLM:

```powershell
python -m pip install -e ".[router]"
adaptivedepth train-phase71 `
  --dataset phase61/router_dataset.jsonl `
  --phase61-gate phase61/gate.json `
  --config configs/phase71.yaml `
  --output experiments/phase71
```

For Colab or another machine, use
[`../notebooks/phase71_multiroute_router_drive.ipynb`](../notebooks/phase71_multiroute_router_drive.ipynb).
A GPU runtime provides no benefit.

## Outputs

- `router.joblib`: four calibrated estimators plus the fail-closed manifest;
- `manifest.json`: hashes, thresholds, route definitions, and enabled state;
- `report.json`: full development, holdout, calibration, and oracle metrics;
- `gate.md`: concise human-readable verdict;
- `development_oof_predictions.jsonl`: auditable threshold-selection data;
- `holdout_predictions.jsonl`: untouched policy decisions.

Only load Joblib files produced by this project. Joblib is a pickle-based
format and must not be accepted from untrusted sources.

## First locked result

The first run over the 400-example Phase 6.1 dataset returned `revise_router`.
Route 20 and Route 23 produced development thresholds, but the combined policy
reached 72.7% held-out safe precision against the locked 80% requirement.
Coverage (13.8%) and gross latency reduction (1.8%) passed; probability
calibration and the development-policy check failed. The serialized artifact
is therefore disabled and must not control Phase 8 or the application runtime.
