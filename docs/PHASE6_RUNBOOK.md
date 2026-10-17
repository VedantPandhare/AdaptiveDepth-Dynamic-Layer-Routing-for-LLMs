# Phase 6 compact router-dataset runbook

## Decision entering Phase 6

The warmed five-repetition confirmation did not validate a globally safe
static route. The corrected checkpoint retained full-depth quality, but the
route measurements were:

| Route | Quality delta | Latency reduction | Teacher KL | Static gate |
|---|---:|---:|---:|---|
| `route_fast_20` | -0.133 | 15.1% | 0.3022 | fail quality |
| `route_target_21` | -0.067 | 11.9% | 0.3973 | near miss on quality |
| `route_balanced_22` | -0.067 | 7.8% | 0.2077 | fail quality and speed |
| `route_safe_23` | -0.033 | 4.0% | 0.1566 | fail speed |

Accordingly, Phase 5 remains `revise_training`; the earlier 18.5% result for
the 23-layer route was a cold/full-baseline timing anomaly. The application
track proceeds to Phase 6 as an explicitly experimental selective-routing
test, not as a claim that the static gate passed.

`route_target_21` is the only useful application candidate: it has a confirmed
speedup above 10%, passes the KL constraint, and misses the aggregate quality
limit by one additional answer on the 30-case evaluation. The full 24-layer
model remains the mandatory fallback.

## What Phase 6 collects

For 200 held-out Dolly examples, the collector compares the corrected model at
full depth with `route_target_21`. These examples start after the 2,000 original
training examples and 300 correction examples in the same deterministic Dolly
shuffle, so they do not overlap either training stage or the 30-case evaluation
set.

Each completed record contains:

- inference-time prompt features;
- reference-response loss for full and routed execution;
- route reference-loss regret;
- token-level `KL(full || route)` on response positions;
- full/route top-token agreement;
- full and routed generated text and text agreement;
- paired fixed-length latency, with measurement order alternated by example;
- a predetermined `safe_route` or `full_model` label.

The safe label requires all three conditions:

- teacher KL no greater than `0.50`;
- reference-loss regret no greater than `0.15`;
- response-token agreement of at least `75%`.

Latency is not part of the per-prompt safety label. It is evaluated separately
at the dataset level.

## Readiness gate

Phase 6 returns `proceed_to_phase7` only when it produces:

- at least 150 valid examples;
- at least 20 `safe_route` examples;
- at least 20 `full_model` fallback examples; and
- at least 10% median paired latency reduction.

These limits are fixed before the cloud run. Do not tune them after seeing the
class balance merely to force a pass.

## Colab procedure

1. Push the Phase 6 implementation to GitHub.
2. Open
   [`../notebooks/phase6_router_dataset_drive.ipynb`](../notebooks/phase6_router_dataset_drive.ipynb)
   in a T4 Colab runtime.
3. Confirm that
   `MyDrive/AdaptiveDepth-results/phase5_corrected_student.zip` exists.
4. Run all cells with `RUN_FULL = False`.
5. A smoke verdict is diagnostic. If the smoke command completes, set
   `RUN_FULL = True` and rerun the final three code cells.
6. Preserve `MyDrive/AdaptiveDepth-results/phase6_artifacts.zip` and provide
   its `gate.md` for review.

The ZIP is small because it contains measurements and the router dataset, not
another copy of the model. The full run is resumable while the Colab runtime
and software environment remain unchanged.

## Outputs

The full workspace contains:

- `experiment.json`: resumable source measurements and provenance;
- `router_dataset.jsonl`: Phase 7 training input;
- `gate.json`: machine-readable readiness result; and
- `gate.md`: human-readable summary.

No Phase 7 router is trained in this phase.
