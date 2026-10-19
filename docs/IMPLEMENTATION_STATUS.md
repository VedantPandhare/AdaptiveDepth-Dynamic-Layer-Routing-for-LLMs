# AdaptiveDepth implementation status

The design memo defines eleven implementation phases. Progress is counted by
working, tested capabilities—not by files created.

Current implementation milestone: **approximately 80%**. Phase 7.2 and Phase 8
are implemented, but the joint two-route application gate returned
`revise_combined_policy`: 90%-precision coverage is only 4.25%, far below the
34% target. Both artifacts remain fail-closed, and Phase 9 is paused until the
label/model mismatch is corrected.

| Phase | Capability | Status |
|---:|---|---|
| 1 | Reproducible baseline and latency harness | Implemented and exercised on T4 |
| 2 | Block influence, ablation routes, route benchmarking, and teacher KL | Implemented and exercised on T4 |
| 3 | Candidate route compiler and Pareto filtering | Implemented and exercised on T4 |
| 4 | Static route evaluation and go/no-go gate | Complete on T4; valid speed-quality frontier measured, verdict `revise_routes` |
| 5 | Route-aware robustness training | Complete; corrected full model retained quality, but stabilized static gate remains `revise_training` |
| 6 | Route-regret dataset generation | Complete; Phase 6.1 produced 400 response-preserving examples for Routes 20-23 and passed readiness |
| 7 | Calibrated risk router | Implemented; standalone low-coverage gate passes, joint 34% application target fails |
| 8 | Rescue predictor | Implemented; narrow candidate gate passes, broad two-route cascade is not sufficiently discriminative |
| 9 | Probe-and-restart fallback runtime | Not started |
| 10 | End-to-end benchmark | Not started |
| 11 | Required interactive prompt demonstration | Specification locked; implementation follows the completed routing and fallback runtime |

## Current boundary

The implemented runtime deliberately supports one fixed route for a complete
request. It does not activate skipped layers midway through generation. This is
the cache-safe V1 behavior specified by the design memo.

The default checkpoint is `Qwen/Qwen2.5-0.5B-Instruct`. The repository can be
tested without downloading it; real experiment commands load it only when run.

## What counts as the next milestone

Phase 6.1 corrected the token-starved labels and yielded 400 valid examples.
Phase 7.2 reaches 80% development precision, 5% coverage, and 0.7% gross
latency reduction with Routes 20 and 23, but Route 23 calibration remains over
the locked bound. Phase 8 then evaluates 1,600 prompt-route outputs with
prompt-grouped folds; on the 20 router candidates it accepts 70% at 100%
observed precision and rescues every observed unsafe candidate. These small,
reused development samples require fresh validation. See
[`PHASE72_PHASE8_RUNBOOK.md`](PHASE72_PHASE8_RUNBOOK.md).

The final project includes a required interactive application where a user
enters a prompt and sees the generated answer, selected route, active/skipped
layers, predicted route risk, latency savings, and any fallback event. Its
acceptance criteria are defined in
[`DEMO_APPLICATION_SPEC.md`](DEMO_APPLICATION_SPEC.md).
