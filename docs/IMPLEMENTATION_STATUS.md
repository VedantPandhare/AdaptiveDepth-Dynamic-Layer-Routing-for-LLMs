# AdaptiveDepth implementation status

The design memo defines eleven implementation phases. Progress is counted by
working, tested capabilities—not by files created.

Current implementation milestone: **approximately 70%**. This measures tested
software capabilities plus completed cloud execution. The response-preserving
Phase 6.1 collection and the first leakage-controlled multi-route Phase 7.1
evaluation are complete. The router gate returned `revise_router`, so its
artifact remains disabled.

| Phase | Capability | Status |
|---:|---|---|
| 1 | Reproducible baseline and latency harness | Implemented and exercised on T4 |
| 2 | Block influence, ablation routes, route benchmarking, and teacher KL | Implemented and exercised on T4 |
| 3 | Candidate route compiler and Pareto filtering | Implemented and exercised on T4 |
| 4 | Static route evaluation and go/no-go gate | Complete on T4; valid speed-quality frontier measured, verdict `revise_routes` |
| 5 | Route-aware robustness training | Complete; corrected full model retained quality, but stabilized static gate remains `revise_training` |
| 6 | Route-regret dataset generation | Complete; Phase 6.1 produced 400 response-preserving examples for Routes 20-23 and passed readiness |
| 7 | Calibrated risk router | Multi-route Phase 7.1 implemented and evaluated; locked verdict `revise_router`, artifact disabled |
| 8 | Rescue predictor | Not started |
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
The first locked Phase 7.1 run enabled development thresholds for Routes 20
and 23, then achieved 72.7% held-out safe precision, 13.8% coverage, and 1.8%
gross latency reduction. Precision and calibration missed the locked gate, so
the saved router fails closed. Phase 8 must not activate routing from this
artifact. Any correction must be evaluated on fresh untouched examples rather
than tuning against the inspected holdout. See
[`PHASE71_RUNBOOK.md`](PHASE71_RUNBOOK.md).

The final project includes a required interactive application where a user
enters a prompt and sees the generated answer, selected route, active/skipped
layers, predicted route risk, latency savings, and any fallback event. Its
acceptance criteria are defined in
[`DEMO_APPLICATION_SPEC.md`](DEMO_APPLICATION_SPEC.md).
