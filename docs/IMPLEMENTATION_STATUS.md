# AdaptiveDepth implementation status

The design memo defines eleven implementation phases. Progress is counted by
working, tested capabilities—not by files created.

Current implementation milestone: **approximately 65%**. This measures tested
software capabilities plus completed cloud execution. Phase 6 collection and
the first leakage-controlled Phase 7 risk-router evaluation are complete. The
router gate returned `revise_router`, and Phase 6.1 is implemented to replace
token-starved labels with a 400-prompt, four-route collection.

| Phase | Capability | Status |
|---:|---|---|
| 1 | Reproducible baseline and latency harness | Implemented and exercised on T4 |
| 2 | Block influence, ablation routes, route benchmarking, and teacher KL | Implemented and exercised on T4 |
| 3 | Candidate route compiler and Pareto filtering | Implemented and exercised on T4 |
| 4 | Static route evaluation and go/no-go gate | Complete on T4; valid speed-quality frontier measured, verdict `revise_routes` |
| 5 | Route-aware robustness training | Complete; corrected full model retained quality, but stabilized static gate remains `revise_training` |
| 6 | Route-regret dataset generation | Initial T4 run exposed token-starved labels; response-preserving four-route Phase 6.1 implementation is ready for its cloud run |
| 7 | Calibrated risk router | Binary version implemented and evaluated with `revise_router`; multi-route revision follows the corrected Phase 6.1 dataset |
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

The first Phase 7 prompt-only classifier had development out-of-fold ROC AUC
`0.5476` and could not achieve 80% safe precision at 5% coverage. Subsequent
inspection found that 62 of 200 Phase 6 examples retained only one reference
token, confounding the safety labels. Phase 6.1 now reserves response tokens,
uses 400 fresh prompts, and measures all four trained routes in one resumable
experiment. Phase 8 must not begin until the resulting multi-route Phase 7.1
router passes a fresh leakage-controlled gate. See
[`PHASE61_RUNBOOK.md`](PHASE61_RUNBOOK.md).

The final project includes a required interactive application where a user
enters a prompt and sees the generated answer, selected route, active/skipped
layers, predicted route risk, latency savings, and any fallback event. Its
acceptance criteria are defined in
[`DEMO_APPLICATION_SPEC.md`](DEMO_APPLICATION_SPEC.md).
