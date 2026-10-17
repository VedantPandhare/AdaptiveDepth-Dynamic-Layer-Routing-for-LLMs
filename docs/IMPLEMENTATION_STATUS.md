# AdaptiveDepth implementation status

The design memo defines eleven implementation phases. Progress is counted by
working, tested capabilities—not by files created.

Current implementation milestone: **approximately 49%**. This measures tested
software capabilities plus completed cloud execution. Phase 5 smoke, the
500-step training run, and its post-training evaluation are complete. A single
bounded correction is implemented and awaits cloud execution.

| Phase | Capability | Status |
|---:|---|---|
| 1 | Reproducible baseline and latency harness | Implemented and exercised on T4 |
| 2 | Block influence, ablation routes, route benchmarking, and teacher KL | Implemented and exercised on T4 |
| 3 | Candidate route compiler and Pareto filtering | Implemented and exercised on T4 |
| 4 | Static route evaluation and go/no-go gate | Complete on T4; valid speed-quality frontier measured, verdict `revise_routes` |
| 5 | Route-aware robustness training | 500-step run and evaluation complete; full-depth retention passed, bounded route correction ready |
| 6 | Route-regret dataset generation | Not started |
| 7 | Calibrated risk router | Not started |
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

The Phase 4 T4 experiment is complete and documented in
[`PHASE4_RESULTS.md`](PHASE4_RESULTS.md). Its `revise_routes` verdict shows that
the raw checkpoint gains latency from reduced depth but does not retain enough
quality or teacher agreement. Phase 5 now has a fixed measured route library,
separate training data, frozen-teacher distillation, deterministic sampling,
atomic resumable checkpoints, bounded checkpoint retention, tests, and a cloud
notebook. Its smoke and 500-step T4 runs completed successfully, and the full
checkpoint passed structural and finite-weight checks. The post-training gate
preserved full-depth quality but returned `revise_training`:
`route_target_21` passed quality and latency but recorded KL `0.6625` against
the `0.5000` limit. The next execution milestone is the one bounded targeted
correction documented in
[`PHASE5_CORRECTION_RUNBOOK.md`](PHASE5_CORRECTION_RUNBOOK.md). Its unchanged
acceptance gate decides whether the corrected checkpoint replaces the original.

The final project includes a required interactive application where a user
enters a prompt and sees the generated answer, selected route, active/skipped
layers, predicted route risk, latency savings, and any fallback event. Its
acceptance criteria are defined in
[`DEMO_APPLICATION_SPEC.md`](DEMO_APPLICATION_SPEC.md).
