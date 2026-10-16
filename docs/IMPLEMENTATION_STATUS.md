# AdaptiveDepth implementation status

The design memo defines eleven implementation phases. Progress is counted by
working, tested capabilities—not by files created.

Current implementation milestone: **approximately 43%**. This measures tested
software capabilities plus the completed Phase 4 cloud experiment. Phase 5 is
implemented locally but its real T4 training run and post-training gate remain.

| Phase | Capability | Status |
|---:|---|---|
| 1 | Reproducible baseline and latency harness | Implemented and exercised on T4 |
| 2 | Block influence, ablation routes, route benchmarking, and teacher KL | Implemented and exercised on T4 |
| 3 | Candidate route compiler and Pareto filtering | Implemented and exercised on T4 |
| 4 | Static route evaluation and go/no-go gate | Complete on T4; valid speed-quality frontier measured, verdict `revise_routes` |
| 5 | Route-aware robustness training | Implemented; T4 smoke/full run pending |
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
notebook. The next execution milestone is its T4 run followed by the same
static gate before any risk router is trained.

The final project includes a required interactive application where a user
enters a prompt and sees the generated answer, selected route, active/skipped
layers, predicted route risk, latency savings, and any fallback event. Its
acceptance criteria are defined in
[`DEMO_APPLICATION_SPEC.md`](DEMO_APPLICATION_SPEC.md).
