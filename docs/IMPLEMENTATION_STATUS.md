# AdaptiveDepth implementation status

The design memo defines eleven implementation phases. Progress is counted by
working, tested capabilities—not by files created.

Current implementation milestone: **approximately 35%**. This measures tested
software capabilities only; it does not claim that real-model experiments have
already been completed.

| Phase | Capability | Status |
|---:|---|---|
| 1 | Reproducible baseline and latency harness | Implemented; real-model run pending checkpoint availability |
| 2 | Block influence, ablation routes, route benchmarking, and teacher KL | Implemented; real measurements pending |
| 3 | Candidate route compiler and Pareto filtering | Implemented; real candidate measurements pending |
| 4 | Static route evaluation and go/no-go gate | Complete on T4; valid speed-quality frontier measured, verdict `revise_routes` |
| 5 | Route-aware robustness training | Not started |
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
quality or teacher agreement. The next implementation milestone is Phase 5:
route-aware robustness training on a compact measured route library, followed
by the same static gate before any risk router is trained.

The final project includes a required interactive application where a user
enters a prompt and sees the generated answer, selected route, active/skipped
layers, predicted route risk, latency savings, and any fallback event. Its
acceptance criteria are defined in
[`DEMO_APPLICATION_SPEC.md`](DEMO_APPLICATION_SPEC.md).
