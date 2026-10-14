# AdaptiveDepth implementation status

The design memo defines eleven implementation phases. Progress is counted by
working, tested capabilities—not by files created.

Current implementation milestone: **approximately 30%**. This measures tested
software capabilities only; it does not claim that real-model experiments have
already been completed.

| Phase | Capability | Status |
|---:|---|---|
| 1 | Reproducible baseline and latency harness | Implemented; real-model run pending checkpoint availability |
| 2 | Block influence, ablation routes, route benchmarking, and teacher KL | Implemented; real measurements pending |
| 3 | Candidate route compiler and Pareto filtering | Implemented; real candidate measurements pending |
| 4 | Static route evaluation and go/no-go gate | Runner, controls, resume safety, reports, and cloud notebook implemented; real GPU run pending |
| 5 | Route-aware robustness training | Not started |
| 6 | Route-regret dataset generation | Not started |
| 7 | Calibrated risk router | Not started |
| 8 | Rescue predictor | Not started |
| 9 | Probe-and-restart fallback runtime | Not started |
| 10 | End-to-end benchmark | Not started |
| 11 | Dashboard | Not started |

## Current boundary

The implemented runtime deliberately supports one fixed route for a complete
request. It does not activate skipped layers midway through generation. This is
the cache-safe V1 behavior specified by the design memo.

The default checkpoint is `Qwen/Qwen2.5-0.5B-Instruct`. The repository can be
tested without downloading it; real experiment commands load it only when run.

## What counts as the next milestone

The immediate next action is to run `adaptivedepth run-phase4` on one T4 or
P100 session and preserve the generated experiment and gate reports. Phase 4 is
only experimentally complete after those artifacts exist and have been
reviewed. A `proceed` verdict advances the project to route-aware robustness
training; `revise_routes` means the compiler or budgets must be adjusted before
training a router.
