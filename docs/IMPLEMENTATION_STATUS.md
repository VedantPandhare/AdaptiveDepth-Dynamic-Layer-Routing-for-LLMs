# AdaptiveDepth implementation status

The design memo defines eleven implementation phases. Progress is counted by
working, tested capabilities—not by files created.

| Phase | Capability | Status |
|---:|---|---|
| 1 | Reproducible baseline and latency harness | Implemented; real-model run pending checkpoint availability |
| 2 | Block influence and layer-analysis foundation | Implemented |
| 3 | Candidate route compiler and Pareto filtering | Implemented foundation; candidate benchmarking is next |
| 4 | Static route evaluation and go/no-go gate | Not started |
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
