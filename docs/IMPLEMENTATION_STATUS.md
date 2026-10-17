# AdaptiveDepth implementation status

The design memo defines eleven implementation phases. Progress is counted by
working, tested capabilities—not by files created.

Current implementation milestone: **approximately 55%**. This measures tested
software capabilities plus completed cloud execution. Phase 5 training,
correction, and stabilized confirmation are complete. The compact application
track Phase 6 collector is implemented and awaits its cloud run.

| Phase | Capability | Status |
|---:|---|---|
| 1 | Reproducible baseline and latency harness | Implemented and exercised on T4 |
| 2 | Block influence, ablation routes, route benchmarking, and teacher KL | Implemented and exercised on T4 |
| 3 | Candidate route compiler and Pareto filtering | Implemented and exercised on T4 |
| 4 | Static route evaluation and go/no-go gate | Complete on T4; valid speed-quality frontier measured, verdict `revise_routes` |
| 5 | Route-aware robustness training | Complete; corrected full model retained quality, but stabilized static gate remains `revise_training` |
| 6 | Route-regret dataset generation | Compact selective-routing collector implemented; cloud smoke/full collection pending |
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

Phase 5 correction reduced `route_target_21` KL to `0.3973` and preserved the
full model, but the stabilized latency confirmation rejected every route under
the static three-constraint gate. The 21-layer route remains a near miss: it
saves `11.9%` latency and passes KL, while its quality drop is `0.067` rather
than the allowed `0.050`. The application track therefore tests selective use
of that route with mandatory full-model fallback. The next execution milestone
is the compact route-regret collection documented in
[`PHASE6_RUNBOOK.md`](PHASE6_RUNBOOK.md).

The final project includes a required interactive application where a user
enters a prompt and sees the generated answer, selected route, active/skipped
layers, predicted route risk, latency savings, and any fallback event. Its
acceptance criteria are defined in
[`DEMO_APPLICATION_SPEC.md`](DEMO_APPLICATION_SPEC.md).
