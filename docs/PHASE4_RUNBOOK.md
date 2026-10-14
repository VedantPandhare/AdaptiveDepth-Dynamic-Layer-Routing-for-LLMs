# Phase 4 cloud runbook

Phase 4 answers one question before any router is trained: does at least one
influence-compiled static route preserve enough quality while producing a real
latency benefit, and is it better than simple static controls?

## Recommended environment

Use a fresh Colab or Kaggle GPU session with a T4 or P100. The default run uses
`Qwen/Qwen2.5-0.5B-Instruct` in float16. A CPU run is supported by the code but
is not useful for the latency decision.

The hosted workflow is in
[`notebooks/phase4_colab_kaggle.ipynb`](../notebooks/phase4_colab_kaggle.ipynb).
It installs the package, confirms that CUDA is visible, runs the experiment,
prints the gate report, and archives the artifacts for download.

From an existing checkout, run:

```bash
pip install -e .
adaptivedepth run-phase4 \
  --config configs/cloud_t4.yaml \
  --dataset data/phase4_eval.jsonl \
  --compiler-config configs/compiler.yaml \
  --evaluation-config configs/evaluation.yaml \
  --workspace experiments/phase4 \
  --resume
```

## Outputs

The workspace contains:

- `block_influence.json`: mean hidden-state change when each block is skipped.
- `candidate_routes.yaml`: routes proposed from the influence ranking.
- `static_suite.yaml`: candidates plus fixed-prefix, importance-only, and
  seeded-random controls at matching budgets.
- `experiment.json`: full-model and per-route quality, latency, teacher KL,
  memory, raw measurements, environment metadata, and completion state.
- `gate.json` and `gate.md`: threshold checks, Pareto routes, control
  comparisons, and the final verdict.

The experiment file is written atomically after every completed route. If the
runtime disconnects, reconnect to the same GPU type, restore the workspace,
and rerun the command with `--resume`. Resume validation rejects incompatible
model, dataset, settings, source revision, or hardware so timings are not
silently mixed.

## Interpreting the verdict

- `proceed`: at least one compiled route meets the quality, latency, and
  teacher-divergence limits and is not dominated by a static control.
- `revise_routes`: the measurements are complete, but no compiled route meets
  all constraints. Inspect the failed constraint before changing budgets or
  compiler preservation rules.
- `insufficient_data`: the full suite has not completed or required
  measurements are absent. Resume the run rather than drawing a conclusion.

The default gate permits at most a 5-point exact-match drop, requires at least
10% latency reduction, limits mean teacher KL to 0.5, and requires an advantage
over the control routes. These are engineering defaults for the practice
project, not research claims; adjust them only before starting a fresh,
separately named experiment.
