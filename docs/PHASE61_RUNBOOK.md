# Phase 6.1 response-preserving multi-route runbook

## Why Phase 6.1 is required

The first Phase 6 run produced 200 valid examples, but 62 examples (31%) used
only one reference token for KL, loss-regret, and token-agreement labels. Long
prompts consumed the original 128-token sequence budget before enough response
tokens could be retained. The resulting labels were valid structurally but too
weak for dependable route-risk supervision.

Phase 6.1 replaces that collection rather than merely adding more examples to
the noisy dataset. It reserves up to 16 response tokens and truncates the
middle of an oversized prompt, retaining both its instruction prefix and its
assistant-generation suffix.

## Multi-route collection

For each of 400 fresh Dolly prompts, one combined experiment evaluates:

- `route_fast_20`;
- `route_target_21`;
- `route_balanced_22`;
- `route_safe_23`; and
- the full 24-layer model.

The full reference logits and full generated answer are calculated once per
prompt and reused for all four comparisons. Routes execute sequentially using
the same loaded checkpoint, so GPU memory is close to the original Phase 6
requirement rather than five times larger.

The generation measurement order rotates across all five paths on successive
examples. This prevents one route from consistently benefiting from GPU clock
or thermal order. Each route records continuous regret, calibrated safety
criteria, generated-answer agreement, fixed-length latency, and memory.

Safety requires the original KL, reference-loss-regret, and response-token
agreement checks plus at least 50% token-set agreement between the full and
routed generated answers. The latter is evaluated on the actual untruncated
prompt, preventing prompt truncation from silently authorizing a route whose
real generated behavior diverges.

After collection, routes are ordered by measured median latency. Each prompt's
target is the cheapest route that passes every safety criterion, or `full` if
none pass.

## Readiness gate

The full configuration requires:

- at least 300 valid examples;
- at least 20 safe and 20 unsafe examples for a trainable route;
- at least two trainable routes with 2% or better median latency reduction;
- at least 20 examples whose cheapest safe target is the full model.

A route that is measured but fails these conditions remains in the report and
dataset but is not declared eligible for Phase 7 routing.

## Runtime and persistence

The completed 400-example, four-route run took approximately 16.5 minutes on a
T4. Future runs should normally allow 15-30 minutes, with additional margin for
Colab provisioning or throttling.

The full workspace is stored directly under
`MyDrive/AdaptiveDepth-results/phase61_workspace`. Every completed prompt is
appended and flushed independently. If Colab disconnects, rerun the notebook
with the same repository commit and checkpoint; `--resume` skips completed
examples. This remains one combined experiment even if infrastructure forces
a resumed session.

## Colab procedure

1. Push the Phase 6.1 implementation to GitHub.
2. Confirm that
   `MyDrive/AdaptiveDepth-results/phase5_corrected_student.zip` exists.
3. Open
   [`../notebooks/phase61_multiroute_drive.ipynb`](../notebooks/phase61_multiroute_drive.ipynb)
   with a T4 GPU.
4. Run the smoke mode once. Its gate is diagnostic.
5. Set `RUN_FULL = True` and rerun the mode, collection, and packaging cells.
6. Preserve `phase61_workspace` until Phase 7.1 is complete.
7. Provide `phase61_artifacts.zip` or its `gate.md` for analysis.

## Outputs

- `experiment.json`: identity, environment, status, and resume metadata;
- `records.jsonl`: append-only raw measurements;
- `router_dataset.jsonl`: finalized examples with safe routes and the
  cheapest-safe target;
- `gate.json`: machine-readable multi-route readiness result;
- `gate.md`: human-readable route table and verdict.

The archive contains measurements only and does not duplicate the model.
