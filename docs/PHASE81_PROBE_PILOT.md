# Phase 8.1 runtime-signal pilot

This pilot answers one narrow question: do runtime confidence signals separate
safe Route 20/23 responses substantially better than the failed prompt-only
router? It is intentionally bounded and has a hard stop condition.

Phase 8 already tested prompt text plus the generated 16-token response as
text. Phase 8.1 adds genuinely new signals:

- normalized next-token entropy;
- top-1 probability and top-1/top-2 margin;
- stability of those quantities over a four-token greedy probe;
- the final prompt-token hidden vector after the first ten shared layers.

The Colab collector reuses the corrected Phase 5 checkpoint, Phase 6.1
artifact, and semantic judgments already stored in Drive. It writes one
resumable file, `phase81_probe_signals.jsonl`.

Run `notebooks/phase81_runtime_signal_pilot_colab.ipynb` on a T4. After adding
the completed signal file to the project, evaluate it locally:

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'src')
python -m adaptivedepth.cli evaluate-phase81-probes `
  --probes phase81_probe_signals.jsonl `
  --dataset experiments/phase72_application_dataset.jsonl `
  --router-oof experiments/phase72_semantic/oof_predictions.jsonl `
  --config configs/phase81_probe_t4.yaml `
  --application-config configs/phase72_application.yaml `
  --output experiments/phase81
```

The evaluator compares logit-only, early-hidden, and combined variants using
five grouped out-of-fold splits. The measured median probe/full-route latency
fraction replaces the earlier assumed 25% probe cost.

The pilot continues only if at least one variant achieves all of:

- at least 90% accepted-route precision;
- at least 30% development coverage, close enough to justify testing the
  locked 34% target on fresh prompts;
- positive projected latency reduction;
- median rejected-probe cost no greater than 50% of a routed request.

Anything else produces `stop_probe_router`. A pass is not deployment approval;
it only justifies collecting a fresh validation set.
