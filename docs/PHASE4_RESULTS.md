# Phase 4 static-route results

The first complete static-route experiment ran on a Tesla T4 using
`Qwen/Qwen2.5-0.5B-Instruct`, the 30-case English evaluation set, fixed-length
16-token latency profiling, and the conservative 20-23-layer route search.

The original saved predictions were rescored with `answer_match_v2`. This
corrects two shortcomings in strict normalized exact match: formatted options
such as `A) animals` are recognized as option A, and short factual answers are
accepted inside harmless explanatory formatting. No inference or latency value
was recomputed during rescoring.

## Gate outcome

**Verdict:** `revise_routes`

No route simultaneously achieved all three requirements:

- quality drop at most 5%;
- fixed-length latency reduction of at least 10%;
- mean teacher KL at most 0.5.

The rescored full-model accuracy was 50% and median fixed-length latency was
496.72 ms. Representative operating points were:

| Route | Layers | Quality change | Latency reduction | Teacher KL |
|---|---:|---:|---:|---:|
| `candidate_b20_v01` | 20 | -40.0% | 15.4% | 2.1234 |
| `random_b21_v00` | 21 | -10.0% | 11.3% | 3.127 |
| `candidate_b22_v00` | 22 | -10.0% | 6.6% | 1.1258 |
| `candidate_b23_v00` | 23 | -3.3% | 5.8% | 0.7015 |
| `candidate_b23_v01` | 23 | +3.3% | 5.2% | 0.7469 |

Positive quality changes on this small set are not evidence that a route is
better than the full model. They indicate sampling variance and occasional
changes in answer formatting.

## Decision

The experiment establishes a real speed-quality frontier, but the untouched
checkpoint is not robust enough for the target operating point. Phase 5 should
train a small measured route library using route-aware dropout and full-model
distillation, then rerun this exact gate without relaxing its thresholds.
