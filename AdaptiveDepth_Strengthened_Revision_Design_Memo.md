# AdaptiveDepth — Strengthened Revision Design Memo

## Risk-Aware Routing over Validated Transformer Depths with Selective Fallback

**Repository / portfolio name:** `AdaptiveDepth`  
**Recommended research title:**  
**AdaptiveDepth: Risk-Aware Routing over Validated Transformer Depths with Selective Fallback**

---

# 1. Purpose of This Document

This document describes the strengthened version of the AdaptiveDepth project and explains **why each design change was made**, **what weakness existed in the previous revision**, and **how the new design improves technical soundness, implementation feasibility, deployment realism, research clarity, and experimental defensibility**.

The earlier revision already improved substantially on the original idea of predicting independent skip/execute decisions for every transformer layer. It introduced:

1. offline layer-importance analysis,
2. a small route library,
3. difficulty-aware routing,
4. layer-dropout / distillation-based robustness,
5. confidence-based fallback,
6. runtime benchmarking, and
7. an execution dashboard.

That revision was a good engineering direction. However, several parts were still too loosely defined for a serious research implementation:

- “difficulty” was not the correct quantity to optimize,
- routes were described too simplistically as shallow/medium/deep,
- fallback behavior was underspecified,
- confidence was treated as if it directly implied correctness,
- switching routes during generation ignored KV-cache and prefix-consistency problems,
- the route library lacked a formal construction objective,
- robustness training was not tied tightly enough to the actual deployed routes,
- latency was treated as a route property rather than an **end-to-end expected cost including fallback**,
- comparison against fine-grained dynamic routing was not yet central to the research story.

The strengthened design fixes those problems.

---

# 2. Executive Summary

The final system should not be framed as:

> “An LLM that detects whether a prompt is easy, medium, or hard and chooses a shallower or deeper network.”

That framing is intuitive but too naive.

Instead, AdaptiveDepth should be framed as:

> **A budget-aware LLM inference system that compiles a small library of empirically validated transformer execution routes offline, predicts the cheapest route likely to preserve full-model behavior for each request, and selectively escalates to deeper execution when the predicted value of rescue justifies the additional compute.**

The system consists of four major ideas:

```text
                  OFFLINE
                    │
                    ▼
          Layer / Block Analysis
                    │
                    ▼
            Route Candidate Search
                    │
                    ▼
             Pareto Route Compiler
                    │
                    ▼
        Validated Route Library
        A → B → C → D → Full
                    │
                    │
                ONLINE
                    ▼
                Prompt
                    │
                    ▼
            Route Risk Predictor
                    │
                    ▼
      Cheapest sufficiently safe route
                    │
                    ▼
             Probe / Generation
                    │
                    ▼
             Rescue Predictor
               │          │
             Accept     Escalate
                          │
                          ▼
                    Deeper Route
```

The important distinction is that the system is not trying to discover an arbitrary transformer topology for every request. Instead, it chooses among a small number of **validated subnetworks whose behavior, latency, and quality are already known**.

This restriction makes the system:

- easier to train,
- easier to benchmark,
- easier to deploy,
- easier to cache,
- easier to explain,
- easier to reproduce,
- more predictable under latency constraints,
- and potentially competitive with more complex fine-grained routing methods.

---

# 3. Evolution of the Project

## 3.1 Original Version

The original concept was approximately:

```text
Prompt
  ↓
Router
  ↓
Predict a skip / execute decision for each layer
  ↓
[1,1,0,1,0,1,...]
  ↓
Execute custom transformer path
  ↓
Answer
```

### Strength

The original version had a strong high-level intuition:

> Different inputs may not require the same amount of transformer computation.

### Weakness

For a 24-layer network, binary layer execution already gives:

\[
2^{24}
\]

possible masks in the worst case.

That means the router must learn a very large structured decision problem.

The original design also creates practical issues:

- enormous path search space,
- many unvalidated layer combinations,
- irregular runtime,
- poor batching characteristics,
- difficult debugging,
- potential representation breakage after arbitrary skips,
- complicated KV-cache semantics,
- hard-to-explain routing decisions,
- more difficult comparison between configurations.

Therefore, the original project was conceptually interesting but too unconstrained.

---

# 4. Previous Revision

The previous revision improved the system by introducing:

```text
Route A → roughly 12 layers
Route B → roughly 16 layers
Route C → roughly 20 layers
Full    → all layers
```

The workflow became:

1. analyze layer importance,
2. create a few validated routes,
3. train the model to tolerate reduced depth,
4. classify prompt difficulty,
5. select a route,
6. generate,
7. fall back if confidence is low.

This was a major improvement because it reduced the online decision space.

However, several conceptual problems remained.

---

# 5. The Main Failure of the Previous Revision

The most important issue is that it implicitly assumed:

\[
\text{prompt difficulty} \approx \text{required transformer depth}
\]

This is not guaranteed.

A prompt can appear difficult but still be solved by a shallow route because the necessary pattern is represented early.

A prompt can also appear simple but depend strongly on representations produced by layers that the shallow route omits.

Consider two prompts:

```text
Prompt A:
"What is 17 × 23?"

Prompt B:
"Which of the following statements logically follows from the passage?"
```

A generic “difficulty classifier” might assign:

```text
Prompt A → medium
Prompt B → medium
```

But what AdaptiveDepth actually needs to know is:

```text
Will Route A behave sufficiently like the full model on this exact request?
```

That is a different problem.

Therefore, the strengthened system does **not** primarily predict task difficulty.

It predicts **route-specific risk**.

---

# 6. Strengthened Change 1 — Replace Difficulty Prediction with Route-Risk / Regret Prediction

## 6.1 Previous Design

```text
Prompt
  ↓
Difficulty classifier
  ↓
Easy / Medium / Hard
  ↓
Route A / B / C
```

## 6.2 Why This Is Weak

“Difficulty” is an indirect proxy.

It does not directly answer:

> How much quality will be lost if I execute this prompt on Route A instead of the full model?

It also creates ambiguity.

What does “hard” mean?

- long prompt?
- mathematical?
- low model confidence?
- high entropy?
- unusual vocabulary?
- many reasoning steps?
- difficult according to humans?
- difficult specifically for this model?

Those are not equivalent.

## 6.3 Strengthened Design

For prompt \(x\) and route \(r\), define a route-regret target:

\[
R(x,r)=D(P_r(y|x),P_F(y|x))
\]

where:

- \(P_r\) is the output behavior of route \(r\),
- \(P_F\) is the full model,
- \(D\) measures disagreement.

Possible disagreement measures:

- KL divergence,
- Jensen-Shannon divergence,
- logit cosine distance,
- top-k token disagreement,
- answer disagreement,
- task-score degradation,
- sequence-level teacher loss.

The router learns:

\[
\hat{R}(x,r)
\]

for each route.

It then chooses:

\[
r^*=\arg\min_r C(r)
\]

subject to:

\[
\hat{R}(x,r)<\tau
\]

where \(C(r)\) represents compute cost or measured latency.

## 6.4 Why This Is Better

The optimization target now matches the actual deployment objective:

> Find the cheapest route predicted to preserve acceptable model behavior.

The router is no longer required to understand abstract task difficulty.

It only needs to predict:

> Is this route safe enough for this prompt?

This makes training labels easier to derive automatically using the full model as a teacher.

---

# 7. Strengthened Change 2 — Build a Route Compiler Instead of Hand-Choosing Depths

## 7.1 Previous Design

The earlier revision suggested routes such as:

```text
Route A = 12 layers
Route B = 16 layers
Route C = 20 layers
Full    = 24 layers
```

This is useful conceptually but too simplistic.

It assumes that the best 12-layer network is merely a prefix or a manually selected set.

That may not be true.

## 7.2 Strengthened Design

AdaptiveDepth should contain an **offline route compiler**.

The compiler takes the full model and produces a small set of validated routes.

```text
Full Model
   │
   ├── Block influence
   ├── Single-layer ablation
   ├── Group ablation
   ├── Teacher-logit divergence
   ├── Task evaluation
   ├── Actual latency
   └── VRAM / memory behavior
              │
              ▼
      Candidate Route Pool
              │
              ▼
         Pareto Filtering
              │
              ▼
       Validated Route Library
```

Example route:

```text
Route A:
1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24
```

rather than necessarily:

```text
1, 2, 3, ..., 12
```

---

# 8. Why Pareto Filtering Matters

Suppose three candidate routes produce:

| Route | Accuracy | Latency | Active layers |
|---|---:|---:|---:|
| R1 | 76.1 | 280 ms | 12 |
| R2 | 75.4 | 310 ms | 14 |
| R3 | 78.7 | 350 ms | 16 |

R2 should probably be removed.

Why?

Because R1 is:

- faster,
- more accurate,
- shallower.

R2 is **dominated**.

AdaptiveDepth should preserve only Pareto-useful routes.

A candidate survives if no alternative route is simultaneously:

- at least as accurate,
- at least as fast,
- and no more expensive.

This converts the route library into a set of meaningful operating points.

---

# 9. Strengthened Change 3 — Route Library Should Represent Validated Subnetworks, Not Merely Depth

The important abstraction should be:

> Route = ordered transformer subnetwork

not:

> Route = number of layers.

Therefore:

```text
Route A ≠ "12 layers"
```

Instead:

```text
Route A = [0,1,2,3,5,7,9,11,14,17,20,23]
```

A route has metadata:

```yaml
route_id: route_a
active_layers: [...]
mean_latency_ms: ...
p95_latency_ms: ...
task_accuracy: ...
teacher_kl: ...
peak_vram_mb: ...
robustness_score: ...
```

This makes routes measurable artifacts rather than vague execution budgets.

---

# 10. ShortGPT-Inspired Layer Analysis

ShortGPT motivates using block-level influence to identify transformer layers whose removal may minimally affect network behavior.

AdaptiveDepth should use this idea as **one signal**, not as the entire route-construction algorithm.

Offline measurements can include:

### A. Block influence

Measure how much representations change after a block.

### B. Single-layer removal

Remove one layer at a time and measure:

- task score,
- logit divergence,
- perplexity,
- latency.

### C. Group removal

Some layers may be individually removable but harmful when removed together.

Therefore test groups.

### D. Teacher-logit divergence

For a route:

\[
D_{KL}(P_F \parallel P_r)
\]

can measure how far the route drifts from the full model.

### E. Task-level behavior

Do not rely only on hidden-state similarity.

A layer can appear representation-redundant while still mattering disproportionately for specific reasoning tasks.

---

# 11. Why Layer Importance Alone Is Not Enough

The previous design could be interpreted as:

> Find unimportant layers and build routes from them.

That is insufficient because layer interactions are non-additive.

For example:

```text
remove L8 → negligible effect
remove L11 → negligible effect
remove L8 + L11 → large degradation
```

Therefore route construction must validate **entire candidate subnetworks**.

The unit of deployment is the route, so the unit of validation must also be the route.

---

# 12. Strengthened Change 4 — Route-Specific Robustness Training

## 12.1 Previous Design

The earlier version proposed generic:

- layer dropout,
- distillation,
- intermediate losses.

That direction was correct but underspecified.

## 12.2 Problem

Random layer dropout may train the model to tolerate many arbitrary configurations that are never used at deployment.

This wastes training capacity.

## 12.3 Strengthened Design

Once the route compiler identifies:

```text
A
B
C
D
Full
```

training should heavily sample **these exact deployment routes**.

Example:

```text
Training batch
   │
   ├── Full route: 30%
   ├── Route D:    20%
   ├── Route C:    20%
   ├── Route B:    15%
   └── Route A:    15%
```

Use distillation:

\[
L=L_{task}+\lambda_{KD}L_{KD}+\lambda_HL_{hidden}
\]

where:

- \(L_{task}\): supervised task / language-model loss,
- \(L_{KD}\): teacher-student logit distillation,
- \(L_{hidden}\): optional representation consistency.

The full model acts as teacher.

---

# 13. LayerSkip-Inspired Robustness, Not an Incorrect Claim of Reimplementation

LayerSkip uses a specific training and inference recipe involving layer dropout, early-exit behavior and self-speculative verification.

AdaptiveDepth should only say:

> “LayerSkip-inspired depth-robust training”

unless the exact LayerSkip procedure is reproduced.

The actual AdaptiveDepth contribution is different:

- validated route library,
- route-level routing,
- route risk prediction,
- selective escalation.

This distinction is important for scientific accuracy.

---

# 14. Strengthened Change 5 — Replace Generic Confidence Fallback with Selective Rescue

## 14.1 Previous Design

```text
Generate using Route A
   ↓
if confidence < threshold:
   use deeper route
```

This is intuitive but problematic.

---

# 15. Why Confidence Alone Is Weak

Confidence can be miscalibrated.

A shallow route can be:

```text
wrong and confident
```

or:

```text
correct and uncertain
```

Therefore:

\[
\text{low confidence}\neq\text{wrong}
\]

and:

\[
\text{high confidence}\neq\text{safe}
\]

Even more importantly:

> Escalation is only useful if the deeper route is expected to improve the answer.

---

# 16. Rescue-Based Escalation

For shallow route \(A\) and deeper route \(B\), define:

\[
G_{A\rightarrow B}(x)=P(B\ correct,A\ wrong)-P(B\ wrong,A\ correct)
\]

Interpretation:

- first term = probability deeper route rescues shallow failure,
- second term = probability escalation damages an already correct result.

Then escalation occurs when:

\[
G_{A\rightarrow B}(x)>\lambda \Delta C
\]

where:

\[
\Delta C=C(B)-C(A)
\]

is the additional compute cost.

This creates a decision based on **expected utility**, not raw confidence.

---

# 17. Strengthened Change 6 — Introduce Two Distinct Predictors

The online controller should conceptually separate:

## Router

Predict:

> What is the cheapest route that is likely to be safe?

## Rescue predictor

Predict:

> Given the current cheap-route behavior, is deeper computation likely to materially improve the result?

These may eventually share an encoder, but separating their responsibilities improves interpretability.

---

# 18. Online Inference Algorithm

A simplified algorithm:

```text
INPUT: prompt x

1. Router predicts risk for A, B, C, D.
2. Select cheapest route r whose predicted risk < threshold.
3. Execute route r.
4. Gather route-level diagnostic signals.
5. Rescue predictor estimates expected gain from deeper route.
6. If expected rescue gain is low:
       accept result.
7. Otherwise:
       escalate to next route or full model.
8. Return final output + execution metadata.
```

---

# 19. Strengthened Change 7 — Fix the Fallback / KV-Cache Problem

This is one of the most important implementation changes.

## 19.1 Previous Revision Problem

The previous description implied:

```text
Generate several tokens with Route A
→ suddenly activate additional layers
→ continue generation
```

This is not generally safe.

Why?

During autoregressive decoding, every active transformer layer maintains KV states for previous tokens.

If Route A skipped layers:

```text
L5, L8, L11
```

those layers may have no valid KV history for the already-generated prefix.

Activating them later is not equivalent to having processed the complete sequence through them from the start.

---

# 20. Safe MVP Fallback Strategy

For Version 1, fallback should use **probe-and-restart**.

```text
Prompt
  ↓
Route A
  ↓
Generate short probe / score candidate
  ↓
Risk / rescue evaluation
  │
  ├── Safe
  │      ↓
  │   continue Route A
  │
  └── Unsafe
         ↓
      discard cheap-route state
         ↓
      restart prompt using Route B/C/full
```

Possible probe mechanisms:

- next-token distribution,
- 1–4 generated tokens,
- multiple-choice logits,
- answer-choice distribution,
- hidden-state stability,
- route agreement score.

This is slower than magical mid-stream switching but correct and reproducible.

---

# 21. Later Runtime Optimization

A later version may investigate:

- cache reconstruction,
- prefix replay,
- hierarchical KV preparation,
- shared prefix computation,
- route-compatible layer organization,
- speculative route verification.

But these should **not** be required for the MVP.

---

# 22. Strengthened Change 8 — Optimize Expected End-to-End Cost, Not Cheap-Route Latency

Suppose:

```text
Route A latency = 200 ms
Full latency    = 400 ms
```

It is tempting to claim:

```text
50% speed improvement
```

But if Route A falls back 60% of the time:

\[
E[T]=T_A+P(F_A)T_F
\]

approximately:

\[
E[T]=200+0.6(400)=440\text{ ms}
\]

Now the adaptive system is slower than full execution.

Therefore every reported efficiency result must include fallback cost.

---

# 23. Expected Latency Objective

For route \(r\):

\[
E[T_r]=T_r+P(F_r)T_{fallback}
\]

More generally:

\[
E[C]=\sum_r P(r)C(r)+\sum_{r_i\rightarrow r_j}P(r_i\rightarrow r_j)\Delta C_{ij}
\]

This expected-cost accounting is mandatory.

---

# 24. Strengthened Change 9 — Treat the Full Model as a Teacher and Safety Anchor

The full model has three roles.

## A. Teacher

Produces training labels for route regret.

## B. Distillation reference

Reduced routes learn to approximate the full model.

## C. Safe fallback

When route risk is too high, use the full network.

This makes the architecture internally coherent.

---

# 25. Route-Risk Training Dataset

For each prompt:

```text
Prompt x
   │
   ├── Full → teacher output
   ├── Route A → output A
   ├── Route B → output B
   ├── Route C → output C
   └── Route D → output D
```

Store labels:

```text
teacher_score
route_task_score
KL divergence
token agreement
answer agreement
latency
memory
```

Example:

```json
{
  "prompt_id": 184,
  "route_a_regret": 0.28,
  "route_b_regret": 0.08,
  "route_c_regret": 0.02,
  "route_d_regret": 0.01,
  "cheapest_safe_route": "B"
}
```

This creates direct supervision.

---

# 26. Strengthened Change 10 — Route Safety Should Be Calibrated

If the router reports:

```text
Route A safe probability = 0.90
```

then approximately 90% of prompts with similar predicted safety should actually satisfy the safety criterion.

Measure:

- Expected Calibration Error,
- Brier score,
- reliability diagrams.

A badly calibrated router can destroy the fallback policy.

---

# 27. OOD / Uncertainty Handling

The router should have an abstention condition.

Examples:

- prompt embedding far from training distribution,
- high route-risk uncertainty,
- high disagreement between predictors,
- unusual context length,
- unsupported task type.

Then:

```text
OOD / uncertain prompt
        ↓
Route D or Full
```

This supports safe degradation.

---

# 28. Final Strengthened Architecture

```text
                         OFFLINE
                           │
                           ▼
                    Full Base Model
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
      Block Influence   Ablations    Task Evaluation
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                    Candidate Routes
                           │
                           ▼
                    Route Benchmarking
            Quality / Latency / VRAM / KL
                           │
                           ▼
                     Pareto Compiler
                           │
                           ▼
                A → B → C → D → Full
                           │
                           ▼
                 Depth-Robust Training
                           │
                           ▼
            Router + Rescue Training Data
                           │
                           │
                         ONLINE
                           ▼
                         Prompt
                           │
                           ▼
                   Route Risk Predictor
                           │
                           ▼
                 Cheapest Safe Route
                           │
                           ▼
                    Probe / Execute
                           │
                           ▼
                     Rescue Predictor
                    │                │
                  ACCEPT          ESCALATE
                    │                │
                    │          deeper route
                    │                │
                    └────────┬───────┘
                             ▼
                           Output
                             │
                             ▼
                   Execution Dashboard
```

---

# 29. Why This Is Better Than the Previous Revision

| Design area | Previous revision | Strengthened revision |
|---|---|---|
| Routing target | Prompt difficulty | Route-specific regret / safety |
| Route definition | Mostly depth buckets | Validated ordered subnetworks |
| Route creation | Manual / heuristic | Offline route compiler |
| Route filtering | Not formal | Pareto frontier |
| Robustness | Generic layer dropout | Route-aware dropout + distillation |
| Fallback | Low-confidence escalation | Expected-rescue escalation |
| Cache handling | Underspecified | Probe-and-restart MVP |
| Efficiency metric | Route latency | Expected end-to-end latency |
| Uncertainty | Confidence threshold | Calibration + abstention + rescue |
| Research comparison | Generic baselines | Fine-grained routing is a core baseline |
| Main contribution | Adaptive depth | Constrained, risk-aware, recoverable adaptive depth |

---

# 30. Research Hypothesis

Primary hypothesis:

> **A small library of offline-validated transformer execution routes, combined with route-risk prediction and selective rescue, can achieve a better quality–latency tradeoff than fixed-depth execution while approaching the flexibility of fine-grained dynamic routing with substantially simpler runtime behavior.**

---

# 31. Main Research Questions

### RQ1

Can offline route compilation identify a small number of execution paths that cover most useful quality–latency operating points?

### RQ2

Can a prompt-level risk predictor reliably identify the cheapest route that preserves full-model behavior?

### RQ3

Does route-aware robustness training significantly improve shallow-route quality?

### RQ4

Does selective fallback recover routing mistakes without eliminating latency gains?

### RQ5

Does rescue-based escalation outperform confidence-only escalation?

### RQ6

Can a 4–6 route system match or approach fine-grained layer routing while being simpler and more predictable?

### RQ7

How much of the theoretical layer reduction translates into **actual batch-size-one latency improvement**?

---

# 32. MVP Scope

Keep Version 1 narrow.

## Model

One decoder-only model:

```text
0.5B–1.5B parameters
```

## Tasks

Two categories:

1. factual / knowledge QA,
2. reasoning / multiple-choice.

## Routes

Four reduced routes plus full model.

Example:

```text
A
B
C
D
Full
```

## Runtime

- batch size = 1,
- fixed route per complete generation,
- no token-level switching,
- probe-and-restart fallback.

## Language

English only.

No multilingual experiment in the first version.

---

# 33. Stage-by-Stage Methodology

## Stage 0 — Reproducible Baseline Harness

Build a baseline before modifying the model.

Measure:

- task accuracy,
- perplexity if appropriate,
- TTFT,
- prefill latency,
- decode latency/token,
- total latency,
- tokens/sec,
- peak VRAM,
- GPU utilization,
- CPU overhead.

All future claims should be relative to this baseline.

---

## Stage 1 — Layer Analysis

Run:

### Experiment 1

Single-layer deletion.

### Experiment 2

Adjacent-group deletion.

### Experiment 3

Importance-ranked deletion.

### Experiment 4

Random deletion.

For each:

```text
quality
teacher KL
latency
memory
```

Purpose:

Understand whether block-importance signals actually correlate with safe removal.

---

## Stage 2 — Candidate Route Generation

Possible strategy:

1. rank layers by importance,
2. generate candidate masks under layer budgets,
3. enforce architectural constraints,
4. benchmark candidates,
5. retain promising routes.

Budget examples:

```text
12 active blocks
14 active blocks
16 active blocks
18 active blocks
20 active blocks
```

For each budget, evaluate multiple combinations.

---

# 34. Architectural Constraints During Route Search

Avoid fully unrestricted masks initially.

Possible constraints:

- always preserve embedding and LM head,
- preserve first N layers,
- preserve final K layers,
- limit long runs of skipped blocks,
- preserve empirically critical blocks,
- maintain original ordering.

These reduce pathological routes.

---

# 35. Stage 3 — Pareto Route Compiler

Candidate metrics:

\[
Q(r)=\text{task quality}
\]

\[
T(r)=\text{measured latency}
\]

\[
M(r)=\text{peak memory}
\]

\[
D(r)=\text{teacher divergence}
\]

Select routes that form a useful frontier.

Do not keep five routes that behave almost identically.

Each route should represent a meaningful budget tier.

---

# 36. Stage 4 — Static Route Benchmark

Before adaptive routing, answer:

> Are the routes themselves worth using?

Compare:

```text
Full
Fixed shallow prefix
Random skip
ShortGPT-inspired static pruning
Route A
Route B
Route C
Route D
```

If the compiled routes do not beat simple baselines, stop and improve route construction.

This is an important **go/no-go gate**.

---

# 37. Stage 5 — Depth-Robust Training

Two recommended experiments.

## Experiment A — Frozen backbone

Train:

- route router only,
- optional lightweight adapters.

Cheaper and useful as baseline.

## Experiment B — Route-aware fine-tuning

During training, sample deployment routes.

Use teacher distillation.

Possible objective:

\[
L=L_{LM}+\lambda_1D_{KL}(P_F||P_r)+\lambda_2L_{hidden}
\]

Measure whether route robustness increases.

---

# 38. Stage 6 — Build Router Labels

For every training prompt:

1. run full model,
2. run every reduced route,
3. compare outputs,
4. determine cheapest route meeting a safety criterion.

Example:

```text
A regret: 0.32 → unsafe
B regret: 0.09 → safe
C regret: 0.03 → safe
D regret: 0.01 → safe

Target route = B
```

This is cleaner than manually labeling “difficulty”.

---

# 39. Stage 7 — Train Route-Risk Predictor

Input features can include:

- prompt embedding,
- prompt length,
- token entropy,
- first-layer hidden-state summary,
- vocabulary rarity,
- task-format indicators.

Start simple.

Example architecture:

```text
Prompt
  ↓
Small encoder / pooled embedding
  ↓
MLP
  ↓
Risk(A), Risk(B), Risk(C), Risk(D)
```

Avoid expensive router architectures that consume the savings you are trying to create.

---

# 40. Router Objective

Regression:

\[
L_{router}=\sum_r(\hat{R}(x,r)-R(x,r))^2
\]

or classification:

```text
safe / unsafe for each route
```

A useful alternative is ordinal prediction:

```text
cheapest safe route:
A < B < C < D < Full
```

Compare these formulations experimentally.

---

# 41. Stage 8 — Rescue Predictor

After shallow execution, collect additional signals:

- route confidence,
- output entropy,
- top-1/top-2 margin,
- agreement with another cheap probe,
- hidden-state norm statistics,
- router uncertainty.

Train a predictor estimating whether deeper execution will improve the result enough to justify cost.

---

# 42. Stage 9 — Safe Fallback

V1 behavior:

```text
Route A chosen
    ↓
short probe
    ↓
rescue gain predicted
    │
    ├── low
    │     ↓
    │  continue A
    │
    └── high
          ↓
      restart using B/C/full
```

Record every fallback event.

---

# 43. Stage 10 — End-to-End Runtime Evaluation

Measure:

- initial route distribution,
- fallback rate,
- average number of executions/request,
- expected latency,
- TTFT,
- token throughput,
- route-switch cost,
- peak VRAM,
- total task quality.

This stage determines whether adaptive routing actually works in practice.

---

# 44. Baselines

The experiment is incomplete without strong baselines.

## B0 — Full model

All layers.

## B1 — Fixed shallow prefix

Example:

```text
first 16 / 24 layers
```

## B2 — Static layer pruning

Importance-based removal.

## B3 — Random skipping

Same layer budget.

## B4 — Best single compiled route

Tests whether adaptation itself matters.

## B5 — Difficulty-aware router

Preserve the earlier revision as a baseline.

This is valuable because you can directly demonstrate whether risk prediction is better.

## B6 — Confidence-only fallback

Tests rescue routing.

## B7 — Fine-grained dynamic router

Where feasible, compare to a representative per-layer routing method.

---

# 45. Critical Ablations

## Ablation A

Without route-aware training.

## Ablation B

Without distillation.

## Ablation C

Without Pareto route compilation.

## Ablation D

Difficulty routing instead of regret routing.

## Ablation E

Confidence fallback instead of rescue fallback.

## Ablation F

No fallback.

## Ablation G

Only 2 routes.

## Ablation H

4 routes.

## Ablation I

6 routes.

This reveals whether route-library size matters.

---

# 46. Primary Metrics

## Quality

- accuracy,
- exact match,
- F1,
- task-specific score,
- teacher agreement.

## Efficiency

- mean latency,
- p50,
- p95,
- TTFT,
- decode latency/token,
- tokens/sec,
- active layers,
- FLOPs estimate,
- peak VRAM.

## Router

- route accuracy,
- under-routing rate,
- over-routing rate,
- route calibration,
- regret prediction error.

## Fallback

- fallback frequency,
- rescue precision,
- rescue recall,
- quality recovered,
- latency overhead.

---

# 47. Define Under-Routing and Over-Routing

## Under-routing

Router chooses a route shallower than the cheapest safe route.

This risks quality.

## Over-routing

Router chooses a deeper route than necessary.

This wastes compute.

These should be measured separately.

A system with high accuracy but severe over-routing may simply be defaulting to the full model.

---

# 48. Safety / Quality Constraint

Define a maximum acceptable degradation:

\[
Q_F-Q_{AdaptiveDepth}\leq\epsilon
\]

Example:

```text
quality degradation ≤ 1 percentage point
```

Then optimize latency subject to that constraint.

This makes evaluation much clearer than merely reporting arbitrary speedups.

---

# 49. Pareto Frontier Evaluation

Plot:

```text
x-axis = measured latency
y-axis = task quality
```

Include:

- full model,
- fixed depth,
- random skipping,
- static pruning,
- each compiled route,
- difficulty router,
- risk router,
- risk router + fallback,
- fine-grained routing.

The ideal system moves toward:

```text
higher quality
+
lower latency
```

If AdaptiveDepth lies on or extends the Pareto frontier, that is a strong result.

---

# 50. What Would Count as a Successful Result?

A compelling result might be:

> At less than 1 percentage point quality loss, AdaptiveDepth reduces median batch-1 latency by a meaningful percentage relative to full execution, while using only four deployable routes and recovering most under-routing errors through selective fallback.

A second strong result:

> AdaptiveDepth matches a fine-grained dynamic router within a small quality/latency margin while using a much simpler runtime configuration.

The second result can be scientifically interesting even if AdaptiveDepth is not absolutely fastest.

---

# 51. What Would Count as a Negative Result?

Negative results are also informative.

Examples:

- fallback overhead removes all speedup,
- route-risk prediction is poorly calibrated,
- compiled routes do not beat simple shallow prefixes,
- reduced routes collapse on reasoning tasks,
- fine-grained routing significantly dominates the route library,
- route-aware training is too expensive relative to savings.

These should be considered legitimate research findings.

---

# 52. Novelty Positioning

Do **not** claim novelty for:

- layer importance,
- layer removal,
- layer dropout,
- adaptive depth,
- confidence routing.

Those areas already have substantial literature.

The strongest contribution is the **combination and systems framing**:

> Offline compilation of a compact, Pareto-validated route library + direct route-regret prediction + selective rescue + explicit expected-cost optimization.

---

# 53. Relationship to Existing Ideas

## ShortGPT

Useful inspiration:

- block redundancy,
- block influence,
- structured depth pruning.

AdaptiveDepth difference:

- ShortGPT-style analysis is used to construct candidate routes rather than simply permanently deleting layers.

## LayerSkip

Useful inspiration:

- layer dropout,
- making partial-depth execution more robust,
- training models to tolerate shallower computation.

AdaptiveDepth difference:

- uses multiple offline-validated route subnetworks and prompt-dependent route assignment.

## Router-Tuning / related dynamic-depth routing

Useful inspiration:

- lightweight routing,
- dynamic depth without retraining the entire model.

AdaptiveDepth difference:

- deliberately constrains the decision space to a small validated route set.

## Dr.LLM

Dr.LLM is especially important because it demonstrates sophisticated learned dynamic layer routing and offline path supervision.

AdaptiveDepth should not compete by pretending the same idea is new.

Instead, compare philosophies:

### Fine-grained routing

```text
many layer-level decisions
```

versus:

### AdaptiveDepth

```text
few route-level decisions
```

The research question becomes:

> How much flexibility is actually required to achieve useful adaptive inference?

---

# 54. Why Constrained Routing May Be Valuable

Fine-grained routing has theoretical flexibility.

However, route-level routing may offer:

- deterministic layer layouts,
- easier kernel/runtime optimization,
- easier caching,
- predictable latency,
- easier profiling,
- easier route-specific training,
- easier regression testing,
- simpler observability,
- fewer routing mistakes,
- smaller router output space.

This practical simplicity is central to the project identity.

---

# 55. Interactive Demonstration as a Technical Component

The interactive application is a required deliverable, not merely a decorative
dashboard. Its primary interaction begins with a user entering a new prompt;
the response and every displayed routing value must come from the real
AdaptiveDepth runtime.

It should expose the runtime decision process.

For each request display:

```text
Prompt
↓
Predicted risk:
A 0.31
B 0.08
C 0.02
D 0.01

Safety threshold: 0.10

Selected route: B
```

Then:

```text
Probe stability: 0.87
Predicted rescue gain B→C: 0.03
Fallback: No
```

---

# 56. Layer Execution Visualization

Display transformer layers:

```text
L01 ✓
L02 ✓
L03 ✓
L04 ✓
L05 ✗
L06 ✓
...
```

Use distinct visualization for:

- active layers,
- skipped layers,
- preserved critical layers.

---

# 57. Fallback Visualization

Example:

```text
Initial route: A
Risk after probe: High

A ───────X
          \
           B ───────✓

Fallback cost: +112 ms
Final quality estimate: recovered
```

This makes the recovery mechanism visually understandable.

---

# 58. Route Comparison Panel

Display:

| Route | Active blocks | Accuracy | Latency | Teacher KL |
|---|---:|---:|---:|---:|
| A | 12 | ... | ... | ... |
| B | 16 | ... | ... | ... |
| C | 20 | ... | ... | ... |
| D | 22 | ... | ... | ... |
| Full | 24 | ... | ... | 0 |

Only fill with measured results.

---

# 59. Research Dashboard

Include plots for:

- quality vs latency,
- route usage frequency,
- fallback rate,
- regret calibration,
- under-routing errors,
- over-routing errors,
- latency distribution,
- layer usage frequency.

This helps the website serve as an experiment explorer, not just a demo.

---

# 60. Recommended Tech Stack

## Core ML

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Accelerate

## Analysis

- NumPy
- pandas
- scikit-learn
- PyTorch Profiler
- optional fvcore / FLOP tooling

## Fine-tuning

- PEFT where useful
- LoRA where appropriate
- custom route dropout
- distillation losses

## Backend

- FastAPI

For MVP, avoid:

- Redis,
- Kubernetes,
- PostgreSQL,
- distributed serving.

Use simple local JSON / SQLite experiment storage initially if needed.

## Frontend

- Next.js
- TypeScript
- Tailwind
- Framer Motion
- Recharts
- optional D3.js

---

# 61. Recommended Repository Structure

```text
adaptivedepth/
│
├── configs/
│   ├── model.yaml
│   ├── routes.yaml
│   └── evaluation.yaml
│
├── src/
│   ├── model/
│   │   ├── routed_model.py
│   │   ├── layer_controller.py
│   │   └── kv_cache.py
│   │
│   ├── analysis/
│   │   ├── block_influence.py
│   │   ├── layer_ablation.py
│   │   └── route_benchmark.py
│   │
│   ├── compiler/
│   │   ├── candidate_search.py
│   │   ├── pareto.py
│   │   └── route_registry.py
│   │
│   ├── training/
│   │   ├── route_dropout.py
│   │   ├── distillation.py
│   │   └── train_robust.py
│   │
│   ├── router/
│   │   ├── dataset.py
│   │   ├── risk_model.py
│   │   └── calibration.py
│   │
│   ├── rescue/
│   │   ├── rescue_model.py
│   │   └── policy.py
│   │
│   ├── runtime/
│   │   ├── engine.py
│   │   ├── fallback.py
│   │   └── metrics.py
│   │
│   └── evaluation/
│       ├── baselines.py
│       ├── tasks.py
│       └── latency.py
│
├── dashboard/
├── experiments/
├── notebooks/
└── README.md
```

---

# 62. Implementation Roadmap

## Phase 1 — Baseline

- load model,
- create task harness,
- measure baseline quality,
- benchmark batch-1 latency.

## Phase 2 — Layer analysis

- block influence,
- deletion tests,
- layer interaction tests.

## Phase 3 — Route compiler

- generate route candidates,
- benchmark,
- Pareto filter,
- retain four routes.

## Phase 4 — Static route evaluation

- compare compiled routes to fixed-prefix and random baselines.

## Phase 5 — Robustness training

- route-aware dropout,
- distillation,
- reevaluate routes.

## Phase 6 — Router dataset

- collect route regret labels.

## Phase 7 — Risk router

- train,
- calibrate,
- evaluate under-/over-routing.

## Phase 8 — Rescue model

- collect shallow/deep disagreement data,
- train escalation predictor.

## Phase 9 — Fallback runtime

- probe-and-restart.

## Phase 10 — Final benchmark

- end-to-end quality,
- expected latency,
- p95 latency,
- fallback cost.

## Phase 11 — Required Interactive Prompt Demonstration

- prompt input and generated answer,
- execution route,
- route reasoning,
- fallback events,
- layer visualization,
- Pareto plots.

---

# 63. Feasibility

## MVP feasibility

**High: approximately 8/10**

The strengthened design is slightly more complex than the previous revision, but much better specified.

## Research-grade feasibility

**Moderate to high: approximately 7/10**

The main uncertainty is whether route-level adaptive execution produces enough real latency improvement after:

- router overhead,
- fallback,
- irregular model execution,
- route robustness losses.

This uncertainty is exactly what makes the project worth testing.

---

# 64. Compute Feasibility

For a 0.5B–1.5B model:

### Feasible for free / low cost

- layer ablation,
- route benchmarking,
- router training,
- small-scale distillation,
- dashboard,
- task evaluation.

### Potentially expensive

- extensive fine-tuning,
- large route search,
- long-context benchmarks,
- repeated GPU latency experiments,
- many seeds / hyperparameter sweeps.

Keep candidate routes constrained to control compute.

---

# 65. Why the Strengthened Version Is Not “Too Naive”

The project is no longer simply:

```text
easy → shallow
hard → deep
```

It is:

```text
offline model analysis
→ constrained route compilation
→ empirical route validation
→ route-aware robustness
→ direct regret prediction
→ calibrated selection
→ selective rescue
→ expected-cost evaluation
```

That is a substantial systems research pipeline.

---

# 66. Why It Is Still Not a Fundamental New Architecture

It is important to remain realistic.

AdaptiveDepth combines and reorganizes existing ideas:

- structured layer pruning,
- dynamic depth,
- routing,
- distillation,
- dropout,
- confidence estimation,
- fallback.

The likely contribution is **not** a fundamentally new transformer primitive.

The contribution is the design and empirical validation of a **practical adaptive inference framework**.

That is a perfectly legitimate research direction.

---

# 67. Strongest Potential Contribution Statement

A defensible contribution statement could eventually be:

> We introduce AdaptiveDepth, a constrained dynamic-inference framework that compiles a compact Pareto-optimal library of transformer execution routes offline and learns to assign prompts to the cheapest route predicted to preserve full-model behavior. Unlike unrestricted per-layer routing, AdaptiveDepth uses validated sequence-level routes with calibrated risk estimation and selective deeper-route rescue, enabling predictable runtime behavior and explicit quality–latency control.

Only make this claim after the actual implementation supports it.

---

# 68. Strongest Experimental Story

The most compelling experiment is:

```text
Full model
        \
Static pruning
          \
Fixed depth
            \
Difficulty routing
              \
Fine-grained routing
                \
AdaptiveDepth
```

all plotted on:

```text
QUALITY
  ↑
  │
  │                    ● Full
  │              ● AdaptiveDepth
  │         ● Fine-grained router
  │    ● Static
  │
  └────────────────────────────→
             LATENCY
```

The actual positions must come from experiments.

If AdaptiveDepth reaches a similar frontier with a simpler runtime, the project succeeds.

---

# 69. Recommended Final Naming

## Repository

**AdaptiveDepth**

## Research title

**AdaptiveDepth: Risk-Aware Routing over Validated Transformer Depths with Selective Fallback**

## Alternative

**AdaptiveDepth: Compiled Execution Routes and Selective Rescue for Efficient LLM Inference**

The second title emphasizes the route-compiler concept.

---

# 70. Final Recommendation

Proceed with the strengthened version.

Do **not** return to the original independent per-layer mask router as the primary design.

The route-library approach is more practical, and the strengthened revisions give it a clearer research identity.

The project should be built around these principles:

1. **Do expensive path discovery offline.**
2. **Deploy only routes that have been empirically validated.**
3. **Predict route-specific risk rather than vague prompt difficulty.**
4. **Train the model specifically for the routes it will use.**
5. **Treat fallback as an expected-utility decision rather than a confidence threshold.**
6. **Handle KV-cache limitations honestly using probe-and-restart in V1.**
7. **Measure real latency, not only FLOPs or skipped layers.**
8. **Include fallback overhead in every efficiency claim.**
9. **Compare against both simple static baselines and fine-grained dynamic routing.**
10. **Judge success by the quality–latency Pareto frontier.**

This combination keeps the implementation feasible while making the research question substantially sharper.

---

# 71. Literature Context Used for the Design

The strengthened design is informed by, but should not be described as reproducing, the following lines of work:

1. **ShortGPT: Layers in Large Language Models are More Redundant Than You Expect** — Xin Men et al., 2024.  
   Relevant idea: Block Influence and structured layer removal.

2. **LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding** — Mostafa Elhoushi et al., 2024.  
   Relevant ideas: layer dropout, early-exit robustness, partial-depth execution.

3. **Router-Tuning: A Simple and Effective Approach for Enabling Dynamic-Depth in Transformers** — Shwai He et al., 2024.  
   Relevant idea: lightweight router adaptation for dynamic depth.

4. **Dr.LLM: Dynamic Layer Routing in LLMs** — Ahmed Heakl et al., 2025.  
   Relevant ideas: supervised dynamic layer routing, offline path discovery, sequence-level routing, compute-aware execution.

These papers are important because they define what AdaptiveDepth should **not** incorrectly claim as new. The goal is to build a distinct constrained-route, risk-aware, selective-rescue system and validate whether that simplification provides a useful deployment-quality tradeoff.

---

# 72. One-Sentence Final Definition

> **AdaptiveDepth is a risk-aware adaptive inference framework that compiles a small set of validated transformer subnetworks offline, selects the cheapest route predicted to preserve full-model behavior for each prompt, and selectively escalates to deeper execution when the expected benefit of rescue outweighs the additional compute cost.**
