# AdaptiveDepth Verified Generation

## Product Design and Technical Overview

**Status:** implementation blueprint  
**Track:** application and engineering project  
**Primary model target:** a compact decoder-only transformer with reusable internal layers  
**Initial decoding mode:** deterministic greedy generation  
**Primary deliverable:** an interactive web application backed by a real adaptive inference engine

---

## 1. Executive summary

AdaptiveDepth Verified Generation is an inference system that uses a reduced-depth version of a language model to **draft** tokens and the complete model to **verify** them before they are shown to the user.

> Use fewer transformer layers to make inexpensive predictions, but never publish an unverified prediction.

When a draft token agrees with the full model, it is accepted. When it disagrees, the full model's token is used instead. Several proposed tokens can be checked together in one full-model forward pass. This can reduce the number of serial full-model decoding steps while preserving the output of ordinary full-depth greedy generation.

The application will make this process visible. A user enters a prompt and receives:

- the verified answer;
- a live view of drafted, accepted, and corrected tokens;
- the transformer layers used by the drafter;
- the draft acceptance rate;
- measured latency, throughput, and memory use;
- an explanation of when speculative drafting was enabled, changed, or bypassed.

This is not a thin chat wrapper. The main product is the inference mechanism and its observability. The web interface exposes a real systems technique rather than merely sending a prompt to an external API.

The project is feasible as a single-GPU demonstration and is deployable. The correctness-preserving decoder is technically realistic. A net speedup, however, cannot be promised in advance: it depends on draft cost, token acceptance, verification-block size, model size, hardware, and cache efficiency. The system will measure its own benefit and automatically fall back to standard full-model decoding when speculation is not worthwhile.

---

## 2. Product definition

### 2.1 The problem

Autoregressive language-model generation is sequential. A full transformer pass is normally required for every new token. Even a relatively small model can spend much of its generation time repeatedly reading weights and updating the key-value cache one token at a time.

Reducing the active layer count makes each step cheaper, but directly returning a reduced-depth model's answer can alter facts, reasoning, formatting, and instruction following. A system that is fast only when it is willing to return unreliable answers is not a compelling application.

AdaptiveDepth separates two responsibilities:

- the **reduced-depth route** proposes likely continuations quickly;
- the **full-depth model** remains the authority over every published token.

### 2.2 Product promise

> AdaptiveDepth attempts to accelerate deterministic generation with layer-skipped draft routes while every published token remains verified against the full-depth model.

The promise deliberately does **not** say that every request will be faster. If speculation is predicted or observed to be unhelpful, the runtime uses the full model directly.

### 2.3 What “verified” means

For the first release, verified means:

- decoding is greedy and deterministic;
- every drafted token is compared with the token selected by the full-depth model;
- mismatches are corrected before any text is emitted;
- the final token sequence must equal the sequence produced by the same full model, tokenizer, prompt template, numerical precision, and stopping rules.

Verification preserves the selected model's output. It does **not** prove that the model's answer is factually correct, harmless, or unbiased. Model-level safety controls remain a separate concern.

### 2.4 What the product is not

AdaptiveDepth is not:

- a prompt classifier that guesses when a weak model might be safe;
- a mode that trades correctness for speed without warning;
- an API wrapper around a hosted chatbot;
- a claim of a new foundation-model architecture;
- a publication-scale research benchmark;
- a guarantee of acceleration on every GPU, prompt, or output length.

---

## 3. Why the project is valuable

### 3.1 User value

The user receives one dependable behavior: adaptive execution never silently changes the full model's greedy answer. The runtime can try a faster path without asking the user to accept a lower-quality response.

### 3.2 Engineering value

The project demonstrates several substantial systems concepts in one application:

- transformer layer routing;
- self-speculative decoding;
- separate draft and target key-value caches;
- batched token verification;
- online control based on observed acceptance and cost;
- correctness invariants and fallback behavior;
- GPU profiling and honest performance reporting;
- a streaming inference API;
- real-time visualization of model execution.

### 3.3 Portfolio value

The impressive part is not simply that the application generates text. It shows **how computation is allocated while generation is happening** and proves that the adaptive result matches the full-model result.

A reviewer can see:

1. which layers drafted a token block;
2. which draft tokens the full model accepted;
3. where the full model corrected the draft;
4. how the controller reacted to recent acceptance;
5. whether speculation saved time on that request;
6. that the final adaptive and baseline token sequences match.

This makes the application explainable, measurable, and visibly different from a conventional chat interface.

---

## 4. Core methodology

### 4.1 Components

The runtime has four main inference components.

#### Full-depth target

The full model is the source of truth. It determines the final greedy token sequence and owns an independent target key-value cache.

#### Reduced-depth drafter

The drafter reuses the same model weights but executes only a configured subset of transformer blocks. It owns a separate draft cache and proposes a small block of future tokens.

Because the drafter and target share weights, this is **self-speculation**: no second language model needs to be stored in GPU memory.

#### Verifier

The verifier evaluates a proposed token block with the full model in one batched forward pass. It finds the longest prefix on which the draft and full model agree.

#### Adaptive controller

The controller selects:

- whether drafting should be used at all;
- which draft route to use;
- how many tokens to draft in the next block;
- when to switch to direct full-model decoding.

Its decisions are based on runtime measurements, not an unsupported assumption that every prompt is easy.

### 4.2 Generation loop

One decoding iteration works as follows:

1. Start from the already verified prefix.
2. Select a draft route and a small draft length, such as 2–6 tokens.
3. Generate the proposed tokens sequentially with the reduced-depth route.
4. Evaluate the proposed block with the full-depth model.
5. Compare draft tokens with full-model greedy choices in order.
6. Accept the matching prefix.
7. At the first mismatch, emit the full model's token instead of the draft token.
8. Crop or rebuild both caches so they represent exactly the verified prefix.
9. Update acceptance and cost statistics.
10. Continue drafting only if the measured policy expects a benefit.

Conceptual pseudocode:

```text
verified_prefix = encoded_prompt
target_state = full_prefill(verified_prefix)
draft_state = route_prefill(verified_prefix)

while not stopped:
    policy = controller.choose(recent_acceptance, output_length, device_metrics)

    if policy.use_full_only:
        token = full_greedy_step(target_state)
        publish(token, source="full")
        synchronize_draft_state_if_needed()
        continue

    proposals = route_draft(policy.route, policy.block_size, draft_state)
    target_choices = full_verify_block(proposals, target_state)
    matching_prefix = longest_equal_prefix(proposals, target_choices)

    publish(matching_prefix, source="accepted_draft")
    if a mismatch exists:
        publish(first_full_model_correction, source="full_correction")

    synchronize_caches_to_verified_prefix()
    controller.observe(costs, accepted_count, rejected_count)
```

No draft token is sent to the user until its corresponding full-model decision is known.

### 4.3 Why this can be faster

Ordinary decoding makes one full-model call per generated token. Speculative decoding proposes multiple tokens cheaply and asks the full model to verify them as a block. GPUs can often process a short block more efficiently than several strictly serial one-token calls.

Performance depends on:

- **draft cost:** how expensive it is to propose each token;
- **acceptance length:** how many proposed tokens survive each verification call.

High acceptance from an expensive drafter may still be unprofitable. A cheap drafter with very low acceptance may also be unprofitable. The controller must optimize the measured combination, not acceptance alone.

### 4.4 Why output quality is protected

For deterministic greedy decoding, the next token is completely determined by the full model and the verified prefix. At every position, AdaptiveDepth either accepts a draft token because it equals that full-model choice or replaces it with the full-model choice.

By induction over generated positions, the emitted sequence equals ordinary full-model greedy decoding, provided cache synchronization and stopping behavior are implemented correctly.

This invariant must be enforced by automated token-ID equality tests. It is not enough for decoded text to merely look similar.

### 4.5 Adaptive behavior

“Adaptive” no longer means choosing a weaker model for the whole request. It means adapting the speculative process while preserving one target output.

The controller can:

- use a more aggressive route after several high-acceptance blocks;
- use a more conservative route when acceptance falls;
- shorten the draft block after early mismatches;
- lengthen it when drafts consistently agree;
- disable drafting for short responses or unfavorable hardware;
- re-enable drafting later if policy rules allow it.

This is finer-grained and safer than routing an entire prompt to one depth.

---

## 5. Route strategy

### 5.1 User-facing profiles

The first application version should expose only verified behaviors:

| Profile | Behavior | User-visible quality |
|---|---|---|
| Verified Auto | Dynamically selects a draft route, block size, or full-only decoding | Full-model greedy output |
| Verified Conservative | Uses shorter blocks and a higher-acceptance draft route; bypasses quickly | Full-model greedy output |
| Full Baseline | Disables drafting and runs ordinary full-depth decoding | Full-model greedy output |

An unverified “Fast” response mode should not be part of the normal product. It may exist only behind a developer flag for route diagnostics and must never be confused with verified output.

### 5.2 Number of routes

Two draft profiles are enough for the first deployable version:

- **aggressive drafter:** fewer active layers and lower proposal cost;
- **conservative drafter:** more active layers and potentially higher acceptance.

The controller may also choose the full model, which is an execution fallback rather than a third draft route. Additional routes should be included only after they demonstrate a distinct measured benefit.

### 5.3 Important feasibility warning

A route that executes 20 or 23 blocks of a 24-block model may be accurate enough to draft well, but it may also be too expensive to yield a speedup after full verification. Such routes are useful starting points for correctness and acceptance testing, not proof of deployment-level acceleration.

Because draft errors are corrected, the drafter can safely be more aggressive than a route used to answer independently. A shallower route may ultimately be required. Finding one is a bounded performance-engineering task: benchmark a small predefined route set against acceptance and end-to-end latency, then lock the best route. It does not require a large prompt-labeling research pipeline.

---

## 6. System architecture

```text
┌──────────────────────── Web client ────────────────────────┐
│ Prompt input · streamed answer · token trace · layer view  │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP + Server-Sent Events
┌────────────────────────────▼────────────────────────────────┐
│                         API service                         │
│ Validation · request queue · cancellation · event stream   │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                 Adaptive generation engine                 │
│ Controller · drafter · full verifier · cache coordinator   │
│ correctness guard · profiler · telemetry                   │
└───────────────┬───────────────────────────┬─────────────────┘
                │                           │
┌───────────────▼──────────────┐ ┌──────────▼─────────────────┐
│ Reduced-depth execution     │ │ Full-depth execution       │
│ shared model weights        │ │ shared model weights       │
│ independent draft KV cache  │ │ independent target KV cache│
└──────────────────────────────┘ └────────────────────────────┘
```

### 6.1 Backend modules

Recommended module boundaries:

```text
src/adaptivedepth/
  speculative/
    engine.py          # Verified generation loop
    cache.py           # Cache crop, rebuild, and synchronization
    verifier.py        # Full-model block verification
    controller.py      # Route and block-size policy
    events.py          # Typed streaming telemetry
    metrics.py         # Timing, acceptance, and memory statistics
  serving/
    api.py             # HTTP endpoints and event streaming
    schemas.py         # Request and response models
    lifecycle.py       # Model load, warm-up, and shutdown
```

The UI must call the same engine used by CLI tests and benchmarks. It must not reimplement inference logic.

### 6.2 Cache design

Cache correctness is the hardest engineering part. The runtime needs independent state for:

- the verified full-model prefix;
- the reduced-route prefix;
- provisional draft tokens;
- target tokens computed during block verification.

After a rejection, cache positions conditioned on rejected draft tokens must be removed. If the installed model or Transformers cache implementation cannot be cropped safely, the correctness-first fallback is to rebuild the affected cache from verified token IDs. Rebuilding is slower but preserves the invariant.

### 6.3 Concurrency model

The current layer-routing approach temporarily changes the active block collection, so the initial server should serialize GPU inference through a request queue. It should not allow concurrent requests to mutate one model instance.

Later deployment options include immutable execution plans, a non-mutating route implementation, or process-level workers that each own a model instance. For a portfolio demonstration, one queued GPU worker is adequate and easier to make correct.

---

## 7. Web application design

### 7.1 Primary screen

The application should be a focused single-page experience:

1. The user enters a prompt.
2. The answer streams into the response panel.
3. A token timeline updates as blocks are verified.
4. A layer map shows the draft route and full verification pass.
5. A performance panel compares the request with a calibrated full baseline.

### 7.2 Required interface elements

#### Prompt composer

- multiline prompt input;
- maximum-token control;
- Verified Auto, Verified Conservative, and Full Baseline profiles;
- deterministic settings shown clearly;
- generate and cancel controls.

#### Verified response

- streamed final answer;
- visible ?full-model verified? state;
- copy action;
- exact-match fingerprint for diagnostics.

#### Token verification ribbon

Each generated token or block is marked as drafted and accepted, drafted and corrected, or generated directly by the full model. Accessible icons or labels must accompany color.

#### Layer execution map

A 24-block strip shows active draft layers, skipped draft layers, full verification sweeps, and route switches over time.

#### Runtime telemetry

- total, drafted, accepted, and corrected token counts;
- mean accepted tokens per verification pass;
- number of full verification passes;
- end-to-end latency and time to first token;
- tokens per second and peak GPU memory;
- measured or calibrated full-baseline latency;
- net speedup or slowdown.

#### Controller explanation

Compact messages must come from real events, for example:

- ?Started with the aggressive drafter.?
- ?Acceptance fell; reduced block size from 4 to 2.?
- ?Drafting bypassed; direct full-model decoding is currently faster.?

### 7.3 Honest display rules

- If the request is slower, display the slowdown.
- If no comparable baseline exists, label the comparison unavailable.
- If drafting is bypassed, show that the full model handled the tokens directly.
- If the engine rebuilds a cache, expose the recovery event in diagnostics.
- If a replayed demonstration is provided, label it as replay data.

---

## 8. API design

### 8.1 Generation request

```json
{
  "prompt": "Explain why the sky appears blue.",
  "profile": "verified_auto",
  "max_new_tokens": 128,
  "request_id": "optional-client-id"
}
```

Sampling controls are intentionally absent from version 1 because the exactness guarantee initially applies to greedy decoding.

### 8.2 Streamed event types

- `request_started`
- `prefill_completed`
- `draft_started`
- `draft_completed`
- `verification_completed`
- `tokens_committed`
- `route_changed`
- `speculation_bypassed`
- `generation_completed`
- `generation_cancelled`
- `generation_failed`

Each event should include a monotonic timestamp and only measurements known at that point.

### 8.3 Completion summary

```json
{
  "text": "...",
  "verified": true,
  "decoding": "greedy",
  "profile": "verified_auto",
  "input_tokens": 18,
  "output_tokens": 84,
  "drafted_tokens": 96,
  "accepted_draft_tokens": 61,
  "corrected_tokens": 23,
  "acceptance_rate": 0.635,
  "verification_passes": 29,
  "latency_ms": 2240.5,
  "baseline_latency_ms": 2710.2,
  "speedup": 1.21,
  "routes_used": ["aggressive", "conservative", "full"],
  "token_id_match_with_baseline": true
}
```

These values illustrate the schema; they are not expected results.

---

## 9. Correctness and safety requirements

### 9.1 Non-negotiable invariant

For every supported prompt in deterministic mode:

```text
adaptive_output_token_ids == full_greedy_output_token_ids
```

The comparison includes EOS handling and maximum-length stopping behavior.

### 9.2 Correctness tests

The engine must test:

- rejection of the first proposed token;
- partial and complete block acceptance;
- EOS proposed inside a block or selected as a correction;
- maximum token limits and prompts near the context limit;
- route changes after partial acceptance;
- draft-cache rebuilds;
- cancellation between verification and commit;
- repeated prompts and fixed seeds;
- CPU and CUDA where supported;
- supported Transformers cache representations.

### 9.3 Failure behavior

Any uncertainty in draft state must disable speculation, not bypass verification:

1. discard uncommitted draft tokens;
2. retain or reconstruct the verified prefix;
3. continue with direct full-depth generation;
4. record the reason in telemetry.

If the verified prefix itself cannot be trusted, stop and return an explicit error instead of emitting uncertain text.

### 9.4 Scope of the guarantee

Exact target equivalence is initially limited to batch size one, greedy decoding, one pinned model family and Transformers version, a fixed prompt template and tokenizer, and the same numerical precision and device path used for comparison.

Top-k, nucleus, temperature sampling, beam search, quantization changes, and cross-device equivalence require separate algorithms and tests. They are out of scope for version 1.

---

## 10. Measurement and acceptance criteria

### 10.1 Correctness metrics

- token-ID exact-match rate against full greedy baseline;
- cache recovery events;
- unverified tokens emitted, which must always be zero;
- deterministic repeatability rate.

### 10.2 Performance metrics

- time to first token and total latency;
- decode milliseconds per committed token;
- committed tokens per second;
- draft acceptance rate and accepted tokens per verification pass;
- full-model calls avoided;
- draft, verification, and cache-management time;
- peak GPU memory;
- percentage of requests where speculation is bypassed;
- p50 and p95 speedup over full greedy generation.

### 10.3 MVP acceptance gate

The application MVP is complete when:

- adaptive token IDs match the full baseline on every correctness test;
- no token is streamed before verification;
- cache rejection and recovery tests pass;
- the UI displays live, real execution events;
- full-only fallback succeeds after forced draft failure;
- one supported GPU deployment runs end to end;
- reports state both speedups and slowdowns honestly.

A speedup is **not** required to prove the correctness MVP.

### 10.4 Performance release gate

The application may advertise acceleration only after a locked benchmark demonstrates:

- 100% token-ID equivalence;
- positive median end-to-end speedup over warmed full greedy decoding;
- no unacceptable p95 regression under the chosen controller;
- identical hardware, precision, prompt set, and generation settings;
- documented warm-up, synchronization, and baseline methods.

If this gate fails, the application remains deployable as an adaptive-computation visualizer, but it must not claim to accelerate the model.

---

## 11. Feasibility assessment

### 11.1 Technical feasibility: high for the working application

The system is implementable with PyTorch and Hugging Face Transformers. It needs one set of model weights, selectable transformer blocks, two logical cache states, a verification loop, and a small web service.

Useful foundations already exist in the repository: model loading, route definitions, guarded layer routing, deterministic greedy generation, profiling utilities, and a compact trained checkpoint.

### 11.2 Correctness feasibility: high but engineering-sensitive

Exact greedy equivalence is a clear, testable property. The algorithm can guarantee it only when cache positions, logits, EOS behavior, and stream-commit timing are correct. Cache code deserves attention before UI polish.

### 11.3 Performance feasibility: uncertain until measured

Acceleration may work because target verification processes multiple positions together, one-token decoding is often memory-bandwidth limited, weights are shared, and a shallow route can propose cheaply.

It may fail because routes remain too deep, acceptance is low, Python/cache overhead dominates a tiny model, target verification still performs full-depth computation, or route switching and cache rebuilding erase gains.

Therefore:

- **deployable application:** yes;
- **correct verified decoder:** realistically achievable;
- **guaranteed output equivalence in the defined scope:** achievable and testable;
- **guaranteed latency improvement:** no;
- **credible chance of measurable improvement after optimization:** yes, but it must be benchmarked.

### 11.4 Resource feasibility

For a model around 0.5B parameters, a 15?16 GB T4-class GPU is sufficient for an initial single-user FP16 service. The design shares weights rather than loading separate full and draft models. Two KV caches and verification tensors add memory but remain modest at short context lengths.

CPU is useful for functional tests, not for the public interactive experience. Longer contexts, concurrency, or larger models require stronger GPUs and explicit memory limits.

### 11.5 Schedule feasibility

| Stage | Expected time | GPU required |
|---|---:|---|
| Decoder contract and mocked tests | 0.5?1 day | No |
| Correctness-first speculative engine | 1.5?3 days | Final check |
| Cache optimization and controller | 1?2 days | Intermittently |
| Streaming API and telemetry | 1 day | Mostly no |
| Web interface | 2?3 days | Mostly no |
| Benchmark, hardening, deployment | 1?3 days | Yes |

A credible local MVP is roughly **5?8 focused development days**. A polished hosted demonstration is roughly **8?12 focused days**, depending mainly on cache compatibility and deployment. GPU use is for short integration and performance runs, not repeated large training or labeling notebooks.

---

## 12. Deployment plan

### 12.1 Deployment topology

The simplest production-shaped deployment is:

- a static or server-rendered web frontend;
- a Python inference API in the same repository;
- one GPU worker with a request queue;
- model and tokenizer loaded once during startup;
- lightweight structured telemetry;
- health and readiness endpoints;
- a container with pinned CUDA, PyTorch, and Transformers versions.

### 12.2 Suitable environments

The application can run on any service supporting a persistent Python process and NVIDIA GPU: a GPU virtual machine, container service, or GPU application-hosting platform. A hosted notebook is suitable for development but not a reliable public deployment because sessions expire and URLs are temporary.

### 12.3 Public-demo limits

The first hosted version should use:

- batch size one;
- a short request queue;
- bounded prompt and output lengths;
- one in-flight generation per worker;
- request cancellation and rate limiting;
- model warm-up before readiness;
- no accounts or saved prompt history by default.

### 12.4 Deployment risks

- Free GPU hosting may sleep or enforce time quotas.
- Cold model loading can delay the first request.
- Concurrent routing against one mutable model instance is unsafe.
- CUDA or Transformers version changes can alter cache behavior.
- GPU memory grows with context and concurrent caches.
- A public prompt endpoint needs input limits and abuse controls.

These are manageable for a portfolio deployment when the environment is pinned and inference is serialized.

---

## 13. Implementation roadmap

### Stage 1 ? Freeze the contract

Deliverables:

- define target equivalence precisely;
- pin the model, tokenizer, prompt template, precision, and library version;
- define typed generation events and the result schema;
- implement a plain full-greedy token generator as reference.

**Exit condition:** the baseline returns reproducible token IDs and timing data.

### Stage 2 ? Build the verified decoder

Deliverables:

- draft generation with one reduced route;
- block verification with the full model;
- accepted-prefix and mismatch handling;
- a correctness-first cache rebuild path;
- delayed streaming until commit;
- token-by-token equivalence tests.

**Exit condition:** adaptive and full token IDs match across every deterministic test.

### Stage 3 ? Optimize caches and adaptation

Deliverables:

- efficient crop or rollback for the pinned cache implementation;
- aggressive and conservative draft profiles;
- adaptive block sizing;
- a rolling acceptance and measured-cost controller;
- automatic speculation bypass;
- latency and memory profiling.

**Exit condition:** correctness remains exact and the controller stops persistently unprofitable speculation.

### Stage 4 ? Add the application API

Deliverables:

- generation endpoint;
- Server-Sent Events or WebSocket stream;
- request queue and cancellation;
- health and readiness endpoints;
- structured metrics and error events.

**Exit condition:** a browser receives both the answer and complete trace from the real engine.

### Stage 5 ? Build the web interface

Deliverables:

- prompt-first landing screen;
- verified response stream;
- token verification ribbon;
- layer execution animation;
- controller event timeline;
- baseline comparison and metric cards;
- responsive layout with clear empty and error states.

**Exit condition:** a reviewer can understand the method without reading source code.

### Stage 6 ? Benchmark and deploy

Deliverables:

- locked benchmark prompts and settings;
- warm baseline and adaptive measurements;
- exactness report;
- container and GPU deployment configuration;
- public-demo limits and failure recovery;
- final README with reproducible commands.

**Exit condition:** the deployed app reports real measurements and makes only supported claims.

---

## 14. Principal challenges and mitigations

| Challenge | Consequence | Mitigation |
|---|---|---|
| Draft route is too expensive | Adaptive mode is slower | Use shallower routes, smaller blocks, and measured auto-bypass |
| Low token acceptance | Draft work is wasted | Shorten blocks, change routes, or decode full-only |
| Cache rollback bug | Output diverges from baseline | Token-ID invariant tests and conservative cache rebuild |
| Mutable routing under concurrency | Requests can interfere | Serialize one worker; later use immutable executors or processes |
| Premature streaming | Rejected text reaches the user | Commit tokens only after full verification |
| EOS mismatch | Truncated or extended answers | Include EOS and length rules in the verifier contract |
| Library cache API changes | Runtime breaks after upgrades | Pin versions and isolate cache adapters |
| Tiny model hides speed benefit | Little or negative gain | Report honestly; test a larger model only after the MVP |
| GPU unavailable | Hosted app cannot infer | Readiness state and optional saved replay clearly labelled |
| ?Verified? is misunderstood | Users assume factual truth | State that verification means target-model equivalence |

---

## 15. Differentiation and impressive factor

The project combines a correctness argument, an inference optimization, and an observable product.

### 15.1 Technical differentiation

- Reuses internal layers of one model instead of requiring a separate draft model.
- Adapts repeatedly during generation, not only once per prompt.
- Converts draft mistakes into corrections rather than unsafe answers.
- Declines to speculate when measured economics are poor.
- Validates its quality invariant at the token-ID level.

### 15.2 Visual differentiation

- The user watches tokens move from proposed to accepted or corrected.
- A layer map connects model depth to actual execution.
- Route and block-size changes appear on a timeline.
- The speed card uses measured timing, not theoretical layer count.
- The app proves ?same answer, different execution path? live.

### 15.3 Demonstration narrative

1. Enter a prompt.
2. Watch a reduced route draft a block.
3. Watch the full model accept tokens and correct a mismatch.
4. Observe that only verified text enters the answer.
5. Compare the final token hash with the full baseline.
6. Inspect latency and acceptance statistics.
7. Try a difficult prompt and watch the controller fall back automatically.

The story is understandable to non-specialists while still revealing substantial engineering depth.

---

## 16. Deliberate scope boundaries

### Included in version 1

- one compact transformer family and checkpoint;
- greedy generation and batch size one;
- two draft profiles plus full-only execution;
- exact target-equivalence tests;
- live telemetry;
- a one-user-at-a-time GPU demo;
- a deployable web application.

### Deferred

- stochastic speculative sampling;
- multilingual specialization;
- multiple model families;
- multi-GPU or distributed serving;
- continuous batching and high concurrency;
- automatic training of draft routes;
- accounts, billing, and persistent chat history;
- claims of universal speedup.

These boundaries keep the application achievable without weakening its central idea.

---

## 17. Final verdict

AdaptiveDepth Verified Generation is a **feasible, valuable, and deployable application project**.

Its strongest achievable claim is not ?a shallow model is usually good enough.? The stronger claim is:

> A shallow route may propose the work, but the full model decides every token the user receives.

This resolves the main product weakness of unsafe whole-prompt routing. The application can safely attempt adaptive computation on every request and bypass it when it offers no benefit.

The major engineering risk is performance, not answer integrity. Correctness can be specified and tested. Speed depends on the hardware and efficiency of the draft/verification implementation. The project must therefore be presented in two layers:

1. **Guaranteed result:** full-model-equivalent deterministic output with visible adaptive execution.
2. **Measured result:** the speedup or slowdown observed for a particular model, route, workload, and GPU.

Even if the first route configuration does not accelerate the compact model, the system remains an impressive inference-engineering application and diagnostic visualizer. If the performance gate passes, it also becomes a credible acceleration demo. In both cases, the interface remains honest, the methodology remains technically meaningful, and the project is substantially more than an AI wrapper.

