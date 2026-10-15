# AdaptiveDepth interactive demonstration

The interactive application is a required final deliverable. It will be built
after the risk router and probe-and-restart runtime are functional, because the
interface must report real routing decisions rather than placeholder values.

## Primary interaction

The user enters a prompt and submits it to AdaptiveDepth. The application
returns the generated answer together with an explanation of how computation
was allocated.

## Required request view

Every completed request must display:

- the generated answer;
- the initially selected route and final route;
- active and skipped transformer layers;
- predicted risk for every available route;
- the configured safety threshold;
- whether fallback occurred and why;
- routed latency, a comparable full-model latency, and measured savings;
- active-layer count and peak GPU memory when available.

## Required visualizations

1. A layer strip that clearly distinguishes active, skipped, and protected
   layers.
2. A route-decision panel showing route risks and the selected operating point.
3. A fallback timeline showing the initial attempt, restart, and final route.
4. A route comparison table populated only from measured experiment artifacts.

## Behavioral requirements

- The answer must come from the same AdaptiveDepth runtime used by the CLI.
- The application must never invent latency, quality, confidence, or route
  measurements.
- If routing or fallback is unavailable, the interface must label the request
  as full-model execution instead of simulating an adaptive decision.
- A deterministic demonstration mode may replay saved experiment records, but
  it must be visibly labelled as a replay.
- Runtime failures must fall back safely to the full route and expose the error
  state without losing the prompt.

## Implementation boundary

The first version should remain small: one prompt page, one inference API, and
local experiment artifacts. Authentication, multi-user persistence,
distributed serving, and billing are out of scope. The implementation choice
between a Python UI and a separate web frontend will be made after the runtime
API stabilizes; the inference engine must remain UI-independent.

## Acceptance criteria

The application is complete when a reviewer can enter a new prompt, receive a
real generated response, identify every executed layer, see why a route was
chosen, observe any fallback, and compare end-to-end cost with full-model
execution.
