from __future__ import annotations

import random

from adaptivedepth.routes import RouteDefinition


def generate_candidate_routes(
    num_layers: int,
    influence: dict[int, float],
    budgets: list[int],
    preserve_first: int = 2,
    preserve_last: int = 2,
    variants_per_budget: int = 4,
    seed: int = 17,
) -> list[RouteDefinition]:
    """Generate constrained candidates; every route still requires benchmarking."""

    if set(influence) != set(range(num_layers)):
        raise ValueError("influence scores must contain every model layer exactly once")
    protected = set(range(min(preserve_first, num_layers)))
    protected.update(range(max(0, num_layers - preserve_last), num_layers))
    ranked = sorted(range(num_layers), key=lambda index: (-influence[index], index))
    rng = random.Random(seed)
    routes: list[RouteDefinition] = []
    seen: set[tuple[int, ...]] = set()

    for budget in sorted(set(budgets)):
        if budget < len(protected) or budget > num_layers:
            raise ValueError(
                f"budget {budget} cannot satisfy {len(protected)} protected layers "
                f"for a {num_layers}-layer model"
            )
        remaining = budget - len(protected)
        eligible = [index for index in ranked if index not in protected]
        base = tuple(sorted(protected | set(eligible[:remaining])))
        candidates = [base]

        # Produce bounded deterministic perturbations around the ranked route.
        for _ in range(max(variants_per_budget - 1, 0)):
            chosen = set(base)
            swappable = [index for index in chosen if index not in protected]
            omitted = [index for index in eligible if index not in chosen]
            if swappable and omitted:
                remove_index = rng.choice(swappable[max(0, len(swappable) // 2) :])
                add_index = rng.choice(omitted[: max(1, len(omitted) // 2)])
                chosen.remove(remove_index)
                chosen.add(add_index)
            candidates.append(tuple(sorted(chosen)))

        variant_number = 0
        for active_layers in candidates:
            if active_layers in seen:
                continue
            seen.add(active_layers)
            routes.append(
                RouteDefinition(
                    route_id=f"candidate_b{budget:02d}_v{variant_number:02d}",
                    active_layers=list(active_layers),
                    source="influence_constrained_search",
                    metadata={
                        "budget": budget,
                        "preserve_first": preserve_first,
                        "preserve_last": preserve_last,
                    },
                )
            )
            variant_number += 1
    return routes
