from __future__ import annotations

from adaptivedepth.analysis.ablation import random_deletion_routes
from adaptivedepth.routes import RouteDefinition, RouteLibrary


def fixed_prefix_routes(num_layers: int, budgets: list[int]) -> list[RouteDefinition]:
    routes: list[RouteDefinition] = []
    for budget in sorted(set(budgets)):
        if budget < 1 or budget >= num_layers:
            raise ValueError("prefix budgets must be between 1 and num_layers - 1")
        routes.append(
            RouteDefinition(
                route_id=f"fixed_prefix_b{budget:02d}",
                active_layers=list(range(budget)),
                source="fixed_prefix_baseline",
                metadata={"budget": budget},
            )
        )
    return routes


def importance_static_routes(
    num_layers: int,
    influence: dict[int, float],
    budgets: list[int],
    preserve_first: int,
    preserve_last: int,
) -> list[RouteDefinition]:
    if set(influence) != set(range(num_layers)):
        raise ValueError("influence scores must contain every model layer exactly once")
    protected = set(range(min(preserve_first, num_layers)))
    protected.update(range(max(0, num_layers - preserve_last), num_layers))
    ranked = sorted(range(num_layers), key=lambda index: (-influence[index], index))
    routes: list[RouteDefinition] = []
    for budget in sorted(set(budgets)):
        if budget < len(protected) or budget >= num_layers:
            raise ValueError("importance budgets must preserve protected layers and skip a layer")
        selected = set(protected)
        selected.update(
            index
            for index in ranked
            if index not in protected
        )
        # Preserve rank order only for selection; execution order remains original.
        ranked_unprotected = [index for index in ranked if index not in protected]
        selected = protected | set(ranked_unprotected[: budget - len(protected)])
        routes.append(
            RouteDefinition(
                route_id=f"importance_static_b{budget:02d}",
                active_layers=sorted(selected),
                source="importance_static_baseline",
                metadata={
                    "budget": budget,
                    "preserve_first": preserve_first,
                    "preserve_last": preserve_last,
                },
            )
        )
    return routes


def build_static_route_suite(
    candidates: RouteLibrary,
    num_layers: int,
    influence: dict[int, float],
    budgets: list[int],
    preserve_first: int = 2,
    preserve_last: int = 2,
    random_routes_per_budget: int = 2,
    seed: int = 17,
) -> RouteLibrary:
    """Combine compiled candidates with the required Phase 4 baselines."""

    protected = set(range(min(preserve_first, num_layers)))
    protected.update(range(max(0, num_layers - preserve_last), num_layers))
    routes = list(candidates.routes)
    routes.extend(fixed_prefix_routes(num_layers, budgets))
    routes.extend(
        importance_static_routes(
            num_layers,
            influence,
            budgets,
            preserve_first,
            preserve_last,
        )
    )
    for budget in sorted(set(budgets)):
        random_routes = random_deletion_routes(
            num_layers=num_layers,
            active_budget=budget,
            samples=random_routes_per_budget,
            protected_layers=protected,
            seed=seed + budget,
        )
        for index, route in enumerate(random_routes):
            routes.append(
                route.model_copy(
                    update={
                        "route_id": f"random_b{budget:02d}_v{index:02d}",
                        "source": "random_baseline",
                    }
                )
            )
    library = RouteLibrary(routes=routes)
    library.validate_for_depth(num_layers)
    return library
