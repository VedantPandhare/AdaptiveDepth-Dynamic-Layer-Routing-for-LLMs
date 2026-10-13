from __future__ import annotations

import random

from adaptivedepth.routes import RouteDefinition


def single_layer_deletion_routes(
    num_layers: int, protected_layers: set[int] | None = None
) -> list[RouteDefinition]:
    protected = protected_layers or set()
    all_layers = set(range(num_layers))
    return [
        RouteDefinition(
            route_id=f"skip_layer_{index:02d}",
            active_layers=sorted(all_layers - {index}),
            source="single_layer_ablation",
            metadata={"removed_layers": [index]},
        )
        for index in range(num_layers)
        if index not in protected
    ]


def adjacent_group_deletion_routes(
    num_layers: int,
    group_size: int,
    protected_layers: set[int] | None = None,
) -> list[RouteDefinition]:
    if group_size < 1 or group_size >= num_layers:
        raise ValueError("group_size must be between 1 and num_layers - 1")
    protected = protected_layers or set()
    all_layers = set(range(num_layers))
    routes: list[RouteDefinition] = []
    for start in range(0, num_layers - group_size + 1):
        removed = set(range(start, start + group_size))
        if removed & protected:
            continue
        routes.append(
            RouteDefinition(
                route_id=f"skip_group_{start:02d}_{start + group_size - 1:02d}",
                active_layers=sorted(all_layers - removed),
                source="adjacent_group_ablation",
                metadata={"removed_layers": sorted(removed)},
            )
        )
    return routes


def random_deletion_routes(
    num_layers: int,
    active_budget: int,
    samples: int,
    protected_layers: set[int] | None = None,
    seed: int = 17,
) -> list[RouteDefinition]:
    protected = protected_layers or set()
    if active_budget < len(protected) or active_budget >= num_layers:
        raise ValueError("active_budget must preserve protected layers and skip at least one layer")
    eligible = sorted(set(range(num_layers)) - protected)
    choose = active_budget - len(protected)
    rng = random.Random(seed)
    seen: set[tuple[int, ...]] = set()
    routes: list[RouteDefinition] = []
    attempts = 0
    max_attempts = max(samples * 20, 100)
    while len(routes) < samples and attempts < max_attempts:
        attempts += 1
        active = tuple(sorted(protected | set(rng.sample(eligible, choose))))
        if active in seen:
            continue
        seen.add(active)
        routes.append(
            RouteDefinition(
                route_id=f"random_b{active_budget:02d}_v{len(routes):02d}",
                active_layers=list(active),
                source="random_ablation",
                metadata={"budget": active_budget, "seed": seed},
            )
        )
    if len(routes) < samples:
        raise ValueError("requested more unique random routes than the constraints permit")
    return routes
