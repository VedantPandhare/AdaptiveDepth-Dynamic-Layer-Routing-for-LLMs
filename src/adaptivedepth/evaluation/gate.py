from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adaptivedepth.compiler.pareto import (
    RouteMeasurement,
    dominates,
    pareto_frontier,
)
from adaptivedepth.config import EvaluationSettings


CANDIDATE_SOURCE = "influence_constrained_search"
BASELINE_SOURCES = {
    "fixed_prefix_baseline",
    "importance_static_baseline",
    "random_baseline",
}


def _as_pareto_measurement(value: dict[str, Any]) -> RouteMeasurement:
    return RouteMeasurement(
        route_id=value["route_id"],
        quality=value["quality"],
        latency_ms=value["latency_ms"],
        teacher_divergence=value["teacher_divergence"],
        peak_memory_mb=value.get("peak_memory_mb"),
    )


def evaluate_static_gate(
    experiment: dict[str, Any],
    settings: EvaluationSettings,
) -> dict[str, Any]:
    full = experiment.get("full_baseline")
    route_entries = list(experiment.get("routes", {}).values())
    if full is None or not route_entries:
        return {
            "verdict": "insufficient_data",
            "reason": "full baseline and route measurements are required",
        }

    measurements = [full["measurement"]] + [
        entry["measurement"] for entry in route_entries
    ]
    pareto_ids = [
        item.route_id
        for item in pareto_frontier([_as_pareto_measurement(item) for item in measurements])
    ]
    baselines = [
        item for item in measurements if item.get("source") in BASELINE_SOURCES
    ]
    candidates = [
        item for item in measurements if item.get("source") == CANDIDATE_SOURCE
    ]

    assessed_candidates = []
    for candidate in candidates:
        constraint_pass = (
            candidate["quality_delta"] >= -settings.max_quality_drop
            and candidate["latency_reduction"] >= settings.min_latency_reduction
            and candidate["teacher_divergence"] <= settings.max_teacher_divergence
        )
        candidate_measurement = _as_pareto_measurement(candidate)
        dominating_baselines = [
            baseline["route_id"]
            for baseline in baselines
            if dominates(_as_pareto_measurement(baseline), candidate_measurement)
        ]
        has_advantage = not dominating_baselines
        passes = constraint_pass and (
            has_advantage or not settings.require_candidate_advantage
        )
        assessed_candidates.append(
            {
                **candidate,
                "constraint_pass": constraint_pass,
                "dominating_baselines": dominating_baselines,
                "candidate_advantage": has_advantage,
                "passes_gate": passes,
            }
        )

    passing = [item for item in assessed_candidates if item["passes_gate"]]
    if experiment.get("status") != "completed":
        verdict = "insufficient_data"
        reason = "the route suite has not completed"
    elif passing:
        verdict = "proceed"
        reason = "at least one compiled route satisfies quality, latency, and baseline constraints"
    else:
        verdict = "revise_routes"
        reason = "no compiled route satisfies all Phase 4 constraints"

    return {
        "verdict": verdict,
        "reason": reason,
        "constraints": settings.model_dump(mode="json"),
        "full_baseline": full["measurement"],
        "pareto_route_ids": pareto_ids,
        "passing_candidate_ids": [item["route_id"] for item in passing],
        "candidates": assessed_candidates,
        "route_count": len(route_entries),
    }


def gate_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# AdaptiveDepth Phase 4 gate",
        "",
        f"**Verdict:** `{report['verdict']}`",
        "",
        report["reason"],
        "",
    ]
    if "constraints" not in report:
        return "\n".join(lines)
    constraints = report["constraints"]
    lines.extend(
        [
            "## Constraints",
            "",
            f"- Maximum quality drop: {constraints['max_quality_drop']:.3f}",
            f"- Minimum latency reduction: {constraints['min_latency_reduction']:.1%}",
            f"- Maximum teacher divergence: {constraints['max_teacher_divergence']:.3f}",
            "",
            "## Candidate assessment",
            "",
            "| Route | Quality Δ | Latency reduction | Teacher KL | Gate |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for candidate in report["candidates"]:
        lines.append(
            "| {route_id} | {quality_delta:+.3f} | {latency_reduction:.1%} | "
            "{teacher_divergence:.4f} | {gate} |".format(
                **candidate,
                gate="pass" if candidate["passes_gate"] else "fail",
            )
        )
    lines.extend(
        [
            "",
            "## Pareto routes",
            "",
            ", ".join(f"`{route_id}`" for route_id in report["pareto_route_ids"]),
            "",
        ]
    )
    return "\n".join(lines)


def save_gate_report(
    report: dict[str, Any],
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    json_destination = Path(json_path)
    markdown_destination = Path(markdown_path)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    markdown_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_destination.write_text(gate_markdown(report), encoding="utf-8")
