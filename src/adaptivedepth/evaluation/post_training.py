from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adaptivedepth.compiler.pareto import RouteMeasurement, pareto_frontier
from adaptivedepth.config import Phase5EvaluationSettings


def _pareto_measurement(value: dict[str, Any]) -> RouteMeasurement:
    return RouteMeasurement(
        route_id=value["route_id"],
        quality=value["quality"],
        latency_ms=value["latency_ms"],
        teacher_divergence=value["teacher_divergence"],
        peak_memory_mb=value.get("peak_memory_mb"),
    )


def evaluate_post_training_gate(
    experiment: dict[str, Any],
    reference: dict[str, Any],
    settings: Phase5EvaluationSettings,
) -> dict[str, Any]:
    if experiment.get("status") != "completed":
        return {
            "verdict": "insufficient_data",
            "reason": "the post-training route experiment has not completed",
        }
    full_entry = experiment.get("full_baseline")
    route_entries = experiment.get("routes", {})
    if full_entry is None or not route_entries:
        return {
            "verdict": "insufficient_data",
            "reason": "full baseline and trained route measurements are required",
        }
    if experiment.get("identity", {}).get("dataset_sha256") != reference.get(
        "dataset_sha256"
    ):
        raise ValueError("post-training and reference evaluation datasets differ")

    full = full_entry["measurement"]
    original_full = reference["full_baseline"]
    full_quality_change = full["quality"] - original_full["quality"]
    full_preserved = full_quality_change >= -settings.max_full_quality_drop

    assessed: list[dict[str, Any]] = []
    measurements = [full]
    for route_id, entry in route_entries.items():
        route = entry["route"]
        measurement = entry["measurement"]
        measurements.append(measurement)
        original_id = route.get("metadata", {}).get("original_route_id")
        if not original_id or original_id not in reference["routes"]:
            raise ValueError(f"route {route_id!r} has no Phase 4 reference mapping")
        original = reference["routes"][original_id]
        constraint_pass = (
            measurement["quality_delta"] >= -settings.max_route_quality_drop
            and measurement["latency_reduction"] >= settings.min_latency_reduction
            and measurement["teacher_divergence"] <= settings.max_teacher_divergence
        )
        assessed.append(
            {
                **measurement,
                "original_route_id": original_id,
                "pretraining_quality": original["quality"],
                "quality_gain": measurement["quality"] - original["quality"],
                "pretraining_quality_delta": original["quality_delta"],
                "quality_delta_improvement": (
                    measurement["quality_delta"] - original["quality_delta"]
                ),
                "pretraining_teacher_divergence": original["teacher_divergence"],
                "teacher_divergence_reduction": (
                    original["teacher_divergence"]
                    - measurement["teacher_divergence"]
                ),
                "constraint_pass": constraint_pass,
                "passes_gate": full_preserved and constraint_pass,
            }
        )

    passing = [item for item in assessed if item["passes_gate"]]
    if not full_preserved:
        verdict = "revise_training"
        reason = "full-depth quality regressed beyond the allowed Phase 5 limit"
    elif passing:
        verdict = "proceed_to_phase6"
        reason = "full-depth quality is preserved and at least one trained route passes"
    else:
        verdict = "revise_training"
        reason = "full-depth quality is preserved, but no trained route passes all constraints"

    pareto_ids = [
        item.route_id
        for item in pareto_frontier(
            [_pareto_measurement(measurement) for measurement in measurements]
        )
    ]
    return {
        "verdict": verdict,
        "reason": reason,
        "constraints": settings.model_dump(mode="json"),
        "latency_basis": full.get(
            "latency_basis", "legacy_variable_generation_length"
        ),
        "original_full_quality": original_full["quality"],
        "trained_full": full,
        "full_quality_change": full_quality_change,
        "full_quality_preserved": full_preserved,
        "passing_route_ids": [item["route_id"] for item in passing],
        "pareto_route_ids": pareto_ids,
        "routes": assessed,
    }


def post_training_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# AdaptiveDepth Phase 5 post-training gate",
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
            "## Full-model retention",
            "",
            f"- Original full quality: {report['original_full_quality']:.3f}",
            f"- Trained full quality: {report['trained_full']['quality']:.3f}",
            f"- Quality change: {report['full_quality_change']:+.3f}",
            f"- Maximum allowed drop: {constraints['max_full_quality_drop']:.3f}",
            f"- Retention gate: {'pass' if report['full_quality_preserved'] else 'fail'}",
            "",
            "## Trained route assessment",
            "",
            "| Route | Quality Δ | Latency reduction | Teacher KL | Quality gain vs pre | KL reduction vs pre | Gate |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for route in report["routes"]:
        lines.append(
            "| {route_id} | {quality_delta:+.3f} | {latency_reduction:.1%} | "
            "{teacher_divergence:.4f} | {quality_gain:+.3f} | "
            "{teacher_divergence_reduction:+.4f} | {gate} |".format(
                **route, gate="pass" if route["passes_gate"] else "fail"
            )
        )
    lines.extend(
        [
            "",
            "## Constraints",
            "",
            f"- Maximum route quality drop: {constraints['max_route_quality_drop']:.3f}",
            f"- Minimum latency reduction: {constraints['min_latency_reduction']:.1%}",
            f"- Maximum teacher divergence: {constraints['max_teacher_divergence']:.3f}",
            f"- Latency basis: `{report['latency_basis']}`",
            "",
            "## Post-training Pareto routes",
            "",
            ", ".join(f"`{route_id}`" for route_id in report["pareto_route_ids"]),
            "",
        ]
    )
    return "\n".join(lines)


def save_post_training_report(
    report: dict[str, Any], json_path: str | Path, markdown_path: str | Path
) -> None:
    json_destination = Path(json_path)
    markdown_destination = Path(markdown_path)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    markdown_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_destination.write_text(
        post_training_markdown(report), encoding="utf-8"
    )
