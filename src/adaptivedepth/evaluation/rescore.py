from __future__ import annotations

from copy import deepcopy
from typing import Any

from adaptivedepth.evaluation.baseline import is_answer_match, is_exact_match


SCORING_METHOD = "answer_match_v2"


def rescore_report(report: dict[str, Any]) -> dict[str, Any]:
    """Recompute quality from saved predictions without running the model."""

    rescored = deepcopy(report)
    results = rescored.get("results", [])
    if not results:
        raise ValueError("cannot rescore a report without case results")
    for result in results:
        result["exact_match"] = is_exact_match(
            result["prediction"], result["reference"]
        )
        result["answer_match"] = is_answer_match(
            result["prediction"], result["reference"], result["category"]
        )
    rescored["accuracy"] = sum(item["answer_match"] for item in results) / len(results)
    rescored["scoring_method"] = SCORING_METHOD
    return rescored


def rescore_experiment(experiment: dict[str, Any]) -> dict[str, Any]:
    """Rescore full and routed reports while retaining measured GPU metrics."""

    rescored = deepcopy(experiment)
    full = rescored.get("full_baseline")
    if full is None:
        raise ValueError("experiment does not contain a full baseline")
    full["report"] = rescore_report(full["report"])
    full_quality = full["report"]["accuracy"]
    full["measurement"]["quality"] = full_quality
    full["measurement"]["quality_delta"] = 0.0

    for entry in rescored.get("routes", {}).values():
        entry["report"] = rescore_report(entry["report"])
        route_quality = entry["report"]["accuracy"]
        entry["measurement"]["quality"] = route_quality
        entry["measurement"]["quality_delta"] = route_quality - full_quality

    rescored["scoring_method"] = SCORING_METHOD
    return rescored
