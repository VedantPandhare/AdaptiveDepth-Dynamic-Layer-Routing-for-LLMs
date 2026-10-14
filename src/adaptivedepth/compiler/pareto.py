from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RouteMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route_id: str
    quality: float
    latency_ms: float = Field(gt=0)
    teacher_divergence: float = Field(ge=0)
    peak_memory_mb: float | None = Field(default=None, gt=0)


def dominates(left: RouteMeasurement, right: RouteMeasurement) -> bool:
    comparisons = [
        left.quality >= right.quality,
        left.latency_ms <= right.latency_ms,
        left.teacher_divergence <= right.teacher_divergence,
    ]
    strict = [
        left.quality > right.quality,
        left.latency_ms < right.latency_ms,
        left.teacher_divergence < right.teacher_divergence,
    ]
    if left.peak_memory_mb is not None and right.peak_memory_mb is not None:
        comparisons.append(left.peak_memory_mb <= right.peak_memory_mb)
        strict.append(left.peak_memory_mb < right.peak_memory_mb)
    return all(comparisons) and any(strict)


def pareto_frontier(measurements: list[RouteMeasurement]) -> list[RouteMeasurement]:
    """Return non-dominated routes ordered from lowest to highest latency."""

    frontier = [
        candidate
        for candidate in measurements
        if not any(
            other.route_id != candidate.route_id and dominates(other, candidate)
            for other in measurements
        )
    ]
    return sorted(frontier, key=lambda item: (item.latency_ms, -item.quality))
