"""Prompt-level route-risk data and routing utilities."""

from typing import Any


__all__ = ["collect_phase6_dataset"]


def __getattr__(name: str) -> Any:
    """Keep CPU-only router tooling from importing Torch eagerly."""
    if name == "collect_phase6_dataset":
        from adaptivedepth.routing.dataset import collect_phase6_dataset

        return collect_phase6_dataset
    raise AttributeError(name)
