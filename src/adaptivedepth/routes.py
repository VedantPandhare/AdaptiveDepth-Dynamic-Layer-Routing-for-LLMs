from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RouteDefinition(BaseModel):
    """An ordered, validated transformer subnetwork."""

    model_config = ConfigDict(extra="forbid")

    route_id: str = Field(min_length=1)
    active_layers: list[int]
    source: str = "compiled"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("active_layers")
    @classmethod
    def validate_active_layers(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("a route must contain at least one active layer")
        if any(index < 0 for index in value):
            raise ValueError("layer indexes must be non-negative")
        if value != sorted(set(value)):
            raise ValueError("active_layers must be sorted and unique")
        return value

    def validate_for_depth(self, num_layers: int) -> None:
        if self.active_layers[-1] >= num_layers:
            raise ValueError(
                f"route {self.route_id!r} references layer {self.active_layers[-1]}, "
                f"but the model has {num_layers} layers"
            )


class RouteLibrary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    routes: list[RouteDefinition] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_route_ids(self) -> "RouteLibrary":
        route_ids = [route.route_id for route in self.routes]
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("route_id values must be unique")
        return self

    def get(self, route_id: str) -> RouteDefinition:
        for route in self.routes:
            if route.route_id == route_id:
                return route
        raise KeyError(f"unknown route: {route_id}")

    def validate_for_depth(self, num_layers: int) -> None:
        for route in self.routes:
            route.validate_for_depth(num_layers)

    @classmethod
    def load(cls, path: str | Path) -> "RouteLibrary":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle) or {})

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                self.model_dump(mode="json"),
                handle,
                sort_keys=False,
                allow_unicode=True,
            )
