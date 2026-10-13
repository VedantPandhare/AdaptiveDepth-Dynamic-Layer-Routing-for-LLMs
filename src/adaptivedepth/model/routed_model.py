from __future__ import annotations

from contextlib import contextmanager
from threading import RLock
from typing import Any, Iterator

from torch import nn

from adaptivedepth.model.layers import LayerStack, discover_layer_stack
from adaptivedepth.routes import RouteDefinition


class RoutedCausalLM:
    """Apply one fixed transformer route for an entire request.

    V1 swaps the model's ModuleList inside a guarded context. This keeps each
    original block and its original cache layer index intact while avoiding
    omitted-block computation. It is single-worker code by design; a later
    serving runtime should use immutable model replicas per route.
    """

    def __init__(self, model: nn.Module, layer_path: str | None = None) -> None:
        self.model = model
        self.stack: LayerStack = discover_layer_stack(model, layer_path)
        self._full_layers = self.stack.layers
        self._lock = RLock()
        self._active_route: str | None = None

    @property
    def num_layers(self) -> int:
        return len(self._full_layers)

    @property
    def layer_path(self) -> str:
        return self.stack.path

    @contextmanager
    def use_route(self, route: RouteDefinition | None) -> Iterator[nn.Module]:
        """Use ``route`` until the context exits, then restore the full model."""

        if route is None:
            yield self.model
            return
        route.validate_for_depth(self.num_layers)
        selected = nn.ModuleList([self._full_layers[index] for index in route.active_layers])
        with self._lock:
            if self._active_route is not None:
                raise RuntimeError("nested route contexts are not supported")
            self._active_route = route.route_id
            self.stack.replace(selected)
            try:
                yield self.model
            finally:
                self.stack.replace(self._full_layers)
                self._active_route = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)
