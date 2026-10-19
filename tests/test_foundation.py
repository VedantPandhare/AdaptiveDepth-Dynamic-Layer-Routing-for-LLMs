from __future__ import annotations

from pathlib import Path

from torch import nn

from adaptivedepth.config import load_project_settings
from adaptivedepth.model.routed_model import RoutedCausalLM
from adaptivedepth.routes import RouteDefinition, RouteLibrary


class ToyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(4)])


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = ToyBackbone()


def test_route_context_restores_full_model_after_exit() -> None:
    model = ToyModel()
    routed = RoutedCausalLM(model, "model.layers")
    route = RouteDefinition(route_id="draft", active_layers=[0, 2, 3])

    with routed.use_route(route):
        assert len(model.model.layers) == 3

    assert len(model.model.layers) == 4


def test_route_library_round_trip(tmp_path: Path) -> None:
    library = RouteLibrary(
        routes=[RouteDefinition(route_id="draft", active_layers=[0, 2, 3])]
    )
    destination = tmp_path / "routes.yaml"
    library.save(destination)

    restored = RouteLibrary.load(destination)
    assert restored.get("draft").active_layers == [0, 2, 3]


def test_project_config_loads_initial_verified_generation_defaults() -> None:
    settings = load_project_settings("configs/model.yaml")
    assert settings.model.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert settings.benchmark.max_new_tokens == 64
