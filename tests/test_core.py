from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from adaptivedepth.analysis.block_influence import BlockInfluenceAnalyzer
from adaptivedepth.compiler.candidate_search import generate_candidate_routes
from adaptivedepth.compiler.pareto import RouteMeasurement, pareto_frontier
from adaptivedepth.config import BenchmarkSettings
from adaptivedepth.evaluation.baseline import BenchmarkCase, run_benchmark
from adaptivedepth.model.routed_model import RoutedCausalLM
from adaptivedepth.routes import RouteDefinition, RouteLibrary
from adaptivedepth.runtime.generation import profile_greedy_generation


class AddBlock(nn.Module):
    def __init__(self, amount: float) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, hidden_states: torch.Tensor, **_kwargs) -> torch.Tensor:
        return hidden_states + self.amount


class ToyBackbone(nn.Module):
    def __init__(self, depth: int = 6) -> None:
        super().__init__()
        self.layers = nn.ModuleList(AddBlock(index + 1.0) for index in range(depth))


class ToyCausalLM(nn.Module):
    def __init__(self, depth: int = 6) -> None:
        super().__init__()
        self.model = ToyBackbone(depth)

    def forward(self, input_ids: torch.Tensor, **_kwargs):
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 3)
        for layer in self.model.layers:
            hidden = layer(hidden)
        logits = torch.stack([hidden[..., 0], -hidden[..., 0]], dim=-1)
        return SimpleNamespace(logits=logits, past_key_values=None)


class ToyTokenizer:
    eos_token_id = None

    def __call__(self, _text: str, return_tensors: str = "pt"):
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3),
        }

    def decode(self, _token_ids, skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens
        return "A"


def test_route_context_executes_only_selected_layers_and_restores_model() -> None:
    routed = RoutedCausalLM(ToyCausalLM(depth=6), "model.layers")
    route = RouteDefinition(route_id="small", active_layers=[0, 2, 5])
    input_ids = torch.tensor([[1]])

    full_logits = routed(input_ids=input_ids).logits
    with routed.use_route(route):
        assert len(routed.model.model.layers) == 3
        route_logits = routed(input_ids=input_ids).logits
    assert len(routed.model.model.layers) == 6
    assert full_logits[0, 0, 0].item() == pytest.approx(22.0)
    assert route_logits[0, 0, 0].item() == pytest.approx(11.0)


def test_route_context_restores_after_failure() -> None:
    routed = RoutedCausalLM(ToyCausalLM(depth=4), "model.layers")
    route = RouteDefinition(route_id="small", active_layers=[0, 3])
    with pytest.raises(RuntimeError):
        with routed.use_route(route):
            raise RuntimeError("boom")
    assert len(routed.model.model.layers) == 4


def test_route_validation_and_library_round_trip(tmp_path) -> None:
    route = RouteDefinition(route_id="route_a", active_layers=[0, 2, 3])
    library = RouteLibrary(routes=[route])
    path = tmp_path / "routes.yaml"
    library.save(path)
    loaded = RouteLibrary.load(path)
    assert loaded.get("route_a") == route
    with pytest.raises(ValueError):
        RouteDefinition(route_id="bad", active_layers=[2, 1])


def test_block_influence_observes_every_layer() -> None:
    routed = RoutedCausalLM(ToyCausalLM(depth=4), "model.layers")
    analyzer = BlockInfluenceAnalyzer(routed)
    scores = analyzer.analyze_prompts(
        ToyTokenizer(), ["one", "two"], torch.device("cpu")
    )
    assert [score.layer_index for score in scores] == [0, 1, 2, 3]
    assert all(score.observations == 2 for score in scores)
    assert all(score.influence >= 0 for score in scores)


def test_candidate_generation_is_constrained_and_deterministic() -> None:
    influence = {index: float(index) for index in range(10)}
    first = generate_candidate_routes(
        10,
        influence,
        [5, 7],
        preserve_first=1,
        preserve_last=1,
        variants_per_budget=3,
        seed=4,
    )
    second = generate_candidate_routes(
        10,
        influence,
        [5, 7],
        preserve_first=1,
        preserve_last=1,
        variants_per_budget=3,
        seed=4,
    )
    assert first == second
    assert first
    for route in first:
        assert 0 in route.active_layers
        assert 9 in route.active_layers
        assert len(route.active_layers) in {5, 7}


def test_pareto_filter_removes_dominated_route() -> None:
    measurements = [
        RouteMeasurement(
            route_id="r1", quality=0.80, latency_ms=100, teacher_divergence=0.2
        ),
        RouteMeasurement(
            route_id="dominated",
            quality=0.70,
            latency_ms=120,
            teacher_divergence=0.3,
        ),
        RouteMeasurement(
            route_id="r2", quality=0.90, latency_ms=150, teacher_divergence=0.1
        ),
    ]
    assert [item.route_id for item in pareto_frontier(measurements)] == ["r1", "r2"]


def test_generation_profiler_and_baseline_harness() -> None:
    routed = RoutedCausalLM(ToyCausalLM(depth=4), "model.layers")
    tokenizer = ToyTokenizer()
    profile = profile_greedy_generation(
        routed, tokenizer, "prompt", torch.device("cpu"), max_new_tokens=3
    )
    assert profile.text == "A"
    assert profile.input_tokens == 3
    assert profile.output_tokens == 3
    assert profile.total_ms >= profile.ttft_ms

    report = run_benchmark(
        routed,
        tokenizer,
        torch.device("cpu"),
        [BenchmarkCase("case-1", "multiple_choice", "prompt", "A")],
        BenchmarkSettings(max_new_tokens=2, warmup_runs=0, measured_runs=2),
    )
    assert report.route_id == "full"
    assert report.accuracy == 1.0
    assert report.active_layers == 4
