from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from adaptivedepth.analysis.block_influence import BlockInfluenceAnalyzer
from adaptivedepth.analysis.ablation import (
    adjacent_group_deletion_routes,
    random_deletion_routes,
    single_layer_deletion_routes,
)
from adaptivedepth.analysis.teacher_divergence import mean_teacher_kl
from adaptivedepth.compiler.candidate_search import generate_candidate_routes
from adaptivedepth.compiler.pareto import RouteMeasurement, pareto_frontier
from adaptivedepth.compiler.static_suite import build_static_route_suite
from adaptivedepth.config import BenchmarkSettings, EvaluationSettings
from adaptivedepth.evaluation.baseline import BenchmarkCase, run_benchmark
from adaptivedepth.evaluation.experiment import run_static_experiment
from adaptivedepth.evaluation.gate import evaluate_static_gate, gate_markdown
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
        self.forward_calls = 0

    def forward(self, input_ids: torch.Tensor, **_kwargs):
        self.forward_calls += 1
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


def test_ablation_route_generators_preserve_constraints() -> None:
    protected = {0, 7}
    singles = single_layer_deletion_routes(8, protected)
    groups = adjacent_group_deletion_routes(8, 2, protected)
    random_routes = random_deletion_routes(
        8, active_budget=5, samples=3, protected_layers=protected, seed=9
    )
    assert len(singles) == 6
    assert groups
    for route in singles + groups + random_routes:
        assert protected.issubset(route.active_layers)
    assert all(len(route.active_layers) == 5 for route in random_routes)


def test_teacher_divergence_is_zero_for_full_route() -> None:
    routed = RoutedCausalLM(ToyCausalLM(depth=4), "model.layers")
    full_route = RouteDefinition(route_id="full-copy", active_layers=[0, 1, 2, 3])
    divergence = mean_teacher_kl(
        routed,
        ToyTokenizer(),
        ["one", "two"],
        torch.device("cpu"),
        full_route,
    )
    assert divergence == pytest.approx(0.0, abs=1e-7)


def test_static_suite_contains_required_baselines() -> None:
    influence = {index: float(index) for index in range(8)}
    candidates = RouteLibrary(
        routes=[RouteDefinition(route_id="candidate_b04_v00", active_layers=[0, 5, 6, 7])]
    )
    suite = build_static_route_suite(
        candidates,
        num_layers=8,
        influence=influence,
        budgets=[4, 6],
        preserve_first=1,
        preserve_last=1,
        random_routes_per_budget=2,
        seed=3,
    )
    sources = {route.source for route in suite.routes}
    assert sources == {
        "compiled",
        "fixed_prefix_baseline",
        "importance_static_baseline",
        "random_baseline",
    }
    assert len(suite.routes) == 9


def test_static_experiment_is_resumable(tmp_path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        '{"id":"one","category":"multiple_choice","prompt":"x","reference":"A"}\n',
        encoding="utf-8",
    )
    model = ToyCausalLM(depth=4)
    routed = RoutedCausalLM(model, "model.layers")
    routes = RouteLibrary(
        routes=[RouteDefinition(route_id="route_a", active_layers=[0, 2, 3])]
    )
    settings = BenchmarkSettings(
        max_new_tokens=2,
        warmup_runs=0,
        measured_runs=1,
        use_chat_template=False,
    )
    output = tmp_path / "experiment.json"
    first = run_static_experiment(
        routed,
        ToyTokenizer(),
        torch.device("cpu"),
        "toy",
        dataset,
        settings,
        routes,
        output,
        workdir=tmp_path,
    )
    calls_after_first = model.forward_calls
    second = run_static_experiment(
        routed,
        ToyTokenizer(),
        torch.device("cpu"),
        "toy",
        dataset,
        settings,
        routes,
        output,
        workdir=tmp_path,
    )
    assert first["status"] == "completed"
    assert second["status"] == "completed"
    assert set(second["routes"]) == {"route_a"}
    assert model.forward_calls == calls_after_first


def test_static_experiment_rejects_changed_route_suite(tmp_path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        '{"id":"one","category":"multiple_choice","prompt":"x","reference":"A"}\n',
        encoding="utf-8",
    )
    routed = RoutedCausalLM(ToyCausalLM(depth=4), "model.layers")
    settings = BenchmarkSettings(
        max_new_tokens=1,
        warmup_runs=0,
        measured_runs=1,
        use_chat_template=False,
    )
    output = tmp_path / "experiment.json"
    original = RouteLibrary(
        routes=[RouteDefinition(route_id="route_a", active_layers=[0, 2, 3])]
    )
    run_static_experiment(
        routed,
        ToyTokenizer(),
        torch.device("cpu"),
        "toy",
        dataset,
        settings,
        original,
        output,
        workdir=tmp_path,
    )
    changed = RouteLibrary(
        routes=[RouteDefinition(route_id="route_a", active_layers=[0, 1, 3])]
    )
    with pytest.raises(ValueError, match="route suite changed"):
        run_static_experiment(
            routed,
            ToyTokenizer(),
            torch.device("cpu"),
            "toy",
            dataset,
            settings,
            changed,
            output,
            workdir=tmp_path,
        )


def test_phase4_gate_proceeds_for_safe_candidate() -> None:
    full = {
        "route_id": "full",
        "source": "full_model",
        "active_layers": 8,
        "quality": 1.0,
        "quality_delta": 0.0,
        "latency_ms": 100.0,
        "latency_reduction": 0.0,
        "speedup": 1.0,
        "teacher_divergence": 0.0,
        "peak_memory_mb": None,
    }
    baseline = {
        **full,
        "route_id": "fixed_prefix_b04",
        "source": "fixed_prefix_baseline",
        "active_layers": 4,
        "quality": 0.80,
        "quality_delta": -0.20,
        "latency_ms": 60.0,
        "latency_reduction": 0.40,
        "speedup": 1.67,
        "teacher_divergence": 0.2,
    }
    candidate = {
        **full,
        "route_id": "candidate_b04_v00",
        "source": "influence_constrained_search",
        "active_layers": 4,
        "quality": 0.98,
        "quality_delta": -0.02,
        "latency_ms": 70.0,
        "latency_reduction": 0.30,
        "speedup": 1.43,
        "teacher_divergence": 0.1,
    }
    experiment = {
        "status": "completed",
        "full_baseline": {"measurement": full},
        "routes": {
            "fixed_prefix_b04": {"measurement": baseline},
            "candidate_b04_v00": {"measurement": candidate},
        },
    }
    report = evaluate_static_gate(experiment, EvaluationSettings())
    assert report["verdict"] == "proceed"
    assert report["passing_candidate_ids"] == ["candidate_b04_v00"]
    assert "Candidate assessment" in gate_markdown(report)
