from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from adaptivedepth.config import Phase61CalibrationSettings, load_phase61_settings
from adaptivedepth.model.routed_model import RoutedCausalLM
from adaptivedepth.routes import RouteDefinition, RouteLibrary
from adaptivedepth.routing.multiroute_dataset import (
    build_phase61_report,
    collect_phase61_dataset,
)
from adaptivedepth.runtime.generation import GenerationProfile
from adaptivedepth.training.data import (
    TrainingExample,
    encode_example_preserving_response,
)


class WordTokenizer:
    chat_template = None
    pad_token_id = 0
    eos_token_id = None

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}

    def _ids(self, text: str) -> list[int]:
        values = []
        for word in text.split():
            if word not in self.vocabulary:
                self.vocabulary[word] = len(self.vocabulary) % 30 + 1
            values.append(self.vocabulary[word])
        return values or [1]

    def __call__(
        self,
        text: str,
        return_tensors=None,
        truncation=False,
        max_length=None,
        **_kwargs,
    ):
        ids = self._ids(text)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if return_tensors == "pt":
            values = torch.tensor([ids], dtype=torch.long)
            return {"input_ids": values, "attention_mask": torch.ones_like(values)}
        return {"input_ids": ids}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens
        return " ".join(str(int(token)) for token in token_ids)


def _route(route_id: str, active_layers: list[int]) -> RouteDefinition:
    return RouteDefinition(route_id=route_id, active_layers=active_layers)


def _measurement(label: str, latency_reduction: float) -> dict:
    return {
        "label": label,
        "latency_reduction": latency_reduction,
        "regret": {
            "teacher_kl": 0.1,
            "reference_loss_regret": 0.02,
            "token_agreement": 0.9,
        },
        "generation": {
            "exact_agreement": label == "safe_route",
            "token_jaccard": 0.8 if label == "safe_route" else 0.2,
        },
    }


def test_phase61_configs_lock_four_routes_and_400_prompt_policy() -> None:
    settings = load_phase61_settings("configs/phase61_t4.yaml")
    routes = RouteLibrary.load("configs/phase61_routes.yaml")
    routes.validate_for_depth(24)
    assert settings.calibration.response_token_budget == 16
    assert settings.calibration.min_examples == 300
    assert [len(route.active_layers) for route in routes.routes] == [20, 21, 22, 23]

    notebook = json.loads(
        Path("notebooks/phase61_multiroute_drive.ipynb").read_text(encoding="utf-8")
    )
    source = "".join(
        line for cell in notebook["cells"] for line in cell.get("source", [])
    )
    assert "RUN_FULL = False" in source
    assert "--max-examples', '400'" in source
    assert "phase61_workspace" in source
    assert "--resume" in source


def test_response_preserving_encoding_reserves_reference_tokens() -> None:
    tokenizer = WordTokenizer()
    example = TrainingExample(
        instruction=" ".join(f"prompt{i}" for i in range(50)),
        response=" ".join(f"answer{i}" for i in range(30)),
    )
    encoded = encode_example_preserving_response(
        tokenizer, example, max_length=20, response_token_budget=6
    )
    assert len(encoded.input_ids) == 20
    assert encoded.prompt_token_count == 14
    assert encoded.available_response_tokens == 30
    assert encoded.used_response_tokens == 6
    assert encoded.prompt_was_truncated
    assert encoded.response_was_truncated
    assert encoded.labels.count(-100) == 14
    assert sum(label != -100 for label in encoded.labels) == 6


def test_phase61_report_selects_cheapest_safe_route() -> None:
    routes = [
        _route("route20", [0]),
        _route("route21", [0, 1]),
        _route("route22", [0, 1, 2]),
        _route("route23", [0, 1, 2, 3]),
    ]
    patterns = [
        dict.fromkeys(("route20",), "safe_route")
        | dict.fromkeys(("route21", "route22", "route23"), "full_model"),
        dict.fromkeys(("route20",), "full_model")
        | dict.fromkeys(("route21", "route22", "route23"), "safe_route"),
        dict.fromkeys(("route20", "route22"), "safe_route")
        | dict.fromkeys(("route21", "route23"), "full_model"),
        dict.fromkeys(("route21", "route23"), "safe_route")
        | dict.fromkeys(("route20", "route22"), "full_model"),
        dict.fromkeys(("route20", "route21", "route22", "route23"), "full_model"),
    ]
    latency = {"route20": 0.20, "route21": 0.15, "route22": 0.08, "route23": 0.04}
    records = [
        {
            "status": "completed",
            "encoding": {
                "prompt_was_truncated": False,
                "available_response_tokens": 16,
                "used_response_tokens": 16,
            },
            "routes": {
                route_id: _measurement(label, latency[route_id])
                for route_id, label in pattern.items()
            },
        }
        for pattern in patterns
    ]
    report = build_phase61_report(
        records,
        routes,
        Phase61CalibrationSettings(
            min_examples=5,
            min_safe_examples=1,
            min_unsafe_examples=1,
            min_eligible_routes=2,
            min_route_latency_reduction=0.01,
        ),
    )
    assert report["verdict"] == "proceed_to_phase7_multiroute"
    assert report["checks"]["response_preservation"]
    assert report["route_order_by_measured_latency"] == [
        "route20",
        "route21",
        "route22",
        "route23",
    ]
    assert report["target_route_counts"] == {
        "route20": 2,
        "route21": 2,
        "route22": 0,
        "route23": 0,
        "full": 1,
    }


def test_tiny_phase61_collection_reuses_full_profile_and_resumes(
    tmp_path, monkeypatch
) -> None:
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    routed = RoutedCausalLM(Qwen2ForCausalLM(config).eval(), "model.layers")
    tokenizer = WordTokenizer()
    routes = [
        _route("route20", [0]),
        _route("route21", [0, 1]),
        _route("route22", [0, 1, 2]),
        _route("route23", [0, 1, 2, 3]),
    ]
    calls = {"full": 0, **{route.route_id: 0 for route in routes}}

    def fake_profile(
        _model, _tokenizer, _prompt, _device, _max_new_tokens, route
    ) -> GenerationProfile:
        route_id = route.route_id if route is not None else "full"
        calls[route_id] += 1
        total_ms = 100.0 if route is None else 70.0 + len(route.active_layers) * 5
        return GenerationProfile(
            text=("unrelated tokens" if route_id == "route23" else "same output"),
            input_tokens=4,
            output_tokens=2,
            profiled_tokens=2,
            ttft_ms=total_ms / 2,
            decode_ms=total_ms / 2,
            total_ms=total_ms,
            decode_ms_per_token=total_ms / 2,
            tokens_per_second=20.0,
            peak_memory_mb=None,
        )

    monkeypatch.setattr(
        "adaptivedepth.routing.multiroute_dataset._profile", fake_profile
    )
    dataset = tmp_path / "calibration.jsonl"
    dataset.write_text(
        '{"instruction":"Name a color","response":"Blue sky"}\n',
        encoding="utf-8",
    )
    workspace = tmp_path / "phase61"
    arguments = {
        "routed_model": routed,
        "tokenizer": tokenizer,
        "device": torch.device("cpu"),
        "model_id": "tiny-qwen",
        "dataset_path": dataset,
        "examples": [TrainingExample(instruction="Name a color", response="Blue sky")],
        "routes": routes,
        "settings": Phase61CalibrationSettings(
            max_seq_length=32,
            response_token_budget=4,
            max_new_tokens=2,
            warmup_runs=0,
            min_examples=1,
            min_safe_examples=1,
            min_unsafe_examples=1,
            min_eligible_routes=1,
            min_route_latency_reduction=0.0,
        ),
        "workspace": workspace,
        "workdir": tmp_path,
        "progress": lambda _message: None,
    }
    _manifest, report = collect_phase61_dataset(**arguments)
    assert report["valid_examples"] == 1
    assert calls == {"full": 1, "route20": 1, "route21": 1, "route22": 1, "route23": 1}
    assert sum(1 for _ in (workspace / "records.jsonl").open(encoding="utf-8")) == 1
    output = json.loads((workspace / "router_dataset.jsonl").read_text(encoding="utf-8"))
    assert set(output["routes"]) == {"route20", "route21", "route22", "route23"}
    assert len(output["execution_order"]) == 5
    assert not output["routes"]["route23"]["criteria"]["generation_token_jaccard"]
    assert output["routes"]["route23"]["label"] == "full_model"

    collect_phase61_dataset(**arguments)
    assert calls == {"full": 1, "route20": 1, "route21": 1, "route22": 1, "route23": 1}
