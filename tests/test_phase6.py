from __future__ import annotations

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from adaptivedepth.config import Phase6CalibrationSettings, load_phase6_settings
from adaptivedepth.model.routed_model import RoutedCausalLM
from adaptivedepth.routes import RouteDefinition, RouteLibrary
from adaptivedepth.routing.dataset import (
    build_phase6_report,
    collect_phase6_dataset,
    label_route_safety,
    prompt_features,
    response_regret_metrics,
)
from adaptivedepth.training.data import TrainingExample


class Phase6TinyTokenizer:
    chat_template = None
    pad_token_id = 0
    eos_token_id = None

    def __call__(self, text: str, return_tensors=None, max_length=32, **_kwargs):
        ids = list(range(1, min(len(text.split()), max_length - 1) + 1)) or [1]
        if return_tensors == "pt":
            values = torch.tensor([ids], dtype=torch.long)
            return {"input_ids": values, "attention_mask": torch.ones_like(values)}
        return {"input_ids": ids}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens
        return " ".join(str(int(token)) for token in token_ids)


def test_phase6_configs_select_only_the_confirmed_near_miss_route() -> None:
    smoke = load_phase6_settings("configs/phase6_smoke.yaml")
    full = load_phase6_settings("configs/phase6_t4.yaml")
    routes = RouteLibrary.load("configs/phase6_route.yaml")
    routes.validate_for_depth(24)
    assert len(routes.routes) == 1
    assert routes.routes[0].route_id == "route_target_21"
    assert len(routes.routes[0].active_layers) == 21
    assert smoke.calibration.min_examples == 4
    assert full.calibration.min_examples == 150
    assert full.calibration.min_median_latency_reduction == pytest.approx(0.10)


def test_identical_route_logits_have_zero_regret_and_are_safe() -> None:
    logits = torch.tensor(
        [[[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [2.0, 0.0, 1.0]]]
    )
    labels = torch.tensor([[-100, 1, 2]])
    metrics = response_regret_metrics(logits, logits, labels, temperature=1.0)
    assert metrics["teacher_kl"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["reference_loss_regret"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["token_agreement"] == pytest.approx(1.0)
    label, criteria = label_route_safety(metrics, Phase6CalibrationSettings())
    assert label == "safe_route"
    assert all(criteria.values())


def test_route_safety_rejects_high_divergence() -> None:
    settings = Phase6CalibrationSettings()
    label, criteria = label_route_safety(
        {
            "teacher_kl": 0.8,
            "reference_loss_regret": 0.1,
            "token_agreement": 0.9,
        },
        settings,
    )
    assert label == "full_model"
    assert not criteria["teacher_kl"]


def test_prompt_features_are_inference_time_available() -> None:
    features = prompt_features("What is 12 + 5?\nExplain.", prompt_tokens=9)
    assert features["prompt_token_count"] == 9
    assert features["question_mark_count"] == 1
    assert features["contains_math_symbol"]
    assert features["digit_fraction"] > 0


def test_phase6_gate_requires_both_router_classes_and_latency() -> None:
    settings = Phase6CalibrationSettings(
        min_examples=4,
        min_safe_examples=2,
        min_unsafe_examples=2,
        min_median_latency_reduction=0.1,
    )
    records = {}
    for index, label in enumerate(
        ["safe_route", "safe_route", "full_model", "full_model"], start=1
    ):
        records[str(index)] = {
            "status": "completed",
            "label": label,
            "latency_reduction": 0.12,
            "regret": {
                "teacher_kl": 0.2,
                "reference_loss_regret": 0.05,
                "token_agreement": 0.9,
            },
        }
    experiment = {
        "identity": {"route": {"route_id": "route_target_21"}},
        "records": records,
    }
    report = build_phase6_report(experiment, settings)
    assert report["verdict"] == "proceed_to_phase7"
    assert report["safe_rate"] == pytest.approx(0.5)

    records["4"]["label"] = "safe_route"
    report = build_phase6_report(experiment, settings)
    assert report["verdict"] == "revise_phase6"
    assert not report["checks"]["minimum_unsafe_examples"]


def test_tiny_phase6_collection_writes_resumable_artifacts(tmp_path) -> None:
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
    dataset = tmp_path / "calibration.jsonl"
    dataset.write_text(
        '{"instruction":"Name a color","response":"Blue"}\n',
        encoding="utf-8",
    )
    workspace = tmp_path / "phase6"
    experiment, report = collect_phase6_dataset(
        routed_model=routed,
        tokenizer=Phase6TinyTokenizer(),
        device=torch.device("cpu"),
        model_id="tiny-qwen",
        dataset_path=dataset,
        examples=[TrainingExample(instruction="Name a color", response="Blue")],
        route=RouteDefinition(route_id="tiny-route", active_layers=[0, 2, 3]),
        settings=Phase6CalibrationSettings(
            max_seq_length=32,
            max_new_tokens=2,
            warmup_runs=0,
            min_examples=1,
            min_safe_examples=1,
            min_unsafe_examples=1,
        ),
        workspace=workspace,
        workdir=tmp_path,
        progress=lambda _message: None,
    )
    assert experiment["status"] == "completed"
    assert len(experiment["records"]) == 1
    assert report["valid_examples"] == 1
    assert (workspace / "experiment.json").exists()
    assert (workspace / "router_dataset.jsonl").exists()
    assert (workspace / "gate.md").exists()
