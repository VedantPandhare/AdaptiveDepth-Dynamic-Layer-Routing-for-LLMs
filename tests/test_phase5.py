from __future__ import annotations

import json

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from adaptivedepth.config import (
    Phase5EvaluationSettings,
    TrainingSettings,
    load_phase5_evaluation_settings,
    load_phase5_settings,
)
from adaptivedepth.evaluation.post_training import (
    evaluate_post_training_gate,
    post_training_markdown,
)
from adaptivedepth.model.routed_model import RoutedCausalLM
from adaptivedepth.routes import RouteDefinition, RouteLibrary
from adaptivedepth.training.data import (
    TrainingExample,
    collate_examples,
    encode_example,
    load_training_examples,
)
from adaptivedepth.training.objective import (
    causal_language_model_loss,
    teacher_student_kl,
)
from adaptivedepth.training.prepare import normalize_dolly_records
from adaptivedepth.training.trainer import (
    find_latest_checkpoint,
    sample_plan,
    train_phase5,
)


class WordTokenizer:
    chat_template = None
    pad_token_id = 0

    def __call__(self, text: str, **_kwargs):
        return {"input_ids": [index + 1 for index, _word in enumerate(text.split())]}

    def save_pretrained(self, path) -> None:
        (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def test_phase5_configs_and_measured_routes_are_valid() -> None:
    settings = load_phase5_settings("configs/phase5_smoke.yaml")
    evaluation = load_phase5_evaluation_settings("configs/phase5_evaluation.yaml")
    routes = RouteLibrary.load("configs/phase5_routes.yaml")
    routes.validate_for_depth(24)
    assert settings.training.max_steps == 4
    assert evaluation.max_full_quality_drop == 0.05
    assert [len(route.active_layers) for route in routes.routes] == [20, 21, 22, 23]


def test_training_dataset_masks_prompt_and_collates(tmp_path) -> None:
    path = tmp_path / "training.jsonl"
    path.write_text(
        json.dumps({"instruction": "Name a color", "response": "Blue"}) + "\n",
        encoding="utf-8",
    )
    examples = load_training_examples(path)
    encoded = encode_example(WordTokenizer(), examples[0], max_length=32)
    assert encoded.labels.count(-100) > 0
    assert any(label != -100 for label in encoded.labels)
    batch = collate_examples([encoded], 0, torch.device("cpu"))
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["attention_mask"].all()


def test_route_and_example_sampling_is_deterministic() -> None:
    routes = RouteLibrary(
        routes=[
            RouteDefinition(route_id="a", active_layers=[0, 2]),
            RouteDefinition(route_id="b", active_layers=[0, 1, 2]),
        ]
    )
    first = sample_plan(9, 20, 2, routes, 0.25, 17)
    second = sample_plan(9, 20, 2, routes, 0.25, 17)
    assert first == second
    assert len(first.example_indexes) == 2


def test_distillation_is_zero_for_identical_logits() -> None:
    logits = torch.tensor([[[1.0, 2.0], [3.0, -1.0], [0.5, 0.5]]])
    mask = torch.ones(1, 3, dtype=torch.long)
    divergence = teacher_student_kl(logits, logits, mask, temperature=2.0)
    assert divergence.item() == pytest.approx(0.0, abs=1e-6)
    labels = torch.tensor([[-100, 1, 0]])
    assert causal_language_model_loss(logits, labels).item() > 0


def test_latest_checkpoint_ignores_incomplete_directories(tmp_path) -> None:
    incomplete = tmp_path / "checkpoint-step-000003"
    complete = tmp_path / "checkpoint-step-000002"
    newer = tmp_path / "checkpoint-step-000010"
    incomplete.mkdir()
    complete.mkdir()
    newer.mkdir()
    (complete / "complete.json").write_text("{}", encoding="utf-8")
    (newer / "complete.json").write_text("{}", encoding="utf-8")
    assert find_latest_checkpoint(tmp_path) == newer


def test_dolly_normalization_is_seeded_and_strict() -> None:
    rows = [
        {"instruction": f"Question {index}", "context": "", "response": "Answer"}
        for index in range(6)
    ]
    first = normalize_dolly_records(rows, max_examples=3, seed=4)
    second = normalize_dolly_records(rows, max_examples=3, seed=4)
    assert first == second
    assert len(first) == 3
    with pytest.raises(ValueError):
        TrainingSettings(language_model_weight=0, distillation_weight=0)


def test_tiny_qwen_teacher_student_step_saves_complete_checkpoint(tmp_path) -> None:
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    teacher_model = Qwen2ForCausalLM(config)
    student_model = Qwen2ForCausalLM(config)
    student_model.load_state_dict(teacher_model.state_dict())
    teacher = RoutedCausalLM(teacher_model, "model.layers")
    student = RoutedCausalLM(student_model, "model.layers")
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        json.dumps({"instruction": "Name one color", "response": "Blue"}) + "\n",
        encoding="utf-8",
    )
    settings = TrainingSettings(
        max_seq_length=32,
        max_steps=1,
        warmup_steps=0,
        checkpoint_every=1,
        keep_last_checkpoints=1,
        gradient_checkpointing=False,
    )
    summary = train_phase5(
        teacher=teacher,
        student=student,
        tokenizer=WordTokenizer(),
        device=torch.device("cpu"),
        model_id="tiny-qwen",
        dataset_path=dataset,
        examples=load_training_examples(dataset),
        routes=RouteLibrary(
            routes=[RouteDefinition(route_id="tiny-route", active_layers=[0, 2, 3])]
        ),
        settings=settings,
        output_dir=tmp_path / "output",
        progress=lambda _message: None,
    )
    checkpoint = tmp_path / "output" / "checkpoint-step-000001"
    assert summary["status"] == "completed"
    assert (checkpoint / "complete.json").exists()
    assert (checkpoint / "training_state.pt").exists()


def _post_training_payloads(
    *, full_quality: float = 0.5, route_quality: float = 0.5
) -> tuple[dict, dict]:
    full_measurement = {
        "route_id": "full",
        "source": "full_model",
        "active_layers": 24,
        "quality": full_quality,
        "quality_delta": 0.0,
        "latency_ms": 100.0,
        "latency_reduction": 0.0,
        "teacher_divergence": 0.0,
    }
    route_measurement = {
        "route_id": "route_target_21",
        "source": "phase4_pareto_selection",
        "active_layers": 21,
        "quality": route_quality,
        "quality_delta": route_quality - full_quality,
        "latency_ms": 85.0,
        "latency_basis": "fixed_generation_length",
        "latency_reduction": 0.15,
        "teacher_divergence": 0.3,
    }
    experiment = {
        "status": "completed",
        "identity": {"dataset_sha256": "same-data"},
        "full_baseline": {"measurement": full_measurement},
        "routes": {
            "route_target_21": {
                "route": {
                    "route_id": "route_target_21",
                    "active_layers": list(range(21)),
                    "source": "phase4_pareto_selection",
                    "metadata": {"original_route_id": "random_b21_v00"},
                },
                "measurement": route_measurement,
            }
        },
    }
    reference = {
        "dataset_sha256": "same-data",
        "full_baseline": {"quality": 0.5},
        "routes": {
            "random_b21_v00": {
                "quality": 0.4,
                "quality_delta": -0.1,
                "teacher_divergence": 3.1,
            }
        },
    }
    return experiment, reference


def test_post_training_gate_proceeds_for_preserved_full_and_safe_route() -> None:
    experiment, reference = _post_training_payloads()
    report = evaluate_post_training_gate(
        experiment, reference, Phase5EvaluationSettings()
    )
    assert report["verdict"] == "proceed_to_phase6"
    assert report["passing_route_ids"] == ["route_target_21"]
    assert report["routes"][0]["quality_gain"] == pytest.approx(0.1)
    assert report["routes"][0]["teacher_divergence_reduction"] == pytest.approx(2.8)
    assert "Full-model retention" in post_training_markdown(report)


def test_post_training_gate_rejects_full_model_regression() -> None:
    experiment, reference = _post_training_payloads(
        full_quality=0.4, route_quality=0.4
    )
    report = evaluate_post_training_gate(
        experiment, reference, Phase5EvaluationSettings()
    )
    assert report["verdict"] == "revise_training"
    assert not report["full_quality_preserved"]
    assert report["passing_route_ids"] == []


def test_post_training_gate_rejects_a_different_evaluation_dataset() -> None:
    experiment, reference = _post_training_payloads()
    experiment["identity"]["dataset_sha256"] = "different-data"
    with pytest.raises(ValueError, match="datasets differ"):
        evaluate_post_training_gate(
            experiment, reference, Phase5EvaluationSettings()
        )
