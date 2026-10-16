from __future__ import annotations

import json

import pytest
import torch

from adaptivedepth.config import TrainingSettings, load_phase5_settings
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
from adaptivedepth.training.trainer import find_latest_checkpoint, sample_plan


class WordTokenizer:
    chat_template = None
    pad_token_id = 0

    def __call__(self, text: str, **_kwargs):
        return {"input_ids": [index + 1 for index, _word in enumerate(text.split())]}


def test_phase5_configs_and_measured_routes_are_valid() -> None:
    settings = load_phase5_settings("configs/phase5_smoke.yaml")
    routes = RouteLibrary.load("configs/phase5_routes.yaml")
    routes.validate_for_depth(24)
    assert settings.training.max_steps == 4
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
