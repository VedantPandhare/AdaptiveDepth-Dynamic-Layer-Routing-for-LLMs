from __future__ import annotations

import json
from pathlib import Path

from adaptivedepth.config import load_project_settings
from adaptivedepth.evaluation.baseline import load_cases


def test_phase4_dataset_and_notebook_are_valid() -> None:
    cases = load_cases("data/phase4_eval.jsonl")
    assert len(cases) == 30
    assert {case.category for case in cases} == {
        "factual_qa",
        "multiple_choice",
        "reasoning",
    }
    notebook = json.loads(
        Path("notebooks/phase4_colab_kaggle.ipynb").read_text(encoding="utf-8")
    )
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 5
    notebook_source = "".join(
        line for cell in notebook["cells"] for line in cell.get("source", [])
    )
    assert "RUN_FULL = False" in notebook_source
    assert "configs/cloud_smoke.yaml" in notebook_source
    assert "data/smoke_eval.jsonl" in notebook_source


def test_cloud_smoke_configuration_is_intentionally_small() -> None:
    smoke_cases = load_cases("data/smoke_eval.jsonl")
    settings = load_project_settings("configs/cloud_smoke.yaml")
    assert len(smoke_cases) == 6
    assert settings.benchmark.max_new_tokens == 8
    assert settings.benchmark.warmup_runs == 1
    assert settings.benchmark.measured_runs == 1
