from __future__ import annotations

import json
from pathlib import Path

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
