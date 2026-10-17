from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from adaptivedepth.config import Phase7RouterSettings, load_phase7_settings
from adaptivedepth.routing.features import prompt_features
from adaptivedepth.routing.risk_router import (
    choose_abstention_threshold,
    expected_calibration_error,
    load_router_artifact,
    route_prompt,
    router_input_row,
    train_phase7_router,
)


class _AlwaysSafeEstimator:
    def predict_proba(self, rows):
        return np.asarray([[0.01, 0.99] for _ in rows])


def _record(index: int, safe: bool) -> dict:
    if safe:
        prompt = f"simple greeting hello friendly short answer example {index}"
        label = "safe_route"
    else:
        prompt = f"complex mathematical proof code multi step calculation example {index}"
        label = "full_model"
    return {
        "status": "completed",
        "example_id": f"example-{index:03d}",
        "prompt": prompt,
        "features": prompt_features(prompt, prompt_tokens=len(prompt.split())),
        "regret": {
            "teacher_kl": 0.01 if safe else 2.0,
            "reference_loss_regret": 0.01 if safe else 1.0,
            "token_agreement": 1.0 if safe else 0.0,
        },
        "generation": {"full_text": "a", "route_text": "a" if safe else "b"},
        "full_latency_ms": 100.0,
        "route_latency_ms": 80.0,
        "latency_reduction": 0.2,
        "label": label,
    }


def test_phase7_config_loads_conservative_policy() -> None:
    settings = load_phase7_settings("configs/phase7.yaml")
    assert settings.target_safe_precision == pytest.approx(0.80)
    assert settings.min_route_coverage == pytest.approx(0.05)
    assert settings.calibration_method == "sigmoid"


def test_router_input_excludes_all_phase6_outcomes() -> None:
    original = _record(1, safe=True)
    changed = copy.deepcopy(original)
    changed["label"] = "full_model"
    changed["regret"]["teacher_kl"] = 999.0
    changed["generation"]["route_text"] = "completely different"
    changed["route_latency_ms"] = 9999.0
    assert router_input_row(original) == router_input_row(changed)


def test_threshold_selection_abstains_when_precision_is_unavailable() -> None:
    labels = np.asarray([1, 0, 1, 0])
    probabilities = np.asarray([0.9, 0.8, 0.7, 0.1])
    threshold, metrics, found = choose_abstention_threshold(
        labels, probabilities, target_precision=0.9, min_coverage=0.25
    )
    assert found
    assert threshold == pytest.approx(0.9)
    assert metrics["safe_precision"] == pytest.approx(1.0)

    threshold, metrics, found = choose_abstention_threshold(
        np.asarray([1, 0]),
        np.asarray([0.6, 0.7]),
        target_precision=1.0,
        min_coverage=0.5,
    )
    assert not found
    assert threshold == pytest.approx(1.0)
    assert metrics["routed_examples"] == 0


def test_expected_calibration_error_is_zero_for_perfect_probabilities() -> None:
    labels = np.asarray([0, 0, 1, 1])
    probabilities = labels.astype(float)
    assert expected_calibration_error(labels, probabilities, bins=5) == pytest.approx(0)


def test_failed_router_artifact_always_falls_back() -> None:
    decision = route_prompt(
        {
            "manifest": {
                "enabled": False,
                "threshold": 0.5,
                "route_id": "route_target_21",
            },
            "estimator": _AlwaysSafeEstimator(),
        },
        "Name a color",
        prompt_token_count=3,
    )
    assert decision["safe_probability"] == pytest.approx(0.99)
    assert decision["decision"] == "full"
    assert decision["abstained"]
    assert not decision["router_enabled"]


def test_phase7_trains_serializes_and_routes_without_gpu(tmp_path) -> None:
    records = [_record(index, safe=index % 2 == 0) for index in range(80)]
    dataset = tmp_path / "router_dataset.jsonl"
    dataset.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "verdict": "proceed_to_phase7",
                "route_id": "route_target_21",
                "valid_examples": len(records),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "phase7"
    report = train_phase7_router(
        dataset_path=dataset,
        phase6_gate_path=gate,
        settings=Phase7RouterSettings(
            outer_cv_folds=3,
            calibration_cv_folds=2,
            target_safe_precision=0.70,
            min_route_coverage=0.10,
            min_gross_latency_reduction=0.01,
            max_expected_calibration_error=0.50,
            max_brier_score=0.50,
        ),
        output_dir=output,
    )
    assert report["verdict"] == "proceed_to_phase8"
    assert (output / "router.joblib").exists()
    assert (output / "manifest.json").exists()
    assert (output / "gate.md").exists()
    assert (output / "development_oof_predictions.jsonl").exists()

    artifact = load_router_artifact(output / "router.joblib")
    decision = route_prompt(
        artifact,
        "simple greeting hello friendly short answer",
        prompt_token_count=6,
    )
    assert decision["decision"] == "route_target_21"
    assert not decision["abstained"]
