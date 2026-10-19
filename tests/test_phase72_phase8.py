from __future__ import annotations

import numpy as np
import pytest
import json
from pathlib import Path

from adaptivedepth.config import (
    load_phase72_settings,
    load_phase8_settings,
    load_phase81_probe_settings,
)
from adaptivedepth.config import load_phase72_application_settings
from adaptivedepth.routing.cascade_optimizer import _policy_payload
from adaptivedepth.routing.application_labels import (
    _parse_judge_response,
    task_family,
)
from adaptivedepth.routing.phase72_router import route_prompt_phase72
from adaptivedepth.routing.rescue import _choose_threshold, predict_rescue


class _Estimator:
    def predict_proba(self, rows):
        return np.asarray([[0.8, 0.2] for _ in range(rows.shape[0])])


def test_application_label_review_contract_and_notebook() -> None:
    assert task_family('What is an ATV?') == 'closed_qa'
    assert task_family('Question?\n\nContext:\nEvidence') == 'context_qa'
    assert task_family('Identify which item is red: apple, sky') == 'classification'
    assert task_family('Write a short paragraph') == 'instruction'
    judgment = _parse_judge_response(
        '```json\n{"verdict":"safe_route","confidence":0.9,'
        '"reason":"Equivalent answer"}\n```'
    )
    assert judgment['verdict'] == 'safe_route'
    notebook = json.loads(
        Path('notebooks/phase72_semantic_label_review_colab.ipynb').read_text(
            encoding='utf-8'
        )
    )
    source = ''.join(
        line for cell in notebook['cells'] for line in cell.get('source', [])
    )
    assert notebook['nbformat'] == 4
    assert 'judge-phase72-label-review' in source
    assert 'GITHUB_TOKEN' in source
    assert 'stderr=subprocess.STDOUT' in source
    assert 'subprocess.run(command,check=True)' not in source


def test_truncated_judge_response_is_rejected_for_conservative_retry() -> None:
    with pytest.raises(ValueError, match='did not return JSON'):
        _parse_judge_response(
            '{"verdict":"full_model","confidence":1,"reason":"truncated'
        )


def test_phase81_runtime_signal_contract_and_notebook() -> None:
    settings = load_phase81_probe_settings("configs/phase81_probe_t4.yaml")
    assert settings.probe_tokens == 4
    assert settings.go_no_go_coverage == pytest.approx(0.30)

    notebook = json.loads(
        Path("notebooks/phase81_runtime_signal_pilot_colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "".join(
        line for cell in notebook["cells"] for line in cell.get("source", [])
    )
    assert notebook["nbformat"] == 4
    assert "collect-phase81-probes" in source
    assert "MAX_EXAMPLES = 400" in source
    assert "phase81_probe_signals.jsonl" in source


def test_phase81_cpu_evaluator_writes_a_bounded_verdict(tmp_path: Path) -> None:
    from adaptivedepth.routing.probe_signals import evaluate_phase81_probe_signals

    dataset = tmp_path / "dataset.jsonl"
    probes = tmp_path / "probes.jsonl"
    router = tmp_path / "router.jsonl"
    dataset_rows = []
    probe_rows = []
    router_rows = []
    for index in range(20):
        example_id = f"example_{index:03d}"
        safe_20 = index % 5 in {0, 1}
        safe_23 = index % 3 == 0
        routes = {}
        probabilities = {}
        for route_id, safe, latency in (
            ("route_fast_20", safe_20, 80.0),
            ("route_safe_23", safe_23, 94.0),
        ):
            routes[route_id] = {
                "label": "safe_route" if safe else "full_model",
                "criteria": {
                    "teacher_kl": safe,
                    "reference_loss_regret": safe,
                    "token_agreement": safe,
                    "generation_token_jaccard": safe,
                },
                "profile": {"total_ms": latency},
            }
            probabilities[route_id] = 0.9 if safe else 0.1
            confidence = 0.05 if safe else 0.95
            probe_rows.append(
                {
                    "example_id": example_id,
                    "route_id": route_id,
                    "probe_tokens": 4,
                    "reached_eos": False,
                    "probe_ms": 20.0,
                    "signals": {
                        "mean_normalized_entropy": confidence,
                        "mean_top1_margin": 1.0 - confidence,
                    },
                    "shared_hidden": [confidence, 1.0 - confidence],
                }
            )
        route_order = ["route_fast_20", "route_safe_23"]
        safe_route_ids = [
            route_id
            for route_id in route_order
            if routes[route_id]["label"] == "safe_route"
        ]
        dataset_rows.append(
            {
                "status": "completed",
                "example_id": example_id,
                "prompt": f"Synthetic prompt {index}",
                "features": {
                    "character_count": 20,
                    "word_count": 3,
                    "prompt_token_count": 4,
                    "newline_count": 0,
                    "question_mark_count": 0,
                    "digit_fraction": 0.05,
                    "contains_code_fence": False,
                    "contains_math_symbol": False,
                },
                "routes": routes,
                "route_cost_order": route_order,
                "safe_route_ids": safe_route_ids,
                "target_route": safe_route_ids[0] if safe_route_ids else "full",
                "full": {"profile": {"total_ms": 100.0}},
            }
        )
        router_rows.append(
            {
                "example_id": example_id,
                "safe_probabilities": probabilities,
            }
        )

    for path, rows in (
        (dataset, dataset_rows),
        (probes, probe_rows),
        (router, router_rows),
    ):
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    probe_settings = load_phase81_probe_settings(
        "configs/phase81_probe_t4.yaml"
    ).model_copy(update={"group_cv_folds": 2})
    application_settings = load_phase72_application_settings(
        "configs/phase72_application.yaml"
    ).model_copy(update={"search_trials": 100, "search_batch_size": 100})
    report = evaluate_phase81_probe_signals(
        dataset_path=dataset,
        probes_path=probes,
        router_oof_path=router,
        settings=probe_settings,
        application_settings=application_settings,
        output_dir=tmp_path / "output",
    )
    assert report["verdict"] in {"continue_probe_router", "stop_probe_router"}
    assert set(report["variants"]) == {
        "probe_logits",
        "early_hidden",
        "combined",
    }
    assert (tmp_path / "output" / "gate.md").is_file()


def test_disabled_phase72_router_fails_closed() -> None:
    settings = load_phase72_settings('configs/phase72.yaml')
    artifact = {
        'manifest': {
            'enabled': False, 'phase8_development_allowed': True,
            'route_order': ['fast'], 'thresholds': {'fast': 0.1},
            'threshold_found': {'fast': True},
            'route_definitions': {'fast': {'active_layers': [0, 1]}},
            'settings': settings.model_dump(mode='json'),
        },
        'estimators': {'fast': _Estimator()},
    }
    result = route_prompt_phase72(artifact, 'simple prompt', 2)
    assert result['candidate_decision'] == 'fast'
    assert result['decision'] == 'full'

    development = route_prompt_phase72(
        artifact, 'simple prompt', 2, allow_development_candidate=True,
    )
    assert development['decision'] == 'fast'
    assert development['active_layers'] == [0, 1]


def test_correction_and_rescue_configs_load() -> None:
    phase72 = load_phase72_settings('configs/phase72.yaml')
    phase8 = load_phase8_settings('configs/phase8.yaml')
    assert phase72.threshold_search_trials == 200000
    assert phase8.target_accept_precision == pytest.approx(0.90)
    application = load_phase72_application_settings(
        'configs/phase72_application.yaml'
    )
    assert application.candidate_routes == [
        'route_fast_20', 'route_safe_23'
    ]
    assert application.target_final_coverage == pytest.approx(0.34)


def test_cascade_policy_accounts_for_rescue_cost() -> None:
    payload = _policy_payload(
        0,
        np.asarray([[0.5, 0.5, 0.5, 0.5]]),
        np.asarray([[True, False]]),
        np.asarray([[False, True]]),
        np.asarray([[True, False]]),
        np.asarray([[False, False]]),
        np.asarray([True, False]),
        np.asarray([False, True]),
        np.asarray([100.0, 100.0]),
        np.asarray([[80.0, 80.0], [90.0, 90.0]]),
        0.25,
    )
    assert payload['candidate_coverage'] == pytest.approx(1.0)
    assert payload['final_coverage'] == pytest.approx(0.5)
    assert payload['final_precision'] == pytest.approx(1.0)
    assert payload['projected_latency_reduction'] == pytest.approx(-0.0125)
    assert payload['full_probe_latency_reduction'] == pytest.approx(-0.35)


def test_rescue_threshold_enforces_precision_and_recall() -> None:
    settings = load_phase8_settings('configs/phase8.yaml')
    probabilities = np.asarray([0.1, 0.2, 0.8, 0.9])
    safe = np.asarray([True, True, False, False])
    threshold, policy = _choose_threshold(probabilities, safe, settings)
    assert threshold == pytest.approx(0.2)
    assert policy['accept_precision'] == pytest.approx(1.0)
    assert policy['rescue_recall'] == pytest.approx(1.0)


def test_disabled_rescue_predictor_fails_closed() -> None:
    settings = load_phase8_settings('configs/phase8.yaml')
    artifact = {
        'manifest': {
            'enabled': False, 'route_order': ['fast'],
            'rescue_threshold': 0.3, 'settings': settings.model_dump(mode='json'),
        },
        'estimator': _Estimator(),
    }
    result = predict_rescue(
        artifact, prompt='simple prompt', route_output='clear answer.',
        route_id='fast', safe_probability=0.9,
    )
    assert result['candidate_action'] == 'accept_route'
    assert result['action'] == 'rescue_to_full'
    assert not result['predictor_enabled']

    development = predict_rescue(
        artifact, prompt='simple prompt', route_output='clear answer.',
        route_id='fast', safe_probability=0.9,
        allow_development_candidate=True,
    )
    assert development['action'] == 'accept_route'
