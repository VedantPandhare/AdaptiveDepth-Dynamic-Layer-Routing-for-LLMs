from __future__ import annotations

import numpy as np
import pytest
import json
from pathlib import Path

from adaptivedepth.config import load_phase72_settings, load_phase8_settings
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
