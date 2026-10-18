from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from adaptivedepth.config import Phase71RouterSettings, load_phase71_settings
from adaptivedepth.routing.features import prompt_features
from adaptivedepth.routing.multiroute_risk_router import (
    load_multiroute_router_artifact,
    load_phase71_records,
    route_prompt_multiroute,
    train_phase71_router,
)


ROUTES = ['route20', 'route21', 'route22', 'route23']


def _record(index: int) -> dict:
    target_index = index % 5
    safe_routes = ROUTES[target_index:] if target_index < 4 else []
    target_words = ['swift', 'target', 'balanced', 'cautious', 'complex']
    prompt = f'{target_words[target_index]} category prompt example {index}'
    latencies = dict(zip(ROUTES, [80.0, 85.0, 92.0, 96.0]))
    layers = dict(zip(ROUTES, [20, 21, 22, 23]))
    route_payload = {}
    for route_id in ROUTES:
        safe = route_id in safe_routes
        route_payload[route_id] = {
            'active_layers': list(range(layers[route_id])),
            'label': 'safe_route' if safe else 'full_model',
            'profile': {'total_ms': latencies[route_id]},
        }
    return {
        'status': 'completed',
        'example_id': f'example-{index:03d}',
        'prompt': prompt,
        'features': prompt_features(prompt, prompt_tokens=len(prompt.split())),
        'full': {'profile': {'total_ms': 100.0}},
        'routes': route_payload,
        'safe_route_ids': safe_routes,
        'target_route': safe_routes[0] if safe_routes else 'full',
        'route_cost_order': ROUTES,
    }


class _ProbabilityEstimator:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, rows):
        return np.asarray(
            [[1.0 - self.probability, self.probability] for _ in rows]
        )


def test_phase71_config_loads_multiroute_policy() -> None:
    settings = load_phase71_settings('configs/phase71.yaml')
    assert settings.target_safe_precision == pytest.approx(0.80)
    assert settings.min_enabled_routes == 2
    assert settings.min_policy_coverage == pytest.approx(0.05)


def test_phase71_loader_rejects_inconsistent_safe_routes(tmp_path) -> None:
    record = _record(0)
    record['safe_route_ids'] = []
    dataset = tmp_path / 'router_dataset.jsonl'
    dataset.write_text(json.dumps(record) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='safe route IDs'):
        load_phase71_records(dataset)


def test_disabled_multiroute_artifact_forces_full_fallback() -> None:
    artifact = {
        'manifest': {
            'enabled': False,
            'route_order': ROUTES,
            'thresholds': dict.fromkeys(ROUTES, 0.8),
            'threshold_found': dict.fromkeys(ROUTES, True),
            'route_definitions': {
                route_id: {'active_layers': list(range(20 + index))}
                for index, route_id in enumerate(ROUTES)
            },
        },
        'estimators': {
            route_id: _ProbabilityEstimator(0.99 if index == 0 else 0.1)
            for index, route_id in enumerate(ROUTES)
        },
    }
    decision = route_prompt_multiroute(artifact, 'swift prompt', 2)
    assert decision['candidate_decision'] == 'route20'
    assert decision['decision'] == 'full'
    assert decision['abstained']
    assert not decision['router_enabled']

    enabled = copy.deepcopy(artifact)
    enabled['manifest']['enabled'] = True
    decision = route_prompt_multiroute(enabled, 'swift prompt', 2)
    assert decision['decision'] == 'route20'
    assert decision['active_layers'] == list(range(20))


def test_phase71_trains_serializes_and_routes_without_gpu(tmp_path) -> None:
    records = [_record(index) for index in range(150)]
    dataset = tmp_path / 'router_dataset.jsonl'
    dataset.write_text(
        ''.join(json.dumps(record) + '\n' for record in records),
        encoding='utf-8',
    )
    gate = tmp_path / 'gate.json'
    gate.write_text(
        json.dumps(
            {
                'verdict': 'proceed_to_phase7_multiroute',
                'valid_examples': len(records),
                'route_order_by_measured_latency': ROUTES,
                'eligible_routes': ROUTES,
                'route_assessments': {
                    route_id: {'median_latency_reduction': 0.20 - 0.04 * index}
                    for index, route_id in enumerate(ROUTES)
                },
            }
        ),
        encoding='utf-8',
    )
    output = tmp_path / 'phase71'
    report = train_phase71_router(
        dataset_path=dataset,
        phase61_gate_path=gate,
        settings=Phase71RouterSettings(
            outer_cv_folds=3,
            calibration_cv_folds=2,
            target_safe_precision=0.70,
            min_route_threshold_coverage=0.02,
            min_policy_coverage=0.05,
            min_gross_latency_reduction=0.001,
            max_expected_calibration_error=0.50,
            max_brier_score=0.50,
        ),
        output_dir=output,
    )
    assert report['verdict'] == 'proceed_to_phase8'
    assert len(report['threshold_enabled_routes']) >= 2
    assert report['holdout']['safe_precision'] >= 0.70
    assert (output / 'router.joblib').exists()
    assert (output / 'manifest.json').exists()
    assert (output / 'gate.md').exists()
    assert (output / 'development_oof_predictions.jsonl').exists()
    assert (output / 'holdout_predictions.jsonl').exists()

    artifact = load_multiroute_router_artifact(output / 'router.joblib')
    decision = route_prompt_multiroute(
        artifact,
        'swift category prompt example',
        prompt_token_count=4,
    )
    assert decision['router_enabled']
    assert decision['decision'] in ROUTES
