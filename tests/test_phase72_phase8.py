from __future__ import annotations

import numpy as np
import pytest

from adaptivedepth.config import load_phase72_settings, load_phase8_settings
from adaptivedepth.routing.rescue import _choose_threshold, predict_rescue


class _Estimator:
    def predict_proba(self, rows):
        return np.asarray([[0.8, 0.2] for _ in range(rows.shape[0])])


def test_correction_and_rescue_configs_load() -> None:
    phase72 = load_phase72_settings('configs/phase72.yaml')
    phase8 = load_phase8_settings('configs/phase8.yaml')
    assert phase72.threshold_search_trials == 200000
    assert phase8.target_accept_precision == pytest.approx(0.90)


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
