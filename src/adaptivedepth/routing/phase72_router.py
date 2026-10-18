from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from adaptivedepth.config import Phase72RouterSettings
from adaptivedepth.routing.features import prompt_features
from adaptivedepth.routing.multiroute_risk_router import (
    _policy_metrics,
    _probability_metrics,
    load_phase71_records,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = 'phase72_shared_router'
NUMERIC_FEATURES = (
    'character_count',
    'word_count',
    'prompt_token_count',
    'newline_count',
    'question_mark_count',
    'digit_fraction',
    'contains_code_fence',
    'contains_math_symbol',
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding='utf-8',
    )
    temporary.replace(path)


def _numeric_row(features: dict[str, Any]) -> list[float]:
    return [
        float(np.log1p(features['character_count']) / 8.0),
        float(np.log1p(features['word_count']) / 5.0),
        float(np.log1p(features['prompt_token_count']) / 6.0),
        float(np.log1p(features['newline_count']) / 3.0),
        float(np.log1p(features['question_mark_count']) / 3.0),
        float(features['digit_fraction']),
        float(features['contains_code_fence']),
        float(features['contains_math_symbol']),
    ]


def _vectorizers(settings: Phase72RouterSettings) -> tuple[Any, Any]:
    word = HashingVectorizer(
        n_features=settings.word_features,
        alternate_sign=False,
        ngram_range=(1, 2),
        norm='l2',
    )
    character = HashingVectorizer(
        n_features=settings.char_features,
        alternate_sign=False,
        analyzer='char_wb',
        ngram_range=(3, 5),
        norm='l2',
    )
    return word, character


def prompt_matrix(
    prompts: list[str],
    features: list[dict[str, Any]],
    settings: Phase72RouterSettings,
):
    word, character = _vectorizers(settings)
    numeric = csr_matrix(np.asarray([_numeric_row(item) for item in features]))
    return hstack(
        [word.transform(prompts), character.transform(prompts), numeric],
        format='csr',
    )


def _estimator(settings: Phase72RouterSettings) -> CalibratedClassifierCV:
    base = LogisticRegression(
        C=settings.logistic_c,
        class_weight='balanced',
        max_iter=2000,
        random_state=settings.seed,
        solver='liblinear',
    )
    calibration = StratifiedKFold(
        n_splits=settings.calibration_cv_folds,
        shuffle=True,
        random_state=settings.seed + 1,
    )
    return CalibratedClassifierCV(
        estimator=base,
        method=settings.calibration_method,
        cv=calibration,
    )


def _labels(records: list[dict[str, Any]], route_id: str) -> np.ndarray:
    return np.asarray(
        [record['routes'][route_id]['label'] == 'safe_route' for record in records],
        dtype=np.int64,
    )


def _decisions_from_thresholds(
    probabilities: dict[str, np.ndarray],
    route_order: list[str],
    thresholds: dict[str, float],
    threshold_found: dict[str, bool],
) -> list[str]:
    size = len(next(iter(probabilities.values())))
    decisions = np.full(size, 'full', dtype=object)
    undecided = np.ones(size, dtype=bool)
    for route_id in route_order:
        if not threshold_found[route_id]:
            continue
        selected = undecided & (probabilities[route_id] >= thresholds[route_id])
        decisions[selected] = route_id
        undecided[selected] = False
    return decisions.tolist()


def choose_joint_thresholds(
    records: list[dict[str, Any]],
    probabilities: dict[str, np.ndarray],
    route_order: list[str],
    settings: Phase72RouterSettings,
) -> tuple[dict[str, float], dict[str, bool], dict[str, Any]]:
    size = len(records)
    choices = list(range(settings.max_dense_route_candidates + 1))
    choices.extend(
        count
        for count in (80, 100, 150, 200, 300, size)
        if count <= size
    )
    choices = sorted(set(choices))
    sorted_probabilities = {
        route_id: np.sort(probabilities[route_id])[::-1]
        for route_id in route_order
    }
    labels = np.stack([_labels(records, route_id) for route_id in route_order])
    full_latency = np.asarray(
        [record['full']['profile']['total_ms'] for record in records], dtype=float
    )
    route_latency = np.stack(
        [
            np.asarray(
                [record['routes'][route_id]['profile']['total_ms'] for record in records],
                dtype=float,
            )
            for route_id in route_order
        ]
    )
    full_total = float(full_latency.sum())
    rng = np.random.default_rng(settings.seed)
    best_feasible: tuple[Any, ...] | None = None
    best_diagnostic: tuple[Any, ...] | None = None

    for _ in range(settings.threshold_search_trials):
        counts = [int(rng.choice(choices)) for _route_id in route_order]
        thresholds = {
            route_id: (
                float(sorted_probabilities[route_id][min(count, size) - 1])
                if count > 0
                else 1.0
            )
            for route_id, count in zip(route_order, counts)
        }
        found = {
            route_id: count > 0 for route_id, count in zip(route_order, counts)
        }
        decisions = np.full(size, -1, dtype=np.int16)
        for route_index, route_id in enumerate(route_order):
            if not found[route_id]:
                continue
            selected = (decisions < 0) & (
                probabilities[route_id] >= thresholds[route_id]
            )
            decisions[selected] = route_index
        routed_indexes = np.flatnonzero(decisions >= 0)
        routed = len(routed_indexes)
        if routed == 0:
            continue
        safe = int(
            sum(labels[decisions[index], index] for index in routed_indexes)
        )
        precision = safe / routed
        coverage = routed / size
        selected_routes = len(set(decisions[routed_indexes].tolist()))
        adaptive = full_latency.copy()
        adaptive[routed_indexes] = route_latency[
            decisions[routed_indexes], routed_indexes
        ]
        gross = 1.0 - float(adaptive.sum()) / max(full_total, 1e-9)
        check_count = sum(
            (
                precision >= settings.target_safe_precision,
                coverage >= settings.min_policy_coverage,
                selected_routes >= settings.min_selected_routes,
                gross >= settings.min_gross_latency_reduction,
            )
        )
        payload = (
            (routed, gross, precision),
            thresholds,
            found,
            decisions.copy(),
            precision,
            coverage,
            selected_routes,
            gross,
        )
        diagnostic_score = (check_count, precision, coverage, gross)
        if best_diagnostic is None or diagnostic_score > best_diagnostic[0]:
            best_diagnostic = (diagnostic_score, payload)
        if check_count == 4 and (
            best_feasible is None or payload[0] > best_feasible[0]
        ):
            best_feasible = payload

    selected = best_feasible or (best_diagnostic[1] if best_diagnostic else None)
    if selected is None:
        raise ValueError('joint threshold search produced no routed policy')
    _, thresholds, found, decisions, precision, coverage, selected_routes, gross = selected
    for route_index, route_id in enumerate(route_order):
        if not bool((decisions == route_index).any()):
            found[route_id] = False
            thresholds[route_id] = 1.0
    summary = {
        'feasible': best_feasible is not None,
        'trials': settings.threshold_search_trials,
        'safe_precision': precision,
        'coverage': coverage,
        'selected_routes': selected_routes,
        'gross_latency_reduction': gross,
    }
    return thresholds, found, summary


def train_phase72_router(
    *,
    dataset_path: str | Path,
    phase61_gate_path: str | Path,
    settings: Phase72RouterSettings,
    output_dir: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    phase61_gate_path = Path(phase61_gate_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, measured_order = load_phase71_records(dataset_path)
    gate = json.loads(phase61_gate_path.read_text(encoding='utf-8'))
    if gate.get('verdict') != 'proceed_to_phase7_multiroute':
        raise ValueError('Phase 6.1 gate does not authorize Phase 7.2 development')
    if gate.get('valid_examples') != len(records):
        raise ValueError('Phase 6.1 gate and router dataset counts differ')
    eligible = set(gate.get('eligible_routes', []))
    route_order = [route_id for route_id in measured_order if route_id in eligible]
    if len(route_order) < settings.min_selected_routes:
        raise ValueError('too few eligible routes for Phase 7.2')
    matrix = prompt_matrix(
        [record['prompt'] for record in records],
        [record['features'] for record in records],
        settings,
    )

    estimators: dict[str, Any] = {}
    probabilities: dict[str, np.ndarray] = {}
    route_reports: dict[str, dict[str, Any]] = {}
    for route_index, route_id in enumerate(route_order):
        labels = _labels(records, route_id)
        outer = StratifiedKFold(
            n_splits=settings.outer_cv_folds,
            shuffle=True,
            random_state=settings.seed + 2 + route_index,
        )
        oof = cross_val_predict(
            _estimator(settings),
            matrix,
            labels,
            cv=outer,
            method='predict_proba',
            n_jobs=1,
        )[:, 1]
        estimator = _estimator(settings)
        estimator.fit(matrix, labels)
        estimators[route_id] = estimator
        probabilities[route_id] = oof
        route_reports[route_id] = {
            'safe_examples': int(labels.sum()),
            'fallback_examples': int(len(labels) - labels.sum()),
            **_probability_metrics(labels, oof, settings.calibration_bins),
        }

    thresholds, threshold_found, search = choose_joint_thresholds(
        records,
        probabilities,
        route_order,
        settings,
    )
    decisions = _decisions_from_thresholds(
        probabilities,
        route_order,
        thresholds,
        threshold_found,
    )
    policy = _policy_metrics(records, decisions, route_order)
    selected_routes = [
        route_id
        for route_id in route_order
        if policy['decision_counts'][route_id] > 0
    ]
    checks = {
        'joint_threshold_policy': search['feasible'],
        'safe_precision': policy['safe_precision'] >= settings.target_safe_precision,
        'policy_coverage': policy['coverage'] >= settings.min_policy_coverage,
        'selected_routes': len(selected_routes) >= settings.min_selected_routes,
        'gross_latency_reduction': (
            policy['gross_latency_reduction']
            >= settings.min_gross_latency_reduction
        ),
        'calibration': all(
            route_reports[route_id]['expected_calibration_error']
            <= settings.max_expected_calibration_error
            and route_reports[route_id]['brier_score'] <= settings.max_brier_score
            for route_id in selected_routes
        ),
    }
    core_checks = [
        checks['joint_threshold_policy'],
        checks['safe_precision'],
        checks['policy_coverage'],
        checks['selected_routes'],
        checks['gross_latency_reduction'],
    ]
    phase8_development_allowed = all(core_checks)
    ready_for_fresh_validation = all(checks.values())
    verdict = (
        'ready_for_fresh_validation'
        if ready_for_fresh_validation
        else (
            'provisional_for_phase8'
            if phase8_development_allowed
            else 'revise_phase72'
        )
    )
    route_definitions = {
        route_id: {
            'active_layers': list(records[0]['routes'][route_id]['active_layers']),
            'median_latency_reduction': gate['route_assessments'][route_id][
                'median_latency_reduction'
            ],
        }
        for route_id in route_order
    }
    manifest = {
        'schema_version': SCHEMA_VERSION,
        'artifact_type': ARTIFACT_TYPE,
        'created_at_utc': datetime.now(UTC).isoformat(),
        'enabled': False,
        'validation_status': 'development_only',
        'requires_fresh_validation': True,
        'phase8_development_allowed': phase8_development_allowed,
        'route_order': route_order,
        'selected_routes': selected_routes,
        'thresholds': thresholds,
        'threshold_found': threshold_found,
        'route_definitions': route_definitions,
        'dataset_sha256': _sha256_file(dataset_path),
        'phase61_gate_sha256': _sha256_file(phase61_gate_path),
        'config_sha256': _sha256_file(config_path) if config_path else None,
        'settings': settings.model_dump(mode='json'),
        'training_examples': len(records),
    }
    artifact = {'manifest': manifest, 'estimators': estimators}
    joblib.dump(artifact, output_dir / 'router.joblib')
    report = {
        'verdict': verdict,
        'reason': (
            'the corrected router is locked and ready for a fresh holdout'
            if ready_for_fresh_validation
            else (
                'the policy gate passes, but calibration limits it to Phase 8 development'
                if phase8_development_allowed
                else 'the corrected development policy still misses operational constraints'
            )
        ),
        'dataset_examples': len(records),
        'route_order': route_order,
        'selected_routes': selected_routes,
        'routes': route_reports,
        'threshold_search': search,
        'thresholds': thresholds,
        'threshold_found': threshold_found,
        'development_policy': policy,
        'checks': checks,
        'phase8_development_allowed': phase8_development_allowed,
        'requires_fresh_validation': True,
        'settings': settings.model_dump(mode='json'),
    }
    _atomic_json_write(output_dir / 'manifest.json', manifest)
    _atomic_json_write(output_dir / 'report.json', report)
    (output_dir / 'gate.md').write_text(phase72_markdown(report), encoding='utf-8')
    with (output_dir / 'oof_predictions.jsonl').open('w', encoding='utf-8') as handle:
        for index, record in enumerate(records):
            decision = str(decisions[index])
            handle.write(json.dumps({
                'example_id': record['example_id'],
                'safe_probabilities': {
                    route_id: float(probabilities[route_id][index])
                    for route_id in route_order
                },
                'candidate_decision': decision,
                'candidate_safe': (
                    bool(record['routes'][decision]['label'] == 'safe_route')
                    if decision != 'full' else True
                ),
            }, sort_keys=True) + '\n')
    return report


def phase72_markdown(report: dict[str, Any]) -> str:
    policy = report['development_policy']
    lines = [
        '# AdaptiveDepth Phase 7.2 correction gate', '',
        f"**Verdict:** `{report['verdict']}`", '', report['reason'], '',
        '## Development policy', '',
        f"- Safe precision: {policy['safe_precision']:.1%}",
        f"- Coverage: {policy['coverage']:.1%}",
        f"- Gross latency reduction: {policy['gross_latency_reduction']:.1%}",
        f"- Selected routes: `{', '.join(report['selected_routes'])}`", '',
        '## Checks', '',
    ]
    lines.extend(
        f"- {name.replace('_', ' ').title()}: {'pass' if passed else 'fail'}"
        for name, passed in report['checks'].items()
    )
    lines += ['', 'The artifact is development-only and remains disabled until fresh validation.', '']
    return '\n'.join(lines)


def load_phase72_router_artifact(path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    manifest = artifact.get('manifest', {})
    if manifest.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('unsupported Phase 7.2 router artifact schema')
    if manifest.get('artifact_type') != ARTIFACT_TYPE:
        raise ValueError('artifact is not a Phase 7.2 shared router')
    if set(artifact.get('estimators', {})) != set(manifest.get('route_order', [])):
        raise ValueError('router estimators do not match the manifest routes')
    return artifact


def route_prompt_phase72(
    artifact: dict[str, Any],
    prompt: str,
    prompt_token_count: int,
    *,
    allow_development_candidate: bool = False,
) -> dict[str, Any]:
    manifest = artifact['manifest']
    settings = Phase72RouterSettings.model_validate(manifest['settings'])
    features = prompt_features(prompt, prompt_token_count)
    row = prompt_matrix([prompt], [features], settings)
    route_order = list(manifest['route_order'])
    probabilities = {
        route_id: float(artifact['estimators'][route_id].predict_proba(row)[0, 1])
        for route_id in route_order
    }
    qualifying = [
        route_id for route_id in route_order
        if manifest['threshold_found'][route_id]
        and probabilities[route_id] >= float(manifest['thresholds'][route_id])
    ]
    candidate = qualifying[0] if qualifying else 'full'
    enabled = bool(manifest.get('enabled', False))
    development = bool(
        allow_development_candidate and manifest.get('phase8_development_allowed', False)
    )
    decision = candidate if enabled or development else 'full'
    definition = manifest['route_definitions'].get(decision)
    return {
        'safe_probabilities': probabilities,
        'qualifying_routes': qualifying,
        'candidate_decision': candidate,
        'decision': decision,
        'active_layers': list(definition['active_layers']) if definition else None,
        'abstained': decision == 'full',
        'router_enabled': enabled,
        'development_override': development and not enabled,
    }
