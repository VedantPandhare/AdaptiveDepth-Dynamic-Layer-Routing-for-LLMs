from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from adaptivedepth.config import Phase71RouterSettings
from adaptivedepth.routing.features import prompt_features
from adaptivedepth.routing.risk_router import (
    NUMERIC_FEATURES,
    choose_abstention_threshold,
    expected_calibration_error,
    router_input_row,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = 'phase71_multiroute_router'


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding='utf-8',
    )
    temporary.replace(path)


def _matrix(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([router_input_row(record) for record in records], dtype=object)


def _labels(records: list[dict[str, Any]], route_id: str) -> np.ndarray:
    return np.asarray(
        [record['routes'][route_id]['label'] == 'safe_route' for record in records],
        dtype=np.int64,
    )


def _base_pipeline(settings: Phase71RouterSettings) -> Pipeline:
    features = ColumnTransformer(
        transformers=[
            (
                'text',
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(settings.min_ngram, settings.max_ngram),
                    max_features=settings.max_text_features,
                    min_df=2,
                    sublinear_tf=True,
                ),
                0,
            ),
            (
                'numeric',
                StandardScaler(),
                list(range(1, len(NUMERIC_FEATURES) + 1)),
            ),
        ]
    )
    classifier = LogisticRegression(
        C=settings.logistic_c,
        class_weight='balanced',
        max_iter=2000,
        random_state=settings.seed,
        solver='liblinear',
    )
    return Pipeline([('features', features), ('classifier', classifier)])


def _calibrated_estimator(
    settings: Phase71RouterSettings,
) -> CalibratedClassifierCV:
    calibration_cv = StratifiedKFold(
        n_splits=settings.calibration_cv_folds,
        shuffle=True,
        random_state=settings.seed + 1,
    )
    return CalibratedClassifierCV(
        estimator=_base_pipeline(settings),
        method=settings.calibration_method,
        cv=calibration_cv,
    )


def _probability_metrics(
    labels: np.ndarray, probabilities: np.ndarray, bins: int
) -> dict[str, float | None]:
    both_classes = len(np.unique(labels)) == 2
    return {
        'roc_auc': float(roc_auc_score(labels, probabilities)) if both_classes else None,
        'average_precision': (
            float(average_precision_score(labels, probabilities))
            if bool(labels.any())
            else None
        ),
        'brier_score': float(brier_score_loss(labels, probabilities)),
        'expected_calibration_error': expected_calibration_error(
            labels, probabilities, bins
        ),
    }


def load_phase71_records(
    path: str | Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    route_order: list[str] | None = None
    with Path(path).open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record.get('status') != 'completed':
                    continue
                if not isinstance(record['prompt'], str) or not record['prompt'].strip():
                    raise ValueError('prompt is empty')
                missing = [
                    name for name in NUMERIC_FEATURES if name not in record['features']
                ]
                if missing:
                    raise ValueError(f'missing features: {missing}')
                current_order = list(record['route_cost_order'])
                if not current_order or len(current_order) != len(set(current_order)):
                    raise ValueError('invalid route cost order')
                if route_order is None:
                    route_order = current_order
                elif current_order != route_order:
                    raise ValueError('inconsistent route cost order')
                for route_id in current_order:
                    if record['routes'][route_id]['label'] not in {
                        'safe_route',
                        'full_model',
                    }:
                        raise ValueError('unknown route label')
                    if float(record['routes'][route_id]['profile']['total_ms']) <= 0:
                        raise ValueError('route latency must be positive')
                if float(record['full']['profile']['total_ms']) <= 0:
                    raise ValueError('full latency must be positive')
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f'invalid Phase 6.1 record at {path}:{line_number}'
                ) from error
            records.append(record)
    if not records or route_order is None:
        raise ValueError(f'Phase 6.1 router dataset is empty: {path}')
    if len({record['example_id'] for record in records}) != len(records):
        raise ValueError('router dataset contains duplicate example IDs')
    for record in records:
        expected_safe = [
            route_id
            for route_id in route_order
            if record['routes'][route_id]['label'] == 'safe_route'
        ]
        if record.get('safe_route_ids') != expected_safe:
            raise ValueError('safe route IDs do not match route labels')
        expected_target = expected_safe[0] if expected_safe else 'full'
        if record.get('target_route') != expected_target:
            raise ValueError('target route is not the cheapest safe route')
    return records, route_order


def _policy_decisions(
    route_order: list[str],
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
    threshold_found: dict[str, bool],
) -> list[str]:
    examples = len(next(iter(probabilities.values())))
    decisions: list[str] = []
    for index in range(examples):
        decision = 'full'
        for route_id in route_order:
            if (
                threshold_found[route_id]
                and float(probabilities[route_id][index]) >= thresholds[route_id]
            ):
                decision = route_id
                break
        decisions.append(decision)
    return decisions


def _policy_metrics(
    records: list[dict[str, Any]],
    decisions: list[str],
    route_order: list[str],
) -> dict[str, Any]:
    routed = sum(decision != 'full' for decision in decisions)
    safe_decisions = sum(
        decision != 'full'
        and record['routes'][decision]['label'] == 'safe_route'
        for record, decision in zip(records, decisions)
    )
    false_safe = routed - safe_decisions
    routable = sum(
        any(record['routes'][route_id]['label'] == 'safe_route' for route_id in route_order)
        for record in records
    )
    route_counts = Counter(decisions)
    per_route: dict[str, dict[str, Any]] = {}
    for route_id in route_order:
        selected = [
            index for index, decision in enumerate(decisions) if decision == route_id
        ]
        safe = sum(
            records[index]['routes'][route_id]['label'] == 'safe_route'
            for index in selected
        )
        per_route[route_id] = {
            'selected_examples': len(selected),
            'safe_examples': safe,
            'false_safe_examples': len(selected) - safe,
            'safe_precision': safe / len(selected) if selected else None,
        }

    full_total = sum(float(record['full']['profile']['total_ms']) for record in records)
    adaptive_total = sum(
        float(record['full']['profile']['total_ms'])
        if decision == 'full'
        else float(record['routes'][decision]['profile']['total_ms'])
        for record, decision in zip(records, decisions)
    )
    target_matches = sum(
        decision == next(
            (
                route_id
                for route_id in route_order
                if record['routes'][route_id]['label'] == 'safe_route'
            ),
            'full',
        )
        for record, decision in zip(records, decisions)
    )
    return {
        'examples': len(records),
        'routable_examples': routable,
        'routed_examples': routed,
        'fallback_examples': len(records) - routed,
        'coverage': routed / len(records),
        'safe_precision': safe_decisions / routed if routed else 0.0,
        'safe_route_recall': safe_decisions / routable if routable else 0.0,
        'false_safe_examples': false_safe,
        'unsafe_decision_rate': false_safe / len(records),
        'target_match_rate': target_matches / len(records),
        'gross_latency_reduction': 1.0 - adaptive_total / max(full_total, 1e-9),
        'decision_counts': {
            route_id: route_counts.get(route_id, 0) for route_id in route_order
        }
        | {'full': route_counts.get('full', 0)},
        'per_route': per_route,
    }


def _write_predictions(
    path: Path,
    records: list[dict[str, Any]],
    probabilities: dict[str, np.ndarray],
    decisions: list[str],
    route_order: list[str],
) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for index, (record, decision) in enumerate(zip(records, decisions)):
            selected_safe = (
                decision == 'full'
                or record['routes'][decision]['label'] == 'safe_route'
            )
            payload = {
                'example_id': record['example_id'],
                'prompt': record['prompt'],
                'expected_safe_routes': [
                    route_id
                    for route_id in route_order
                    if record['routes'][route_id]['label'] == 'safe_route'
                ],
                'expected_target': next(
                    (
                        route_id
                        for route_id in route_order
                        if record['routes'][route_id]['label'] == 'safe_route'
                    ),
                    'full',
                ),
                'safe_probabilities': {
                    route_id: float(probabilities[route_id][index])
                    for route_id in route_order
                },
                'decision': decision,
                'decision_safe': bool(selected_safe),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + '\n')


def _route_probability_report(
    records: list[dict[str, Any]],
    probabilities: dict[str, np.ndarray],
    route_order: list[str],
    settings: Phase71RouterSettings,
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for route_id in route_order:
        labels = _labels(records, route_id)
        report[route_id] = {
            'safe_examples': int(labels.sum()),
            'fallback_examples': int(len(labels) - labels.sum()),
            **_probability_metrics(
                labels,
                probabilities[route_id],
                settings.calibration_bins,
            ),
        }
    return report


def phase71_markdown(report: dict[str, Any]) -> str:
    holdout = report['holdout']
    checks = report['checks']
    verdict = report['verdict']
    lines = [
        '# AdaptiveDepth Phase 7.1 multi-route risk-router gate',
        '',
        f'**Verdict:** `{verdict}`',
        '',
        report['reason'],
        '',
        '## Policy',
        '',
        '- Route order: `' + ' -> '.join(report['route_order'] + ['full']) + '`',
        '- Threshold-enabled routes: `'
        + '`, `'.join(report['threshold_enabled_routes'])
        + '`',
        '- Default decision: `full`',
        '',
        '## Development OOF thresholds',
        '',
        '| Route | Threshold | Found | Precision | Coverage | ROC AUC |',
        '|---|---:|---|---:|---:|---:|',
    ]
    for route_id in report['route_order']:
        item = report['routes'][route_id]
        threshold = item['threshold']
        found = 'yes' if item['threshold_found'] else 'no'
        precision = item['development_threshold']['safe_precision']
        coverage = item['development_threshold']['coverage']
        auc = item['development_oof']['roc_auc']
        auc_text = f'{auc:.4f}' if auc is not None else 'n/a'
        lines.append(
            f'| {route_id} | {threshold:.4f} | {found} | '
            f'{precision:.1%} | {coverage:.1%} | {auc_text} |'
        )
    decision_text = '`, `'.join(
        f'{name}={count}' for name, count in holdout['decision_counts'].items()
    )
    heldout_examples = holdout['examples']
    routed_examples = holdout['routed_examples']
    heldout_coverage = holdout['coverage']
    heldout_precision = holdout['safe_precision']
    heldout_recall = holdout['safe_route_recall']
    false_safe = holdout['false_safe_examples']
    latency = holdout['gross_latency_reduction']
    lines.extend(
        [
            '',
            '## Combined held-out policy',
            '',
            f'- Examples: {heldout_examples}',
            f'- Routed examples: {routed_examples}',
            f'- Coverage: {heldout_coverage:.1%}',
            f'- Safe-route precision: {heldout_precision:.1%}',
            f'- Safe-route recall: {heldout_recall:.1%}',
            f'- False-safe decisions: {false_safe}',
            f'- Gross latency reduction: {latency:.1%}',
            f'- Decisions: `{decision_text}`',
            '',
            '## Gate checks',
            '',
            '- Minimum enabled routes: '
            + ('pass' if checks['enabled_routes'] else 'fail'),
            '- Development policy: '
            + ('pass' if checks['development_policy'] else 'fail'),
            '- Held-out precision: '
            + ('pass' if checks['safe_precision'] else 'fail'),
            '- Held-out coverage: '
            + ('pass' if checks['policy_coverage'] else 'fail'),
            '- Routed holdout examples: '
            + ('pass' if checks['routed_examples'] else 'fail'),
            '- Probability calibration: '
            + ('pass' if checks['calibration'] else 'fail'),
            '- Gross latency benefit: '
            + ('pass' if checks['gross_latency_reduction'] else 'fail'),
            '',
            'A disabled artifact always routes to full depth. A passing artifact still falls back whenever no reduced route clears its threshold.',
        ]
    )
    return '\n'.join(lines) + '\n'


def _validate_development_classes(
    records: list[dict[str, Any]],
    route_order: list[str],
    settings: Phase71RouterSettings,
) -> None:
    for route_id in route_order:
        labels = _labels(records, route_id)
        counts = [int(labels.sum()), int(len(labels) - labels.sum())]
        smallest_outer_training = min(counts) - int(
            np.ceil(min(counts) / settings.outer_cv_folds)
        )
        if min(counts) < settings.outer_cv_folds:
            raise ValueError(f'{route_id} has too few development labels for outer CV')
        if smallest_outer_training < settings.calibration_cv_folds:
            raise ValueError(
                f'{route_id} has too few labels for nested probability calibration'
            )


def train_phase71_router(
    *,
    dataset_path: str | Path,
    phase61_gate_path: str | Path,
    settings: Phase71RouterSettings,
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
        raise ValueError('Phase 6.1 gate does not authorize multi-route training')
    if gate.get('valid_examples') != len(records):
        raise ValueError('Phase 6.1 gate and router dataset counts differ')
    if gate.get('route_order_by_measured_latency') != measured_order:
        raise ValueError('Phase 6.1 gate and dataset route order differ')
    eligible = set(gate.get('eligible_routes', []))
    route_order = [route_id for route_id in measured_order if route_id in eligible]
    if len(route_order) < settings.min_enabled_routes:
        raise ValueError('Phase 6.1 gate exposes too few eligible routes')

    indexes = np.arange(len(records))
    effective_targets = np.asarray(
        [
            next(
                (
                    route_id
                    for route_id in route_order
                    if record['routes'][route_id]['label'] == 'safe_route'
                ),
                'full',
            )
            for record in records
        ]
    )
    target_counts = Counter(effective_targets.tolist())
    stratify = effective_targets if min(target_counts.values()) >= 2 else None
    development_indexes, holdout_indexes = train_test_split(
        indexes,
        test_size=settings.test_fraction,
        random_state=settings.seed,
        stratify=stratify,
    )
    development = [records[int(index)] for index in development_indexes]
    holdout = [records[int(index)] for index in holdout_indexes]
    _validate_development_classes(development, route_order, settings)
    development_matrix = _matrix(development)
    holdout_matrix = _matrix(holdout)

    estimators: dict[str, Any] = {}
    development_probabilities: dict[str, np.ndarray] = {}
    holdout_probabilities: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    threshold_found: dict[str, bool] = {}
    route_reports: dict[str, dict[str, Any]] = {}
    for route_index, route_id in enumerate(route_order):
        development_labels = _labels(development, route_id)
        holdout_labels = _labels(holdout, route_id)
        outer_cv = StratifiedKFold(
            n_splits=settings.outer_cv_folds,
            shuffle=True,
            random_state=settings.seed + 2 + route_index,
        )
        oof = cross_val_predict(
            _calibrated_estimator(settings),
            development_matrix,
            development_labels,
            cv=outer_cv,
            method='predict_proba',
            n_jobs=1,
        )[:, 1]
        threshold, threshold_report, found = choose_abstention_threshold(
            development_labels,
            oof,
            settings.target_safe_precision,
            settings.min_route_threshold_coverage,
        )
        estimator = _calibrated_estimator(settings)
        estimator.fit(development_matrix, development_labels)
        holdout_probability = estimator.predict_proba(holdout_matrix)[:, 1]
        estimators[route_id] = estimator
        development_probabilities[route_id] = oof
        holdout_probabilities[route_id] = holdout_probability
        thresholds[route_id] = threshold
        threshold_found[route_id] = found
        route_reports[route_id] = {
            'threshold': threshold,
            'threshold_found': found,
            'development_threshold': threshold_report,
            'development_oof': _probability_metrics(
                development_labels,
                oof,
                settings.calibration_bins,
            ),
            'holdout': {
                'safe_examples': int(holdout_labels.sum()),
                'fallback_examples': int(len(holdout_labels) - holdout_labels.sum()),
                **_probability_metrics(
                    holdout_labels,
                    holdout_probability,
                    settings.calibration_bins,
                ),
            },
        }

    threshold_enabled_routes = [
        route_id for route_id in route_order if threshold_found[route_id]
    ]
    development_decisions = _policy_decisions(
        route_order,
        development_probabilities,
        thresholds,
        threshold_found,
    )
    holdout_decisions = _policy_decisions(
        route_order,
        holdout_probabilities,
        thresholds,
        threshold_found,
    )
    development_policy = _policy_metrics(
        development, development_decisions, route_order
    )
    holdout_policy = _policy_metrics(holdout, holdout_decisions, route_order)
    oracle_decisions = [
        next(
            (
                route_id
                for route_id in route_order
                if record['routes'][route_id]['label'] == 'safe_route'
            ),
            'full',
        )
        for record in holdout
    ]
    oracle_holdout = _policy_metrics(holdout, oracle_decisions, route_order)
    calibration_ok = all(
        route_reports[route_id]['holdout']['expected_calibration_error']
        <= settings.max_expected_calibration_error
        and route_reports[route_id]['holdout']['brier_score']
        <= settings.max_brier_score
        for route_id in route_order
    )
    development_ok = (
        development_policy['routed_examples'] > 0
        and development_policy['safe_precision'] >= settings.target_safe_precision
        and development_policy['coverage'] >= settings.min_policy_coverage
    )
    checks = {
        'enabled_routes': (
            len(threshold_enabled_routes) >= settings.min_enabled_routes
        ),
        'development_policy': development_ok,
        'safe_precision': (
            holdout_policy['safe_precision'] >= settings.target_safe_precision
        ),
        'policy_coverage': (
            holdout_policy['coverage'] >= settings.min_policy_coverage
        ),
        'routed_examples': (
            holdout_policy['routed_examples']
            >= settings.min_routed_holdout_examples
        ),
        'calibration': calibration_ok,
        'gross_latency_reduction': (
            holdout_policy['gross_latency_reduction']
            >= settings.min_gross_latency_reduction
        ),
    }
    proceed = all(checks.values())

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
        'enabled': proceed,
        'fallback_route': 'full',
        'route_order': route_order,
        'threshold_enabled_routes': threshold_enabled_routes,
        'thresholds': thresholds,
        'threshold_found': threshold_found,
        'route_definitions': route_definitions,
        'numeric_features': list(NUMERIC_FEATURES),
        'dataset_sha256': _sha256_file(dataset_path),
        'phase61_gate_sha256': _sha256_file(phase61_gate_path),
        'config_sha256': _sha256_file(config_path) if config_path else None,
        'settings': settings.model_dump(mode='json'),
        'training_examples': len(development),
        'holdout_examples': len(holdout),
    }
    artifact = {
        'manifest': manifest,
        'estimators': estimators,
    }
    joblib.dump(artifact, output_dir / 'router.joblib')

    if proceed:
        reason = (
            'the calibrated multi-route policy meets safety, coverage, '
            'calibration, and latency constraints'
        )
    elif len(threshold_enabled_routes) < settings.min_enabled_routes:
        reason = 'development OOF predictions support too few safe route thresholds'
    else:
        reason = 'the multi-route policy does not satisfy the conservative holdout gate'
    report = {
        'verdict': 'proceed_to_phase8' if proceed else 'revise_router',
        'reason': reason,
        'dataset_examples': len(records),
        'development_examples': len(development),
        'holdout_examples': len(holdout),
        'route_order': route_order,
        'threshold_enabled_routes': threshold_enabled_routes,
        'effective_target_counts': dict(target_counts),
        'routes': route_reports,
        'development_policy': development_policy,
        'holdout': holdout_policy,
        'oracle_holdout': oracle_holdout,
        'checks': checks,
        'settings': settings.model_dump(mode='json'),
    }
    _atomic_json_write(output_dir / 'manifest.json', manifest)
    _atomic_json_write(output_dir / 'report.json', report)
    (output_dir / 'gate.md').write_text(
        phase71_markdown(report), encoding='utf-8'
    )
    _write_predictions(
        output_dir / 'development_oof_predictions.jsonl',
        development,
        development_probabilities,
        development_decisions,
        route_order,
    )
    _write_predictions(
        output_dir / 'holdout_predictions.jsonl',
        holdout,
        holdout_probabilities,
        holdout_decisions,
        route_order,
    )
    return report


def load_multiroute_router_artifact(path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    manifest = artifact.get('manifest', {})
    if manifest.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('unsupported multi-route router artifact schema')
    if manifest.get('artifact_type') != ARTIFACT_TYPE:
        raise ValueError('artifact is not a Phase 7.1 multi-route router')
    estimators = artifact.get('estimators')
    if not isinstance(estimators, dict):
        raise ValueError('multi-route router artifact has no estimators')
    if set(estimators) != set(manifest.get('route_order', [])):
        raise ValueError('router estimators do not match the manifest routes')
    return artifact


def route_prompt_multiroute(
    artifact: dict[str, Any], prompt: str, prompt_token_count: int
) -> dict[str, Any]:
    manifest = artifact['manifest']
    features = prompt_features(prompt, prompt_token_count)
    row = np.asarray(
        [[prompt] + [float(features[name]) for name in NUMERIC_FEATURES]],
        dtype=object,
    )
    route_order = list(manifest['route_order'])
    probabilities = {
        route_id: float(artifact['estimators'][route_id].predict_proba(row)[0, 1])
        for route_id in route_order
    }
    qualifying = [
        route_id
        for route_id in route_order
        if manifest['threshold_found'][route_id]
        and probabilities[route_id] >= float(manifest['thresholds'][route_id])
    ]
    candidate = qualifying[0] if qualifying else 'full'
    enabled = bool(manifest.get('enabled', False))
    decision = candidate if enabled else 'full'
    route_definition = manifest['route_definitions'].get(decision)
    return {
        'safe_probabilities': probabilities,
        'thresholds': {
            route_id: float(manifest['thresholds'][route_id])
            for route_id in route_order
        },
        'qualifying_routes': qualifying,
        'candidate_decision': candidate,
        'decision': decision,
        'active_layers': (
            list(route_definition['active_layers']) if route_definition else None
        ),
        'abstained': decision == 'full',
        'router_enabled': enabled,
    }
