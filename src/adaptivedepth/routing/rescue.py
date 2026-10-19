from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict

from adaptivedepth.config import Phase8RescueSettings
from adaptivedepth.routing.multiroute_risk_router import load_phase71_records
from adaptivedepth.routing.risk_router import _probability_metrics


SCHEMA_VERSION = 1
ARTIFACT_TYPE = 'phase8_rescue_predictor'
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    temporary.replace(path)


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _numeric(prompt: str, output: str, route_index: int, safe_probability: float) -> list[float]:
    prompt_tokens = _tokens(prompt)
    output_tokens = _tokens(output)
    prompt_set, output_set = set(prompt_tokens), set(output_tokens)
    overlap = len(prompt_set & output_set) / max(len(output_set), 1)
    unique = len(output_set) / max(len(output_tokens), 1)
    repeats = 1.0 - unique
    uncertainty = 1.0 - abs(2.0 * safe_probability - 1.0)
    return [
        len(prompt) / 1000.0,
        len(prompt_tokens) / 200.0,
        len(output) / 500.0,
        len(output_tokens) / 100.0,
        unique,
        repeats,
        overlap,
        float(output.rstrip().endswith(('.', '!', '?'))),
        route_index / 4.0,
        safe_probability,
        uncertainty,
    ]


def _matrix(rows: list[dict[str, Any]], settings: Phase8RescueSettings):
    texts = [f"{row['prompt']} [ROUTE_OUTPUT] {row['output']}" for row in rows]
    word = HashingVectorizer(
        n_features=settings.word_features, alternate_sign=False,
        ngram_range=(1, 2), norm='l2', lowercase=True,
    ).transform(texts)
    char = HashingVectorizer(
        analyzer='char_wb', n_features=settings.char_features,
        alternate_sign=False, ngram_range=(3, 5), norm='l2', lowercase=True,
    ).transform(texts)
    numeric = sparse.csr_matrix(np.asarray([
        _numeric(row['prompt'], row['output'], row['route_index'], row['safe_probability'])
        for row in rows
    ], dtype=np.float64))
    return sparse.hstack((word, char, numeric), format='csr')


def _estimator(settings: Phase8RescueSettings) -> LogisticRegression:
    return LogisticRegression(
        C=settings.logistic_c, class_weight='balanced', solver='liblinear',
        max_iter=2000, random_state=settings.seed,
    )


def _load_oof(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row['example_id']] = row
    return rows


def _choose_threshold(
    probabilities: np.ndarray,
    safe: np.ndarray,
    settings: Phase8RescueSettings,
) -> tuple[float, dict[str, Any]]:
    candidates = sorted(set([0.0, 1.0, *probabilities.tolist()]))
    best: tuple[tuple[float, ...], float, dict[str, Any]] | None = None
    for threshold in candidates:
        accepted = probabilities <= threshold
        rescued = ~accepted
        accepted_count = int(accepted.sum())
        unsafe = ~safe
        precision = float(safe[accepted].mean()) if accepted_count else 0.0
        coverage = accepted_count / len(safe)
        rescue_recall = float((rescued & unsafe).sum() / max(int(unsafe.sum()), 1))
        feasible = (
            precision >= settings.target_accept_precision
            and coverage >= settings.min_accept_coverage
            and rescue_recall >= settings.min_rescue_recall
        )
        metrics = {
            'candidate_examples': len(safe), 'accepted_examples': accepted_count,
            'rescued_examples': int(rescued.sum()), 'accept_precision': precision,
            'accept_coverage': coverage, 'rescue_recall': rescue_recall,
            'feasible': feasible,
        }
        score = (
            float(feasible), accepted_count, precision, rescue_recall
        )
        if best is None or score > best[0]:
            best = (score, float(threshold), metrics)
    assert best is not None
    return best[1], best[2]


def train_phase8_rescue(
    *,
    dataset_path: str | Path,
    phase72_report_path: str | Path,
    phase72_oof_path: str | Path,
    settings: Phase8RescueSettings,
    output_dir: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    dataset_path, phase72_report_path = Path(dataset_path), Path(phase72_report_path)
    phase72_oof_path, output_dir = Path(phase72_oof_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, route_order = load_phase71_records(dataset_path)
    router_report = json.loads(phase72_report_path.read_text(encoding='utf-8'))
    if not router_report.get('phase8_development_allowed'):
        raise ValueError('Phase 7.2 policy does not authorize Phase 8 development')
    router_oof = _load_oof(phase72_oof_path)
    if set(router_oof) != {record['example_id'] for record in records}:
        raise ValueError('Phase 7.2 OOF predictions do not match the dataset')

    rows: list[dict[str, Any]] = []
    for record in records:
        oof = router_oof[record['example_id']]
        for route_index, route_id in enumerate(route_order):
            route = record['routes'][route_id]
            rows.append({
                'example_id': record['example_id'], 'prompt': record['prompt'],
                'route_id': route_id, 'route_index': route_index,
                'output': route.get('generation', {}).get('text', ''),
                'safe_probability': float(oof['safe_probabilities'][route_id]),
                'safe': route['label'] == 'safe_route',
            })
    matrix = _matrix(rows, settings)
    labels = np.asarray([not row['safe'] for row in rows], dtype=np.int8)
    groups = np.asarray([row['example_id'] for row in rows])
    splitter = GroupKFold(n_splits=settings.group_cv_folds)
    oof_probabilities = cross_val_predict(
        _estimator(settings), matrix, labels, cv=splitter, groups=groups,
        method='predict_proba', n_jobs=1,
    )[:, 1]
    estimator = _estimator(settings).fit(matrix, labels)

    row_lookup = {(row['example_id'], row['route_id']): index for index, row in enumerate(rows)}
    candidate_indexes: list[int] = []
    for record in records:
        decision = router_oof[record['example_id']]['candidate_decision']
        if decision != 'full':
            candidate_indexes.append(row_lookup[(record['example_id'], decision)])
    candidate_probabilities = oof_probabilities[candidate_indexes]
    candidate_safe = np.asarray([rows[index]['safe'] for index in candidate_indexes], dtype=bool)
    threshold, policy = _choose_threshold(candidate_probabilities, candidate_safe, settings)
    checks = {
        'minimum_candidates': len(candidate_indexes) >= settings.min_candidate_examples,
        'accept_precision': policy['accept_precision'] >= settings.target_accept_precision,
        'accept_coverage': policy['accept_coverage'] >= settings.min_accept_coverage,
        'rescue_recall': policy['rescue_recall'] >= settings.min_rescue_recall,
    }
    ready = all(checks.values())
    verdict = 'ready_for_fresh_validation' if ready else 'phase8_implemented_needs_data'
    manifest = {
        'schema_version': SCHEMA_VERSION, 'artifact_type': ARTIFACT_TYPE,
        'created_at_utc': datetime.now(UTC).isoformat(), 'enabled': False,
        'validation_status': 'development_only', 'rescue_threshold': threshold,
        'route_order': route_order, 'settings': settings.model_dump(mode='json'),
        'dataset_sha256': _sha256(dataset_path),
        'phase72_report_sha256': _sha256(phase72_report_path),
        'phase72_oof_sha256': _sha256(phase72_oof_path),
        'config_sha256': _sha256(Path(config_path)) if config_path else None,
    }
    joblib.dump({'manifest': manifest, 'estimator': estimator}, output_dir / 'rescue.joblib')
    report = {
        'verdict': verdict,
        'reason': (
            'the rescue predictor meets its development policy gate'
            if ready else 'the rescue path is implemented, but its development gate needs more candidate data or revision'
        ),
        'training_route_examples': len(rows),
        'unique_prompts': len(records),
        'candidate_policy_examples': len(candidate_indexes),
        'rescue_probability_metrics': _probability_metrics(labels, oof_probabilities, settings.calibration_bins),
        'rescue_threshold': threshold, 'policy': policy, 'checks': checks,
        'limitations': [
            'development evaluation reuses Phase 6.1 prompts',
            'the model uses generated-text diagnostics rather than runtime logits',
            'restart latency must be measured by the Phase 9 runtime',
        ],
        'requires_fresh_validation': True,
    }
    _atomic_json(output_dir / 'manifest.json', manifest)
    _atomic_json(output_dir / 'report.json', report)
    (output_dir / 'gate.md').write_text(phase8_markdown(report), encoding='utf-8')
    with (output_dir / 'oof_candidate_predictions.jsonl').open('w', encoding='utf-8') as handle:
        for index, probability in zip(candidate_indexes, candidate_probabilities):
            row = rows[index]
            handle.write(json.dumps({
                'example_id': row['example_id'], 'route_id': row['route_id'],
                'rescue_probability': float(probability),
                'action': 'rescue_to_full' if probability > threshold else 'accept_route',
                'route_was_safe': bool(row['safe']),
            }, sort_keys=True) + '\n')
    with (output_dir / 'oof_route_predictions.jsonl').open('w', encoding='utf-8') as handle:
        for row, probability in zip(rows, oof_probabilities):
            handle.write(json.dumps({
                'example_id': row['example_id'],
                'route_id': row['route_id'],
                'router_safe_probability': row['safe_probability'],
                'rescue_probability': float(probability),
                'route_was_safe': bool(row['safe']),
            }, sort_keys=True) + '\n')
    return report


def phase8_markdown(report: dict[str, Any]) -> str:
    policy = report['policy']
    lines = [
        '# AdaptiveDepth Phase 8 rescue gate', '',
        f"**Verdict:** `{report['verdict']}`", '', report['reason'], '',
        '## Development evaluation', '',
        f"- Candidate examples: {report['candidate_policy_examples']}",
        f"- Accepted-route precision: {policy['accept_precision']:.1%}",
        f"- Accepted-route coverage: {policy['accept_coverage']:.1%}",
        f"- Unsafe-route rescue recall: {policy['rescue_recall']:.1%}", '',
        '## Checks', '',
    ]
    lines.extend(f"- {key.replace('_', ' ').title()}: {'pass' if value else 'fail'}" for key, value in report['checks'].items())
    lines += ['', 'The predictor remains disabled pending fresh validation and Phase 9 runtime measurement.', '']
    return '\n'.join(lines)


def load_phase8_rescue_artifact(path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    manifest = artifact.get('manifest', {})
    if manifest.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('unsupported Phase 8 rescue artifact schema')
    if manifest.get('artifact_type') != ARTIFACT_TYPE:
        raise ValueError('artifact is not a Phase 8 rescue predictor')
    if 'estimator' not in artifact:
        raise ValueError('rescue artifact has no estimator')
    return artifact


def predict_rescue(
    artifact: dict[str, Any],
    *,
    prompt: str,
    route_output: str,
    route_id: str,
    safe_probability: float,
    allow_development_candidate: bool = False,
) -> dict[str, Any]:
    manifest = artifact['manifest']
    route_order = list(manifest['route_order'])
    if route_id not in route_order:
        raise ValueError(f'unknown route: {route_id}')
    settings = Phase8RescueSettings.model_validate(manifest['settings'])
    row = {
        'prompt': prompt, 'output': route_output, 'route_id': route_id,
        'route_index': route_order.index(route_id),
        'safe_probability': float(safe_probability),
    }
    probability = float(artifact['estimator'].predict_proba(_matrix([row], settings))[0, 1])
    candidate = (
        'rescue_to_full'
        if probability > float(manifest['rescue_threshold'])
        else 'accept_route'
    )
    enabled = bool(manifest.get('enabled', False))
    action = candidate if enabled or allow_development_candidate else 'rescue_to_full'
    return {
        'rescue_probability': probability,
        'threshold': float(manifest['rescue_threshold']),
        'candidate_action': candidate,
        'action': action,
        'predictor_enabled': enabled,
        'development_override': allow_development_candidate and not enabled,
    }
