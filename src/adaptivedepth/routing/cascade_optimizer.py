from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from adaptivedepth.config import Phase72ApplicationSettings
from adaptivedepth.routing.multiroute_risk_router import load_phase71_records


SCHEMA_VERSION = 1
ARTIFACT_TYPE = 'phase72_two_route_cascade_policy'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    temporary.replace(path)


def _load_by_example(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row['example_id']] = row
    return rows


def _load_rescue_scores(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[(row['example_id'], row['route_id'])] = row
    return rows


def _threshold_choices(values: np.ndarray, *, greater_is_better: bool) -> np.ndarray:
    unique = np.unique(values)
    if greater_is_better:
        sentinel = np.nextafter(unique[-1], np.inf)
    else:
        sentinel = np.nextafter(unique[0], -np.inf)
    return np.concatenate(([sentinel], unique))


def _policy_payload(
    index: int,
    thresholds: np.ndarray,
    candidate_20: np.ndarray,
    candidate_23: np.ndarray,
    accept_20: np.ndarray,
    accept_23: np.ndarray,
    safe_20: np.ndarray,
    safe_23: np.ndarray,
    full_latency: np.ndarray,
    route_latency: np.ndarray,
    probe_fraction: float,
) -> dict[str, Any]:
    c20, c23 = candidate_20[index], candidate_23[index]
    a20, a23 = accept_20[index], accept_23[index]
    candidate = c20 | c23
    accepted = a20 | a23
    safe_count = int((a20 & safe_20).sum() + (a23 & safe_23).sum())
    accepted_count = int(accepted.sum())
    rescued_20, rescued_23 = c20 & ~a20, c23 & ~a23
    savings = float(
        (full_latency[a20] - route_latency[0, a20]).sum()
        + (full_latency[a23] - route_latency[1, a23]).sum()
    )
    measured_overhead = float(
        route_latency[0, rescued_20].sum() + route_latency[1, rescued_23].sum()
    )
    projected_overhead = probe_fraction * measured_overhead
    full_total = float(full_latency.sum())
    return {
        'router_thresholds': {
            'route_fast_20': float(thresholds[index, 0]),
            'route_safe_23': float(thresholds[index, 1]),
        },
        'rescue_thresholds': {
            'route_fast_20': float(thresholds[index, 2]),
            'route_safe_23': float(thresholds[index, 3]),
        },
        'candidate_examples': int(candidate.sum()),
        'candidate_coverage': float(candidate.mean()),
        'accepted_examples': accepted_count,
        'final_coverage': accepted_count / len(accepted),
        'safe_accepted_examples': safe_count,
        'unsafe_accepted_examples': accepted_count - safe_count,
        'final_precision': safe_count / accepted_count if accepted_count else 0.0,
        'rescued_examples': int((candidate & ~accepted).sum()),
        'accepted_route_counts': {
            'route_fast_20': int(a20.sum()),
            'route_safe_23': int(a23.sum()),
        },
        'accepted_route_safe_counts': {
            'route_fast_20': int((a20 & safe_20).sum()),
            'route_safe_23': int((a23 & safe_23).sum()),
        },
        'accepted_route_savings_ms': savings,
        'full_probe_restart_overhead_ms': measured_overhead,
        'projected_probe_restart_overhead_ms': projected_overhead,
        'full_probe_latency_reduction': (savings - measured_overhead) / full_total,
        'projected_latency_reduction': (savings - projected_overhead) / full_total,
    }


def optimize_two_route_cascade(
    *,
    dataset_path: str | Path,
    router_oof_path: str | Path,
    rescue_oof_path: str | Path,
    settings: Phase72ApplicationSettings,
    output_dir: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    dataset_path, router_oof_path = Path(dataset_path), Path(router_oof_path)
    rescue_oof_path, output_dir = Path(rescue_oof_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, measured_routes = load_phase71_records(dataset_path)
    routes = list(settings.candidate_routes)
    if routes != ['route_fast_20', 'route_safe_23']:
        raise ValueError('the application policy is locked to Route 20 then Route 23')
    if not set(routes).issubset(measured_routes):
        raise ValueError('candidate routes are missing from the measured dataset')
    router = _load_by_example(router_oof_path)
    rescue = _load_rescue_scores(rescue_oof_path)
    example_ids = [record['example_id'] for record in records]
    if set(router) != set(example_ids):
        raise ValueError('router OOF predictions do not match the dataset')
    expected = {(example_id, route_id) for example_id in example_ids for route_id in routes}
    if not expected.issubset(rescue):
        raise ValueError('rescue OOF predictions are incomplete for Route 20/23')

    router_probability = np.asarray([
        [router[example_id]['safe_probabilities'][route_id] for example_id in example_ids]
        for route_id in routes
    ], dtype=float)
    rescue_probability = np.asarray([
        [rescue[(example_id, route_id)]['rescue_probability'] for example_id in example_ids]
        for route_id in routes
    ], dtype=float)
    safe = np.asarray([
        [record['routes'][route_id]['label'] == 'safe_route' for record in records]
        for route_id in routes
    ], dtype=bool)
    full_latency = np.asarray(
        [record['full']['profile']['total_ms'] for record in records], dtype=float
    )
    route_latency = np.asarray([
        [record['routes'][route_id]['profile']['total_ms'] for record in records]
        for route_id in routes
    ], dtype=float)
    choices = [
        _threshold_choices(router_probability[index], greater_is_better=True)
        for index in range(2)
    ] + [
        _threshold_choices(rescue_probability[index], greater_is_better=False)
        for index in range(2)
    ]
    rng = np.random.default_rng(settings.seed)
    best_feasible: tuple[tuple[float, ...], dict[str, Any]] | None = None
    best_safe_coverage: tuple[tuple[float, ...], dict[str, Any]] | None = None
    best_target_coverage: tuple[tuple[float, ...], dict[str, Any]] | None = None
    processed = 0
    while processed < settings.search_trials:
        size = min(settings.search_batch_size, settings.search_trials - processed)
        thresholds = np.column_stack([
            rng.choice(route_choices, size=size) for route_choices in choices
        ])
        c20 = router_probability[0][None, :] >= thresholds[:, 0, None]
        c23 = (~c20) & (
            router_probability[1][None, :] >= thresholds[:, 1, None]
        )
        a20 = c20 & (rescue_probability[0][None, :] <= thresholds[:, 2, None])
        a23 = c23 & (rescue_probability[1][None, :] <= thresholds[:, 3, None])
        candidate_count = (c20 | c23).sum(axis=1)
        accepted_count = (a20 | a23).sum(axis=1)
        safe_count = (a20 & safe[0]).sum(axis=1) + (a23 & safe[1]).sum(axis=1)
        precision = np.divide(
            safe_count, accepted_count,
            out=np.zeros(size, dtype=float), where=accepted_count > 0,
        )
        coverage = accepted_count / len(records)
        candidate_coverage = candidate_count / len(records)
        savings = (
            (a20 * (full_latency - route_latency[0])).sum(axis=1)
            + (a23 * (full_latency - route_latency[1])).sum(axis=1)
        )
        rescued_20, rescued_23 = c20 & ~a20, c23 & ~a23
        measured_overhead = (
            (rescued_20 * route_latency[0]).sum(axis=1)
            + (rescued_23 * route_latency[1]).sum(axis=1)
        )
        projected = (
            savings - settings.projected_probe_cost_fraction * measured_overhead
        ) / full_latency.sum()
        safe_mask = precision >= settings.target_final_precision
        target_mask = coverage >= settings.target_final_coverage
        feasible = (
            safe_mask & target_mask
            & (candidate_coverage >= settings.min_candidate_coverage)
            & (projected >= settings.minimum_projected_latency_reduction)
        )
        for mask, destination, score_columns in (
            (feasible, 'feasible', (projected, precision, coverage)),
            (safe_mask, 'safe', (coverage, projected, precision)),
            (target_mask, 'target', (precision, projected, coverage)),
        ):
            indexes = np.flatnonzero(mask)
            if not len(indexes):
                continue
            index = max(
                indexes,
                key=lambda item: tuple(float(column[item]) for column in score_columns),
            )
            payload = _policy_payload(
                index, thresholds, c20, c23, a20, a23, safe[0], safe[1],
                full_latency, route_latency, settings.projected_probe_cost_fraction,
            )
            score = tuple(float(column[index]) for column in score_columns)
            current = (
                best_feasible if destination == 'feasible'
                else best_safe_coverage if destination == 'safe'
                else best_target_coverage
            )
            if current is None or score > current[0]:
                if destination == 'feasible':
                    best_feasible = (score, payload)
                elif destination == 'safe':
                    best_safe_coverage = (score, payload)
                else:
                    best_target_coverage = (score, payload)
        processed += size

    selected = best_feasible[1] if best_feasible else best_safe_coverage[1]
    checks = {
        'final_coverage': selected['final_coverage'] >= settings.target_final_coverage,
        'final_precision': selected['final_precision'] >= settings.target_final_precision,
        'candidate_coverage': selected['candidate_coverage'] >= settings.min_candidate_coverage,
        'projected_latency_reduction': (
            selected['projected_latency_reduction']
            >= settings.minimum_projected_latency_reduction
        ),
    }
    ready = best_feasible is not None
    label_audit = {}
    for route_id in routes:
        failures = Counter(
            tuple(
                name
                for name, passed in record['routes'][route_id]['criteria'].items()
                if not passed
            )
            for record in records
            if record['routes'][route_id]['label'] != 'safe_route'
        )
        label_audit[route_id] = {
            'unsafe_examples': sum(failures.values()),
            'failure_patterns': [
                {'failed_criteria': list(pattern), 'examples': count}
                for pattern, count in failures.most_common()
            ],
        }
    report = {
        'verdict': 'ready_for_fresh_data' if ready else 'revise_combined_policy',
        'reason': (
            'a two-route cascade satisfies the locked development targets'
            if ready else 'no searched two-route cascade satisfies the locked coverage, safety, and latency targets'
        ),
        'examples': len(records),
        'oracle_safe_on_either_route': int((safe[0] | safe[1]).sum()),
        'oracle_safe_coverage': float((safe[0] | safe[1]).mean()),
        'selected_policy': selected,
        'best_policy_at_precision_target': best_safe_coverage[1],
        'best_policy_at_coverage_target': (
            best_target_coverage[1] if best_target_coverage else None
        ),
        'checks': checks,
        'label_audit': label_audit,
        'settings': settings.model_dump(mode='json'),
        'limitations': [
            'thresholds were searched on previously inspected development examples',
            'projected latency scales measured route cost by an assumed probe fraction',
            'fresh examples are required before enabling this policy',
        ],
    }
    manifest = {
        'schema_version': SCHEMA_VERSION,
        'artifact_type': ARTIFACT_TYPE,
        'created_at_utc': datetime.now(UTC).isoformat(),
        'enabled': False,
        'validation_status': 'development_only',
        'verdict': report['verdict'],
        'policy': selected,
        'dataset_sha256': _sha256(dataset_path),
        'router_oof_sha256': _sha256(router_oof_path),
        'rescue_oof_sha256': _sha256(rescue_oof_path),
        'config_sha256': _sha256(Path(config_path)) if config_path else None,
    }
    _write_json(output_dir / 'report.json', report)
    _write_json(output_dir / 'policy.json', manifest)
    (output_dir / 'gate.md').write_text(cascade_markdown(report), encoding='utf-8')
    return report


def cascade_markdown(report: dict[str, Any]) -> str:
    selected = report['selected_policy']
    lines = [
        '# AdaptiveDepth Phase 7.2 two-route application gate', '',
        f"**Verdict:** `{report['verdict']}`", '', report['reason'], '',
        '## Selected development policy', '',
        f"- Candidate coverage: {selected['candidate_coverage']:.1%}",
        f"- Final reduced-route coverage: {selected['final_coverage']:.1%}",
        f"- Final accepted precision: {selected['final_precision']:.1%}",
        f"- Projected latency reduction: {selected['projected_latency_reduction']:.1%}",
        f"- Full-probe latency reduction: {selected['full_probe_latency_reduction']:.1%}", '',
        '## Locked targets', '',
    ]
    lines.extend(
        f"- {name.replace('_', ' ').title()}: {'pass' if passed else 'fail'}"
        for name, passed in report['checks'].items()
    )
    lines += [
        '',
        f"Oracle safe coverage for Route 20 or Route 23: {report['oracle_safe_coverage']:.1%}.",
        '', 'The policy remains disabled and development-only.', '',
    ]
    return '\n'.join(lines)
