from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from adaptivedepth.routing.multiroute_risk_router import load_phase71_records


SCHEMA_VERSION = 1
APPLICATION_ROUTES = ('route_fast_20', 'route_safe_23')
CORE_CRITERIA = ('teacher_kl', 'reference_loss_regret', 'token_agreement')


def task_family(prompt: str) -> str:
    lowered = prompt.lower()
    if '\n\ncontext:' in lowered:
        return 'context_qa'
    if lowered.startswith('identify which') or 'classify ' in lowered:
        return 'classification'
    if '?' in prompt or re.match(
        r'^(what|who|where|when|why|how|which|is|are|do|does|did|can|could|would|should)\b',
        lowered,
    ):
        return 'closed_qa'
    return 'instruction'


def _review_key(example_id: str, route_id: str) -> str:
    return f'{example_id}::{route_id}'


def _needs_review(route: dict[str, Any]) -> bool:
    criteria = route['criteria']
    return (
        all(bool(criteria[name]) for name in CORE_CRITERIA)
        and not bool(criteria['generation_token_jaccard'])
    )


def prepare_application_label_review(
    *, dataset_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    records, measured_routes = load_phase71_records(dataset_path)
    if not set(APPLICATION_ROUTES).issubset(measured_routes):
        raise ValueError('Route 20 and Route 23 are required for application review')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    family_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    count = 0
    with output_path.open('w', encoding='utf-8') as handle:
        for record in records:
            family = task_family(record['prompt'])
            for route_id in APPLICATION_ROUTES:
                route = record['routes'][route_id]
                if not _needs_review(route):
                    continue
                row = {
                    'schema_version': SCHEMA_VERSION,
                    'review_key': _review_key(record['example_id'], route_id),
                    'example_id': record['example_id'],
                    'route_id': route_id,
                    'task_family': family,
                    'prompt': record['prompt'],
                    'reference': record['reference'],
                    'full_response': record['full']['generation']['text'],
                    'route_response': route['generation']['text'],
                    'teacher_kl': route['regret']['teacher_kl'],
                    'reference_loss_regret': route['regret']['reference_loss_regret'],
                    'token_agreement': route['regret']['token_agreement'],
                    'generation_token_jaccard': route['generation']['token_jaccard'],
                    'review_instruction': (
                        'Mark safe only when the route response is not materially worse '
                        'than the full response for the prompt/reference and introduces '
                        'no new critical factual error.'
                    ),
                }
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
                count += 1
                family_counts[family] += 1
                route_counts[route_id] += 1
    return {
        'review_examples': count,
        'route_counts': dict(route_counts),
        'task_family_counts': dict(family_counts),
        'output': str(output_path),
    }


def load_review_judgments(path: str | Path) -> dict[str, dict[str, Any]]:
    judgments: dict[str, dict[str, Any]] = {}
    with Path(path).open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get('review_key') or _review_key(
                row.get('example_id', ''), row.get('route_id', '')
            )
            if key in judgments:
                raise ValueError(f'duplicate judgment for {key}')
            if row.get('verdict') not in {'safe_route', 'full_model'}:
                raise ValueError(
                    f'line {line_number}: verdict must be safe_route or full_model'
                )
            confidence = float(row.get('confidence', 0.0))
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f'line {line_number}: confidence must be in [0, 1]')
            if not str(row.get('reason', '')).strip():
                raise ValueError(f'line {line_number}: reason is required')
            judgments[key] = {
                'verdict': row['verdict'],
                'confidence': confidence,
                'reason': str(row['reason']).strip(),
                'reviewer': str(row.get('reviewer', 'unspecified')),
            }
    return judgments


def apply_application_label_review(
    *,
    dataset_path: str | Path,
    judgments_path: str | Path,
    output_path: str | Path,
    minimum_safe_confidence: float = 0.80,
) -> dict[str, Any]:
    if not 0.0 <= minimum_safe_confidence <= 1.0:
        raise ValueError('minimum_safe_confidence must be in [0, 1]')
    records, _ = load_phase71_records(dataset_path)
    judgments = load_review_judgments(judgments_path)
    expected = {
        _review_key(record['example_id'], route_id)
        for record in records
        for route_id in APPLICATION_ROUTES
        if _needs_review(record['routes'][route_id])
    }
    missing = expected - set(judgments)
    unexpected = set(judgments) - expected
    if missing or unexpected:
        raise ValueError(
            f'judgment keys do not match review set: missing={len(missing)}, '
            f'unexpected={len(unexpected)}'
        )

    route_safe_counts: Counter[str] = Counter()
    accepted_reviews = 0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        for record in records:
            for route_id in APPLICATION_ROUTES:
                route = record['routes'][route_id]
                route['original_label'] = route['label']
                if _needs_review(route):
                    judgment = judgments[_review_key(record['example_id'], route_id)]
                    accepted = (
                        judgment['verdict'] == 'safe_route'
                        and judgment['confidence'] >= minimum_safe_confidence
                    )
                    route['label'] = 'safe_route' if accepted else 'full_model'
                    route['application_review'] = {
                        **judgment,
                        'accepted_as_safe': accepted,
                        'minimum_safe_confidence': minimum_safe_confidence,
                    }
                    accepted_reviews += int(accepted)
                if route['label'] == 'safe_route':
                    route_safe_counts[route_id] += 1
            safe_route_ids = [
                route_id for route_id in record['route_cost_order']
                if record['routes'][route_id]['label'] == 'safe_route'
            ]
            record['safe_route_ids'] = safe_route_ids
            record['safe_route_ids_unordered'] = list(safe_route_ids)
            record['target_route'] = safe_route_ids[0] if safe_route_ids else 'full'
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
    return {
        'examples': len(records),
        'reviewed_pairs': len(expected),
        'accepted_review_pairs': accepted_reviews,
        'minimum_safe_confidence': minimum_safe_confidence,
        'route_safe_counts': dict(route_safe_counts),
        'output': str(output_path),
    }
