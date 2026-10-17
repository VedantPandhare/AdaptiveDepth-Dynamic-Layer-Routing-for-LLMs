from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from adaptivedepth.config import Phase7RouterSettings
from adaptivedepth.routing.features import prompt_features


SCHEMA_VERSION = 1
NUMERIC_FEATURES = (
    "character_count",
    "word_count",
    "prompt_token_count",
    "newline_count",
    "question_mark_count",
    "digit_fraction",
    "contains_code_fence",
    "contains_math_symbol",
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_router_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record.get("status") != "completed":
                    continue
                if record["label"] not in {"safe_route", "full_model"}:
                    raise ValueError("unknown route label")
                if not isinstance(record["prompt"], str) or not record["prompt"].strip():
                    raise ValueError("prompt is empty")
                missing = [name for name in NUMERIC_FEATURES if name not in record["features"]]
                if missing:
                    raise ValueError(f"missing features: {missing}")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid Phase 6 record at {path}:{line_number}") from error
            records.append(record)
    if not records:
        raise ValueError(f"router dataset is empty: {path}")
    if len({record["example_id"] for record in records}) != len(records):
        raise ValueError("router dataset contains duplicate example IDs")
    return records


def router_input_row(record: dict[str, Any]) -> list[Any]:
    features = record["features"]
    return [record["prompt"]] + [float(features[name]) for name in NUMERIC_FEATURES]


def _matrix(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([router_input_row(record) for record in records], dtype=object)


def _labels(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [1 if record["label"] == "safe_route" else 0 for record in records],
        dtype=np.int64,
    )


def _base_pipeline(settings: Phase7RouterSettings) -> Pipeline:
    features = ColumnTransformer(
        transformers=[
            (
                "text",
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
                "numeric",
                StandardScaler(),
                list(range(1, len(NUMERIC_FEATURES) + 1)),
            ),
        ]
    )
    classifier = LogisticRegression(
        C=settings.logistic_c,
        class_weight="balanced",
        max_iter=2000,
        random_state=settings.seed,
        solver="liblinear",
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def _calibrated_estimator(settings: Phase7RouterSettings) -> CalibratedClassifierCV:
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


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int
) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    if total == 0:
        return 0.0
    error = 0.0
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        if index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not bool(mask.any()):
            continue
        observed = float(labels[mask].mean())
        confidence = float(probabilities[mask].mean())
        error += float(mask.mean()) * abs(observed - confidence)
    return error


def threshold_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    predictions = probabilities >= threshold
    routed = int(predictions.sum())
    false_safe = int(((labels == 0) & predictions).sum())
    false_fallback = int(((labels == 1) & ~predictions).sum())
    return {
        "threshold": float(threshold),
        "routed_examples": routed,
        "coverage": float(predictions.mean()),
        "safe_precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "safe_recall": float(recall_score(labels, predictions, zero_division=0)),
        "safe_f1": float(f1_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "false_safe_examples": false_safe,
        "false_fallback_examples": false_fallback,
    }


def choose_abstention_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_precision: float,
    min_coverage: float,
) -> tuple[float, dict[str, Any], bool]:
    candidates = sorted({float(value) for value in probabilities} | {0.0, 1.0})
    viable: list[dict[str, Any]] = []
    for threshold in candidates:
        metrics = threshold_metrics(labels, probabilities, threshold)
        if (
            metrics["routed_examples"] > 0
            and metrics["safe_precision"] >= target_precision
            and metrics["coverage"] >= min_coverage
        ):
            viable.append(metrics)
    if not viable:
        threshold = 1.0
        return threshold, threshold_metrics(labels, probabilities, threshold), False
    selected = max(
        viable,
        key=lambda item: (
            item["coverage"],
            item["safe_precision"],
            item["threshold"],
        ),
    )
    return float(selected["threshold"]), selected, True


def _probability_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    calibration_bins: int,
) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "expected_calibration_error": expected_calibration_error(
            labels, probabilities, calibration_bins
        ),
    }


def _gross_latency_reduction(
    records: list[dict[str, Any]], predictions: np.ndarray
) -> float:
    full_total = sum(float(record["full_latency_ms"]) for record in records)
    adaptive_total = sum(
        float(record["route_latency_ms"] if use_route else record["full_latency_ms"])
        for record, use_route in zip(records, predictions)
    )
    return 1.0 - adaptive_total / max(full_total, 1e-9)


def _write_predictions(
    path: Path,
    records: list[dict[str, Any]],
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    route_id: str,
) -> None:
    predictions = probabilities >= threshold
    with path.open("w", encoding="utf-8") as handle:
        for record, expected, probability, predicted in zip(
            records, labels, probabilities, predictions
        ):
            payload = {
                "example_id": record["example_id"],
                "prompt": record["prompt"],
                "expected_label": (
                    "safe_route" if int(expected) == 1 else "full_model"
                ),
                "safe_probability": float(probability),
                "decision": route_id if predicted else "full",
                "correct": bool(predicted == bool(expected)),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def phase7_markdown(report: dict[str, Any]) -> str:
    test = report["holdout"]
    checks = report["checks"]
    lines = [
        "# AdaptiveDepth Phase 7 risk-router gate",
        "",
        f"**Verdict:** `{report['verdict']}`",
        "",
        report["reason"],
        "",
        "## Router policy",
        "",
        f"- Candidate route: `{report['route_id']}`",
        f"- Safe probability threshold: {report['threshold']:.4f}",
        f"- Threshold found from development OOF predictions: {report['threshold_found']}",
        f"- Default decision: `full_model`",
        "",
        "## Held-out evaluation",
        "",
        f"- Examples: {test['examples']}",
        f"- Safe examples: {test['safe_examples']}",
        f"- Routed examples: {test['routed_examples']}",
        f"- Route coverage: {test['coverage']:.1%}",
        f"- Safe-route precision: {test['safe_precision']:.1%}",
        f"- Safe-route recall: {test['safe_recall']:.1%}",
        f"- False-safe decisions: {test['false_safe_examples']}",
        f"- ROC AUC: {test['roc_auc']:.4f}",
        f"- Average precision: {test['average_precision']:.4f}",
        f"- Brier score: {test['brier_score']:.4f}",
        f"- Expected calibration error: {test['expected_calibration_error']:.4f}",
        f"- Estimated gross latency reduction: {test['gross_latency_reduction']:.1%}",
        "",
        "## Gate checks",
        "",
        f"- Development threshold available: {'pass' if checks['threshold_found'] else 'fail'}",
        f"- Held-out precision: {'pass' if checks['safe_precision'] else 'fail'}",
        f"- Held-out coverage: {'pass' if checks['route_coverage'] else 'fail'}",
        f"- Minimum routed test examples: {'pass' if checks['routed_examples'] else 'fail'}",
        f"- Probability calibration: {'pass' if checks['calibration'] else 'fail'}",
        f"- Gross latency benefit: {'pass' if checks['gross_latency_reduction'] else 'fail'}",
        "",
        "The router abstains to full depth whenever safe-route confidence is below the threshold.",
    ]
    return "\n".join(lines) + "\n"


def train_phase7_router(
    *,
    dataset_path: str | Path,
    phase6_gate_path: str | Path,
    settings: Phase7RouterSettings,
    output_dir: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    phase6_gate_path = Path(phase6_gate_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_router_records(dataset_path)
    phase6_gate = json.loads(phase6_gate_path.read_text(encoding="utf-8"))
    if phase6_gate.get("verdict") != "proceed_to_phase7":
        raise ValueError("Phase 6 gate does not authorize router training")
    if phase6_gate.get("valid_examples") != len(records):
        raise ValueError("Phase 6 gate and router dataset counts differ")

    labels = _labels(records)
    safe_count = int(labels.sum())
    unsafe_count = len(labels) - safe_count
    minimum_class = min(safe_count, unsafe_count)
    required = max(settings.outer_cv_folds, settings.calibration_cv_folds)
    if minimum_class < required:
        raise ValueError("router dataset has too few examples in one class")

    indexes = np.arange(len(records))
    development_indexes, test_indexes = train_test_split(
        indexes,
        test_size=settings.test_fraction,
        random_state=settings.seed,
        stratify=labels,
    )
    development = [records[int(index)] for index in development_indexes]
    holdout = [records[int(index)] for index in test_indexes]
    development_labels = labels[development_indexes]
    holdout_labels = labels[test_indexes]
    development_matrix = _matrix(development)
    holdout_matrix = _matrix(holdout)

    outer_cv = StratifiedKFold(
        n_splits=settings.outer_cv_folds,
        shuffle=True,
        random_state=settings.seed + 2,
    )
    development_probabilities = cross_val_predict(
        _calibrated_estimator(settings),
        development_matrix,
        development_labels,
        cv=outer_cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    threshold, development_threshold_metrics, threshold_found = (
        choose_abstention_threshold(
            development_labels,
            development_probabilities,
            settings.target_safe_precision,
            settings.min_route_coverage,
        )
    )

    estimator = _calibrated_estimator(settings)
    estimator.fit(development_matrix, development_labels)
    holdout_probabilities = estimator.predict_proba(holdout_matrix)[:, 1]
    holdout_predictions = holdout_probabilities >= threshold
    holdout_threshold_metrics = threshold_metrics(
        holdout_labels, holdout_probabilities, threshold
    )
    holdout_probability_metrics = _probability_metrics(
        holdout_labels,
        holdout_probabilities,
        settings.calibration_bins,
    )
    gross_latency_reduction = _gross_latency_reduction(
        holdout, holdout_predictions
    )
    holdout_report = {
        "examples": len(holdout),
        "safe_examples": int(holdout_labels.sum()),
        **holdout_threshold_metrics,
        **holdout_probability_metrics,
        "gross_latency_reduction": gross_latency_reduction,
    }
    checks = {
        "threshold_found": threshold_found,
        "safe_precision": (
            holdout_report["safe_precision"] >= settings.target_safe_precision
        ),
        "route_coverage": (
            holdout_report["coverage"] >= settings.min_route_coverage
        ),
        "routed_examples": (
            holdout_report["routed_examples"]
            >= settings.min_routed_test_examples
        ),
        "calibration": (
            holdout_report["expected_calibration_error"]
            <= settings.max_expected_calibration_error
            and holdout_report["brier_score"] <= settings.max_brier_score
        ),
        "gross_latency_reduction": (
            gross_latency_reduction >= settings.min_gross_latency_reduction
        ),
    }
    proceed = all(checks.values())
    route_id = str(phase6_gate["route_id"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "route_id": route_id,
        "fallback_route": "full",
        "enabled": proceed,
        "threshold": threshold,
        "numeric_features": list(NUMERIC_FEATURES),
        "dataset_sha256": _sha256_file(dataset_path),
        "phase6_gate_sha256": _sha256_file(phase6_gate_path),
        "config_sha256": _sha256_file(config_path) if config_path else None,
        "settings": settings.model_dump(mode="json"),
        "training_examples": len(development),
        "holdout_examples": len(holdout),
    }
    artifact = {"manifest": manifest, "estimator": estimator}
    joblib.dump(artifact, output_dir / "router.joblib")

    report = {
        "verdict": "proceed_to_phase8" if proceed else "revise_router",
        "reason": (
            "the calibrated router meets precision, coverage, and latency constraints"
            if proceed
            else (
                "development out-of-fold predictions could not identify a safe routing threshold"
                if not threshold_found
                else "the router does not yet satisfy the conservative application gate"
            )
        ),
        "route_id": route_id,
        "dataset_examples": len(records),
        "safe_examples": safe_count,
        "fallback_examples": unsafe_count,
        "development_examples": len(development),
        "threshold": threshold,
        "threshold_found": threshold_found,
        "development_oof": {
            **development_threshold_metrics,
            **_probability_metrics(
                development_labels,
                development_probabilities,
                settings.calibration_bins,
            ),
        },
        "holdout": holdout_report,
        "checks": checks,
        "settings": settings.model_dump(mode="json"),
    }
    _atomic_json_write(output_dir / "manifest.json", manifest)
    _atomic_json_write(output_dir / "report.json", report)
    (output_dir / "gate.md").write_text(
        phase7_markdown(report), encoding="utf-8"
    )
    _write_predictions(
        output_dir / "development_oof_predictions.jsonl",
        development,
        development_labels,
        development_probabilities,
        threshold,
        route_id,
    )
    _write_predictions(
        output_dir / "holdout_predictions.jsonl",
        holdout,
        holdout_labels,
        holdout_probabilities,
        threshold,
        route_id,
    )
    return report


def load_router_artifact(path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    if artifact.get("manifest", {}).get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported router artifact schema")
    if "estimator" not in artifact:
        raise ValueError("router artifact has no estimator")
    return artifact


def route_prompt(
    artifact: dict[str, Any], prompt: str, prompt_token_count: int
) -> dict[str, Any]:
    features = prompt_features(prompt, prompt_token_count)
    row = np.asarray(
        [[prompt] + [float(features[name]) for name in NUMERIC_FEATURES]],
        dtype=object,
    )
    probability = float(artifact["estimator"].predict_proba(row)[0, 1])
    manifest = artifact["manifest"]
    enabled = bool(manifest.get("enabled", False))
    use_route = enabled and probability >= float(manifest["threshold"])
    return {
        "safe_probability": probability,
        "threshold": float(manifest["threshold"]),
        "decision": manifest["route_id"] if use_route else "full",
        "abstained": not use_route,
        "router_enabled": enabled,
    }
