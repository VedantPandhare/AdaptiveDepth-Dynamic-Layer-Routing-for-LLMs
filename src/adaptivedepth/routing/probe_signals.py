from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from adaptivedepth.config import (
    Phase72ApplicationSettings,
    Phase81ProbeSettings,
)
from adaptivedepth.routes import RouteDefinition
from adaptivedepth.routing.cascade_optimizer import optimize_two_route_cascade


SCHEMA_VERSION = 1
PILOT_ROUTES = ("route_fast_20", "route_safe_23")


def _sync(device: Any) -> None:
    if device.type == "cuda":
        import torch

        torch.cuda.synchronize(device)


def _eos_ids(tokenizer: Any) -> set[int]:
    value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        return set()
    return {value} if isinstance(value, int) else set(value)


def logit_signal(logits: Any) -> dict[str, float]:
    """Return bounded confidence statistics for one next-token distribution."""
    import torch.nn.functional as functional

    values = logits.float()
    log_probabilities = functional.log_softmax(values, dim=-1)
    probabilities = log_probabilities.exp()
    top_probabilities, _ = probabilities.topk(k=2, dim=-1)
    entropy = -(probabilities * log_probabilities).sum(dim=-1)
    normalized_entropy = entropy / float(np.log(values.shape[-1]))
    return {
        "normalized_entropy": float(normalized_entropy.item()),
        "top1_probability": float(top_probabilities[0, 0].item()),
        "top2_probability": float(top_probabilities[0, 1].item()),
        "top1_margin": float(
            (top_probabilities[0, 0] - top_probabilities[0, 1]).item()
        ),
        "top1_log_probability": float(log_probabilities.max(dim=-1).values.item()),
    }


def _aggregate_token_signals(signals: list[dict[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in signals[0]:
        values = np.asarray([signal[name] for signal in signals], dtype=np.float64)
        result[f"mean_{name}"] = float(values.mean())
        result[f"min_{name}"] = float(values.min())
        result[f"max_{name}"] = float(values.max())
        result[f"std_{name}"] = float(values.std())
    return result


def _collect_route_probe(
    *,
    routed_model: Any,
    tokenizer: Any,
    device: Any,
    prompt: str,
    route: RouteDefinition,
    probe_tokens: int,
    shared_hidden_index: int,
) -> dict[str, Any]:
    import torch

    from adaptivedepth.runtime.generation import encode_prompt

    encoded = encode_prompt(tokenizer, prompt, device, use_chat_template=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    stop_ids = _eos_ids(tokenizer)
    token_signals: list[dict[str, float]] = []
    generated: list[torch.Tensor] = []
    reached_eos = False

    _sync(device)
    started = perf_counter()
    with routed_model.use_route(route):
        outputs = routed_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None or shared_hidden_index >= len(hidden_states):
            raise ValueError(
                f"shared_hidden_index={shared_hidden_index} is unavailable for "
                f"{route.route_id}; hidden state count={len(hidden_states or ())}"
            )
        shared_hidden = (
            hidden_states[shared_hidden_index][:, -1, :]
            .float()
            .squeeze(0)
            .cpu()
            .tolist()
        )
        for _ in range(probe_tokens):
            next_logits = outputs.logits[:, -1, :]
            token_signals.append(logit_signal(next_logits))
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            reached_eos = reached_eos or int(next_token.item()) in stop_ids
            if reached_eos or len(generated) >= probe_tokens:
                break
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token, device=device)], dim=-1
            )
            outputs = routed_model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=outputs.past_key_values,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
    _sync(device)
    elapsed_ms = (perf_counter() - started) * 1000.0
    generated_ids = torch.cat(generated, dim=-1)
    return {
        "probe_text": tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        ).strip(),
        "probe_tokens": len(generated),
        "reached_eos": reached_eos,
        "probe_ms": elapsed_ms,
        "input_tokens": int(input_ids.shape[-1]),
        "signals": _aggregate_token_signals(token_signals),
        "shared_hidden_index": shared_hidden_index,
        "shared_hidden": shared_hidden,
    }


def collect_route_probe(**kwargs: Any) -> dict[str, Any]:
    import torch

    with torch.inference_mode():
        return _collect_route_probe(**kwargs)


def collect_phase81_probe_signals(
    *,
    routed_model: Any,
    tokenizer: Any,
    device: Any,
    dataset_path: str | Path,
    routes: list[RouteDefinition],
    output_path: str | Path,
    settings: Phase81ProbeSettings,
    max_examples: int | None = None,
    resume: bool = True,
    progress: Any | None = None,
) -> dict[str, Any]:
    route_lookup = {route.route_id: route for route in routes}
    if not set(PILOT_ROUTES).issubset(route_lookup):
        raise ValueError("Phase 8.1 requires Route 20 and Route 23 definitions")
    records = [
        json.loads(line)
        for line in Path(dataset_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max_examples is not None:
        records = records[:max_examples]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed: set[tuple[str, str]] = set()
    if output_path.exists() and resume:
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed.add((row["example_id"], row["route_id"]))
    elif output_path.exists():
        output_path.unlink()

    written = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for position, record in enumerate(records, start=1):
            for route_id in PILOT_ROUTES:
                key = (record["example_id"], route_id)
                if key in completed:
                    continue
                probe = collect_route_probe(
                    routed_model=routed_model,
                    tokenizer=tokenizer,
                    device=device,
                    prompt=record["prompt"],
                    route=route_lookup[route_id],
                    probe_tokens=settings.probe_tokens,
                    shared_hidden_index=settings.shared_hidden_index,
                )
                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "example_id": record["example_id"],
                    "route_id": route_id,
                    **probe,
                }
                handle.write(
                    json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
                )
                handle.flush()
                completed.add(key)
                written += 1
            if progress:
                progress(f"Probed {position}/{len(records)} prompts")
    return {
        "examples": len(records),
        "expected_route_rows": len(records) * len(PILOT_ROUTES),
        "completed_route_rows": len(completed),
        "new_route_rows": written,
        "probe_tokens": settings.probe_tokens,
        "output": str(output_path),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _probe_numeric(row: dict[str, Any], router_probability: float) -> list[float]:
    signals = row["signals"]
    names = sorted(signals)
    return [
        *(float(signals[name]) for name in names),
        float(row["probe_tokens"]) / 4.0,
        float(row["reached_eos"]),
        float(router_probability),
        float(row["route_id"] == "route_safe_23"),
    ]


def evaluate_phase81_probe_signals(
    *,
    dataset_path: str | Path,
    probes_path: str | Path,
    router_oof_path: str | Path,
    settings: Phase81ProbeSettings,
    application_settings: Phase72ApplicationSettings,
    output_dir: str | Path,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    probes_path = Path(probes_path)
    router_oof_path = Path(router_oof_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _read_jsonl(dataset_path)
    probes = {
        (row["example_id"], row["route_id"]): row
        for row in _read_jsonl(probes_path)
    }
    router = {
        row["example_id"]: row for row in _read_jsonl(router_oof_path)
    }
    expected = {
        (record["example_id"], route_id)
        for record in records
        for route_id in PILOT_ROUTES
    }
    if set(probes) != expected:
        raise ValueError(
            f"probe rows do not match dataset: missing={len(expected - set(probes))}, "
            f"unexpected={len(set(probes) - expected)}"
        )
    if set(router) != {record["example_id"] for record in records}:
        raise ValueError("router OOF predictions do not match the probe dataset")

    rows: list[dict[str, Any]] = []
    probe_ratios: list[float] = []
    for record in records:
        example_id = record["example_id"]
        for route_id in PILOT_ROUTES:
            probe = probes[(example_id, route_id)]
            route = record["routes"][route_id]
            probability = float(router[example_id]["safe_probabilities"][route_id])
            rows.append(
                {
                    "example_id": example_id,
                    "route_id": route_id,
                    "unsafe": route["label"] != "safe_route",
                    "probe": _probe_numeric(probe, probability),
                    "hidden": [float(value) for value in probe["shared_hidden"]],
                }
            )
            route_ms = float(route["profile"]["total_ms"])
            probe_ratios.append(float(probe["probe_ms"]) / max(route_ms, 1e-9))

    labels = np.asarray([row["unsafe"] for row in rows], dtype=np.int8)
    groups = np.asarray([row["example_id"] for row in rows])
    probe_matrix = np.asarray([row["probe"] for row in rows], dtype=np.float64)
    hidden_matrix = np.asarray([row["hidden"] for row in rows], dtype=np.float64)
    route_and_router = probe_matrix[:, -2:]
    matrices = {
        "probe_logits": probe_matrix,
        "early_hidden": np.column_stack((route_and_router, hidden_matrix)),
        "combined": np.column_stack((probe_matrix, hidden_matrix)),
    }
    splitter = GroupKFold(n_splits=settings.group_cv_folds)
    variant_reports: dict[str, Any] = {}
    for name, matrix in matrices.items():
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=settings.logistic_c,
                class_weight="balanced",
                solver="liblinear",
                max_iter=3000,
                random_state=settings.seed,
            ),
        )
        probabilities = cross_val_predict(
            estimator,
            matrix,
            labels,
            cv=splitter,
            groups=groups,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        variant_dir = output_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        oof_path = variant_dir / "oof_route_predictions.jsonl"
        with oof_path.open("w", encoding="utf-8") as handle:
            for row, probability in zip(rows, probabilities):
                handle.write(
                    json.dumps(
                        {
                            "example_id": row["example_id"],
                            "route_id": row["route_id"],
                            "rescue_probability": float(probability),
                            "route_was_safe": not bool(row["unsafe"]),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        median_probe_fraction = float(np.median(probe_ratios))
        measured_application = application_settings.model_copy(
            update={
                "projected_probe_cost_fraction": min(
                    max(median_probe_fraction, 1e-6), 1.0
                )
            }
        )
        cascade = optimize_two_route_cascade(
            dataset_path=dataset_path,
            router_oof_path=router_oof_path,
            rescue_oof_path=oof_path,
            settings=measured_application,
            output_dir=variant_dir / "cascade",
        )
        selected = cascade["selected_policy"]
        go = (
            selected["final_precision"] >= settings.target_precision
            and selected["final_coverage"] >= settings.go_no_go_coverage
            and selected["projected_latency_reduction"] > 0.0
            and median_probe_fraction <= settings.max_median_probe_fraction
        )
        variant_reports[name] = {
            "roc_auc_unsafe": float(roc_auc_score(labels, probabilities)),
            "average_precision_unsafe": float(
                average_precision_score(labels, probabilities)
            ),
            "median_probe_fraction": median_probe_fraction,
            "selected_policy": selected,
            "go": go,
            "oof_predictions": str(oof_path),
        }

    best_name = max(
        variant_reports,
        key=lambda name: (
            variant_reports[name]["go"],
            variant_reports[name]["selected_policy"]["final_coverage"],
            variant_reports[name]["selected_policy"]["projected_latency_reduction"],
            variant_reports[name]["selected_policy"]["final_precision"],
        ),
    )
    verdict = "continue_probe_router" if variant_reports[best_name]["go"] else "stop_probe_router"
    report = {
        "verdict": verdict,
        "reason": (
            "at least one new runtime signal clears the bounded pilot gate"
            if verdict == "continue_probe_router"
            else "no new runtime signal clears the bounded pilot gate"
        ),
        "examples": len(records),
        "route_rows": len(rows),
        "best_variant": best_name,
        "targets": {
            "minimum_precision": settings.target_precision,
            "minimum_development_coverage": settings.go_no_go_coverage,
            "positive_projected_latency": True,
            "maximum_median_probe_fraction": settings.max_median_probe_fraction,
        },
        "variants": variant_reports,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# AdaptiveDepth Phase 8.1 runtime-signal pilot",
        "",
        f"**Verdict:** `{verdict}`",
        "",
        report["reason"],
        "",
        "| Variant | Unsafe AUC | Coverage | Precision | Latency | Go |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, result in variant_reports.items():
        policy = result["selected_policy"]
        lines.append(
            f"| {name} | {result['roc_auc_unsafe']:.3f} | "
            f"{policy['final_coverage']:.1%} | {policy['final_precision']:.1%} | "
            f"{policy['projected_latency_reduction']:.1%} | "
            f"{'yes' if result['go'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "This is a development go/no-go test on previously inspected prompts,",
            "not a deployment claim. A passing signal still requires fresh validation.",
        ]
    )
    (output_dir / "gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
