from __future__ import annotations

import hashlib
import json
import os
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import torch

from adaptivedepth.config import Phase61CalibrationSettings
from adaptivedepth.model.routed_model import RoutedCausalLM
from adaptivedepth.routes import RouteDefinition
from adaptivedepth.routing.dataset import (
    _text_agreement,
    label_route_safety,
    render_prompt,
    response_regret_metrics,
)
from adaptivedepth.routing.features import prompt_features
from adaptivedepth.runtime.environment import capture_environment
from adaptivedepth.runtime.generation import GenerationProfile, profile_greedy_generation
from adaptivedepth.training.data import (
    TrainingExample,
    collate_examples,
    encode_example_preserving_response,
)


SCHEMA_VERSION = 2
ProgressCallback = Callable[[str], None]


def _now() -> str:
    return datetime.now(UTC).isoformat()


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


class Phase61Store:
    """Append-only per-example journal suitable for a Drive-backed workspace."""

    def __init__(
        self,
        manifest_path: Path,
        records_path: Path,
        manifest: dict[str, Any],
        records: dict[str, dict[str, Any]],
    ) -> None:
        self.manifest_path = manifest_path
        self.records_path = records_path
        self.manifest = manifest
        self.records = records

    @classmethod
    def open(
        cls,
        workspace: str | Path,
        identity: dict[str, Any],
        environment: dict[str, Any],
        resume: bool,
    ) -> "Phase61Store":
        root = Path(workspace)
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "experiment.json"
        records_path = root / "records.jsonl"
        if manifest_path.exists() and resume:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("Phase 6.1 dataset schema version does not match")
            if manifest.get("identity") != identity:
                raise ValueError("cannot resume Phase 6.1 with a different identity")
            previous = manifest.get("environment", {})
            stable_keys = ("device", "gpu_name", "torch", "transformers", "git_commit")
            mismatches = [
                key
                for key in stable_keys
                if previous.get(key) is not None
                and environment.get(key) is not None
                and previous.get(key) != environment.get(key)
            ]
            if mismatches:
                raise ValueError(
                    "cannot resume Phase 6.1 on a different environment: "
                    + ", ".join(mismatches)
                )
            records = cls._load_and_repair_journal(records_path)
            return cls(manifest_path, records_path, manifest, records)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "started_at_utc": _now(),
            "updated_at_utc": _now(),
            "identity": identity,
            "environment": environment,
            "completed_records": 0,
        }
        records_path.write_text("", encoding="utf-8")
        _atomic_json_write(manifest_path, manifest)
        return cls(manifest_path, records_path, manifest, {})

    @staticmethod
    def _load_and_repair_journal(path: Path) -> dict[str, dict[str, Any]]:
        if not path.exists():
            return {}
        lines = path.read_text(encoding="utf-8").splitlines()
        records: dict[str, dict[str, Any]] = {}
        valid_lines: list[str] = []
        repaired = False
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                if index != len(lines) - 1:
                    raise ValueError(
                        "Phase 6.1 journal is corrupt before its final line"
                    ) from error
                repaired = True
                break
            example_id = str(record.get("example_id", ""))
            if not example_id or example_id in records:
                raise ValueError("Phase 6.1 journal has a missing or duplicate example ID")
            records[example_id] = record
            valid_lines.append(json.dumps(record, ensure_ascii=False, allow_nan=False))
        if repaired:
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(
                "".join(line + "\n" for line in valid_lines), encoding="utf-8"
            )
            temporary.replace(path)
        return records

    def save_record(self, record: dict[str, Any]) -> None:
        example_id = str(record["example_id"])
        if example_id in self.records:
            raise ValueError(f"Phase 6.1 record already exists: {example_id}")
        line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        with self.records_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                # Some hosted-drive FUSE mounts do not expose fsync.
                pass
        self.records[example_id] = record
        self.manifest["completed_records"] = len(self.records)
        self.manifest["updated_at_utc"] = _now()
        _atomic_json_write(self.manifest_path, self.manifest)

    def mark_complete(self) -> None:
        self.manifest["status"] = "completed"
        self.manifest["completed_records"] = len(self.records)
        self.manifest["updated_at_utc"] = _now()
        _atomic_json_write(self.manifest_path, self.manifest)


def _rotate(values: list[str], offset: int) -> list[str]:
    if not values:
        return []
    offset %= len(values)
    return values[offset:] + values[:offset]


def _profile(
    routed_model: RoutedCausalLM,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    route: RouteDefinition | None,
) -> GenerationProfile:
    if route is None:
        return profile_greedy_generation(
            routed_model,
            tokenizer,
            prompt,
            device,
            max_new_tokens,
            True,
            True,
        )
    with routed_model.use_route(route):
        return profile_greedy_generation(
            routed_model,
            tokenizer,
            prompt,
            device,
            max_new_tokens,
            True,
            True,
        )


def _completed_records(store: Phase61Store) -> list[dict[str, Any]]:
    return [
        record for record in store.records.values() if record.get("status") == "completed"
    ]


def build_phase61_report(
    records: list[dict[str, Any]],
    routes: list[RouteDefinition],
    settings: Phase61CalibrationSettings,
) -> dict[str, Any]:
    completed = [record for record in records if record.get("status") == "completed"]
    route_assessments: dict[str, dict[str, Any]] = {}
    for route in routes:
        measurements = [record["routes"][route.route_id] for record in completed]
        safe = [item for item in measurements if item["label"] == "safe_route"]
        latency_reductions = [item["latency_reduction"] for item in measurements]
        median_latency = statistics.median(latency_reductions) if measurements else 0.0
        checks = {
            "minimum_safe_examples": len(safe) >= settings.min_safe_examples,
            "minimum_unsafe_examples": (
                len(measurements) - len(safe) >= settings.min_unsafe_examples
            ),
            "latency_reduction": median_latency >= settings.min_route_latency_reduction,
        }
        route_assessments[route.route_id] = {
            "active_layers": len(route.active_layers),
            "safe_examples": len(safe),
            "unsafe_examples": len(measurements) - len(safe),
            "safe_rate": len(safe) / len(measurements) if measurements else 0.0,
            "median_latency_reduction": median_latency,
            "median_teacher_kl": (
                statistics.median(item["regret"]["teacher_kl"] for item in measurements)
                if measurements
                else None
            ),
            "median_reference_loss_regret": (
                statistics.median(
                    item["regret"]["reference_loss_regret"] for item in measurements
                )
                if measurements
                else None
            ),
            "median_token_agreement": (
                statistics.median(
                    item["regret"]["token_agreement"] for item in measurements
                )
                if measurements
                else 0.0
            ),
            "median_generation_token_jaccard": (
                statistics.median(
                    item["generation"]["token_jaccard"] for item in measurements
                )
                if measurements and all("generation" in item for item in measurements)
                else None
            ),
            "generation_exact_agreement_rate": (
                sum(
                    bool(item["generation"]["exact_agreement"])
                    for item in measurements
                )
                / len(measurements)
                if measurements and all("generation" in item for item in measurements)
                else None
            ),
            "checks": checks,
            "eligible": all(checks.values()),
        }

    route_order = sorted(
        (route.route_id for route in routes),
        key=lambda route_id: (
            -route_assessments[route_id]["median_latency_reduction"],
            route_assessments[route_id]["active_layers"],
            route_id,
        ),
    )
    target_counts = {route_id: 0 for route_id in route_order}
    target_counts["full"] = 0
    for record in completed:
        target = next(
            (
                route_id
                for route_id in route_order
                if record["routes"][route_id]["label"] == "safe_route"
            ),
            "full",
        )
        target_counts[target] += 1

    eligible_routes = [
        route_id
        for route_id in route_order
        if route_assessments[route_id]["eligible"]
    ]
    response_preservation_failures = sum(
        record["encoding"]["used_response_tokens"]
        != min(
            record["encoding"]["available_response_tokens"],
            settings.response_token_budget,
        )
        for record in completed
    )
    checks = {
        "minimum_examples": len(completed) >= settings.min_examples,
        "response_preservation": response_preservation_failures == 0,
        "minimum_eligible_routes": len(eligible_routes) >= settings.min_eligible_routes,
        "minimum_full_fallback_examples": (
            target_counts["full"] >= settings.min_unsafe_examples
        ),
    }
    proceed = all(checks.values())
    return {
        "verdict": "proceed_to_phase7_multiroute" if proceed else "revise_phase61",
        "reason": (
            "the response-preserving dataset supports multi-route risk prediction"
            if proceed
            else "the multi-route dataset does not yet meet router-readiness constraints"
        ),
        "valid_examples": len(completed),
        "skipped_examples": len(records) - len(completed),
        "response_token_budget": settings.response_token_budget,
        "prompt_truncated_examples": sum(
            bool(record["encoding"]["prompt_was_truncated"]) for record in completed
        ),
        "median_used_response_tokens": (
            statistics.median(
                record["encoding"]["used_response_tokens"] for record in completed
            )
            if completed
            else 0
        ),
        "response_preservation_failures": response_preservation_failures,
        "route_order_by_measured_latency": route_order,
        "eligible_routes": eligible_routes,
        "target_route_counts": target_counts,
        "route_assessments": route_assessments,
        "checks": checks,
        "thresholds": settings.model_dump(mode="json"),
    }


def phase61_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# AdaptiveDepth Phase 6.1 multi-route dataset gate",
        "",
        f"**Verdict:** `{report['verdict']}`",
        "",
        report["reason"],
        "",
        "## Dataset",
        "",
        f"- Valid examples: {report['valid_examples']}",
        f"- Skipped examples: {report['skipped_examples']}",
        f"- Response-token budget: {report['response_token_budget']}",
        f"- Prompts truncated in the middle: {report['prompt_truncated_examples']}",
        f"- Median response tokens used: {report['median_used_response_tokens']}",
        f"- Response-preservation failures: {report['response_preservation_failures']}",
        "- Route cost order: `"
        + " -> ".join(report["route_order_by_measured_latency"] + ["full"])
        + "`",
        "- Eligible routes: `" + "`, `".join(report["eligible_routes"]) + "`",
        "",
        "## Route assessment",
        "",
        "| Route | Layers | Safe | Fallback | Safe rate | Latency reduction "
        "| KL | Gen. Jaccard | Eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for route_id in report["route_order_by_measured_latency"]:
        item = report["route_assessments"][route_id]
        median_kl = item["median_teacher_kl"]
        median_kl_text = f"{median_kl:.4f}" if median_kl is not None else "n/a"
        generation_jaccard = item["median_generation_token_jaccard"]
        generation_jaccard_text = (
            f"{generation_jaccard:.1%}" if generation_jaccard is not None else "n/a"
        )
        lines.append(
            f"| {route_id} | {item['active_layers']} | {item['safe_examples']} | "
            f"{item['unsafe_examples']} | {item['safe_rate']:.1%} | "
            f"{item['median_latency_reduction']:.1%} | {median_kl_text} | "
            f"{generation_jaccard_text} | "
            f"{'yes' if item['eligible'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Cheapest-safe-route targets",
            "",
            *[
                f"- `{route_id}`: {count}"
                for route_id, count in report["target_route_counts"].items()
            ],
            "",
            "## Readiness checks",
            "",
            "- Minimum valid examples: "
            + ("pass" if report["checks"]["minimum_examples"] else "fail"),
            "- Response preservation: "
            + ("pass" if report["checks"]["response_preservation"] else "fail"),
            "- Minimum eligible routes: "
            + ("pass" if report["checks"]["minimum_eligible_routes"] else "fail"),
            "- Minimum full fallbacks: "
            + (
                "pass"
                if report["checks"]["minimum_full_fallback_examples"]
                else "fail"
            ),
            "",
            "The full model remains the mandatory fallback for prompts with no safe reduced route.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_router_dataset(
    path: Path, records: list[dict[str, Any]], route_order: list[str]
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: item["example_id"]):
            if record.get("status") != "completed":
                continue
            safe_route_ids = [
                route_id
                for route_id in route_order
                if record["routes"][route_id]["label"] == "safe_route"
            ]
            payload = {
                **record,
                "safe_route_ids": safe_route_ids,
                "target_route": safe_route_ids[0] if safe_route_ids else "full",
                "route_cost_order": route_order,
            }
            handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


@torch.inference_mode()
def collect_phase61_dataset(
    *,
    routed_model: RoutedCausalLM,
    tokenizer: Any,
    device: torch.device,
    model_id: str,
    dataset_path: str | Path,
    examples: list[TrainingExample],
    routes: list[RouteDefinition],
    settings: Phase61CalibrationSettings,
    workspace: str | Path,
    resume: bool = True,
    workdir: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(routes) < 2:
        raise ValueError("Phase 6.1 requires at least two reduced routes")
    route_ids = [route.route_id for route in routes]
    if len(route_ids) != len(set(route_ids)):
        raise ValueError("Phase 6.1 route IDs must be unique")
    if "full" in route_ids:
        raise ValueError("Phase 6.1 reserves the route ID 'full' for fallback")
    for route in routes:
        route.validate_for_depth(routed_model.num_layers)

    workspace = Path(workspace)
    identity = {
        "model_id": model_id,
        "dataset_path": Path(dataset_path).name,
        "dataset_sha256": _sha256_file(dataset_path),
        "num_layers": routed_model.num_layers,
        "routes": [route.model_dump(mode="json") for route in routes],
        "calibration": settings.model_dump(mode="json"),
    }
    store = Phase61Store.open(
        workspace,
        identity,
        capture_environment(device, workdir),
        resume,
    )
    routed_model.model.eval()
    routes_by_id = {route.route_id: route for route in routes}
    pending = sum(
        f"calibration_{index:04d}" not in store.records
        for index in range(1, len(examples) + 1)
    )
    if progress:
        progress(
            f"Phase 6.1 workspace has {len(store.records)} completed/skipped records; "
            f"{pending} prompts remain"
        )

    if pending and examples and settings.warmup_runs:
        warmup_prompt = render_prompt(examples[0])
        if progress:
            progress("Warming full generation and all reduced routes")
        for _ in range(settings.warmup_runs):
            _profile(
                routed_model,
                tokenizer,
                warmup_prompt,
                device,
                min(settings.max_new_tokens, 4),
                None,
            )
            for route in routes:
                _profile(
                    routed_model,
                    tokenizer,
                    warmup_prompt,
                    device,
                    min(settings.max_new_tokens, 4),
                    route,
                )

    execution_ids = ["full", *route_ids]
    for index, example in enumerate(examples, start=1):
        example_id = f"calibration_{index:04d}"
        if example_id in store.records:
            continue
        prompt = render_prompt(example)
        try:
            encoded = encode_example_preserving_response(
                tokenizer,
                example,
                settings.max_seq_length,
                settings.response_token_budget,
            )
            batch = collate_examples([encoded], tokenizer.pad_token_id, device)
        except ValueError as error:
            store.save_record(
                {
                    "status": "skipped",
                    "example_id": example_id,
                    "reason": str(error),
                }
            )
            continue

        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "use_cache": False,
        }
        full_logits = routed_model(**inputs).logits
        route_metrics: dict[str, dict[str, Any]] = {}
        for route in routes:
            with routed_model.use_route(route):
                route_logits = routed_model(**inputs).logits
            regret = response_regret_metrics(
                full_logits,
                route_logits,
                batch["labels"],
                settings.distillation_temperature,
            )
            label, criteria = label_route_safety(regret, settings)
            route_metrics[route.route_id] = {
                "active_layers": route.active_layers,
                "regret": regret,
                "label": label,
                "criteria": criteria,
            }
            del route_logits
        del full_logits, batch

        order = _rotate(execution_ids, (index - 1) % len(execution_ids))
        profiles: dict[str, GenerationProfile] = {}
        for route_id in order:
            profiles[route_id] = _profile(
                routed_model,
                tokenizer,
                prompt,
                device,
                settings.max_new_tokens,
                routes_by_id.get(route_id),
            )
        full_profile = profiles["full"]
        for route_id in route_ids:
            route_profile = profiles[route_id]
            exact, jaccard = _text_agreement(full_profile.text, route_profile.text)
            route_metrics[route_id].update(
                {
                    "generation": {
                        "text": route_profile.text,
                        "exact_agreement": exact,
                        "token_jaccard": jaccard,
                    },
                    "profile": route_profile.to_dict(),
                    "latency_reduction": 1.0
                    - route_profile.total_ms / max(full_profile.total_ms, 1e-9),
                }
            )
            generation_safe = jaccard >= settings.min_generation_token_jaccard
            route_metrics[route_id]["criteria"]["generation_token_jaccard"] = (
                generation_safe
            )
            if not generation_safe:
                route_metrics[route_id]["label"] = "full_model"

        safe_route_ids = [
            route_id
            for route_id in route_ids
            if route_metrics[route_id]["label"] == "safe_route"
        ]
        record = {
            "status": "completed",
            "example_id": example_id,
            "prompt": prompt,
            "reference": example.response,
            "features": prompt_features(prompt, encoded.prompt_token_count),
            "encoding": {
                "prompt_token_count": encoded.prompt_token_count,
                "available_response_tokens": encoded.available_response_tokens,
                "used_response_tokens": encoded.used_response_tokens,
                "prompt_was_truncated": encoded.prompt_was_truncated,
                "response_was_truncated": encoded.response_was_truncated,
            },
            "full": {
                "generation": {"text": full_profile.text},
                "profile": full_profile.to_dict(),
            },
            "routes": route_metrics,
            "safe_route_ids_unordered": safe_route_ids,
            "execution_order": order,
        }
        store.save_record(record)
        if progress:
            safe_text = ",".join(safe_route_ids) if safe_route_ids else "full-only"
            progress(f"[{index}/{len(examples)}] {example_id} safe={safe_text}")

    store.mark_complete()
    records = list(store.records.values())
    report = build_phase61_report(records, routes, settings)
    _atomic_json_write(workspace / "gate.json", report)
    (workspace / "gate.md").write_text(phase61_markdown(report), encoding="utf-8")
    _write_router_dataset(
        workspace / "router_dataset.jsonl",
        records,
        report["route_order_by_measured_latency"],
    )
    return store.manifest, report
