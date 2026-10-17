from __future__ import annotations

import hashlib
import json
import re
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import torch
from torch.nn import functional as F

from adaptivedepth.config import Phase6CalibrationSettings
from adaptivedepth.model.routed_model import RoutedCausalLM
from adaptivedepth.routes import RouteDefinition
from adaptivedepth.runtime.environment import capture_environment
from adaptivedepth.runtime.generation import profile_greedy_generation
from adaptivedepth.training.data import (
    TrainingExample,
    collate_examples,
    encode_example,
)
from adaptivedepth.training.objective import causal_language_model_loss


SCHEMA_VERSION = 1
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


def render_prompt(example: TrainingExample) -> str:
    instruction = example.instruction.strip()
    if not example.context.strip():
        return instruction
    return f"{instruction}\n\nContext:\n{example.context.strip()}"


def prompt_features(prompt: str, prompt_tokens: int) -> dict[str, float | int | bool]:
    characters = len(prompt)
    digits = sum(character.isdigit() for character in prompt)
    return {
        "character_count": characters,
        "word_count": len(prompt.split()),
        "prompt_token_count": prompt_tokens,
        "newline_count": prompt.count("\n"),
        "question_mark_count": prompt.count("?"),
        "digit_fraction": digits / max(characters, 1),
        "contains_code_fence": "```" in prompt,
        "contains_math_symbol": bool(re.search(r"[=+*/^]", prompt)),
    }


def response_regret_metrics(
    full_logits: torch.Tensor,
    route_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> dict[str, float]:
    mask = labels[:, 1:] != -100
    if not bool(mask.any()):
        raise ValueError("example has no response tokens after shifting")

    full_loss = causal_language_model_loss(full_logits, labels).float()
    route_loss = causal_language_model_loss(route_logits, labels).float()
    full = full_logits[:, :-1, :].float() / temperature
    route = route_logits[:, :-1, :].float() / temperature
    token_kl = F.kl_div(
        F.log_softmax(route, dim=-1),
        F.softmax(full, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    teacher_kl = (token_kl[mask].mean() * (temperature**2)).clamp_min(0)
    agreement = (full.argmax(dim=-1) == route.argmax(dim=-1))[mask].float().mean()
    return {
        "full_reference_loss": float(full_loss.cpu()),
        "route_reference_loss": float(route_loss.cpu()),
        "reference_loss_regret": float((route_loss - full_loss).cpu()),
        "teacher_kl": float(teacher_kl.cpu()),
        "token_agreement": float(agreement.cpu()),
        "reference_token_count": int(mask.sum().cpu()),
    }


def label_route_safety(
    metrics: dict[str, float], settings: Phase6CalibrationSettings
) -> tuple[str, dict[str, bool]]:
    criteria = {
        "teacher_kl": metrics["teacher_kl"] <= settings.max_teacher_kl,
        "reference_loss_regret": (
            metrics["reference_loss_regret"]
            <= settings.max_reference_loss_regret
        ),
        "token_agreement": (
            metrics["token_agreement"] >= settings.min_token_agreement
        ),
    }
    return ("safe_route" if all(criteria.values()) else "full_model"), criteria


def _normalized_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _text_agreement(full_text: str, route_text: str) -> tuple[bool, float]:
    full_normalized = re.findall(r"[a-z0-9]+", full_text.lower())
    route_normalized = re.findall(r"[a-z0-9]+", route_text.lower())
    full_tokens = _normalized_tokens(full_text)
    route_tokens = _normalized_tokens(route_text)
    exact = full_normalized == route_normalized
    union = full_tokens | route_tokens
    jaccard = len(full_tokens & route_tokens) / len(union) if union else 1.0
    return exact, jaccard


class Phase6Store:
    def __init__(self, path: Path, payload: dict[str, Any]) -> None:
        self.path = path
        self.payload = payload

    @classmethod
    def open(
        cls,
        path: str | Path,
        identity: dict[str, Any],
        environment: dict[str, Any],
        resume: bool,
    ) -> "Phase6Store":
        destination = Path(path)
        if destination.exists() and resume:
            payload = json.loads(destination.read_text(encoding="utf-8"))
            if payload.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("Phase 6 dataset schema version does not match")
            if payload.get("identity") != identity:
                raise ValueError("cannot resume Phase 6 with a different identity")
            previous = payload.get("environment", {})
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
                    "cannot resume Phase 6 on a different environment: "
                    + ", ".join(mismatches)
                )
            return cls(destination, payload)

        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "started_at_utc": _now(),
            "updated_at_utc": _now(),
            "identity": identity,
            "environment": environment,
            "records": {},
        }
        store = cls(destination, payload)
        store.save()
        return store

    def save_record(self, example_id: str, record: dict[str, Any]) -> None:
        self.payload["records"][example_id] = record
        self.save()

    def mark_complete(self) -> None:
        self.payload["status"] = "completed"
        self.save()

    def save(self) -> None:
        self.payload["updated_at_utc"] = _now()
        _atomic_json_write(self.path, self.payload)


def build_phase6_report(
    experiment: dict[str, Any], settings: Phase6CalibrationSettings
) -> dict[str, Any]:
    records = [
        record
        for record in experiment["records"].values()
        if record.get("status") == "completed"
    ]
    safe = [record for record in records if record["label"] == "safe_route"]
    unsafe = [record for record in records if record["label"] == "full_model"]
    latency_reductions = [record["latency_reduction"] for record in records]
    median_latency_reduction = (
        statistics.median(latency_reductions) if latency_reductions else 0.0
    )
    checks = {
        "minimum_examples": len(records) >= settings.min_examples,
        "minimum_safe_examples": len(safe) >= settings.min_safe_examples,
        "minimum_unsafe_examples": len(unsafe) >= settings.min_unsafe_examples,
        "latency_reduction": (
            median_latency_reduction >= settings.min_median_latency_reduction
        ),
    }
    proceed = all(checks.values())
    return {
        "verdict": "proceed_to_phase7" if proceed else "revise_phase6",
        "reason": (
            "the compact dataset has enough safe and fallback examples"
            if proceed
            else "the compact dataset is not sufficient for a calibrated router"
        ),
        "route_id": experiment["identity"]["route"]["route_id"],
        "total_source_examples": len(experiment["records"]),
        "valid_examples": len(records),
        "skipped_examples": len(experiment["records"]) - len(records),
        "safe_examples": len(safe),
        "unsafe_examples": len(unsafe),
        "safe_rate": len(safe) / len(records) if records else 0.0,
        "median_latency_reduction": median_latency_reduction,
        "median_teacher_kl": (
            statistics.median(record["regret"]["teacher_kl"] for record in records)
            if records
            else None
        ),
        "median_reference_loss_regret": (
            statistics.median(
                record["regret"]["reference_loss_regret"] for record in records
            )
            if records
            else None
        ),
        "median_token_agreement": (
            statistics.median(
                record["regret"]["token_agreement"] for record in records
            )
            if records
            else 0.0
        ),
        "checks": checks,
        "thresholds": settings.model_dump(mode="json"),
    }


def phase6_markdown(report: dict[str, Any]) -> str:
    checks = report["checks"]
    teacher_kl = report["median_teacher_kl"]
    loss_regret = report["median_reference_loss_regret"]
    lines = [
        "# AdaptiveDepth Phase 6 compact router dataset",
        "",
        f"**Verdict:** `{report['verdict']}`",
        "",
        report["reason"],
        "",
        "## Dataset",
        "",
        f"- Route: `{report['route_id']}`",
        f"- Valid examples: {report['valid_examples']}",
        f"- Skipped examples: {report['skipped_examples']}",
        f"- Safe-route labels: {report['safe_examples']}",
        f"- Full-model fallback labels: {report['unsafe_examples']}",
        f"- Safe-route rate: {report['safe_rate']:.1%}",
        "",
        "## Aggregate measurements",
        "",
        f"- Median latency reduction: {report['median_latency_reduction']:.1%}",
        f"- Median teacher KL: {teacher_kl:.4f}" if teacher_kl is not None else "- Median teacher KL: n/a",
        f"- Median reference-loss regret: {loss_regret:+.4f}" if loss_regret is not None else "- Median reference-loss regret: n/a",
        f"- Median response-token agreement: {report['median_token_agreement']:.1%}",
        "",
        "## Phase 7 readiness",
        "",
        f"- Minimum valid examples: {'pass' if checks['minimum_examples'] else 'fail'}",
        f"- Minimum safe examples: {'pass' if checks['minimum_safe_examples'] else 'fail'}",
        f"- Minimum fallback examples: {'pass' if checks['minimum_unsafe_examples'] else 'fail'}",
        f"- Median latency target: {'pass' if checks['latency_reduction'] else 'fail'}",
        "",
        "This is an application-track viability gate, not a claim that the static Phase 5 gate passed.",
    ]
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def collect_phase6_dataset(
    *,
    routed_model: RoutedCausalLM,
    tokenizer: Any,
    device: torch.device,
    model_id: str,
    dataset_path: str | Path,
    examples: list[TrainingExample],
    route: RouteDefinition,
    settings: Phase6CalibrationSettings,
    workspace: str | Path,
    resume: bool = True,
    workdir: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    experiment_path = workspace / "experiment.json"
    route.validate_for_depth(routed_model.num_layers)
    identity = {
        "model_id": model_id,
        "dataset_path": Path(dataset_path).name,
        "dataset_sha256": _sha256_file(dataset_path),
        "num_layers": routed_model.num_layers,
        "route": route.model_dump(mode="json"),
        "calibration": settings.model_dump(mode="json"),
    }
    store = Phase6Store.open(
        experiment_path,
        identity,
        capture_environment(device, workdir),
        resume,
    )
    routed_model.model.eval()

    if not store.payload["records"] and examples and settings.warmup_runs:
        warmup_prompt = render_prompt(examples[0])
        if progress:
            progress("Warming full and routed generation")
        for _ in range(settings.warmup_runs):
            profile_greedy_generation(
                routed_model,
                tokenizer,
                warmup_prompt,
                device,
                min(settings.max_new_tokens, 4),
                True,
                True,
            )
            with routed_model.use_route(route):
                profile_greedy_generation(
                    routed_model,
                    tokenizer,
                    warmup_prompt,
                    device,
                    min(settings.max_new_tokens, 4),
                    True,
                    True,
                )

    for index, example in enumerate(examples, start=1):
        example_id = f"calibration_{index:04d}"
        if example_id in store.payload["records"]:
            if progress:
                progress(f"Skipping completed {example_id}")
            continue
        prompt = render_prompt(example)
        try:
            encoded = encode_example(tokenizer, example, settings.max_seq_length)
            batch = collate_examples([encoded], tokenizer.pad_token_id, device)
        except ValueError as error:
            store.save_record(
                example_id,
                {
                    "status": "skipped",
                    "example_id": example_id,
                    "reason": str(error),
                },
            )
            continue

        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "use_cache": False,
        }
        full_logits = routed_model(**inputs).logits
        with routed_model.use_route(route):
            route_logits = routed_model(**inputs).logits
        regret = response_regret_metrics(
            full_logits,
            route_logits,
            batch["labels"],
            settings.distillation_temperature,
        )
        label, criteria = label_route_safety(regret, settings)
        del full_logits, route_logits, batch

        def profile_full():
            return profile_greedy_generation(
                routed_model,
                tokenizer,
                prompt,
                device,
                settings.max_new_tokens,
                True,
                True,
            )

        def profile_route():
            with routed_model.use_route(route):
                return profile_greedy_generation(
                    routed_model,
                    tokenizer,
                    prompt,
                    device,
                    settings.max_new_tokens,
                    True,
                    True,
                )

        # Alternate order to prevent GPU clock or thermal drift from favoring
        # the route that is always measured second.
        if index % 2:
            full_profile, route_profile = profile_full(), profile_route()
        else:
            route_profile, full_profile = profile_route(), profile_full()
        generation_exact, generation_jaccard = _text_agreement(
            full_profile.text, route_profile.text
        )
        latency_reduction = 1.0 - (
            route_profile.total_ms / max(full_profile.total_ms, 1e-9)
        )
        prompt_token_count = len(encoded.input_ids) - int(
            regret["reference_token_count"]
        )
        record = {
            "status": "completed",
            "example_id": example_id,
            "prompt": prompt,
            "reference": example.response,
            "features": prompt_features(prompt, prompt_token_count),
            "regret": regret,
            "generation": {
                "full_text": full_profile.text,
                "route_text": route_profile.text,
                "exact_agreement": generation_exact,
                "token_jaccard": generation_jaccard,
            },
            "full_latency_ms": full_profile.total_ms,
            "route_latency_ms": route_profile.total_ms,
            "latency_reduction": latency_reduction,
            "label": label,
            "criteria": criteria,
        }
        store.save_record(example_id, record)
        if progress:
            progress(
                f"{example_id} label={label} kl={regret['teacher_kl']:.4f} "
                f"regret={regret['reference_loss_regret']:+.4f}"
            )

    store.mark_complete()
    report = build_phase6_report(store.payload, settings)
    _atomic_json_write(workspace / "gate.json", report)
    (workspace / "gate.md").write_text(phase6_markdown(report), encoding="utf-8")
    router_dataset = workspace / "router_dataset.jsonl"
    with router_dataset.open("w", encoding="utf-8") as handle:
        for example_id in sorted(store.payload["records"]):
            record = store.payload["records"][example_id]
            if record.get("status") == "completed":
                handle.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )
    return store.payload, report
