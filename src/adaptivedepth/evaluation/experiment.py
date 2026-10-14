from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import torch

from adaptivedepth.analysis.teacher_divergence import mean_teacher_kl
from adaptivedepth.config import BenchmarkSettings
from adaptivedepth.evaluation.baseline import load_cases, run_benchmark
from adaptivedepth.model.routed_model import RoutedCausalLM
from adaptivedepth.routes import RouteDefinition, RouteLibrary
from adaptivedepth.runtime.environment import capture_environment


SCHEMA_VERSION = 1
ProgressCallback = Callable[[str], None]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


class ExperimentStore:
    """Crash-safe JSON store that rejects incompatible resume attempts."""

    def __init__(self, path: str | Path, payload: dict[str, Any]) -> None:
        self.path = Path(path)
        self.payload = payload

    @classmethod
    def open(
        cls,
        path: str | Path,
        identity: dict[str, Any],
        environment: dict[str, Any],
        resume: bool = True,
    ) -> "ExperimentStore":
        destination = Path(path)
        if destination.exists() and resume:
            payload = json.loads(destination.read_text(encoding="utf-8"))
            if payload.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("experiment schema version does not match")
            if payload.get("identity") != identity:
                raise ValueError(
                    "cannot resume: model, dataset, layer count, benchmark settings, "
                    "or route suite changed"
                )
            previous_environment = payload.get("environment", {})
            stable_keys = ("device", "gpu_name", "torch", "transformers", "git_commit")
            mismatches = [
                key
                for key in stable_keys
                if previous_environment.get(key) is not None
                and environment.get(key) is not None
                and previous_environment.get(key) != environment.get(key)
            ]
            if mismatches:
                raise ValueError(
                    "cannot resume latency experiment on a different environment: "
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
            "full_baseline": None,
            "routes": {},
        }
        store = cls(destination, payload)
        store.save()
        return store

    @property
    def completed_route_ids(self) -> set[str]:
        return set(self.payload["routes"])

    def save_full_baseline(self, report: dict[str, Any]) -> None:
        self.payload["full_baseline"] = {
            "report": report,
            "measurement": {
                "route_id": "full",
                "source": "full_model",
                "active_layers": report["total_layers"],
                "quality": report["accuracy"],
                "quality_delta": 0.0,
                "latency_ms": report["median_total_ms"],
                "latency_reduction": 0.0,
                "speedup": 1.0,
                "teacher_divergence": 0.0,
                "peak_memory_mb": report["peak_memory_mb"],
            },
        }
        self.save()

    def save_route(
        self,
        route: RouteDefinition,
        report: dict[str, Any],
        teacher_divergence: float,
    ) -> None:
        baseline = self.payload["full_baseline"]
        if baseline is None:
            raise RuntimeError("full baseline must be saved before route results")
        full_report = baseline["report"]
        full_latency = full_report["median_total_ms"]
        route_latency = report["median_total_ms"]
        self.payload["routes"][route.route_id] = {
            "route": route.model_dump(mode="json"),
            "report": report,
            "measurement": {
                "route_id": route.route_id,
                "source": route.source,
                "active_layers": len(route.active_layers),
                "quality": report["accuracy"],
                "quality_delta": report["accuracy"] - full_report["accuracy"],
                "latency_ms": route_latency,
                "latency_reduction": 1.0 - (route_latency / full_latency),
                "speedup": full_latency / route_latency,
                "teacher_divergence": teacher_divergence,
                "peak_memory_mb": report["peak_memory_mb"],
            },
        }
        self.save()

    def mark_complete(self) -> None:
        self.payload["status"] = "completed"
        self.save()

    def save(self) -> None:
        self.payload["updated_at_utc"] = _now()
        _atomic_json_write(self.path, self.payload)


def _experiment_identity(
    model_id: str,
    dataset_path: str | Path,
    settings: BenchmarkSettings,
    num_layers: int,
    routes: RouteLibrary,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "dataset_path": Path(dataset_path).name,
        "dataset_sha256": file_sha256(dataset_path),
        "benchmark": settings.model_dump(mode="json"),
        "num_layers": num_layers,
        "routes": [route.model_dump(mode="json") for route in routes.routes],
    }


def run_static_experiment(
    routed_model: RoutedCausalLM,
    tokenizer: Any,
    device: torch.device,
    model_id: str,
    dataset_path: str | Path,
    settings: BenchmarkSettings,
    routes: RouteLibrary,
    output: str | Path,
    resume: bool = True,
    workdir: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run and persist the full baseline plus every static route."""

    routes.validate_for_depth(routed_model.num_layers)
    cases = load_cases(dataset_path)
    identity = _experiment_identity(
        model_id, dataset_path, settings, routed_model.num_layers, routes
    )
    store = ExperimentStore.open(
        output,
        identity,
        capture_environment(device, workdir),
        resume,
    )

    if store.payload["full_baseline"] is None:
        if progress:
            progress("Running full-model baseline")
        report = run_benchmark(
            routed_model, tokenizer, device, cases, settings, route=None
        )
        store.save_full_baseline(report.to_dict())
    elif progress:
        progress("Resuming with existing full-model baseline")

    prompts = [case.prompt for case in cases]
    for index, route in enumerate(routes.routes, start=1):
        existing = store.payload["routes"].get(route.route_id)
        if existing is not None:
            if existing["route"]["active_layers"] != route.active_layers:
                raise ValueError(
                    f"route {route.route_id!r} changed since the experiment started"
                )
            if progress:
                progress(f"Skipping completed route {route.route_id}")
            continue
        if progress:
            progress(f"Benchmarking route {index}/{len(routes.routes)}: {route.route_id}")
        report = run_benchmark(
            routed_model, tokenizer, device, cases, settings, route=route
        )
        divergence = mean_teacher_kl(
            routed_model,
            tokenizer,
            prompts,
            device,
            route,
            settings.use_chat_template,
        )
        store.save_route(route, report.to_dict(), divergence)

    expected = {route.route_id for route in routes.routes}
    if expected.issubset(store.completed_route_ids):
        store.mark_complete()
    return store.payload
