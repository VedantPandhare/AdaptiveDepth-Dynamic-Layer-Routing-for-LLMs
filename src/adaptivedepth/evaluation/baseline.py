from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from adaptivedepth.config import BenchmarkSettings
from adaptivedepth.model.routed_model import RoutedCausalLM
from adaptivedepth.routes import RouteDefinition
from adaptivedepth.runtime.generation import GenerationProfile, profile_greedy_generation


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    category: str
    prompt: str
    reference: str


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    category: str
    reference: str
    prediction: str
    exact_match: bool
    profile: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkReport:
    route_id: str
    active_layers: int
    total_layers: int
    accuracy: float
    median_total_ms: float
    p95_total_ms: float
    median_ttft_ms: float
    median_decode_ms_per_token: float
    results: list[CaseResult]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def load_cases(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            try:
                cases.append(
                    BenchmarkCase(
                        case_id=value["id"],
                        category=value["category"],
                        prompt=value["prompt"],
                        reference=value["reference"],
                    )
                )
            except KeyError as error:
                raise ValueError(f"missing {error.args[0]!r} on line {line_number}") from error
    if not cases:
        raise ValueError(f"benchmark dataset is empty: {path}")
    return cases


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _is_exact_match(prediction: str, reference: str) -> bool:
    normalized_prediction = normalize_answer(prediction)
    normalized_reference = normalize_answer(reference)
    if len(normalized_reference) == 1:
        return normalized_prediction.split(" ", 1)[0] == normalized_reference
    return normalized_prediction == normalized_reference


def _percentile_95(values: list[float]) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = round(0.95 * (len(ordered) - 1))
    return ordered[index]


def _median_profile(profiles: list[GenerationProfile]) -> GenerationProfile:
    median_total = statistics.median(item.total_ms for item in profiles)
    return min(profiles, key=lambda profile: abs(profile.total_ms - median_total))


def run_benchmark(
    routed_model: RoutedCausalLM,
    tokenizer: Any,
    device: torch.device,
    cases: list[BenchmarkCase],
    settings: BenchmarkSettings,
    route: RouteDefinition | None = None,
) -> BenchmarkReport:
    torch.manual_seed(settings.seed)
    route_id = route.route_id if route else "full"
    active_layers = len(route.active_layers) if route else routed_model.num_layers

    with routed_model.use_route(route):
        for _ in range(settings.warmup_runs):
            profile_greedy_generation(
                routed_model,
                tokenizer,
                cases[0].prompt,
                device,
                min(settings.max_new_tokens, 4),
            )

        results: list[CaseResult] = []
        for case in cases:
            profiles = [
                profile_greedy_generation(
                    routed_model,
                    tokenizer,
                    case.prompt,
                    device,
                    settings.max_new_tokens,
                )
                for _ in range(settings.measured_runs)
            ]
            profile = _median_profile(profiles)
            results.append(
                CaseResult(
                    case_id=case.case_id,
                    category=case.category,
                    reference=case.reference,
                    prediction=profile.text,
                    exact_match=_is_exact_match(profile.text, case.reference),
                    profile=profile.to_dict(),
                )
            )

    totals = [result.profile["total_ms"] for result in results]
    return BenchmarkReport(
        route_id=route_id,
        active_layers=active_layers,
        total_layers=routed_model.num_layers,
        accuracy=sum(result.exact_match for result in results) / len(results),
        median_total_ms=statistics.median(totals),
        p95_total_ms=_percentile_95(totals),
        median_ttft_ms=statistics.median(result.profile["ttft_ms"] for result in results),
        median_decode_ms_per_token=statistics.median(
            result.profile["decode_ms_per_token"] for result in results
        ),
        results=results,
    )
