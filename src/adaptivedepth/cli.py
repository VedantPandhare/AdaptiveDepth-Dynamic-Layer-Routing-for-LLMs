from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from adaptivedepth.analysis.block_influence import BlockInfluenceAnalyzer
from adaptivedepth.analysis.ablation import (
    adjacent_group_deletion_routes,
    random_deletion_routes,
    single_layer_deletion_routes,
)
from adaptivedepth.compiler.candidate_search import generate_candidate_routes
from adaptivedepth.compiler.pareto import RouteMeasurement, pareto_frontier
from adaptivedepth.config import load_compiler_settings, load_project_settings
from adaptivedepth.evaluation.baseline import load_cases, run_benchmark
from adaptivedepth.model.loading import load_model
from adaptivedepth.routes import RouteLibrary

app = typer.Typer(no_args_is_help=True, help="AdaptiveDepth experiment CLI")


@app.command("inspect-model")
def inspect_model(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/model.yaml"),
) -> None:
    settings = load_project_settings(config)
    routed_model, _tokenizer, device = load_model(settings.model)
    typer.echo(
        json.dumps(
            {
                "model_id": settings.model.model_id,
                "device": str(device),
                "layer_path": routed_model.layer_path,
                "num_layers": routed_model.num_layers,
            },
            indent=2,
        )
    )


@app.command()
def baseline(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/model.yaml"),
    dataset: Annotated[Path, typer.Option(exists=True)] = Path("data/smoke_eval.jsonl"),
    output: Annotated[Path, typer.Option()] = Path("experiments/baseline.json"),
) -> None:
    settings = load_project_settings(config)
    routed_model, tokenizer, device = load_model(settings.model)
    report = run_benchmark(
        routed_model,
        tokenizer,
        device,
        load_cases(dataset),
        settings.benchmark,
    )
    report.save(output)
    typer.echo(f"Saved baseline report to {output}")
    typer.echo(
        json.dumps(
            {
                "accuracy": report.accuracy,
                "median_total_ms": report.median_total_ms,
                "p95_total_ms": report.p95_total_ms,
            },
            indent=2,
        )
    )


@app.command("analyze-layers")
def analyze_layers(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/model.yaml"),
    dataset: Annotated[Path, typer.Option(exists=True)] = Path("data/smoke_eval.jsonl"),
    output: Annotated[Path, typer.Option()] = Path("experiments/block_influence.json"),
) -> None:
    settings = load_project_settings(config)
    routed_model, tokenizer, device = load_model(settings.model)
    prompts = [case.prompt for case in load_cases(dataset)]
    scores = BlockInfluenceAnalyzer(routed_model).analyze_prompts(tokenizer, prompts, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "model_id": settings.model.model_id,
                "layers": [score.to_dict() for score in scores],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    typer.echo(f"Saved block influence scores to {output}")


@app.command("compile-candidates")
def compile_candidates(
    influence_file: Annotated[Path, typer.Option(exists=True)],
    compiler_config: Annotated[Path, typer.Option(exists=True)] = Path("configs/compiler.yaml"),
    output: Annotated[Path, typer.Option()] = Path("experiments/candidate_routes.yaml"),
) -> None:
    payload = json.loads(influence_file.read_text(encoding="utf-8"))
    scores = {
        int(item["layer_index"]): float(item["influence"])
        for item in payload["layers"]
    }
    settings = load_compiler_settings(compiler_config)
    routes = generate_candidate_routes(
        num_layers=len(scores),
        influence=scores,
        budgets=settings.budgets,
        preserve_first=settings.preserve_first,
        preserve_last=settings.preserve_last,
        variants_per_budget=settings.variants_per_budget,
        seed=settings.seed,
    )
    RouteLibrary(routes=routes).save(output)
    typer.echo(f"Saved {len(routes)} candidates to {output}")


@app.command("generate-ablations")
def generate_ablations(
    strategy: Annotated[str, typer.Option(help="single, adjacent, or random")],
    num_layers: Annotated[int, typer.Option(min=2)],
    output: Annotated[Path, typer.Option()] = Path("experiments/ablation_routes.yaml"),
    group_size: Annotated[int, typer.Option(min=1)] = 2,
    active_budget: Annotated[int | None, typer.Option()] = None,
    samples: Annotated[int, typer.Option(min=1)] = 8,
    preserve_first: Annotated[int, typer.Option(min=0)] = 1,
    preserve_last: Annotated[int, typer.Option(min=0)] = 1,
    seed: Annotated[int, typer.Option()] = 17,
) -> None:
    protected = set(range(min(preserve_first, num_layers)))
    protected.update(range(max(0, num_layers - preserve_last), num_layers))
    if strategy == "single":
        routes = single_layer_deletion_routes(num_layers, protected)
    elif strategy == "adjacent":
        routes = adjacent_group_deletion_routes(num_layers, group_size, protected)
    elif strategy == "random":
        if active_budget is None:
            raise typer.BadParameter("--active-budget is required for random ablations")
        routes = random_deletion_routes(
            num_layers,
            active_budget,
            samples,
            protected,
            seed,
        )
    else:
        raise typer.BadParameter("strategy must be one of: single, adjacent, random")
    RouteLibrary(routes=routes).save(output)
    typer.echo(f"Saved {len(routes)} ablation routes to {output}")


@app.command("benchmark-routes")
def benchmark_routes(
    routes_file: Annotated[Path, typer.Option(exists=True)],
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/model.yaml"),
    dataset: Annotated[Path, typer.Option(exists=True)] = Path("data/smoke_eval.jsonl"),
    output: Annotated[Path, typer.Option()] = Path("experiments/route_benchmarks.json"),
) -> None:
    settings = load_project_settings(config)
    routed_model, tokenizer, device = load_model(settings.model)
    library = RouteLibrary.load(routes_file)
    library.validate_for_depth(routed_model.num_layers)
    cases = load_cases(dataset)
    reports = [
        run_benchmark(
            routed_model,
            tokenizer,
            device,
            cases,
            settings.benchmark,
            route,
        ).to_dict()
        for route in library.routes
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    typer.echo(f"Saved {len(reports)} route benchmarks to {output}")


@app.command("pareto-filter")
def pareto_filter(
    measurements_file: Annotated[Path, typer.Option(exists=True)],
    output: Annotated[Path, typer.Option()] = Path("experiments/pareto_routes.json"),
) -> None:
    payload = json.loads(measurements_file.read_text(encoding="utf-8"))
    measurements = [RouteMeasurement.model_validate(item) for item in payload]
    frontier = pareto_frontier(measurements)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([item.model_dump(mode="json") for item in frontier], indent=2),
        encoding="utf-8",
    )
    typer.echo(f"Saved {len(frontier)} Pareto routes to {output}")


if __name__ == "__main__":
    app()
