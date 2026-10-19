from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from adaptivedepth.analysis.ablation import (
    adjacent_group_deletion_routes,
    random_deletion_routes,
    single_layer_deletion_routes,
)
from adaptivedepth.compiler.candidate_search import generate_candidate_routes
from adaptivedepth.compiler.pareto import RouteMeasurement, pareto_frontier
from adaptivedepth.config import (
    load_compiler_settings,
    load_evaluation_settings,
    load_phase5_evaluation_settings,
    load_phase5_settings,
    load_phase6_settings,
    load_phase61_settings,
    load_phase71_settings,
    load_phase72_settings,
    load_phase72_application_settings,
    load_phase8_settings,
    load_phase7_settings,
    load_project_settings,
)
from adaptivedepth.routes import RouteLibrary

app = typer.Typer(no_args_is_help=True, help="AdaptiveDepth experiment CLI")


@app.command("inspect-model")
def inspect_model(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/model.yaml"),
) -> None:
    from adaptivedepth.model.loading import load_model

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
    from adaptivedepth.evaluation.baseline import load_cases, run_benchmark
    from adaptivedepth.model.loading import load_model

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
    from adaptivedepth.analysis.block_influence import BlockInfluenceAnalyzer
    from adaptivedepth.evaluation.baseline import load_cases
    from adaptivedepth.model.loading import load_model

    settings = load_project_settings(config)
    routed_model, tokenizer, device = load_model(settings.model)
    prompts = [case.prompt for case in load_cases(dataset)]
    scores = BlockInfluenceAnalyzer(routed_model).analyze_prompts(
        tokenizer,
        prompts,
        device,
        settings.benchmark.use_chat_template,
    )
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
    from adaptivedepth.analysis.teacher_divergence import mean_teacher_kl
    from adaptivedepth.evaluation.baseline import load_cases, run_benchmark
    from adaptivedepth.model.loading import load_model

    settings = load_project_settings(config)
    routed_model, tokenizer, device = load_model(settings.model)
    library = RouteLibrary.load(routes_file)
    library.validate_for_depth(routed_model.num_layers)
    cases = load_cases(dataset)
    reports = []
    measurements = []
    prompts = [case.prompt for case in cases]
    for route in library.routes:
        report = run_benchmark(
            routed_model,
            tokenizer,
            device,
            cases,
            settings.benchmark,
            route,
        )
        teacher_divergence = mean_teacher_kl(
            routed_model,
            tokenizer,
            prompts,
            device,
            route,
            settings.benchmark.use_chat_template,
        )
        reports.append(report.to_dict())
        measurements.append(
            {
                "route_id": route.route_id,
                "quality": report.accuracy,
                "latency_ms": report.median_total_ms,
                "teacher_divergence": teacher_divergence,
                "peak_memory_mb": report.peak_memory_mb,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"measurements": measurements, "reports": reports}, indent=2),
        encoding="utf-8",
    )
    typer.echo(f"Saved {len(reports)} route benchmarks to {output}")


@app.command("pareto-filter")
def pareto_filter(
    measurements_file: Annotated[Path, typer.Option(exists=True)],
    output: Annotated[Path, typer.Option()] = Path("experiments/pareto_routes.json"),
) -> None:
    payload = json.loads(measurements_file.read_text(encoding="utf-8"))
    values = payload["measurements"] if isinstance(payload, dict) else payload
    measurements = [RouteMeasurement.model_validate(item) for item in values]
    frontier = pareto_frontier(measurements)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([item.model_dump(mode="json") for item in frontier], indent=2),
        encoding="utf-8",
    )
    typer.echo(f"Saved {len(frontier)} Pareto routes to {output}")


@app.command("build-static-suite")
def build_static_suite(
    influence_file: Annotated[Path, typer.Option(exists=True)],
    candidates_file: Annotated[Path, typer.Option(exists=True)],
    compiler_config: Annotated[Path, typer.Option(exists=True)] = Path("configs/compiler.yaml"),
    evaluation_config: Annotated[Path, typer.Option(exists=True)] = Path("configs/evaluation.yaml"),
    output: Annotated[Path, typer.Option()] = Path("experiments/static_suite.yaml"),
) -> None:
    from adaptivedepth.compiler.static_suite import build_static_route_suite

    payload = json.loads(influence_file.read_text(encoding="utf-8"))
    influence = {
        int(item["layer_index"]): float(item["influence"])
        for item in payload["layers"]
    }
    compiler = load_compiler_settings(compiler_config)
    evaluation = load_evaluation_settings(evaluation_config)
    suite = build_static_route_suite(
        candidates=RouteLibrary.load(candidates_file),
        num_layers=len(influence),
        influence=influence,
        budgets=compiler.budgets,
        preserve_first=compiler.preserve_first,
        preserve_last=compiler.preserve_last,
        random_routes_per_budget=evaluation.random_routes_per_budget,
        seed=compiler.seed,
    )
    suite.save(output)
    typer.echo(f"Saved {len(suite.routes)} candidate and baseline routes to {output}")


@app.command("run-static-experiment")
def run_static_experiment_command(
    routes_file: Annotated[Path, typer.Option(exists=True)],
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/model.yaml"),
    dataset: Annotated[Path, typer.Option(exists=True)] = Path("data/phase4_eval.jsonl"),
    output: Annotated[Path, typer.Option()] = Path("experiments/phase4/experiment.json"),
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
) -> None:
    from adaptivedepth.evaluation.experiment import run_static_experiment
    from adaptivedepth.model.loading import load_model

    settings = load_project_settings(config)
    routed_model, tokenizer, device = load_model(settings.model)
    run_static_experiment(
        routed_model=routed_model,
        tokenizer=tokenizer,
        device=device,
        model_id=settings.model.model_id,
        dataset_path=dataset,
        settings=settings.benchmark,
        routes=RouteLibrary.load(routes_file),
        output=output,
        resume=resume,
        workdir=Path.cwd(),
        progress=typer.echo,
    )
    typer.echo(f"Saved resumable experiment to {output}")


@app.command("evaluate-static-experiment")
def evaluate_static_experiment_command(
    experiment_file: Annotated[Path, typer.Option(exists=True)],
    evaluation_config: Annotated[Path, typer.Option(exists=True)] = Path("configs/evaluation.yaml"),
    output_json: Annotated[Path, typer.Option()] = Path("experiments/phase4/gate.json"),
    output_markdown: Annotated[Path, typer.Option()] = Path("experiments/phase4/gate.md"),
) -> None:
    from adaptivedepth.evaluation.gate import evaluate_static_gate, save_gate_report

    experiment = json.loads(experiment_file.read_text(encoding="utf-8"))
    report = evaluate_static_gate(
        experiment, load_evaluation_settings(evaluation_config)
    )
    save_gate_report(report, output_json, output_markdown)
    typer.echo(f"Phase 4 verdict: {report['verdict']}")
    typer.echo(f"Saved gate reports to {output_json} and {output_markdown}")


@app.command("rescore-static-experiment")
def rescore_static_experiment_command(
    experiment_file: Annotated[Path, typer.Option(exists=True)],
    evaluation_config: Annotated[Path, typer.Option(exists=True)] = Path("configs/evaluation.yaml"),
    output_experiment: Annotated[Path, typer.Option()] = Path("experiments/phase4/experiment_rescored.json"),
    output_json: Annotated[Path, typer.Option()] = Path("experiments/phase4/gate_rescored.json"),
    output_markdown: Annotated[Path, typer.Option()] = Path("experiments/phase4/gate_rescored.md"),
) -> None:
    """Rescore saved predictions and rebuild the gate without GPU inference."""

    from adaptivedepth.evaluation.gate import evaluate_static_gate, save_gate_report
    from adaptivedepth.evaluation.rescore import rescore_experiment

    experiment = rescore_experiment(
        json.loads(experiment_file.read_text(encoding="utf-8"))
    )
    output_experiment.parent.mkdir(parents=True, exist_ok=True)
    output_experiment.write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    report = evaluate_static_gate(
        experiment, load_evaluation_settings(evaluation_config)
    )
    save_gate_report(report, output_json, output_markdown)
    typer.echo(f"Rescored Phase 4 verdict: {report['verdict']}")
    typer.echo(f"Saved rescored experiment to {output_experiment}")
    typer.echo(f"Saved rescored gate reports to {output_json} and {output_markdown}")


@app.command("run-phase4")
def run_phase4(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/cloud_t4.yaml"),
    dataset: Annotated[Path, typer.Option(exists=True)] = Path("data/phase4_eval.jsonl"),
    compiler_config: Annotated[Path, typer.Option(exists=True)] = Path("configs/compiler.yaml"),
    evaluation_config: Annotated[Path, typer.Option(exists=True)] = Path("configs/evaluation.yaml"),
    workspace: Annotated[Path, typer.Option()] = Path("experiments/phase4"),
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
) -> None:
    """Run the complete cloud-ready Phase 4 pipeline with one model load."""

    from adaptivedepth.analysis.block_influence import BlockInfluenceAnalyzer
    from adaptivedepth.compiler.static_suite import build_static_route_suite
    from adaptivedepth.evaluation.baseline import load_cases
    from adaptivedepth.evaluation.experiment import run_static_experiment
    from adaptivedepth.evaluation.gate import evaluate_static_gate, save_gate_report
    from adaptivedepth.model.loading import load_model

    workspace.mkdir(parents=True, exist_ok=True)
    influence_path = workspace / "block_influence.json"
    candidates_path = workspace / "candidate_routes.yaml"
    suite_path = workspace / "static_suite.yaml"
    experiment_path = workspace / "experiment.json"
    gate_json_path = workspace / "gate.json"
    gate_markdown_path = workspace / "gate.md"

    project = load_project_settings(config)
    compiler = load_compiler_settings(compiler_config)
    evaluation = load_evaluation_settings(evaluation_config)
    routed_model, tokenizer, device = load_model(project.model)
    cases = load_cases(dataset)

    if resume and influence_path.exists():
        typer.echo("Reusing saved block-influence measurements")
        influence_payload = json.loads(influence_path.read_text(encoding="utf-8"))
    else:
        typer.echo("Measuring block influence")
        scores = BlockInfluenceAnalyzer(routed_model).analyze_prompts(
            tokenizer,
            [case.prompt for case in cases],
            device,
            project.benchmark.use_chat_template,
        )
        influence_payload = {
            "model_id": project.model.model_id,
            "layers": [score.to_dict() for score in scores],
        }
        influence_path.write_text(
            json.dumps(influence_payload, indent=2), encoding="utf-8"
        )

    influence = {
        int(item["layer_index"]): float(item["influence"])
        for item in influence_payload["layers"]
    }
    candidates = RouteLibrary(
        routes=generate_candidate_routes(
            num_layers=routed_model.num_layers,
            influence=influence,
            budgets=compiler.budgets,
            preserve_first=compiler.preserve_first,
            preserve_last=compiler.preserve_last,
            variants_per_budget=compiler.variants_per_budget,
            seed=compiler.seed,
        )
    )
    candidates.save(candidates_path)
    suite = build_static_route_suite(
        candidates=candidates,
        num_layers=routed_model.num_layers,
        influence=influence,
        budgets=compiler.budgets,
        preserve_first=compiler.preserve_first,
        preserve_last=compiler.preserve_last,
        random_routes_per_budget=evaluation.random_routes_per_budget,
        seed=compiler.seed,
    )
    suite.save(suite_path)
    typer.echo(f"Static suite contains {len(suite.routes)} routes")

    experiment = run_static_experiment(
        routed_model=routed_model,
        tokenizer=tokenizer,
        device=device,
        model_id=project.model.model_id,
        dataset_path=dataset,
        settings=project.benchmark,
        routes=suite,
        output=experiment_path,
        resume=resume,
        workdir=Path.cwd(),
        progress=typer.echo,
    )
    gate = evaluate_static_gate(experiment, evaluation)
    save_gate_report(gate, gate_json_path, gate_markdown_path)
    typer.echo(f"Phase 4 verdict: {gate['verdict']}")
    typer.echo(f"All artifacts saved under {workspace}")


@app.command("run-phase5")
def run_phase5_command(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/phase5_t4.yaml"),
    routes_file: Annotated[Path, typer.Option(exists=True)] = Path("configs/phase5_routes.yaml"),
    dataset: Annotated[Path, typer.Option(exists=True)] = Path("data/phase5_train.jsonl"),
    output: Annotated[Path, typer.Option()] = Path("experiments/phase5"),
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
) -> None:
    """Train a route-robust student against a frozen full-depth teacher."""

    from adaptivedepth.model.loading import load_teacher_student
    from adaptivedepth.training import (
        find_latest_checkpoint,
        load_training_examples,
        train_phase5,
    )

    settings = load_phase5_settings(config)
    routes = RouteLibrary.load(routes_file)
    checkpoint = find_latest_checkpoint(output) if resume else None
    if checkpoint is not None:
        typer.echo(f"Found resumable checkpoint: {checkpoint}")
    teacher, student, tokenizer, device = load_teacher_student(
        settings.model,
        student_model_id=str(checkpoint) if checkpoint is not None else None,
    )
    summary = train_phase5(
        teacher=teacher,
        student=student,
        tokenizer=tokenizer,
        device=device,
        model_id=settings.model.model_id,
        dataset_path=dataset,
        examples=load_training_examples(dataset),
        routes=routes,
        settings=settings.training,
        output_dir=output,
        resume_checkpoint=checkpoint,
        progress=typer.echo,
    )
    typer.echo(f"Phase 5 complete: {summary['final_checkpoint']}")


@app.command("run-phase5-correction")
def run_phase5_correction_command(
    checkpoint: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/phase5_correction_t4.yaml"),
    routes_file: Annotated[Path, typer.Option(exists=True)] = Path("configs/phase5_correction_route.yaml"),
    dataset: Annotated[Path, typer.Option(exists=True)] = Path("data/phase5_correction_train.jsonl"),
    output: Annotated[Path, typer.Option()] = Path("experiments/phase5_correction"),
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
) -> None:
    """Fine-tune one target route from an existing Phase 5 checkpoint."""

    from adaptivedepth.model.loading import load_teacher_student
    from adaptivedepth.training import (
        find_latest_checkpoint,
        load_training_examples,
        train_phase5,
    )

    settings = load_phase5_settings(config)
    routes = RouteLibrary.load(routes_file)
    if len(routes.routes) != 1:
        raise typer.BadParameter(
            "the correction run requires exactly one target route",
            param_hint="--routes-file",
        )
    base_checkpoint = checkpoint.resolve()
    latest = find_latest_checkpoint(output) if resume else None
    if latest is not None:
        typer.echo(f"Found resumable correction checkpoint: {latest}")
    local_model = settings.model.model_copy(update={"local_files_only": True})
    teacher, student, tokenizer, device = load_teacher_student(
        local_model,
        teacher_model_id=str(base_checkpoint),
        student_model_id=str(latest or base_checkpoint),
    )
    summary = train_phase5(
        teacher=teacher,
        student=student,
        tokenizer=tokenizer,
        device=device,
        model_id=str(base_checkpoint),
        dataset_path=dataset,
        examples=load_training_examples(dataset),
        routes=routes,
        settings=settings.training,
        output_dir=output,
        resume_checkpoint=latest,
        progress=typer.echo,
    )
    typer.echo(f"Phase 5 correction complete: {summary['final_checkpoint']}")


@app.command("prepare-phase5-data")
def prepare_phase5_data_command(
    output: Annotated[Path, typer.Option()] = Path("data/phase5_train.jsonl"),
    max_examples: Annotated[int, typer.Option(min=1)] = 2000,
    seed: Annotated[int, typer.Option()] = 17,
) -> None:
    """Create the separate English Phase 5 training set from Dolly 15k."""

    from adaptivedepth.training.prepare import prepare_dolly_dataset

    count = prepare_dolly_dataset(output, max_examples=max_examples, seed=seed)
    typer.echo(f"Saved {count} training examples to {output}")


@app.command("prepare-phase5-correction-data")
def prepare_phase5_correction_data_command(
    output: Annotated[Path, typer.Option()] = Path("data/phase5_correction_train.jsonl"),
    dolly_examples: Annotated[int, typer.Option(min=1)] = 300,
    balanced_examples: Annotated[int, typer.Option(min=1)] = 200,
    skip_dolly_examples: Annotated[int, typer.Option(min=0)] = 2000,
    seed: Annotated[int, typer.Option()] = 17,
) -> None:
    """Create held-out and balanced data for the bounded Phase 5 correction."""

    from adaptivedepth.training.prepare import prepare_phase5_correction_dataset

    count = prepare_phase5_correction_dataset(
        output,
        dolly_examples=dolly_examples,
        balanced_examples=balanced_examples,
        skip_dolly_examples=skip_dolly_examples,
        seed=seed,
    )
    typer.echo(f"Saved {count} correction examples to {output}")


@app.command("run-phase5-evaluation")
def run_phase5_evaluation_command(
    checkpoint: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/cloud_t4.yaml"),
    routes_file: Annotated[Path, typer.Option(exists=True)] = Path("configs/phase5_routes.yaml"),
    dataset: Annotated[Path, typer.Option(exists=True)] = Path("data/phase4_eval.jsonl"),
    reference_file: Annotated[Path, typer.Option(exists=True)] = Path("data/phase4_reference_metrics.json"),
    evaluation_config: Annotated[Path, typer.Option(exists=True)] = Path("configs/phase5_evaluation.yaml"),
    workspace: Annotated[Path, typer.Option()] = Path("experiments/phase5_evaluation"),
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
) -> None:
    """Evaluate the trained student and decide whether Phase 6 may begin."""

    from adaptivedepth.evaluation.experiment import run_static_experiment
    from adaptivedepth.evaluation.post_training import (
        evaluate_post_training_gate,
        save_post_training_report,
    )
    from adaptivedepth.model.loading import load_model

    workspace.mkdir(parents=True, exist_ok=True)
    experiment_path = workspace / "experiment.json"
    gate_json_path = workspace / "gate.json"
    gate_markdown_path = workspace / "gate.md"

    project = load_project_settings(config)
    checkpoint_path = checkpoint.resolve()
    model_settings = project.model.model_copy(
        update={"model_id": str(checkpoint_path), "local_files_only": True}
    )
    routed_model, tokenizer, device = load_model(model_settings)
    routes = RouteLibrary.load(routes_file)
    experiment = run_static_experiment(
        routed_model=routed_model,
        tokenizer=tokenizer,
        device=device,
        model_id=str(checkpoint_path),
        dataset_path=dataset,
        settings=project.benchmark,
        routes=routes,
        output=experiment_path,
        resume=resume,
        workdir=Path.cwd(),
        progress=typer.echo,
    )
    reference = json.loads(reference_file.read_text(encoding="utf-8"))
    gate = evaluate_post_training_gate(
        experiment,
        reference,
        load_phase5_evaluation_settings(evaluation_config),
    )
    save_post_training_report(gate, gate_json_path, gate_markdown_path)
    typer.echo(f"Phase 5 post-training verdict: {gate['verdict']}")
    typer.echo(f"Saved evaluation artifacts under {workspace}")


@app.command("prepare-phase6-data")
def prepare_phase6_data_command(
    output: Annotated[Path, typer.Option()] = Path("data/phase6_calibration.jsonl"),
    max_examples: Annotated[int, typer.Option(min=1)] = 200,
    skip_examples: Annotated[int, typer.Option(min=0)] = 2300,
    seed: Annotated[int, typer.Option()] = 17,
) -> None:
    """Prepare held-out English prompts for compact route-risk collection."""

    from adaptivedepth.training.prepare import prepare_dolly_dataset

    count = prepare_dolly_dataset(
        output,
        max_examples=max_examples,
        seed=seed,
        skip_examples=skip_examples,
    )
    typer.echo(f"Saved {count} held-out Phase 6 examples to {output}")


@app.command("run-phase6")
def run_phase6_command(
    checkpoint: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/phase6_t4.yaml"),
    route_file: Annotated[Path, typer.Option(exists=True)] = Path("configs/phase6_route.yaml"),
    dataset: Annotated[Path, typer.Option(exists=True)] = Path("data/phase6_calibration.jsonl"),
    workspace: Annotated[Path, typer.Option()] = Path("experiments/phase6"),
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
) -> None:
    """Collect a compact full-vs-route regret dataset for the application router."""

    from adaptivedepth.model.loading import load_model
    from adaptivedepth.routing.dataset import collect_phase6_dataset
    from adaptivedepth.training import load_training_examples

    settings = load_phase6_settings(config)
    routes = RouteLibrary.load(route_file)
    if len(routes.routes) != 1:
        raise typer.BadParameter(
            "Phase 6 requires exactly one experimental fast route",
            param_hint="--route-file",
        )
    checkpoint_path = checkpoint.resolve()
    model_settings = settings.model.model_copy(
        update={"model_id": str(checkpoint_path), "local_files_only": True}
    )
    routed_model, tokenizer, device = load_model(model_settings)
    _experiment, report = collect_phase6_dataset(
        routed_model=routed_model,
        tokenizer=tokenizer,
        device=device,
        model_id=str(checkpoint_path),
        dataset_path=dataset,
        examples=load_training_examples(dataset),
        route=routes.routes[0],
        settings=settings.calibration,
        workspace=workspace,
        resume=resume,
        workdir=Path.cwd(),
        progress=typer.echo,
    )
    typer.echo(f"Phase 6 verdict: {report['verdict']}")
    typer.echo(f"Saved Phase 6 artifacts under {workspace}")


@app.command("prepare-phase61-data")
def prepare_phase61_data_command(
    output: Annotated[Path, typer.Option()] = Path("data/phase61_calibration.jsonl"),
    max_examples: Annotated[int, typer.Option(min=1)] = 400,
    skip_examples: Annotated[int, typer.Option(min=0)] = 2500,
    seed: Annotated[int, typer.Option()] = 17,
) -> None:
    """Prepare fresh held-out prompts for response-preserving multi-route collection."""

    from adaptivedepth.training.prepare import prepare_dolly_dataset

    count = prepare_dolly_dataset(
        output,
        max_examples=max_examples,
        seed=seed,
        skip_examples=skip_examples,
    )
    typer.echo(f"Saved {count} held-out Phase 6.1 examples to {output}")


@app.command("run-phase61")
def run_phase61_command(
    checkpoint: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/phase61_t4.yaml"),
    routes_file: Annotated[Path, typer.Option(exists=True)] = Path(
        "configs/phase61_routes.yaml"
    ),
    dataset: Annotated[Path, typer.Option(exists=True)] = Path(
        "data/phase61_calibration.jsonl"
    ),
    workspace: Annotated[Path, typer.Option()] = Path("experiments/phase61"),
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
) -> None:
    """Collect response-preserving regret labels for four reduced routes."""

    from adaptivedepth.model.loading import load_model
    from adaptivedepth.routing.multiroute_dataset import collect_phase61_dataset
    from adaptivedepth.training import load_training_examples

    settings = load_phase61_settings(config)
    routes = RouteLibrary.load(routes_file)
    layer_counts = {len(route.active_layers) for route in routes.routes}
    if len(routes.routes) != 4 or layer_counts != {20, 21, 22, 23}:
        raise typer.BadParameter(
            "Phase 6.1 requires exactly the 20-, 21-, 22-, and 23-layer routes",
            param_hint="--routes-file",
        )
    checkpoint_path = checkpoint.resolve()
    model_settings = settings.model.model_copy(
        update={"model_id": str(checkpoint_path), "local_files_only": True}
    )
    routed_model, tokenizer, device = load_model(model_settings)
    _experiment, report = collect_phase61_dataset(
        routed_model=routed_model,
        tokenizer=tokenizer,
        device=device,
        model_id=str(checkpoint_path),
        dataset_path=dataset,
        examples=load_training_examples(dataset),
        routes=routes.routes,
        settings=settings.calibration,
        workspace=workspace,
        resume=resume,
        workdir=Path.cwd(),
        progress=typer.echo,
    )
    typer.echo(f"Phase 6.1 verdict: {report['verdict']}")
    typer.echo(f"Saved Phase 6.1 artifacts under {workspace}")


@app.command("train-phase7")
def train_phase7_command(
    dataset: Annotated[Path, typer.Option(exists=True)] = Path(
        "phase6/router_dataset.jsonl"
    ),
    phase6_gate: Annotated[Path, typer.Option(exists=True)] = Path(
        "phase6/gate.json"
    ),
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/phase7.yaml"),
    output: Annotated[Path, typer.Option()] = Path("experiments/phase7"),
) -> None:
    """Train and evaluate the conservative Phase 7 risk router."""

    from adaptivedepth.routing.risk_router import train_phase7_router

    report = train_phase7_router(
        dataset_path=dataset,
        phase6_gate_path=phase6_gate,
        settings=load_phase7_settings(config),
        output_dir=output,
        config_path=config,
    )
    typer.echo(f"Phase 7 verdict: {report['verdict']}")
    typer.echo(f"Router: {output / 'router.joblib'}")
    typer.echo(f"Report: {output / 'gate.md'}")


@app.command("train-phase71")
def train_phase71_command(
    dataset: Annotated[Path, typer.Option(exists=True)] = Path(
        "phase61/router_dataset.jsonl"
    ),
    phase61_gate: Annotated[Path, typer.Option(exists=True)] = Path(
        "phase61/gate.json"
    ),
    config: Annotated[Path, typer.Option(exists=True)] = Path(
        "configs/phase71.yaml"
    ),
    output: Annotated[Path, typer.Option()] = Path("experiments/phase71"),
) -> None:
    """Train and evaluate the conservative Phase 7.1 multi-route router."""

    from adaptivedepth.routing.multiroute_risk_router import train_phase71_router

    report = train_phase71_router(
        dataset_path=dataset,
        phase61_gate_path=phase61_gate,
        settings=load_phase71_settings(config),
        output_dir=output,
        config_path=config,
    )
    typer.echo(f"Phase 7.1 verdict: {report['verdict']}")
    typer.echo(f"Router: {output / 'router.joblib'}")
    typer.echo(f"Report: {output / 'gate.md'}")


@app.command("train-phase72")
def train_phase72_command(
    dataset: Annotated[Path, typer.Option(exists=True)] = Path("phase61/router_dataset.jsonl"),
    phase61_gate: Annotated[Path, typer.Option(exists=True)] = Path("phase61/gate.json"),
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/phase72.yaml"),
    output: Annotated[Path, typer.Option()] = Path("experiments/phase72"),
) -> None:
    """Correct the multi-route router with shared hashed features and joint thresholds."""
    from adaptivedepth.routing.phase72_router import train_phase72_router

    report = train_phase72_router(
        dataset_path=dataset, phase61_gate_path=phase61_gate,
        settings=load_phase72_settings(config), output_dir=output,
        config_path=config,
    )
    typer.echo(f"Phase 7.2 verdict: {report['verdict']}")
    typer.echo(f"Report: {output / 'gate.md'}")


@app.command("train-phase8")
def train_phase8_command(
    dataset: Annotated[Path, typer.Option(exists=True)] = Path("phase61/router_dataset.jsonl"),
    phase72_report: Annotated[Path, typer.Option(exists=True)] = Path("experiments/phase72/report.json"),
    phase72_oof: Annotated[Path, typer.Option(exists=True)] = Path("experiments/phase72/oof_predictions.jsonl"),
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/phase8.yaml"),
    output: Annotated[Path, typer.Option()] = Path("experiments/phase8"),
) -> None:
    """Train the Phase 8 post-route rescue predictor without GPU compute."""
    from adaptivedepth.routing.rescue import train_phase8_rescue

    report = train_phase8_rescue(
        dataset_path=dataset, phase72_report_path=phase72_report,
        phase72_oof_path=phase72_oof, settings=load_phase8_settings(config),
        output_dir=output, config_path=config,
    )
    typer.echo(f"Phase 8 verdict: {report['verdict']}")
    typer.echo(f"Report: {output / 'gate.md'}")


@app.command("optimize-phase72")
def optimize_phase72_command(
    dataset: Annotated[Path, typer.Option(exists=True)] = Path("phase61/router_dataset.jsonl"),
    router_oof: Annotated[Path, typer.Option(exists=True)] = Path("experiments/phase72/oof_predictions.jsonl"),
    rescue_oof: Annotated[Path, typer.Option(exists=True)] = Path("experiments/phase8/oof_route_predictions.jsonl"),
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/phase72_application.yaml"),
    output: Annotated[Path, typer.Option()] = Path("experiments/phase72_application"),
) -> None:
    """Jointly optimize the locked Route 20/23 candidate and rescue policy."""
    from adaptivedepth.routing.cascade_optimizer import optimize_two_route_cascade

    report = optimize_two_route_cascade(
        dataset_path=dataset, router_oof_path=router_oof,
        rescue_oof_path=rescue_oof,
        settings=load_phase72_application_settings(config),
        output_dir=output, config_path=config,
    )
    typer.echo(f"Phase 7.2 application verdict: {report['verdict']}")
    typer.echo(f"Report: {output / 'gate.md'}")


@app.command("prepare-phase72-label-review")
def prepare_phase72_label_review_command(
    dataset: Annotated[Path, typer.Option(exists=True)] = Path("phase61/router_dataset.jsonl"),
    output: Annotated[Path, typer.Option()] = Path("experiments/phase72_label_review.jsonl"),
) -> None:
    """Export ambiguous Route 20/23 labels for semantic review."""
    from adaptivedepth.routing.application_labels import prepare_application_label_review

    summary = prepare_application_label_review(
        dataset_path=dataset, output_path=output
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command("apply-phase72-label-review")
def apply_phase72_label_review_command(
    dataset: Annotated[Path, typer.Option(exists=True)] = Path("phase61/router_dataset.jsonl"),
    judgments: Annotated[Path, typer.Option(exists=True)] = Path("experiments/phase72_label_judgments.jsonl"),
    output: Annotated[Path, typer.Option()] = Path("experiments/phase72_application_dataset.jsonl"),
    minimum_safe_confidence: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.80,
) -> None:
    """Apply complete reviewed judgments and build an application dataset."""
    from adaptivedepth.routing.application_labels import apply_application_label_review

    summary = apply_application_label_review(
        dataset_path=dataset, judgments_path=judgments, output_path=output,
        minimum_safe_confidence=minimum_safe_confidence,
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command("judge-phase72-label-review")
def judge_phase72_label_review_command(
    review: Annotated[Path, typer.Option(exists=True)] = Path("experiments/phase72_label_review.jsonl"),
    output: Annotated[Path, typer.Option()] = Path("experiments/phase72_label_judgments.jsonl"),
    model_id: Annotated[str, typer.Option()] = "Qwen/Qwen2.5-3B-Instruct",
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
) -> None:
    """Judge ambiguous Phase 7.2 labels on a CUDA/Colab runtime."""
    from adaptivedepth.routing.application_labels import judge_application_label_review

    summary = judge_application_label_review(
        review_path=review, output_path=output, model_id=model_id,
        resume=resume, progress=typer.echo,
    )
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
