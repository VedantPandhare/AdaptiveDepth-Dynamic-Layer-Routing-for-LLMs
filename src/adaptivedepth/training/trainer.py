from __future__ import annotations

import json
import random
import shutil
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from adaptivedepth.config import TrainingSettings
from adaptivedepth.model.routed_model import RoutedCausalLM
from adaptivedepth.routes import RouteDefinition, RouteLibrary
from adaptivedepth.training.data import (
    TrainingExample,
    collate_examples,
    encode_example,
    sha256_file,
)
from adaptivedepth.training.objective import route_training_loss


@dataclass(frozen=True)
class SamplePlan:
    example_indexes: list[int]
    route: RouteDefinition | None

    @property
    def route_id(self) -> str:
        return "full" if self.route is None else self.route.route_id


def sample_plan(
    micro_step: int,
    num_examples: int,
    batch_size: int,
    routes: RouteLibrary,
    full_route_probability: float,
    seed: int,
) -> SamplePlan:
    if num_examples < 1:
        raise ValueError("num_examples must be positive")
    if not routes.routes:
        raise ValueError("route library must not be empty")
    rng = random.Random(seed * 1_000_003 + micro_step)
    indexes = [rng.randrange(num_examples) for _ in range(batch_size)]
    route = None if rng.random() < full_route_probability else rng.choice(routes.routes)
    return SamplePlan(indexes, route)


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir)
    checkpoints: list[tuple[int, Path]] = []
    if not root.exists():
        return None
    for candidate in root.glob("checkpoint-step-*"):
        if not candidate.is_dir() or not (candidate / "complete.json").exists():
            continue
        try:
            step = int(candidate.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        checkpoints.append((step, candidate))
    return max(checkpoints, default=(0, None), key=lambda item: item[0])[1]


def _identity(
    model_id: str,
    dataset_path: Path,
    routes: RouteLibrary,
    settings: TrainingSettings,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "dataset_sha256": sha256_file(dataset_path),
        "routes": routes.model_dump(mode="json"),
        "training": settings.model_dump(mode="json"),
    }


def _autocast(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _save_checkpoint(
    output_dir: Path,
    step: int,
    student: RoutedCausalLM,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    identity: dict[str, Any],
    keep_last: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / f"checkpoint-step-{step:06d}"
    if final.exists():
        return final
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=output_dir))
    try:
        student.model.save_pretrained(temporary, safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        torch.save(
            {
                "step": step,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "identity": identity,
            },
            temporary / "training_state.pt",
        )
        (temporary / "complete.json").write_text(
            json.dumps({"step": step}, indent=2), encoding="utf-8"
        )
        temporary.replace(final)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    completed = sorted(
        (
            candidate
            for candidate in output_dir.glob("checkpoint-step-*")
            if candidate.is_dir() and (candidate / "complete.json").exists()
        ),
        key=lambda path: path.name,
    )
    for expired in completed[:-keep_last]:
        shutil.rmtree(expired)
    return final


def _scheduler_lambda(step: int, warmup_steps: int, max_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    remaining = max(max_steps - warmup_steps, 1)
    return max(0.0, float(max_steps - step) / float(remaining))


def train_phase5(
    *,
    teacher: RoutedCausalLM,
    student: RoutedCausalLM,
    tokenizer: Any,
    device: torch.device,
    model_id: str,
    dataset_path: str | Path,
    examples: list[TrainingExample],
    routes: RouteLibrary,
    settings: TrainingSettings,
    output_dir: str | Path,
    resume_checkpoint: str | Path | None = None,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    identity = _identity(model_id, dataset_path, routes, settings)
    routes.validate_for_depth(student.num_layers)
    if teacher.num_layers != student.num_layers:
        raise ValueError("teacher and student depths differ")

    teacher.model.eval()
    teacher.model.requires_grad_(False)
    student.model.train()
    if hasattr(student.model.config, "use_cache"):
        student.model.config.use_cache = False
    if settings.gradient_checkpointing and hasattr(student.model, "gradient_checkpointing_enable"):
        student.model.gradient_checkpointing_enable()

    encoded = [encode_example(tokenizer, item, settings.max_seq_length) for item in examples]
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer has no pad token")

    optimizer = torch.optim.AdamW(
        student.model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _scheduler_lambda(step, settings.warmup_steps, settings.max_steps),
    )
    parameter_dtype = next(student.model.parameters()).dtype
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and parameter_dtype == torch.float16
    )
    start_step = 0
    if resume_checkpoint is not None:
        state_path = Path(resume_checkpoint) / "training_state.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("checkpoint identity does not match this Phase 5 run")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_step = int(state["step"])
        progress(f"Resuming after optimizer step {start_step}")

    history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step, settings.max_steps):
        accumulated_total = 0.0
        accumulated_lm = 0.0
        accumulated_kd = 0.0
        route_ids: list[str] = []
        for accumulation_index in range(settings.gradient_accumulation_steps):
            micro_step = step * settings.gradient_accumulation_steps + accumulation_index
            plan = sample_plan(
                micro_step,
                len(encoded),
                settings.batch_size,
                routes,
                settings.full_route_probability,
                settings.seed,
            )
            route_ids.append(plan.route_id)
            batch = collate_examples(
                [encoded[index] for index in plan.example_indexes], pad_token_id, device
            )
            model_inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "use_cache": False,
            }
            with torch.no_grad(), _autocast(device, parameter_dtype):
                teacher_logits = teacher(**model_inputs).logits
            with student.use_route(plan.route), _autocast(device, parameter_dtype):
                student_logits = student(**model_inputs).logits
                losses = route_training_loss(
                    student_logits,
                    teacher_logits,
                    batch["labels"],
                    batch["attention_mask"],
                    settings.language_model_weight,
                    settings.distillation_weight,
                    settings.distillation_temperature,
                )
                scaled_loss = losses.total / settings.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            accumulated_total += float(losses.total.detach())
            accumulated_lm += float(losses.language_model.detach())
            accumulated_kd += float(losses.distillation.detach())
            del teacher_logits, student_logits, losses, batch

        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            student.model.parameters(), settings.gradient_clip_norm
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        completed_step = step + 1
        divisor = settings.gradient_accumulation_steps
        record = {
            "step": completed_step,
            "loss": accumulated_total / divisor,
            "language_model_loss": accumulated_lm / divisor,
            "distillation_loss": accumulated_kd / divisor,
            "gradient_norm": float(gradient_norm),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "routes": route_ids,
        }
        history.append(record)
        if completed_step % settings.log_every == 0 or completed_step == settings.max_steps:
            progress(
                f"step={completed_step}/{settings.max_steps} "
                f"loss={record['loss']:.4f} routes={','.join(route_ids)}"
            )
        if completed_step % settings.checkpoint_every == 0:
            checkpoint = _save_checkpoint(
                output_dir,
                completed_step,
                student,
                tokenizer,
                optimizer,
                scheduler,
                scaler,
                identity,
                settings.keep_last_checkpoints,
            )
            progress(f"Saved {checkpoint}")

    final_step = settings.max_steps
    final_checkpoint = _save_checkpoint(
        output_dir,
        final_step,
        student,
        tokenizer,
        optimizer,
        scheduler,
        scaler,
        identity,
        settings.keep_last_checkpoints,
    )
    summary = {
        "status": "completed",
        "start_step": start_step,
        "final_step": final_step,
        "final_checkpoint": str(final_checkpoint),
        "new_history": history,
        "identity": identity,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
