from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "auto"
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    trust_remote_code: bool = False
    local_files_only: bool = False
    layer_path: str | None = None


class BenchmarkSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_new_tokens: int = Field(default=32, ge=1, le=2048)
    warmup_runs: int = Field(default=1, ge=0)
    measured_runs: int = Field(default=3, ge=1)
    seed: int = 17
    use_chat_template: bool = True
    fixed_generation_length: bool = True


class ProjectSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelSettings = Field(default_factory=ModelSettings)
    benchmark: BenchmarkSettings = Field(default_factory=BenchmarkSettings)


class CompilerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    budgets: list[int] = Field(default_factory=lambda: [12, 14, 16, 18, 20])
    preserve_first: int = Field(default=2, ge=0)
    preserve_last: int = Field(default=2, ge=0)
    variants_per_budget: int = Field(default=4, ge=1)
    seed: int = 17


class EvaluationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_quality_drop: float = Field(default=0.05, ge=0, le=1)
    min_latency_reduction: float = Field(default=0.10, ge=0, lt=1)
    max_teacher_divergence: float = Field(default=0.50, ge=0)
    random_routes_per_budget: int = Field(default=2, ge=1)
    require_candidate_advantage: bool = True


class TrainingSettings(BaseModel):
    """Resource-conscious settings for route-aware teacher/student training."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 17
    max_seq_length: int = Field(default=128, ge=16, le=2048)
    batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    max_steps: int = Field(default=500, ge=1)
    learning_rate: float = Field(default=2e-5, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_steps: int = Field(default=25, ge=0)
    gradient_clip_norm: float = Field(default=1.0, gt=0)
    language_model_weight: float = Field(default=0.3, ge=0)
    distillation_weight: float = Field(default=0.7, ge=0)
    distillation_temperature: float = Field(default=2.0, gt=0)
    full_route_probability: float = Field(default=0.2, ge=0, le=1)
    checkpoint_every: int = Field(default=50, ge=1)
    keep_last_checkpoints: int = Field(default=2, ge=1, le=10)
    log_every: int = Field(default=5, ge=1)
    gradient_checkpointing: bool = True

    @model_validator(mode="after")
    def validate_objective(self) -> "TrainingSettings":
        if self.language_model_weight + self.distillation_weight <= 0:
            raise ValueError("at least one training loss weight must be positive")
        if self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be smaller than max_steps")
        return self


class Phase5Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelSettings = Field(default_factory=ModelSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)


class Phase5EvaluationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_full_quality_drop: float = Field(default=0.05, ge=0, le=1)
    max_route_quality_drop: float = Field(default=0.05, ge=0, le=1)
    min_latency_reduction: float = Field(default=0.10, ge=0, lt=1)
    max_teacher_divergence: float = Field(default=0.50, ge=0)


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    return value or {}


def load_project_settings(path: str | Path) -> ProjectSettings:
    return ProjectSettings.model_validate(_load_yaml(path))


def load_compiler_settings(path: str | Path) -> CompilerSettings:
    return CompilerSettings.model_validate(_load_yaml(path))


def load_evaluation_settings(path: str | Path) -> EvaluationSettings:
    return EvaluationSettings.model_validate(_load_yaml(path))


def load_phase5_settings(path: str | Path) -> Phase5Settings:
    return Phase5Settings.model_validate(_load_yaml(path))


def load_phase5_evaluation_settings(path: str | Path) -> Phase5EvaluationSettings:
    return Phase5EvaluationSettings.model_validate(_load_yaml(path))
