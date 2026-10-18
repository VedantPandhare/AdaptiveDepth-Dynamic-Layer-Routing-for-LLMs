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


class Phase6CalibrationSettings(BaseModel):
    """Compact route-regret collection settings for the application track."""

    model_config = ConfigDict(extra="forbid")

    max_seq_length: int = Field(default=128, ge=16, le=2048)
    max_new_tokens: int = Field(default=16, ge=1, le=256)
    warmup_runs: int = Field(default=4, ge=0)
    distillation_temperature: float = Field(default=1.0, gt=0)
    max_teacher_kl: float = Field(default=0.50, ge=0)
    max_reference_loss_regret: float = Field(default=0.15, ge=0)
    min_token_agreement: float = Field(default=0.75, ge=0, le=1)
    min_examples: int = Field(default=150, ge=1)
    min_safe_examples: int = Field(default=20, ge=1)
    min_unsafe_examples: int = Field(default=20, ge=1)
    min_median_latency_reduction: float = Field(default=0.10, ge=0, lt=1)


class Phase6Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelSettings = Field(default_factory=ModelSettings)
    calibration: Phase6CalibrationSettings = Field(
        default_factory=Phase6CalibrationSettings
    )


class Phase61CalibrationSettings(Phase6CalibrationSettings):
    """Multi-route regret collection with response-preserving supervision."""

    response_token_budget: int = Field(default=16, ge=2, le=256)
    min_generation_token_jaccard: float = Field(default=0.50, ge=0, le=1)
    min_eligible_routes: int = Field(default=2, ge=1)
    min_route_latency_reduction: float = Field(default=0.02, ge=0, lt=1)

    @model_validator(mode="after")
    def validate_response_budget(self) -> "Phase61CalibrationSettings":
        if self.response_token_budget >= self.max_seq_length:
            raise ValueError("response_token_budget must be below max_seq_length")
        return self


class Phase61Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelSettings = Field(default_factory=ModelSettings)
    calibration: Phase61CalibrationSettings = Field(
        default_factory=Phase61CalibrationSettings
    )


class Phase7RouterSettings(BaseModel):
    """Leakage-controlled prompt-risk router training settings."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 17
    test_fraction: float = Field(default=0.20, gt=0, lt=0.5)
    outer_cv_folds: int = Field(default=5, ge=2, le=10)
    calibration_cv_folds: int = Field(default=3, ge=2, le=10)
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid"
    max_text_features: int = Field(default=2000, ge=100)
    min_ngram: int = Field(default=1, ge=1, le=3)
    max_ngram: int = Field(default=2, ge=1, le=3)
    logistic_c: float = Field(default=1.0, gt=0)
    target_safe_precision: float = Field(default=0.80, gt=0, le=1)
    min_route_coverage: float = Field(default=0.05, gt=0, lt=1)
    min_routed_test_examples: int = Field(default=2, ge=1)
    min_gross_latency_reduction: float = Field(default=0.005, ge=0, lt=1)
    max_expected_calibration_error: float = Field(default=0.15, ge=0, le=1)
    max_brier_score: float = Field(default=0.20, ge=0, le=1)
    calibration_bins: int = Field(default=10, ge=2, le=50)

    @model_validator(mode="after")
    def validate_ngrams(self) -> "Phase7RouterSettings":
        if self.min_ngram > self.max_ngram:
            raise ValueError("min_ngram must not exceed max_ngram")
        return self


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


def load_phase6_settings(path: str | Path) -> Phase6Settings:
    return Phase6Settings.model_validate(_load_yaml(path))


def load_phase61_settings(path: str | Path) -> Phase61Settings:
    return Phase61Settings.model_validate(_load_yaml(path))


def load_phase7_settings(path: str | Path) -> Phase7RouterSettings:
    return Phase7RouterSettings.model_validate(_load_yaml(path))
