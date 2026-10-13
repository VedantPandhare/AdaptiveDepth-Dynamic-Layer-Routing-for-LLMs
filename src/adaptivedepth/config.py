from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


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


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    return value or {}


def load_project_settings(path: str | Path) -> ProjectSettings:
    return ProjectSettings.model_validate(_load_yaml(path))


def load_compiler_settings(path: str | Path) -> CompilerSettings:
    return CompilerSettings.model_validate(_load_yaml(path))
