from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModelSettings(BaseModel):
    """Model-loading settings shared by the local engine and future service."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "auto"
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    trust_remote_code: bool = False
    local_files_only: bool = False
    layer_path: str | None = None


class BenchmarkSettings(BaseModel):
    """Deterministic reference-generation and profiling settings."""

    model_config = ConfigDict(extra="forbid")

    max_new_tokens: int = Field(default=64, ge=1, le=2048)
    warmup_runs: int = Field(default=1, ge=0)
    measured_runs: int = Field(default=3, ge=1)
    use_chat_template: bool = True
    fixed_generation_length: bool = False


class ProjectSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelSettings = Field(default_factory=ModelSettings)
    benchmark: BenchmarkSettings = Field(default_factory=BenchmarkSettings)


def load_project_settings(path: str | Path) -> ProjectSettings:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return ProjectSettings.model_validate(payload)
