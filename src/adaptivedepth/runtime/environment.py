from __future__ import annotations

import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import transformers


def _git_commit(workdir: str | Path | None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def capture_environment(
    device: torch.device,
    workdir: str | Path | None = None,
) -> dict[str, Any]:
    gpu_name = None
    gpu_memory_mb = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu_name = properties.name
        gpu_memory_mb = round(properties.total_memory / (1024**2), 2)
    return {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "gpu_memory_mb": gpu_memory_mb,
        "git_commit": _git_commit(workdir),
    }
