from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from adaptivedepth.config import ModelSettings
from adaptivedepth.model.routed_model import RoutedCausalLM


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested != "auto":
        return DTYPES[requested]
    return torch.float16 if device.type == "cuda" else torch.float32


def load_model(settings: ModelSettings) -> tuple[RoutedCausalLM, object, torch.device]:
    device = resolve_device(settings.device)
    dtype = resolve_dtype(settings.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(
        settings.model_id,
        trust_remote_code=settings.trust_remote_code,
        local_files_only=settings.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_id,
        torch_dtype=dtype,
        trust_remote_code=settings.trust_remote_code,
        local_files_only=settings.local_files_only,
    )
    model.to(device)
    model.eval()
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return RoutedCausalLM(model, settings.layer_path), tokenizer, device


def load_teacher_student(
    settings: ModelSettings,
    student_model_id: str | None = None,
    teacher_model_id: str | None = None,
) -> tuple[RoutedCausalLM, RoutedCausalLM, object, torch.device]:
    """Load an immutable full-depth teacher and an independently trainable student."""

    device = resolve_device(settings.device)
    dtype = resolve_dtype(settings.dtype, device)
    teacher_source = teacher_model_id or settings.model_id
    student_source = student_model_id or teacher_source
    common = {
        "trust_remote_code": settings.trust_remote_code,
        "local_files_only": settings.local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        teacher_source,
        trust_remote_code=settings.trust_remote_code,
        local_files_only=settings.local_files_only,
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_source, torch_dtype=dtype, **common
    )
    teacher_model.to(device)
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    # AMP needs FP32 master parameters. Loading the trainable model directly in
    # FP16 creates FP16 gradients, which GradScaler intentionally refuses to
    # unscale. The frozen teacher remains compact while autocast keeps student
    # matrix operations in the configured compute dtype.
    student_dtype = (
        torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
    )
    student_model = AutoModelForCausalLM.from_pretrained(
        student_source,
        torch_dtype=student_dtype,
        **common,
    )
    student_model.to(device)
    student_model.train()
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return (
        RoutedCausalLM(teacher_model, settings.layer_path),
        RoutedCausalLM(student_model, settings.layer_path),
        tokenizer,
        device,
    )
