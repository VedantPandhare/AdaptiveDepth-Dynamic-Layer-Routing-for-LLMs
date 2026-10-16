from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LossValues:
    total: torch.Tensor
    language_model: torch.Tensor
    distillation: torch.Tensor


def causal_language_model_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def teacher_student_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student = student_logits[:, :-1, :].float() / temperature
    teacher = teacher_logits[:, :-1, :].float() / temperature
    token_kl = F.kl_div(
        F.log_softmax(student, dim=-1),
        F.softmax(teacher, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    valid = attention_mask[:, 1:].to(token_kl.dtype)
    return (token_kl * valid).sum() / valid.sum().clamp_min(1.0) * (temperature**2)


def route_training_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    language_model_weight: float,
    distillation_weight: float,
    temperature: float,
) -> LossValues:
    language_model = causal_language_model_loss(student_logits, labels)
    distillation = teacher_student_kl(
        student_logits, teacher_logits, attention_mask, temperature
    )
    total = language_model_weight * language_model + distillation_weight * distillation
    return LossValues(total, language_model, distillation)
