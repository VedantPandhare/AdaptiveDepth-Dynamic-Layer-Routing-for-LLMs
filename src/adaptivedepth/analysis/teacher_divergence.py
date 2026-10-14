from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional

from adaptivedepth.model.routed_model import RoutedCausalLM
from adaptivedepth.routes import RouteDefinition
from adaptivedepth.runtime.generation import encode_prompt


@torch.inference_mode()
def mean_teacher_kl(
    routed_model: RoutedCausalLM,
    tokenizer: Any,
    prompts: list[str],
    device: torch.device,
    route: RouteDefinition,
    use_chat_template: bool = True,
) -> float:
    """Mean token-level KL(full || route) over calibration prompts."""

    if not prompts:
        raise ValueError("at least one prompt is required")
    divergences: list[float] = []
    for prompt in prompts:
        encoded = encode_prompt(tokenizer, prompt, device, use_chat_template)
        teacher_logits = routed_model(**encoded, use_cache=False).logits.float()
        with routed_model.use_route(route):
            route_logits = routed_model(**encoded, use_cache=False).logits.float()
        teacher_log_prob = functional.log_softmax(teacher_logits, dim=-1)
        route_log_prob = functional.log_softmax(route_logits, dim=-1)
        teacher_prob = teacher_log_prob.exp()
        token_kl = (teacher_prob * (teacher_log_prob - route_log_prob)).sum(dim=-1)
        divergences.append(max(0.0, float(token_kl.mean().cpu())))
    return sum(divergences) / len(divergences)
