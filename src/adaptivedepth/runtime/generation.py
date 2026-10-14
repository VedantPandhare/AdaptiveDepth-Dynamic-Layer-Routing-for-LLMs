from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import torch


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def encode_prompt(
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    use_chat_template: bool = True,
) -> dict[str, Any]:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        batch = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        batch = tokenizer(prompt, return_tensors="pt")
    return _move_batch(dict(batch), device)


def _eos_ids(tokenizer: Any) -> set[int]:
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {eos}
    return set(eos)


@dataclass(frozen=True)
class GenerationProfile:
    text: str
    input_tokens: int
    output_tokens: int
    ttft_ms: float
    decode_ms: float
    total_ms: float
    decode_ms_per_token: float
    tokens_per_second: float
    peak_memory_mb: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@torch.inference_mode()
def profile_greedy_generation(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    use_chat_template: bool = True,
) -> GenerationProfile:
    """Profile deterministic generation while separating prefill/TTFT and decode."""

    encoded = encode_prompt(tokenizer, prompt, device, use_chat_template)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    started = perf_counter()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    _synchronize(device)
    ttft_ms = (perf_counter() - started) * 1000

    generated = [next_token]
    stop_ids = _eos_ids(tokenizer)
    decode_started = perf_counter()
    while len(generated) < max_new_tokens and int(next_token.item()) not in stop_ids:
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token, device=device)], dim=-1
        )
        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=outputs.past_key_values,
            use_cache=True,
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
    _synchronize(device)
    decode_ms = (perf_counter() - decode_started) * 1000

    generated_ids = torch.cat(generated, dim=-1)
    output_tokens = generated_ids.shape[-1]
    total_ms = ttft_ms + decode_ms
    decode_steps = max(output_tokens - 1, 0)
    return GenerationProfile(
        text=tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip(),
        input_tokens=input_ids.shape[-1],
        output_tokens=output_tokens,
        ttft_ms=ttft_ms,
        decode_ms=decode_ms,
        total_ms=total_ms,
        decode_ms_per_token=decode_ms / decode_steps if decode_steps else 0.0,
        tokens_per_second=(output_tokens / total_ms * 1000) if total_ms else 0.0,
        peak_memory_mb=(
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else None
        ),
    )
