from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field


class TrainingExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(min_length=1)
    response: str = Field(min_length=1)
    context: str = ""


@dataclass(frozen=True)
class EncodedExample:
    input_ids: list[int]
    labels: list[int]


@dataclass(frozen=True)
class ResponsePreservingEncoding:
    input_ids: list[int]
    labels: list[int]
    prompt_token_count: int
    available_response_tokens: int
    used_response_tokens: int
    prompt_was_truncated: bool
    response_was_truncated: bool


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_examples(path: str | Path) -> list[TrainingExample]:
    source = Path(path)
    examples: list[TrainingExample] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                examples.append(TrainingExample.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"invalid training example at {source}:{line_number}") from error
    if not examples:
        raise ValueError(f"training dataset is empty: {source}")
    return examples


def _user_content(example: TrainingExample) -> str:
    if not example.context.strip():
        return example.instruction.strip()
    return f"{example.instruction.strip()}\n\nContext:\n{example.context.strip()}"


def _render_texts(tokenizer: Any, example: TrainingExample) -> tuple[str, str, bool]:
    user = {"role": "user", "content": _user_content(example)}
    assistant = {"role": "assistant", "content": example.response.strip()}
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [user], tokenize=False, add_generation_prompt=True
        )
        full = tokenizer.apply_chat_template(
            [user, assistant], tokenize=False, add_generation_prompt=False
        )
        return prompt, full, False
    prompt = f"Instruction: {user['content']}\nResponse:"
    return prompt, f"{prompt} {example.response.strip()}", True


def _token_ids(tokenizer: Any, text: str, max_length: int, add_special_tokens: bool) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        truncation=True,
        max_length=max_length,
    )
    values = encoded["input_ids"]
    if isinstance(values, torch.Tensor):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def encode_example(tokenizer: Any, example: TrainingExample, max_length: int) -> EncodedExample:
    prompt, full, add_special_tokens = _render_texts(tokenizer, example)
    prompt_ids = _token_ids(tokenizer, prompt, max_length, add_special_tokens)
    input_ids = _token_ids(tokenizer, full, max_length, add_special_tokens)
    if len(input_ids) < 2:
        raise ValueError("training example contains fewer than two tokens")
    prompt_length = min(len(prompt_ids), len(input_ids) - 1)
    labels = [-100] * prompt_length + input_ids[prompt_length:]
    if all(label == -100 for label in labels[1:]):
        raise ValueError("training example was truncated before its response")
    return EncodedExample(input_ids=input_ids, labels=labels)


def _untruncated_token_ids(
    tokenizer: Any, text: str, add_special_tokens: bool
) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        truncation=False,
    )
    values = encoded["input_ids"]
    if isinstance(values, torch.Tensor):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    common = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        common += 1
    return common


def _truncate_prompt_middle(tokens: list[int], budget: int) -> list[int]:
    if len(tokens) <= budget:
        return tokens
    if budget < 2:
        raise ValueError("response reservation leaves no usable prompt budget")
    head = (budget + 1) // 2
    tail = budget - head
    return tokens[:head] + (tokens[-tail:] if tail else [])


def encode_example_preserving_response(
    tokenizer: Any,
    example: TrainingExample,
    max_length: int,
    response_token_budget: int,
) -> ResponsePreservingEncoding:
    """Encode an example without allowing a long prompt to erase its response.

    Up to ``response_token_budget`` available response tokens are retained.
    When the prompt is too long, its middle is removed so both the instruction
    prefix and the assistant-generation suffix remain represented.
    """
    if response_token_budget < 1 or response_token_budget >= max_length:
        raise ValueError("response_token_budget must be between 1 and max_length - 1")
    prompt, full, add_special_tokens = _render_texts(tokenizer, example)
    prompt_ids = _untruncated_token_ids(tokenizer, prompt, add_special_tokens)
    full_ids = _untruncated_token_ids(tokenizer, full, add_special_tokens)
    common = _common_prefix_length(prompt_ids, full_ids)
    if common == 0:
        raise ValueError("prompt and completed example have no common token prefix")
    response_ids = full_ids[common:]
    if not response_ids:
        raise ValueError("example has no response tokens")

    used_response = response_ids[:response_token_budget]
    prompt_budget = max_length - len(used_response)
    prompt_prefix = full_ids[:common]
    used_prompt = _truncate_prompt_middle(prompt_prefix, prompt_budget)
    input_ids = used_prompt + used_response
    labels = [-100] * len(used_prompt) + used_response
    return ResponsePreservingEncoding(
        input_ids=input_ids,
        labels=labels,
        prompt_token_count=len(used_prompt),
        available_response_tokens=len(response_ids),
        used_response_tokens=len(used_response),
        prompt_was_truncated=len(used_prompt) < len(prompt_prefix),
        response_was_truncated=len(used_response) < len(response_ids),
    )


def collate_examples(
    examples: list[EncodedExample], pad_token_id: int, device: torch.device
) -> dict[str, torch.Tensor]:
    max_length = max(len(example.input_ids) for example in examples)
    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for example in examples:
        padding = max_length - len(example.input_ids)
        input_rows.append(example.input_ids + [pad_token_id] * padding)
        label_rows.append(example.labels + [-100] * padding)
        mask_rows.append([1] * len(example.input_ids) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(mask_rows, dtype=torch.long, device=device),
        "labels": torch.tensor(label_rows, dtype=torch.long, device=device),
    }
