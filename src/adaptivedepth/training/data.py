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
