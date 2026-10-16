from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

from adaptivedepth.training.data import TrainingExample


def normalize_dolly_records(
    records: Iterable[dict[str, Any]], max_examples: int, seed: int
) -> list[TrainingExample]:
    values = list(records)
    random.Random(seed).shuffle(values)
    examples: list[TrainingExample] = []
    for row in values:
        try:
            example = TrainingExample(
                instruction=str(row.get("instruction", "")).strip(),
                context=str(row.get("context", "")).strip(),
                response=str(row.get("response", "")).strip(),
            )
        except ValueError:
            continue
        examples.append(example)
        if len(examples) >= max_examples:
            break
    if len(examples) < max_examples:
        raise ValueError(
            f"requested {max_examples} examples, but only {len(examples)} valid records exist"
        )
    return examples


def save_training_examples(examples: list[TrainingExample], output: str | Path) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.model_dump(mode="json"), ensure_ascii=False) + "\n")


def prepare_dolly_dataset(
    output: str | Path, max_examples: int = 2000, seed: int = 17
) -> int:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("install the data extra with: pip install -e '.[data]'") from error
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    examples = normalize_dolly_records(dataset, max_examples, seed)
    save_training_examples(examples, output)
    return len(examples)
