from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

from adaptivedepth.training.data import TrainingExample


def normalize_dolly_records(
    records: Iterable[dict[str, Any]],
    max_examples: int,
    seed: int,
    skip_examples: int = 0,
) -> list[TrainingExample]:
    if skip_examples < 0:
        raise ValueError("skip_examples must be non-negative")
    values = list(records)
    random.Random(seed).shuffle(values)
    valid_examples: list[TrainingExample] = []
    for row in values:
        try:
            example = TrainingExample(
                instruction=str(row.get("instruction", "")).strip(),
                context=str(row.get("context", "")).strip(),
                response=str(row.get("response", "")).strip(),
            )
        except ValueError:
            continue
        valid_examples.append(example)
        if len(valid_examples) >= skip_examples + max_examples:
            break
    examples = valid_examples[skip_examples : skip_examples + max_examples]
    if len(examples) < max_examples:
        raise ValueError(
            f"requested {max_examples} examples, but only {len(examples)} valid records exist"
        )
    return examples


def generate_balanced_multiple_choice_examples(
    count: int = 200, seed: int = 23
) -> list[TrainingExample]:
    """Generate deterministic arithmetic examples with balanced A-D labels."""

    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    letters = "ABCD"
    examples: list[TrainingExample] = []
    for index in range(count):
        operation = index % 3
        left = rng.randint(6, 35)
        right = rng.randint(2, 12)
        if operation == 0:
            expression, answer = f"{left} + {right}", left + right
        elif operation == 1:
            expression, answer = f"{left} - {right}", left - right
        else:
            left = rng.randint(2, 12)
            expression, answer = f"{left} x {right}", left * right

        correct_index = index % 4
        distractors = [answer - 2, answer - 1, answer + 1]
        options = distractors[:]
        options.insert(correct_index, answer)
        rendered = " ".join(
            f"{letter}) {value}" for letter, value in zip(letters, options)
        )
        examples.append(
            TrainingExample(
                instruction=(
                    f"What is {expression}? Choose the correct option and begin "
                    f"your response with its letter. {rendered}"
                ),
                response=f"{letters[correct_index]}) {answer}",
            )
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


def prepare_phase5_correction_dataset(
    output: str | Path,
    dolly_examples: int = 300,
    balanced_examples: int = 200,
    skip_dolly_examples: int = 2000,
    seed: int = 17,
) -> int:
    """Build a held-out correction set plus balanced structured-answer examples."""

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("install the data extra with: pip install -e '.[data]'") from error
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    examples = normalize_dolly_records(
        dataset,
        max_examples=dolly_examples,
        seed=seed,
        skip_examples=skip_dolly_examples,
    )
    examples.extend(
        generate_balanced_multiple_choice_examples(balanced_examples, seed=seed + 6)
    )
    random.Random(seed + 1).shuffle(examples)
    save_training_examples(examples, output)
    return len(examples)
