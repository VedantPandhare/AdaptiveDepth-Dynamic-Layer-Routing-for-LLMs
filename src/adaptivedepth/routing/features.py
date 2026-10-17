from __future__ import annotations

import re


def prompt_features(prompt: str, prompt_tokens: int) -> dict[str, float | int | bool]:
    """Return inexpensive features available before model generation."""
    characters = len(prompt)
    digits = sum(character.isdigit() for character in prompt)
    return {
        "character_count": characters,
        "word_count": len(prompt.split()),
        "prompt_token_count": prompt_tokens,
        "newline_count": prompt.count("\n"),
        "question_mark_count": prompt.count("?"),
        "digit_fraction": digits / max(characters, 1),
        "contains_code_fence": "```" in prompt,
        "contains_math_symbol": bool(re.search(r"[=+*/^]", prompt)),
    }
