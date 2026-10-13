from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn


KNOWN_LAYER_PATHS = (
    "model.layers",  # Llama, Mistral, Qwen2/3, Gemma, Phi-3
    "model.decoder.layers",  # OPT and encoder-decoder-style wrappers
    "transformer.h",  # GPT-2 family
    "transformer.blocks",
    "gpt_neox.layers",
)


def _resolve_parent(root: Any, path: str) -> tuple[Any, str, Any]:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        if not hasattr(parent, part):
            raise AttributeError(path)
        parent = getattr(parent, part)
    attribute = parts[-1]
    if not hasattr(parent, attribute):
        raise AttributeError(path)
    return parent, attribute, getattr(parent, attribute)


@dataclass
class LayerStack:
    parent: Any
    attribute: str
    layers: nn.ModuleList
    path: str

    @property
    def depth(self) -> int:
        return len(self.layers)

    def replace(self, layers: nn.ModuleList) -> None:
        setattr(self.parent, self.attribute, layers)


def discover_layer_stack(model: nn.Module, explicit_path: str | None = None) -> LayerStack:
    candidates = (explicit_path,) if explicit_path else KNOWN_LAYER_PATHS
    for path in candidates:
        if path is None:
            continue
        try:
            parent, attribute, value = _resolve_parent(model, path)
        except AttributeError:
            continue
        if isinstance(value, nn.ModuleList):
            if not value:
                raise ValueError(f"transformer layer stack at {path!r} is empty")
            return LayerStack(parent, attribute, value, path)
    searched = ", ".join(path for path in candidates if path)
    raise ValueError(
        "could not locate a transformer block ModuleList; "
        f"searched: {searched}. Set model.layer_path explicitly."
    )
