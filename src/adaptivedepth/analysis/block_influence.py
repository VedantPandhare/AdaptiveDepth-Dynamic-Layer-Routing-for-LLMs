from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as functional

from adaptivedepth.model.routed_model import RoutedCausalLM
from adaptivedepth.runtime.generation import encode_prompt


def _hidden_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    if hasattr(value, "last_hidden_state"):
        return value.last_hidden_state
    raise TypeError(f"cannot extract hidden states from {type(value).__name__}")


@dataclass(frozen=True)
class LayerInfluence:
    layer_index: int
    influence: float
    observations: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BlockInfluenceAnalyzer:
    """Measure mean 1-cosine(input, output) for every transformer block."""

    def __init__(self, routed_model: RoutedCausalLM) -> None:
        self.routed_model = routed_model
        self._totals = [0.0] * routed_model.num_layers
        self._counts = [0] * routed_model.num_layers
        self._handles: list[Any] = []

    def _hook(self, layer_index: int):
        def measure(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            before = _hidden_tensor(inputs[0]).detach().float()
            after = _hidden_tensor(output).detach().float()
            score = 1.0 - functional.cosine_similarity(before, after, dim=-1).mean()
            self._totals[layer_index] += max(0.0, float(score.cpu()))
            self._counts[layer_index] += 1

        return measure

    def __enter__(self) -> "BlockInfluenceAnalyzer":
        for index, layer in enumerate(self.routed_model.stack.layers):
            self._handles.append(layer.register_forward_hook(self._hook(index)))
        return self

    def __exit__(self, *_args: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @torch.inference_mode()
    def analyze_prompts(
        self,
        tokenizer: Any,
        prompts: list[str],
        device: torch.device,
        use_chat_template: bool = True,
    ) -> list[LayerInfluence]:
        with self:
            for prompt in prompts:
                encoded = encode_prompt(tokenizer, prompt, device, use_chat_template)
                self.routed_model(**encoded, use_cache=False)
        return [
            LayerInfluence(
                layer_index=index,
                influence=self._totals[index] / self._counts[index]
                if self._counts[index]
                else 0.0,
                observations=self._counts[index],
            )
            for index in range(self.routed_model.num_layers)
        ]
