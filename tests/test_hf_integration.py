from __future__ import annotations

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from adaptivedepth.model.routed_model import RoutedCausalLM
from adaptivedepth.routes import RouteDefinition
from adaptivedepth.runtime.generation import profile_greedy_generation


class TinyTokenizer:
    eos_token_id = None

    def __call__(self, _text: str, return_tensors: str = "pt"):
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([[2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens
        return " ".join(str(int(token)) for token in token_ids)


def test_fixed_route_works_with_real_qwen2_cache_semantics() -> None:
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = Qwen2ForCausalLM(config).eval()
    routed = RoutedCausalLM(model, "model.layers")
    route = RouteDefinition(route_id="sparse", active_layers=[0, 2, 3])

    with routed.use_route(route):
        profile = profile_greedy_generation(
            routed,
            TinyTokenizer(),
            "ignored",
            torch.device("cpu"),
            max_new_tokens=3,
        )

    assert profile.output_tokens == 3
    assert len(model.model.layers) == 4
