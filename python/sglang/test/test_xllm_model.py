import unittest
from collections import namedtuple
from types import SimpleNamespace

import torch

from sglang.srt.models.xllm import (
    EntryClass,
    XllmAttention,
    XllmForCausalLM,
    XllmSparseMoeBlock,
)


TopKOutput = namedtuple("TopKOutput", ["topk_weights"])


class _RecordingTopK:
    def __init__(self):
        self.native_calls = 0
        self.generic_calls = 0
        self.output = TopKOutput(topk_weights=torch.tensor([[0.25, 0.75]]))

    def forward_native(self, hidden_states, router_logits):
        self.native_calls += 1
        self.hidden_shape = tuple(hidden_states.shape)
        self.router_shape = tuple(router_logits.shape)
        return self.output

    def __call__(self, *args, **kwargs):
        self.generic_calls += 1
        raise AssertionError("xLLM MoE should use TopK.forward_native")


class _RecordingExperts:
    def __call__(self, hidden_states, topk_output):
        self.seen_topk_output = topk_output
        return torch.zeros_like(hidden_states)


class _IdentityRotary:
    def __call__(self, positions, q, k):
        self.positions_shape = tuple(positions.shape)
        self.q_shape = tuple(q.shape)
        self.k_shape = tuple(k.shape)
        return q, k


class TestXllmModel(unittest.TestCase):
    def test_entry_class_and_expert_location_metadata(self):
        self.assertIs(EntryClass, XllmForCausalLM)

        config = SimpleNamespace(num_hidden_layers=61, num_experts=192)
        expert_location = XllmForCausalLM.get_model_config_for_expert_location(
            config
        )
        self.assertEqual(expert_location.num_layers, 61)
        self.assertEqual(expert_location.num_logical_experts, 192)
        self.assertIsNone(expert_location.num_groups)

        dense_config = SimpleNamespace(num_hidden_layers=61, num_experts=0)
        self.assertIsNone(
            XllmForCausalLM.get_model_config_for_expert_location(dense_config)
        )

    def test_partial_rope_round_trips_non_rotary_dimensions(self):
        attention = object.__new__(XllmAttention)
        torch.nn.Module.__init__(attention)
        attention.num_heads = 2
        attention.num_kv_heads = 1
        attention.head_dim = 8
        attention.rope_head_dim = 4
        attention.rotary_emb = _IdentityRotary()

        positions = torch.arange(3)
        q = torch.randn(3, attention.num_heads * attention.head_dim)
        k = torch.randn(3, attention.num_kv_heads * attention.head_dim)

        q_out, k_out = XllmAttention._apply_partial_rope(attention, positions, q, k)

        torch.testing.assert_close(q_out, q)
        torch.testing.assert_close(k_out, k)
        self.assertEqual(attention.rotary_emb.positions_shape, (3,))
        self.assertEqual(
            attention.rotary_emb.q_shape,
            (3, attention.num_heads * attention.rope_head_dim),
        )
        self.assertEqual(
            attention.rotary_emb.k_shape,
            (3, attention.num_kv_heads * attention.rope_head_dim),
        )

    def test_moe_uses_native_topk_contract_and_scales_after_renormalization(self):
        block = object.__new__(XllmSparseMoeBlock)
        torch.nn.Module.__init__(block)
        block.layer_id = 3
        block.router_scaling_factor = 2.5
        block.topk = _RecordingTopK()
        block.experts = _RecordingExperts()
        block.gate = lambda hidden_states: torch.randn(hidden_states.shape[0], 4)
        block._forward_shared_experts = lambda hidden_states: torch.ones_like(
            hidden_states
        )

        hidden_states = torch.randn(1, 4)
        out = XllmSparseMoeBlock._forward_deepep(
            block,
            hidden_states,
            SimpleNamespace(num_token_non_padded=hidden_states.shape[0]),
        )

        self.assertEqual(block.topk.native_calls, 1)
        self.assertEqual(block.topk.generic_calls, 0)
        self.assertEqual(block.topk.hidden_shape, (1, 4))
        self.assertEqual(block.topk.router_shape, (1, 4))
        torch.testing.assert_close(out, torch.ones_like(hidden_states))
        torch.testing.assert_close(
            block.experts.seen_topk_output.topk_weights,
            block.topk.output.topk_weights * block.router_scaling_factor,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
