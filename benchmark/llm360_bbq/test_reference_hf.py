#!/usr/bin/env python3
"""CPU-only tests for narrowly scoped Hugging Face oracle compatibility."""

from __future__ import annotations

import importlib.util
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers.modeling_utils import PreTrainedModel

PINNED_XLLM_375B_PATH = Path(
    "/mnt/weka/home/mrunner/workspace/checkpoints/huggingface/"
    "k2moe375B_mid4_v2_200B_256nodes_seed42_bsz32M_seq512k_jais250k_"
    "ep8_dot_te_bestfit/checkpoints/checkpoint_0006000"
)

MODULE_PATH = Path(__file__).with_name("reference_hf.py")
SPEC = importlib.util.spec_from_file_location("bbq_reference_hf", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
reference = importlib.util.module_from_spec(SPEC)
with patch.dict(
    os.environ,
    {
        "BBQ_MODEL_NAME": "k2moe-375b",
        "BBQ_MODEL_PATH": "/tmp/bbq-unit-model",
        "BBQ_SGLANG_RESULT": "/tmp/bbq-unit-sglang.json",
        "BBQ_HF_OUTPUT_DIR": "/tmp/bbq-unit-hf-output",
    },
):
    SPEC.loader.exec_module(reference)


def exact_xllm_375b_config(**overrides: object) -> SimpleNamespace:
    values = {
        "architectures": ["XllmForCausalLM"],
        "model_type": "xllm",
        "hidden_size": 6144,
        "head_dim": 128,
        "rope_head_dim": 64,
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "num_hidden_layers": 61,
        "num_experts": 192,
        "mlp_only_layers": [0, 1, 2],
        "max_position_embeddings": 524288,
        "rope_theta": 10000000.0,
        "rope_scaling": None,
        "partial_rotary_factor": None,
        "initializer_range": 0.02,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class XllmPartialRopeCompatibilityTests(unittest.TestCase):
    def test_post_load_replay_uses_partial_rope_dimension(self) -> None:
        config = exact_xllm_375b_config()
        expected = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.rope_head_dim, 2, dtype=torch.int64).float()
                / config.rope_head_dim
            )
        )
        replayed_buffer, scaling = reference._xllm_375b_partial_rope_parameters(config)

        self.assertEqual(expected.shape, (32,))
        self.assertEqual(replayed_buffer.shape, (32,))
        self.assertEqual(scaling, 1.0)
        self.assertTrue(torch.equal(replayed_buffer, expected))

        class XllmRotaryEmbedding(torch.nn.Module):
            compute_default_rope_parameters = staticmethod(
                reference._xllm_375b_partial_rope_parameters
            )

            def __init__(self) -> None:
                super().__init__()
                self.config = config
                self.rope_type = "default"
                self.register_buffer("inv_freq", torch.zeros_like(expected))
                self.original_inv_freq = self.inv_freq

        rotary = XllmRotaryEmbedding()
        owner = SimpleNamespace(config=config)

        # Exercise the exact Transformers-5 generic replay operation that failed
        # after all checkpoint weights had loaded in job 2243741.
        PreTrainedModel._init_weights(owner, rotary)
        self.assertTrue(torch.equal(rotary.inv_freq, expected))
        self.assertTrue(torch.equal(rotary.original_inv_freq, expected))

    def test_generic_replay_would_reproduce_the_observed_mismatch(self) -> None:
        config = exact_xllm_375b_config()
        generic_buffer, _ = reference._default_rope_parameters(config)
        self.assertEqual(generic_buffer.shape, (64,))
        with self.assertRaisesRegex(RuntimeError, "size of tensor"):
            torch.empty(32).copy_(generic_buffer)

    def test_partial_rope_shim_rejects_nearby_config(self) -> None:
        config = exact_xllm_375b_config(rope_head_dim=32)
        self.assertFalse(reference._is_exact_xllm_375b_partial_rope_config(config))
        with self.assertRaisesRegex(
            ValueError,
            r"'rope_head_dim': \{'expected': 64, 'actual': 32\}",
        ):
            reference._xllm_375b_partial_rope_parameters(config)

    def test_tf5_normalized_unscaled_rope_is_the_only_accepted_alias(self) -> None:
        normalized = {
            "rope_type": "default",
            "rope_theta": 10000000.0,
        }
        config = exact_xllm_375b_config(rope_scaling=normalized)
        self.assertTrue(reference._is_exact_xllm_375b_partial_rope_config(config))

        invalid_aliases = (
            {**normalized, "factor": 1.0},
            {**normalized, "partial_rotary_factor": 1.0},
            {"rope_type": "linear", "rope_theta": 10000000.0},
            {"rope_type": "default", "rope_theta": 10000.0},
        )
        for rope_scaling in invalid_aliases:
            with self.subTest(rope_scaling=rope_scaling):
                config = exact_xllm_375b_config(rope_scaling=rope_scaling)
                self.assertFalse(
                    reference._is_exact_xllm_375b_partial_rope_config(config)
                )

    @unittest.skipUnless(
        (PINNED_XLLM_375B_PATH / "config.json").is_file(),
        "pinned xllm-375b artifact is not mounted",
    )
    def test_pinned_auto_config_normalization_matches_exact_contract(self) -> None:
        raw_config = json.loads((PINNED_XLLM_375B_PATH / "config.json").read_text())
        self.assertIsNone(raw_config["rope_scaling"])

        config = reference.AutoConfig.from_pretrained(
            PINNED_XLLM_375B_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )
        expected_normalized = dict(reference._XLLM_375B_PARTIAL_ROPE_EXPECTED)
        expected_normalized["rope_scaling"] = {
            "rope_type": "default",
            "rope_theta": 10000000.0,
        }
        actual = {
            key: getattr(config, key, None)
            for key in reference._XLLM_375B_PARTIAL_ROPE_EXPECTED
        }
        self.assertEqual(actual, expected_normalized)
        self.assertTrue(reference._is_exact_xllm_375b_partial_rope_config(config))


if __name__ == "__main__":
    unittest.main()
