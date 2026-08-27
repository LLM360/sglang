#!/usr/bin/env python3
"""CPU-only tests for narrowly scoped Hugging Face oracle compatibility."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
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
PINNED_K2_AURORA_1B_PATH = Path("/mnt/weka/shrd/k2m/junlin.chen/xllm_1b_final/model")

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


class K2AuroraDense1BCompatibilityTests(unittest.TestCase):
    @staticmethod
    def exact_raw_config() -> dict[str, object]:
        return json.loads(json.dumps(reference._K2_AURORA_DENSE_1B_RAW_EXPECTED))

    def test_rope_normalizer_injects_theta_and_revalidates(self) -> None:
        raw_config = self.exact_raw_config()
        config = SimpleNamespace(**raw_config)
        del config.rope_theta

        injected = reference._normalize_k2_aurora_dense_1b_rope_config(
            config, raw_config
        )

        self.assertTrue(injected)
        self.assertEqual(config.rope_theta, 1000000.0)
        self.assertEqual(config.rope_parameters["rope_theta"], 1000000.0)
        self.assertFalse(
            reference._normalize_k2_aurora_dense_1b_rope_config(config, raw_config)
        )

    def test_rope_normalizer_rejects_raw_or_normalized_conflicts(self) -> None:
        raw_config = self.exact_raw_config()
        invalid_raw = {**raw_config, "vocab_size": 64257}
        with self.assertRaisesRegex(
            ValueError, "exact pinned dense-1B topology/YaRN contract"
        ):
            reference._normalize_k2_aurora_dense_1b_rope_config(
                SimpleNamespace(**invalid_raw), invalid_raw
            )

        conflicting_config = SimpleNamespace(**raw_config)
        conflicting_config.rope_parameters = {
            **conflicting_config.rope_parameters,
            "rope_theta": 10000.0,
        }
        with self.assertRaisesRegex(ValueError, "conflicting nested rope_theta"):
            reference._normalize_k2_aurora_dense_1b_rope_config(
                conflicting_config, raw_config
            )

        conflicting_top_level = SimpleNamespace(**raw_config)
        conflicting_top_level.rope_theta = 10000.0
        with self.assertRaisesRegex(ValueError, "conflicting top-level rope_theta"):
            reference._normalize_k2_aurora_dense_1b_rope_config(
                conflicting_top_level, raw_config
            )

        wrong_shape_config = SimpleNamespace(**raw_config)
        wrong_shape_config.rope_parameters = []
        with self.assertRaisesRegex(ValueError, "rope_parameters as a dict"):
            reference._normalize_k2_aurora_dense_1b_rope_config(
                wrong_shape_config, raw_config
            )

        mismatched_config = SimpleNamespace(**raw_config)
        mismatched_config.hidden_size = 1537
        with self.assertRaisesRegex(ValueError, "post-normalization validation"):
            reference._normalize_k2_aurora_dense_1b_rope_config(
                mismatched_config, raw_config
            )

    def test_dense_signature_does_not_capture_large_k2_aurora(self) -> None:
        mova_config = {
            "model_type": "k2_aurora",
            "architectures": ["K2AuroraForCausalLM"],
            "hidden_size": 7168,
            "num_hidden_layers": 61,
            "num_experts": 0,
            "mova_num_experts": 128,
        }
        self.assertFalse(reference._is_k2_aurora_dense_1b_candidate(mova_config))
        config = SimpleNamespace(rope_parameters=None)
        self.assertFalse(
            reference._normalize_k2_aurora_dense_1b_rope_config(config, mova_config)
        )

    def test_output_capturing_reexports_generic_output_recorder(self) -> None:
        raw_config = self.exact_raw_config()

        class ExistingOutputRecorder:
            pass

        generic_module = SimpleNamespace(OutputRecorder=ExistingOutputRecorder)
        output_module_name = "transformers.utils.output_capturing"
        real_import_module = reference.importlib.import_module

        def import_module(name: str):
            if name == output_module_name:
                raise ModuleNotFoundError(
                    f"No module named {name!r}", name=output_module_name
                )
            if name == "transformers.utils.generic":
                return generic_module
            return real_import_module(name)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(raw_config), encoding="utf-8")
            with (
                patch.object(reference, "MODEL_PATH", root),
                patch.object(reference.importlib, "import_module", import_module),
                patch.dict(reference.sys.modules, {output_module_name: None}),
            ):
                self.assertTrue(reference._install_output_recorder_compat())
                self.assertIs(
                    reference.sys.modules[output_module_name].OutputRecorder,
                    ExistingOutputRecorder,
                )

    @unittest.skipUnless(
        (PINNED_K2_AURORA_1B_PATH / "config.json").is_file(),
        "pinned K2Aurora dense-1B artifact is not mounted",
    )
    def test_pinned_auto_config_gets_exact_yarn_theta(self) -> None:
        raw_config = json.loads((PINNED_K2_AURORA_1B_PATH / "config.json").read_text())
        self.assertNotIn("rope_theta", raw_config["rope_parameters"])

        with patch.object(reference, "MODEL_PATH", PINNED_K2_AURORA_1B_PATH):
            reference._install_k2_aurora_strict_compat()
            config = reference.AutoConfig.from_pretrained(
                PINNED_K2_AURORA_1B_PATH,
                trust_remote_code=True,
                local_files_only=True,
            )

        self.assertFalse(hasattr(config, "rope_theta"))
        self.assertNotIn("rope_theta", config.rope_parameters)
        self.assertTrue(
            reference._normalize_k2_aurora_dense_1b_rope_config(config, raw_config)
        )
        self.assertEqual(config.rope_theta, 1000000.0)
        self.assertEqual(config.rope_parameters["rope_theta"], 1000000.0)
        inv_freq, attention_scaling = reference.ROPE_INIT_FUNCTIONS["yarn"](
            config, device=None
        )
        self.assertEqual(inv_freq.shape, (32,))
        self.assertTrue(torch.isfinite(inv_freq).all())
        self.assertAlmostEqual(float(attention_scaling), 1.2772588722239782)


class CheckpointFormatCompatibilityTests(unittest.TestCase):
    def test_dense_safetensors_layout_needs_no_router_bias_restore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard_name = "model-00001-of-00001.safetensors"
            (root / shard_name).write_bytes(b"safetensors-placeholder")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"model.embed_tokens.weight": shard_name}}),
                encoding="utf-8",
            )
            layout = reference.resolve_checkpoint_layout(root)

            self.assertTrue(layout.use_safetensors)
            with patch.object(reference.torch, "load") as load:
                self.assertEqual(reference._restore_router_biases(object(), layout), [])
            load.assert_not_called()

    def test_safetensors_router_bias_restore_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard_name = "model-00001-of-00001.safetensors"
            (root / shard_name).write_bytes(b"safetensors-placeholder")
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {"weight_map": {"model.layers.0.mlp.gate.bias": shard_name}}
                ),
                encoding="utf-8",
            )
            layout = reference.resolve_checkpoint_layout(root)

            with self.assertRaisesRegex(ValueError, "only for PyTorch .bin"):
                reference._restore_router_biases(object(), layout)

    def test_pytorch_bin_router_bias_restore_preserves_fp32_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parameter_name = "model.layers.0.mlp.gate.bias"
            shard_name = "pytorch_model-00001-of-00001.bin"
            expected = torch.tensor([0.125, -0.25], dtype=torch.float32)
            torch.save({parameter_name: expected}, root / shard_name)
            (root / "pytorch_model.bin.index.json").write_text(
                json.dumps({"weight_map": {parameter_name: shard_name}}),
                encoding="utf-8",
            )
            layout = reference.resolve_checkpoint_layout(root)
            parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))

            class Model:
                @staticmethod
                def get_parameter(name: str) -> torch.nn.Parameter:
                    if name != parameter_name:
                        raise KeyError(name)
                    return parameter

            restored = reference._restore_router_biases(Model(), layout)

            self.assertFalse(layout.use_safetensors)
            self.assertEqual(parameter.dtype, torch.float32)
            self.assertTrue(torch.equal(parameter, expected))
            self.assertEqual([row["name"] for row in restored], [parameter_name])


if __name__ == "__main__":
    unittest.main()
