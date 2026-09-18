# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native dense K2 FP8 loading with public metadata and synthetic CPU tensors."""

import copy
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from k2_fp8_native_utils import native_cpu_context

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.k2_horizon import K2HorizonConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.layers.parameter import BlockQuantScaleParameter, ModelWeightParameter
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Fp8,
)
from sglang.srt.model_loader import loader as model_loader
from sglang.srt.model_loader.loader import PreshardedModelLoader, _post_load_weights
from sglang.srt.models import xllm
from sglang.srt.models.xllm import K2HorizonForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

FIXTURES = {
    "7b": "15d02f6cf02859da2cbd6170acc8195e03a43a51229a27059dd59f9df19dfefd",
    "32b": "c86f2175c4faa80e94e0e7207c25b833cbe58764a044351d61ed811d6d8c8fe0",
}


def _small_config(kind):
    path = Path(__file__).parent / f"fixtures/k2_horizon_{kind}_fp8_config.json"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURES[kind]
    metadata = json.loads(raw)
    divisor = 8 if kind == "7b" else 4
    for name in (
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
    ):
        assert metadata[name] % divisor == 0
        metadata[name] //= divisor
    metadata.update(
        num_hidden_layers=1,
        mlp_only_layers=[0],
        vocab_size=256,
        max_position_embeddings=32,
    )
    return K2HorizonConfig.from_dict(metadata)


def _source_tensors(config):
    hidden = config.hidden_size
    width = config.intermediate_size
    q_width = config.num_attention_heads * config.head_dim
    kv_width = config.num_key_value_heads * config.head_dim
    prefix = "model.layers.0"
    shapes = {
        "model.embed_tokens.weight": (config.vocab_size, hidden),
        "lm_head.weight": (config.vocab_size, hidden),
        "model.norm.weight": (hidden,),
        f"{prefix}.input_layernorm.weight": (hidden,),
        f"{prefix}.post_attention_layernorm.weight": (hidden,),
    }
    projections = {
        "self_attn.q_proj": (q_width, hidden),
        "self_attn.k_proj": (kv_width, hidden),
        "self_attn.v_proj": (kv_width, hidden),
        "self_attn.o_proj": (hidden, q_width),
        "mlp.gate_proj": (width, hidden),
        "mlp.up_proj": (width, hidden),
        "mlp.down_proj": (hidden, width),
    }
    for name, shape in projections.items():
        assert all(size % 128 == 0 for size in shape)
        shapes[f"{prefix}.{name}.weight"] = shape
        shapes[f"{prefix}.{name}.weight_scale"] = tuple(size // 128 for size in shape)
    tensors = {}
    for index, (name, shape) in enumerate(shapes.items()):
        rows = torch.arange(shape[0], dtype=torch.float32)
        values = rows if len(shape) == 1 else rows[:, None]
        if len(shape) == 2:
            values = values + 3 * torch.arange(shape[1], dtype=torch.float32)[None, :]
        values = ((values + index) % 29 + 1) / 4
        dtype = torch.bfloat16
        if name.startswith(prefix) and name.endswith(".weight") and len(shape) == 2:
            dtype = torch.float8_e4m3fn
        tensors[name] = values.to(dtype)
    assert len(tensors) == 19
    source_bytes = sum(
        value.numel() * value.element_size() for value in tensors.values()
    )
    assert source_bytes < 40 << 20
    return tensors


def _model_and_tensors(kind):
    config = _small_config(kind)
    metadata = copy.deepcopy(config.quantization_config)
    metadata["packed_modules_mapping"] = copy.deepcopy(
        K2HorizonForCausalLM.packed_modules_mapping
    )
    quantization = CompressedTensorsConfig.from_config(metadata)
    model = K2HorizonForCausalLM(config, quant_config=quantization)
    return model, _source_tensors(config)


def _expected_storage(config, tensors, tp, rank):
    prefix = "model.layers.0"
    result = {}
    for name in (
        "model.norm.weight",
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.post_attention_layernorm.weight",
    ):
        result[name] = tensors[name]
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        width = config.vocab_size // tp
        result[name] = tensors[name][rank * width : (rank + 1) * width]
    for suffix, unit in (("weight", 1), ("weight_scale", 128)):
        q_size = config.num_attention_heads * config.head_dim // tp // unit
        kv_size = max(1, config.num_key_value_heads // tp) * config.head_dim // unit
        kv_rank = rank if config.num_key_value_heads >= tp else 0
        pieces = []
        for projection, size, owner in (
            ("q_proj", q_size, rank),
            ("k_proj", kv_size, kv_rank),
            ("v_proj", kv_size, kv_rank),
        ):
            source = tensors[f"{prefix}.self_attn.{projection}.{suffix}"]
            pieces.append(source[owner * size : (owner + 1) * size].float())
        result[f"{prefix}.self_attn.qkv_proj.{suffix}"] = torch.cat(pieces, dim=0)
        width = config.intermediate_size // tp // unit
        pieces = [
            tensors[f"{prefix}.mlp.{projection}.{suffix}"][
                rank * width : (rank + 1) * width
            ].float()
            for projection in ("gate_proj", "up_proj")
        ]
        result[f"{prefix}.mlp.gate_up_proj.{suffix}"] = torch.cat(pieces, dim=0)
        for projection in ("self_attn.o_proj", "mlp.down_proj"):
            name = f"{prefix}.{projection}.{suffix}"
            source = tensors[name]
            width = source.shape[1] // tp
            result[name] = source[:, rank * width : (rank + 1) * width]
    return {name: value.float().clone() for name, value in result.items()}


@pytest.mark.parametrize("kind", ("7b", "32b"))
@pytest.mark.parametrize(
    ("tp", "rank"), [(1, 0), (2, 0), (2, 1)], ids=("tp1", "tp2-r0", "tp2-r1")
)
def test_dense_native_storage(monkeypatch, kind, tp, rank):
    with native_cpu_context(monkeypatch, tp=tp, rank=rank):
        model, tensors = _model_and_tensors(kind)
        expected = _expected_storage(
            config=model.config, tensors=tensors, tp=tp, rank=rank
        )
        model.load_weights(reversed(tuple(tensors.items())))
        actual = dict(model.named_parameters())
        assert actual.keys() == expected.keys()
        assert len(actual) == 13
        for name, value in expected.items():
            parameter = actual[name]
            if name.endswith(".weight_scale"):
                assert isinstance(parameter, BlockQuantScaleParameter)
                assert parameter.dtype == torch.float32
                assert callable(parameter.weight_loader)
            elif value.ndim == 2 and name.startswith("model.layers."):
                assert isinstance(parameter, ModelWeightParameter)
                assert parameter.dtype == torch.float8_e4m3fn
                assert callable(parameter.weight_loader)
            else:
                assert parameter.dtype == torch.bfloat16
            torch.testing.assert_close(parameter.float(), value.float(), rtol=0, atol=0)
        layer = model.model.layers[0]
        for linear in (
            layer.self_attn.qkv_proj,
            layer.self_attn.o_proj,
            layer.mlp.gate_up_proj,
            layer.mlp.down_proj,
        ):
            assert isinstance(linear.quant_method, CompressedTensorsLinearMethod)
            assert isinstance(linear.scheme, CompressedTensorsW8A8Fp8)
            assert linear.weight_block_size == [128, 128]
            assert getattr(linear, "input_scale", None) is None
        assert not hasattr(model.lm_head, "weight_scale")
        assert not hasattr(model.model.embed_tokens, "weight_scale")


@pytest.mark.parametrize("kind", ("7b", "32b"))
@pytest.mark.parametrize(
    "missing",
    ("self_attn.q_proj.weight", "mlp.up_proj.weight_scale"),
    ids=("q-weight", "up-scale"),
)
def test_dense_native_missing_tensor(monkeypatch, kind, missing):
    with native_cpu_context(monkeypatch):
        model, tensors = _model_and_tensors(kind)
        name = f"model.layers.0.{missing}"
        del tensors[name]
        with pytest.raises(ValueError, match=re.escape(name)):
            model.load_weights(tensors.items())


def _assert_dense_expected(model, expected):
    actual = dict(model.named_parameters())
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(actual[name].float(), value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "suffix", ("self_attn.q_proj.weight", "mlp.up_proj.weight_scale")
)
def test_dense_native_partial_update(monkeypatch, suffix):
    with native_cpu_context(monkeypatch, tp=2, rank=1):
        model, tensors = _model_and_tensors("7b")
        model.load_weights(tensors.items())
        name = f"model.layers.0.{suffix}"
        tensors[name] = torch.full_like(tensors[name].float(), 2).to(
            tensors[name].dtype
        )
        expected = _expected_storage(model.config, tensors, tp=2, rank=1)
        model.load_weights([(name, tensors[name])])
        _assert_dense_expected(model, expected)


def test_dense_native_failed_load_stays_pending(monkeypatch):
    with native_cpu_context(monkeypatch):
        model, tensors = _model_and_tensors("7b")
        name = "model.layers.0.self_attn.q_proj.weight"
        incomplete = {key: value for key, value in tensors.items() if key != name}
        with pytest.raises(ValueError, match=re.escape(name)):
            model.load_weights(incomplete.items())
        with pytest.raises(ValueError, match=re.escape("lm_head.weight")):
            model.load_weights([(name, tensors[name])])
        expected = _expected_storage(model.config, tensors, tp=1, rank=0)
        model.load_weights(tensors.items())
        _assert_dense_expected(model, expected)


def test_dense_native_packed_initial_load(monkeypatch):
    with native_cpu_context(monkeypatch):
        model, tensors = _model_and_tensors("7b")
        expected = _expected_storage(model.config, tensors, tp=1, rank=0)
        for packed, projections in (
            (
                "self_attn.qkv_proj",
                ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
            ),
            ("mlp.gate_up_proj", ("mlp.gate_proj", "mlp.up_proj")),
        ):
            for suffix in ("weight", "weight_scale"):
                values = [
                    tensors.pop(f"model.layers.0.{projection}.{suffix}")
                    for projection in projections
                ]
                tensors[f"model.layers.0.{packed}.{suffix}"] = torch.cat(
                    [value.float() for value in values], dim=0
                ).to(values[0].dtype)
        model.load_weights(tensors.items())
        _assert_dense_expected(model, expected)


def test_dense_native_fill_completion_allows_update(monkeypatch):
    with native_cpu_context(monkeypatch):
        model, tensors = _model_and_tensors("7b")
        expected = _expected_storage(model.config, tensors, tp=1, rank=0)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(expected[name].to(parameter.dtype))
        _post_load_weights(model)
        name = "model.layers.0.mlp.up_proj.weight_scale"
        tensors[name] = torch.full_like(tensors[name], 2)
        expected = _expected_storage(model.config, tensors, tp=1, rank=0)
        model.load_weights([(name, tensors[name])])
        _assert_dense_expected(model, expected)


@pytest.mark.parametrize("rank", (0, 1), ids=("pp-first", "pp-last"))
def test_dense_native_tied_pp_ownership(monkeypatch, rank):
    with native_cpu_context(monkeypatch), monkeypatch.context() as patch:
        group = SimpleNamespace(
            world_size=2,
            rank_in_group=rank,
            is_first_rank=rank == 0,
            is_last_rank=rank == 1,
        )
        patch.setattr(xllm, "get_pp_group", lambda: group)
        config = _small_config("7b")
        config.num_hidden_layers = 2
        config.mlp_only_layers = [0, 1]
        config.tie_word_embeddings = True
        metadata = copy.deepcopy(config.quantization_config)
        metadata["packed_modules_mapping"] = copy.deepcopy(
            K2HorizonForCausalLM.packed_modules_mapping
        )
        model = K2HorizonForCausalLM(
            config, quant_config=CompressedTensorsConfig.from_config(metadata)
        )
        tensors = _source_tensors(config)
        expected = _expected_storage(config, tensors, tp=1, rank=0)
        expected["lm_head.weight"] = (
            tensors["model.embed_tokens.weight"].float().clone()
        )
        tensors.update(
            {
                name.replace("layers.0.", "layers.1."): value
                for name, value in tuple(tensors.items())
                if name.startswith("model.layers.0.")
            }
        )
        prefix = f"model.layers.{rank}."
        owned = {
            name: value
            for name, value in tensors.items()
            if name.startswith(prefix)
            or name == "model.embed_tokens.weight"
            or (rank == 1 and name == "model.norm.weight")
        }
        expected = {
            name.replace("layers.0.", f"layers.{rank}."): value
            for name, value in expected.items()
            if name.startswith("model.layers.0.")
            or (rank == 0 and name == "model.embed_tokens.weight")
            or (rank == 1 and name in ("lm_head.weight", "model.norm.weight"))
        }
        model.load_weights(owned.items())
        _assert_dense_expected(model, expected)


def _prepare_presharded_restore(monkeypatch, tmp_path, model, weights, fault=None):
    load_config = LoadConfig(
        load_format="presharded",
        model_loader_extra_config={
            "presharded_path": str(tmp_path),
            "verify_on_load": True,
            "hash_num_threads": 1,
        },
    )
    loader = PreshardedModelLoader(load_config)
    config = SimpleNamespace(
        model_path=str(tmp_path),
        is_draft_model=False,
        dtype=next(model.parameters()).dtype,
    )
    shard_config = {"tp": 1}
    monkeypatch.setattr(loader, "_collect_shard_config", lambda config: shard_config)
    monkeypatch.setattr(model_loader, "_get_quantization_config", lambda *args: None)
    monkeypatch.setattr(model_loader, "_initialize_model", lambda *args: model)
    folder = Path(loader._presharded_dir(config, shard_config))
    folder.mkdir()
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    weights = dict(weights)
    if fault == "missing":
        del weights["model.norm.weight"]
    manifest = {
        name: {
            "checksum": loader._hash_tensor(tensor),
            "size": tensor.numel() * tensor.element_size(),
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        for name, tensor in weights.items()
    }
    (manifests / "manifest_00000.json").write_text(json.dumps(manifest))
    plan = loader._build_dump_plan(
        world_size=1, tmp_dir=str(manifests), max_file_bytes=1 << 30
    )
    plan["shard_config"] = shard_config
    if fault == "checksum":
        checksum = int(plan["rank_checksums"]["0"], 16) ^ 1
        plan["rank_checksums"]["0"] = f"{checksum:016x}"
    loader._dump_files_for_rank(weights, plan, 0, str(folder))
    (folder / loader.CHECKSUM_FILENAME).write_text(json.dumps(plan))
    (folder / loader.READY_FILENAME).write_text("{}")
    return loader, config


def _reject_post_load_transform():
    raise AssertionError("A restored native tensor must not be processed again")


def test_dense_native_presharded_update(monkeypatch, tmp_path):
    with native_cpu_context(monkeypatch):
        model, tensors = _model_and_tensors("7b")
        expected = _expected_storage(model.config, tensors, tp=1, rank=0)
        weights = {
            name: expected[name].to(parameter.dtype)
            for name, parameter in model.named_parameters()
        }
        monkeypatch.setattr(model, "post_load_weights", _reject_post_load_transform)
        loader, config = _prepare_presharded_restore(
            monkeypatch, tmp_path, model, weights
        )
        restored = loader.load_model(
            model_config=config, device_config=DeviceConfig("cpu")
        )
        assert restored is model
        assert model.training is False
        _assert_dense_expected(model, expected)
        name = "model.layers.0.mlp.up_proj.weight_scale"
        tensors[name] = torch.full_like(tensors[name], 2)
        expected = _expected_storage(model.config, tensors, tp=1, rank=0)
        model.load_weights([(name, tensors[name])])
        _assert_dense_expected(model, expected)


@pytest.mark.parametrize("fault", ("missing", "checksum"))
def test_dense_native_presharded_failure_stays_pending(monkeypatch, tmp_path, fault):
    with native_cpu_context(monkeypatch):
        model, tensors = _model_and_tensors("7b")
        expected = _expected_storage(model.config, tensors, tp=1, rank=0)
        weights = {
            name: expected[name].to(parameter.dtype)
            for name, parameter in model.named_parameters()
        }
        loader, config = _prepare_presharded_restore(
            monkeypatch, tmp_path, model, weights, fault
        )
        message = "Missing keys" if fault == "missing" else "checksum mismatch"
        with pytest.raises(ValueError, match=message):
            loader.load_model(model_config=config, device_config=DeviceConfig("cpu"))
        name = "model.layers.0.self_attn.q_proj.weight"
        with pytest.raises(ValueError, match="Dense K2 FP8 initial load is missing"):
            model.load_weights([(name, tensors[name])])


def test_dense_native_presharded_does_not_repeat_transforms(monkeypatch, tmp_path):
    model = torch.nn.Linear(2, 2, bias=False, dtype=torch.float32)
    parameter = model.weight
    expected = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    monkeypatch.setattr(
        model, "post_load_weights", _reject_post_load_transform, raising=False
    )
    loader, config = _prepare_presharded_restore(
        monkeypatch, tmp_path, model, {"weight": expected}
    )
    restored = loader.load_model(model_config=config, device_config=DeviceConfig("cpu"))
    assert restored is model
    assert model.weight is parameter
    assert model.training is False
    torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)


if __name__ == "__main__":
    import sys

    args = [arg for arg in sys.argv[1:] if arg != "-f"]
    sys.exit(pytest.main([__file__, "-v", *args]))
