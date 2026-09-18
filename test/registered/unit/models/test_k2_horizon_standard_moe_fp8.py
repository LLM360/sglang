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

"""Native standard-MoE loading with public K2 FP8 metadata and synthetic tensors."""

import copy
import hashlib
import json
import os
import re
from pathlib import Path

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

CONFIG_SHA256 = "938611cdbfdcd3bc860543fb2294d848c59d06d3d57c9668bab718ed31daad3e"
CONTEXT_SHA256 = "2bb3c15e965fd742596e5b9db71ecf1f6ead1276811fac26402fe2b27912e825"
EXPERT_PREFIX = "model.layers.3.mlp.experts"
BF16_SHAPES = {
    "lm_head.weight": (256, 768),
    "model.embed_tokens.weight": (256, 768),
    "model.layers.0.input_layernorm.weight": (768,),
    "model.layers.0.mlp.down_proj.weight": (768, 512),
    "model.layers.0.mlp.gate_proj.weight": (512, 768),
    "model.layers.0.mlp.up_proj.weight": (512, 768),
    "model.layers.0.post_attention_layernorm.weight": (768,),
    "model.layers.0.self_attn.k_proj.weight": (128, 768),
    "model.layers.0.self_attn.o_proj.weight": (768, 768),
    "model.layers.0.self_attn.q_proj.weight": (768, 768),
    "model.layers.0.self_attn.v_proj.weight": (128, 768),
    "model.layers.1.input_layernorm.weight": (768,),
    "model.layers.1.mlp.down_proj.weight": (768, 512),
    "model.layers.1.mlp.gate_proj.weight": (512, 768),
    "model.layers.1.mlp.up_proj.weight": (512, 768),
    "model.layers.1.post_attention_layernorm.weight": (768,),
    "model.layers.1.self_attn.k_proj.weight": (128, 768),
    "model.layers.1.self_attn.o_proj.weight": (768, 768),
    "model.layers.1.self_attn.q_proj.weight": (768, 768),
    "model.layers.1.self_attn.v_proj.weight": (128, 768),
    "model.layers.2.input_layernorm.weight": (768,),
    "model.layers.2.mlp.down_proj.weight": (768, 512),
    "model.layers.2.mlp.gate_proj.weight": (512, 768),
    "model.layers.2.mlp.up_proj.weight": (512, 768),
    "model.layers.2.post_attention_layernorm.weight": (768,),
    "model.layers.2.self_attn.k_proj.weight": (128, 768),
    "model.layers.2.self_attn.o_proj.weight": (768, 768),
    "model.layers.2.self_attn.q_proj.weight": (768, 768),
    "model.layers.2.self_attn.v_proj.weight": (128, 768),
    "model.layers.3.input_layernorm.weight": (768,),
    "model.layers.3.mlp.gate.bias": (4,),
    "model.layers.3.mlp.gate.weight": (4, 768),
    "model.layers.3.mlp.shared_experts.down_proj.weight": (768, 512),
    "model.layers.3.mlp.shared_experts.gate_proj.weight": (512, 768),
    "model.layers.3.mlp.shared_experts.up_proj.weight": (512, 768),
    "model.layers.3.post_attention_layernorm.weight": (768,),
    "model.layers.3.self_attn.k_proj.weight": (128, 768),
    "model.layers.3.self_attn.o_proj.weight": (768, 768),
    "model.layers.3.self_attn.q_proj.weight": (768, 768),
    "model.layers.3.self_attn.v_proj.weight": (128, 768),
    "model.norm.weight": (768,),
}
BF16_NAMES = tuple(sorted(BF16_SHAPES))
ROUTED_ROLES = (
    ("gate_proj", 0, (512, 768)),
    ("up_proj", 1, (512, 768)),
    ("down_proj", 2, (768, 512)),
)


def _make_model(*, experts=4, expert_width=512):
    from sglang.srt.configs.k2_horizon import K2HorizonConfig
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from sglang.srt.models.xllm import K2HorizonForCausalLM

    path = Path(__file__).parent / "fixtures/k2_horizon_375b_fp8_config.json"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CONFIG_SHA256
    public = json.loads(raw)
    metadata = copy.deepcopy(public)
    metadata.update(
        num_hidden_layers=4,
        hidden_size=768,
        intermediate_size=512,
        moe_intermediate_size=expert_width,
        num_experts=experts,
        num_experts_per_tok=2,
        num_attention_heads=6,
        num_key_value_heads=1,
        vocab_size=256,
        max_position_embeddings=128,
    )
    config = K2HorizonConfig.from_dict(metadata)
    quant_metadata = copy.deepcopy(config.quantization_config)
    quant_metadata["packed_modules_mapping"] = copy.deepcopy(
        K2HorizonForCausalLM.packed_modules_mapping
    )
    quantization = CompressedTensorsConfig.from_config(quant_metadata)
    model = K2HorizonForCausalLM(config, quant_config=quantization)
    assert config.quantization_config == public["quantization_config"]
    return model


def _source_identity(module):
    path = Path(module.__file__).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def native_moe_model(monkeypatch, request):
    import_environment = os.environ.get("SGLANG_USE_CPU_ENGINE")

    import k2_fp8_native_utils

    from sglang.srt.layers import vocab_parallel_embedding
    from sglang.srt.layers.moe import (
        MoeA2ABackend,
        MoeRunnerBackend,
        get_moe_a2a_backend,
        get_moe_runner_backend,
    )
    from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer
    from sglang.srt.layers.quantization.compressed_tensors import compressed_tensors
    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w8a8_fp8_moe,
    )
    from sglang.srt.models import xllm
    from sglang.srt.runtime_context import get_parallel

    tp, rank, ep = getattr(request, "param", (1, 0, 1))
    moe_tp = tp // ep
    with (
        k2_fp8_native_utils.native_cpu_context(monkeypatch, tp=tp, rank=rank),
        get_parallel().override(
            moe_tp_size=moe_tp,
            moe_tp_rank=rank % moe_tp,
            moe_ep_size=ep,
            moe_ep_rank=rank // moe_tp,
        ),
        torch.device("cpu"),
    ):
        observed = {
            "cpu_engine_before_native_import": import_environment,
            "cpu_engine_after_native_import": os.environ.get("SGLANG_USE_CPU_ENGINE"),
            "moe_cpu": fused_moe_layer._is_cpu,
            "moe_aiter": fused_moe_layer._use_aiter,
            "vocabulary_cpu": vocab_parallel_embedding._is_cpu,
            "runner": get_moe_runner_backend().value,
            "a2a": get_moe_a2a_backend().value,
            "tp": tp,
            "rank": rank,
            "moe_tp": moe_tp,
            "ep": ep,
            "modules": {
                module.__name__: _source_identity(module)
                for module in (
                    k2_fp8_native_utils,
                    xllm,
                    fused_moe_layer,
                    vocab_parallel_embedding,
                    compressed_tensors,
                    compressed_tensors_w8a8_fp8_moe,
                )
            },
        }
        print("K2 native branch before construction: " + json.dumps(observed))
        assert import_environment != "1", observed
        assert (
            observed["cpu_engine_after_native_import"] == import_environment
        ), observed
        assert fused_moe_layer._is_cpu is False, observed
        assert fused_moe_layer._use_aiter is False, observed
        assert vocab_parallel_embedding._is_cpu is False, observed
        assert get_moe_runner_backend() == MoeRunnerBackend.TRITON, observed
        assert get_moe_a2a_backend() == MoeA2ABackend.NONE, observed
        assert observed["modules"]["k2_fp8_native_utils"]["sha256"] == CONTEXT_SHA256
        model = _make_model()
        experts = model.model.layers[3].mlp.experts
        observed.update(
            flashinfer=experts.use_flashinfer_trtllm_moe,
            padded_loading=experts.use_padded_loading,
        )
        print("K2 native branch before loading: " + json.dumps(observed))
        assert experts.use_flashinfer_trtllm_moe is False, observed
        assert experts.use_padded_loading is False, observed
        yield model, tp, rank, ep


def _coordinates(shape, row_start=0, column_start=0):
    rows = torch.arange(row_start, row_start + shape[0], device="cpu")
    if len(shape) == 1:
        return rows, 0
    columns = torch.arange(column_start, column_start + shape[1], device="cpu")
    return rows[:, None], columns[None, :]


def _decode_bf16_words(words):
    fraction = (words & 127).to(torch.float32)
    exponent = ((words >> 7) & 255).to(torch.int32) - 127
    return torch.ldexp(1.0 + fraction / 128, exponent)


def _bf16_values(name, shape, row_start=0, column_start=0):
    rows, columns = _coordinates(shape, row_start, column_start)
    index = BF16_NAMES.index(name)
    words = 0x3F00 + 32 * index + (3 * rows + 7 * columns) % 29
    return _decode_bf16_words(words)


def _routed_values(expert, role, shape, row_start=0, column_start=0):
    rows, columns = _coordinates(shape, row_start, column_start)
    values = (
        3 * expert
        + 5 * role
        + rows // 128
        + 2 * (columns // 128)
        + rows % 7
        + columns % 5
    )
    return (1 + values % 15).to(torch.float32)


def _scale_values(expert, role, shape, row_start=0, column_start=0):
    rows, columns = _coordinates(shape, row_start, column_start)
    words = 0x3C00 + 256 * expert + 64 * role + 8 * rows + columns
    return _decode_bf16_words(words)


def _source_tensors():
    tensors = {
        name: _bf16_values(name, shape).to(torch.bfloat16)
        for name, shape in BF16_SHAPES.items()
    }
    for expert in range(4):
        for projection, role, shape in ROUTED_ROLES:
            prefix = f"{EXPERT_PREFIX}.{expert}.{projection}"
            tensors[f"{prefix}.weight"] = _routed_values(expert, role, shape).to(
                torch.float8_e4m3fn
            )
            scale_shape = tuple(size // 128 for size in shape)
            tensors[f"{prefix}.weight_scale"] = _scale_values(
                expert, role, scale_shape
            ).to(torch.bfloat16)
    assert len(tensors) == 65
    assert sum(t.numel() * t.element_size() for t in tensors.values()) == 25_972_808
    return dict(sorted(tensors.items()))


def _expected_mlp(prefix, tp, rank):
    width = 512 // tp
    pieces = [
        _bf16_values(
            f"{prefix}.{projection}.weight", (width, 768), row_start=rank * width
        )
        for projection in ("gate_proj", "up_proj")
    ]
    return {
        f"{prefix}.gate_up_proj.weight": torch.cat(pieces, dim=0),
        f"{prefix}.down_proj.weight": _bf16_values(
            f"{prefix}.down_proj.weight", (768, width), column_start=rank * width
        ),
    }


def _expected_bf16_storage(tp, rank):
    result = {}
    for name, shape in BF16_SHAPES.items():
        if len(shape) == 1 or name == "model.layers.3.mlp.gate.weight":
            result[name] = _bf16_values(name, shape)
    for name in ("lm_head.weight", "model.embed_tokens.weight"):
        width = 256 // tp
        result[name] = _bf16_values(name, (width, 768), row_start=rank * width)
    for layer in range(4):
        prefix = f"model.layers.{layer}"
        query_width = 768 // tp
        pieces = [
            _bf16_values(
                f"{prefix}.self_attn.q_proj.weight",
                (query_width, 768),
                row_start=rank * query_width,
            )
        ]
        for projection in ("k_proj", "v_proj"):
            pieces.append(
                _bf16_values(f"{prefix}.self_attn.{projection}.weight", (128, 768))
            )
        result[f"{prefix}.self_attn.qkv_proj.weight"] = torch.cat(pieces, dim=0)
        result[f"{prefix}.self_attn.o_proj.weight"] = _bf16_values(
            f"{prefix}.self_attn.o_proj.weight",
            (768, query_width),
            column_start=rank * query_width,
        )
        mlp = f"{prefix}.mlp" if layer < 3 else f"{prefix}.mlp.shared_experts"
        result.update(_expected_mlp(mlp, tp, rank))
    assert len(result) == 29
    return result


def _expected_routed_storage(tp, rank, ep):
    moe_tp = tp // ep
    moe_rank = rank % moe_tp
    local_experts = 4 // ep
    first_expert = (rank // moe_tp) * local_experts
    width = 512 // moe_tp
    result = {
        name: []
        for name in (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
        )
    }
    for expert in range(first_expert, first_expert + local_experts):
        for suffix, unit, values in (
            ("weight", 1, _routed_values),
            ("weight_scale", 128, _scale_values),
        ):
            local_width = width // unit
            hidden = 768 // unit
            pieces = [
                values(
                    expert,
                    role,
                    (local_width, hidden),
                    row_start=moe_rank * local_width,
                )
                for role in (0, 1)
            ]
            result[f"w13_{suffix}"].append(torch.cat(pieces, dim=0))
            result[f"w2_{suffix}"].append(
                values(
                    expert,
                    2,
                    (hidden, local_width),
                    column_start=moe_rank * local_width,
                )
            )
    return {name: torch.stack(values, dim=0) for name, values in result.items()}


def _assert_routed_storage(model, expected):
    from sglang.srt.layers.moe.fused_moe_triton import (
        FusedMoE,
        FusedMoeWeightScaleSupported,
    )
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsFusedMoEMethod,
    )
    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsW8A8Fp8MoE,
    )

    experts = model.model.layers[3].mlp.experts
    assert isinstance(experts, FusedMoE)
    assert isinstance(experts.quant_method, CompressedTensorsFusedMoEMethod)
    assert isinstance(experts.scheme, CompressedTensorsW8A8Fp8MoE)
    assert experts.quant_method.quantization_config is model.quant_config
    assert experts.weight_block_size == [128, 128]
    assert experts.w13_input_scale is None
    assert experts.w2_input_scale is None
    for name, value in expected.items():
        parameter = getattr(experts, name)
        assert isinstance(parameter, torch.nn.Parameter)
        assert parameter.weight_loader.__self__ is experts
        if name.endswith("weight_scale"):
            assert parameter.dtype == torch.float32
            assert parameter.quant_method == FusedMoeWeightScaleSupported.BLOCK.value
        else:
            assert parameter.dtype == torch.float8_e4m3fn
        torch.testing.assert_close(parameter.float(), value, rtol=0, atol=0)


def _assert_bf16_storage(model, expected, *, check_router_bias=True):
    parameters = dict(model.named_parameters())
    for name, value in expected.items():
        parameter = parameters[name]
        # The dedicated dtype case owns router-bias precision.
        if check_router_bias or name != "model.layers.3.mlp.gate.bias":
            assert parameter.dtype == torch.bfloat16, name
        torch.testing.assert_close(parameter.float(), value, rtol=0, atol=0)


def _assert_unquantized_methods(model):
    from sglang.srt.layers.quantization.unquant import (
        UnquantizedEmbeddingMethod,
        UnquantizedLinearMethod,
    )

    for layer_id, layer in enumerate(model.model.layers):
        mlp = layer.mlp if layer_id < 3 else layer.mlp.shared_experts
        for projection in (
            layer.self_attn.qkv_proj,
            layer.self_attn.o_proj,
            mlp.gate_up_proj,
            mlp.down_proj,
        ):
            assert isinstance(projection.quant_method, UnquantizedLinearMethod)
            assert not hasattr(projection, "weight_scale")
    for projection in (model.model.embed_tokens, model.lm_head):
        assert isinstance(projection.quant_method, UnquantizedEmbeddingMethod)
        assert not hasattr(projection, "weight_scale")


@pytest.mark.parametrize(
    "native_moe_model",
    [(1, 0, 1), (2, 0, 1), (2, 1, 1), (2, 0, 2), (2, 1, 2)],
    ids=("tp1", "tp2-r0", "tp2-r1", "ep2-r0", "ep2-r1"),
    indirect=True,
)
def test_standard_moe_native_storage(native_moe_model):
    model, tp, rank, ep = native_moe_model
    tensors = _source_tensors()
    expected = _expected_routed_storage(tp, rank, ep)
    bf16_expected = _expected_bf16_storage(tp, rank)
    stream = tuple(tensors.items())
    if (tp, rank, ep) in ((2, 0, 1), (2, 1, 2)):
        stream = reversed(stream)
    model.load_weights(stream)
    expected_names = set(bf16_expected) | {
        f"{EXPERT_PREFIX}.{name}" for name in expected
    }
    assert set(dict(model.named_parameters())) == expected_names
    _assert_routed_storage(model, expected)
    _assert_bf16_storage(model, bf16_expected, check_router_bias=False)
    _assert_unquantized_methods(model)
    experts = model.model.layers[3].mlp.experts
    experts.quant_method.process_weights_after_loading(experts)
    _assert_routed_storage(model, expected)
    _assert_bf16_storage(model, bf16_expected, check_router_bias=False)


def test_standard_moe_bf16_exclusions(native_moe_model):
    model, tp, rank, _ = native_moe_model
    mlp = model.model.layers[3].mlp
    assert mlp.topk.topk_config.correction_bias is mlp.gate.bias
    tensors = _source_tensors()
    expected = _expected_bf16_storage(tp, rank)
    model.load_weights(tensors.items())
    _assert_bf16_storage(model, expected)
    _assert_unquantized_methods(model)


@pytest.mark.parametrize(
    ("native_moe_model", "invalid"),
    [((2, 0, 1), "fp8-block"), ((2, 0, 2), "expert-count")],
    ids=("fp8-block", "expert-count"),
    indirect=["native_moe_model"],
)
def test_standard_moe_rejects_partial_partition(native_moe_model, invalid):
    # Fixture setup constructs and observes the valid twin first.
    if invalid == "fp8-block":
        with pytest.raises(ValueError) as caught:
            _make_model(expert_width=384)
        for word in ("192", "128", "gate", "up"):
            assert word in str(caught.value)
    else:
        with pytest.raises(AssertionError) as caught:
            _make_model(experts=3)
        assert str(caught.traceback[-1].path).endswith(
            "sglang/srt/layers/moe/fused_moe_triton/layer.py"
        )
        assert caught.traceback[-1].frame.code.name == "__init__"


@pytest.mark.parametrize(
    ("projection", "suffix"),
    [
        ("gate_proj", "weight"),
        ("gate_proj", "weight_scale"),
        ("up_proj", "weight"),
        ("up_proj", "weight_scale"),
        ("down_proj", "weight"),
        ("down_proj", "weight_scale"),
    ],
    ids=(
        "gate-weight",
        "gate-scale",
        "up-weight",
        "up-scale",
        "down-weight",
        "down-scale",
    ),
)
def test_standard_moe_rejects_missing_owned_tensor(
    native_moe_model, projection, suffix
):
    model, _, _, _ = native_moe_model
    tensors = _source_tensors()
    name = f"{EXPERT_PREFIX}.0.{projection}.{suffix}"
    del tensors[name]
    with pytest.raises(ValueError, match=re.escape(name)):
        model.load_weights(tensors.items())


@pytest.mark.parametrize(
    ("native_moe_model", "short"),
    [((2, 1, 1), "gate-weight-row"), ((2, 1, 1), "down-scale-column")],
    ids=("gate-weight-row", "down-scale-column"),
    indirect=["native_moe_model"],
)
def test_standard_moe_rejects_partial_tensor(native_moe_model, short):
    model, _, _, _ = native_moe_model
    tensors = _source_tensors()
    if short == "gate-weight-row":
        name = f"{EXPERT_PREFIX}.0.gate_proj.weight"
        tensors[name] = tensors[name][:-1, :].clone()
        assert tuple(tensors[name].shape) == (511, 768)
    else:
        name = f"{EXPERT_PREFIX}.0.down_proj.weight_scale"
        tensors[name] = tensors[name][:, :-1].clone()
        assert tuple(tensors[name].shape) == (6, 3)
    with pytest.raises(ValueError, match=re.escape(name)):
        model.load_weights(tensors.items())


@pytest.mark.parametrize(
    ("native_moe_model", "owner"),
    [((2, 1, 2), True), ((2, 0, 2), False)],
    ids=("owner", "nonowner"),
    indirect=["native_moe_model"],
)
def test_standard_moe_missing_remote_expert(native_moe_model, owner):
    model, tp, rank, ep = native_moe_model
    tensors = _source_tensors()
    missing_prefix = f"{EXPERT_PREFIX}.3."
    missing_names = [name for name in tensors if name.startswith(missing_prefix)]
    assert len(missing_names) == 6
    for name in missing_names:
        del tensors[name]
    if owner:
        with pytest.raises(ValueError, match=re.escape(missing_prefix)):
            model.load_weights(tensors.items())
    else:
        expected = _expected_routed_storage(tp, rank, ep)
        model.load_weights(tensors.items())
        _assert_routed_storage(model, expected)


@pytest.mark.parametrize(
    ("native_moe_model", "projection", "suffix"),
    [((2, 1, 1), "gate_proj", "weight"), ((2, 1, 1), "up_proj", "weight_scale")],
    ids=("gate-weight", "up-scale"),
    indirect=["native_moe_model"],
)
def test_standard_moe_partial_update(native_moe_model, projection, suffix):
    model, tp, rank, ep = native_moe_model
    tensors = _source_tensors()
    model.load_weights(tensors.items())
    name = f"{EXPERT_PREFIX}.0.{projection}.{suffix}"
    value = 2 if suffix == "weight" else 0.125
    tensors[name] = torch.full_like(tensors[name].float(), value).to(tensors[name].dtype)
    expected = _expected_routed_storage(tp, rank, ep)
    if suffix == "weight":
        expected["w13_weight"][0, :256, :] = value
    else:
        expected["w13_weight_scale"][0, 2:, :] = value
    model.load_weights([(name, tensors[name])])
    _assert_routed_storage(model, expected)
    _assert_bf16_storage(model, _expected_bf16_storage(tp, rank))


def test_standard_moe_failed_load_stays_pending(native_moe_model):
    model, tp, rank, ep = native_moe_model
    tensors = _source_tensors()
    name = f"{EXPERT_PREFIX}.0.gate_proj.weight"
    incomplete = {key: value for key, value in tensors.items() if key != name}
    with pytest.raises(ValueError, match=re.escape(name)):
        model.load_weights(incomplete.items())
    missing = f"{EXPERT_PREFIX}.0.down_proj.weight"
    with pytest.raises(ValueError, match=re.escape(missing)):
        model.load_weights([(name, tensors[name])])
    expected = _expected_routed_storage(tp, rank, ep)
    model.load_weights(tensors.items())
    _assert_routed_storage(model, expected)
    _assert_bf16_storage(model, _expected_bf16_storage(tp, rank))


@pytest.mark.parametrize(
    "native_moe_model", [(2, 1, 2)], ids=("ep2-r1",), indirect=True
)
def test_standard_moe_presharded_restore_allows_update(
    native_moe_model, monkeypatch, tmp_path
):
    import test_k2_horizon_compressed_fp8 as dense_tests

    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.model_loader import loader as model_loader

    print(
        "K2 presharded providers: "
        + json.dumps(
            {
                "tests": _source_identity(dense_tests),
                "loader": _source_identity(model_loader),
            }
        )
    )
    model, tp, rank, ep = native_moe_model
    expected = _expected_routed_storage(tp, rank, ep)
    bf16_expected = _expected_bf16_storage(tp, rank)
    all_expected = dict(bf16_expected)
    all_expected.update(
        {f"{EXPERT_PREFIX}.{name}": value for name, value in expected.items()}
    )
    parameters = dict(model.named_parameters())
    assert parameters.keys() == all_expected.keys()
    weights = {
        name: all_expected[name].to(parameter.dtype)
        for name, parameter in parameters.items()
    }
    monkeypatch.setattr(
        model, "post_load_weights", dense_tests._reject_post_load_transform
    )
    loader, config = dense_tests._prepare_presharded_restore(
        monkeypatch, tmp_path, model, weights
    )
    restored = loader.load_model(model_config=config, device_config=DeviceConfig("cpu"))
    assert restored is model
    assert model.training is False
    _assert_routed_storage(model, expected)
    _assert_bf16_storage(model, bf16_expected)
    name = f"{EXPERT_PREFIX}.3.down_proj.weight_scale"
    replacement = torch.full((6, 4), 0.125, dtype=torch.bfloat16)
    expected["w2_weight_scale"][1, :, :] = 0.125
    model.load_weights([(name, replacement)])
    _assert_routed_storage(model, expected)
    _assert_bf16_storage(model, bf16_expected)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
