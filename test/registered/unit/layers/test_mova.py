import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import PretrainedConfig

from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.mova import (
    RoutedValueExperts,
    mova_router_topk,
    routed_linear,
    routed_linear_reference,
)
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.model_loader.parameter_mapper import ParameterMapper
from sglang.srt.models.xllm import (
    XllmGroupRMSNorm,
    XllmMoEGate,
    XllmMoVAAttention,
    XllmQKGParallelLinear,
    _XLLM_SOURCE_ROUTER_PARTITIONS_CONFIG_KEY,
    _get_xllm_source_router_gemm_partitions,
    _interleave_rope_weight,
    _validate_mova_config,
    _xllm_router_gemm,
    _xllm_stacked_params_mapping,
    _XllmMoVAAttentionBase,
)


@pytest.fixture(autouse=True)
def _server_args_for_kernel_helpers(monkeypatch):
    """ModelRunner normally installs these globals before model creation."""

    monkeypatch.setattr(
        "sglang.srt.server_args._global_server_args",
        SimpleNamespace(
            enable_deterministic_inference=False,
            rl_on_policy_target=None,
        ),
    )


def _valid_mova_config(**overrides):
    values = dict(
        num_values=64,
        num_values_per_tok=4,
        num_hidden_layers=48,
        num_dense_layers=3,
        mlp_only_layers=[0, 1, 2],
        decoder_sparse_step=1,
        num_experts=100,
        hidden_size=2560,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        rope_head_dim=128,
        attention_bias=False,
        query_key_norm=False,
        apply_attn_gate=True,
        attn_gate_func="softplus",
        rope_scaling=None,
        sliding_window=None,
        use_sliding_window=False,
        router_score_func="sigmoid",
        router_scaling_factor=2.5,
        layernorm_num_groups=2,
        xllm_source_router_gemm_partitions=2,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _native_mp2_router_gemm_reference(hidden, weight):
    hidden_parts = hidden.chunk(2, dim=-1)
    weight_parts = weight.chunk(2, dim=-1)
    return sum(
        F.linear(hidden_part.contiguous(), weight_part.contiguous()).float()
        for hidden_part, weight_part in zip(hidden_parts, weight_parts)
    )


def _native_router_topk_reference(
    router_logits,
    router_bias,
    *,
    top_k,
    scaling_factor,
    output_dtype=torch.float32,
):
    scores = torch.sigmoid(router_logits.float())
    selected = torch.topk(scores + router_bias.float(), top_k, dim=-1).indices
    weights = torch.gather(scores, dim=-1, index=selected)
    if top_k > 1:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return (
        (weights * scaling_factor).to(output_dtype),
        selected.to(torch.int32),
    )


def _canonical_routes(weights, selected):
    order = torch.argsort(selected.long(), dim=-1)
    return (
        torch.gather(weights, dim=-1, index=order),
        torch.gather(selected.long(), dim=-1, index=order),
    )


def _real_shape_boundary_router_case(*, num_routes, top_k, device):
    """Build a literal BF16 rounding boundary where MP2 flips the last route."""

    hidden_size = 2560
    split = hidden_size // 2
    native_candidate = top_k - 1
    full_gemm_candidate = top_k

    hidden = torch.zeros(1, hidden_size, device=device, dtype=torch.bfloat16)
    weight = torch.zeros(
        num_routes, hidden_size, device=device, dtype=torch.bfloat16
    )
    hidden[0, 0] = 1.0
    hidden[0, split] = 1.0
    weight[native_candidate, 0] = 1.0
    weight[native_candidate, split] = 2**-8
    weight[full_gemm_candidate, 0] = 1.0

    native_logits = _native_mp2_router_gemm_reference(hidden, weight)
    full_gemm_logits = F.linear(hidden, weight).float()
    # Analytic literal golden for native xLLM 5494c84 MP2 ordering: each BF16
    # partial is rounded before its FP32 cast and the FP32 all-reduce sum.
    expected_native_pair = torch.tensor(
        [[1.0 + 2**-8, 1.0]], device=device, dtype=torch.float32
    )
    expected_full_pair = torch.ones(1, 2, device=device, dtype=torch.float32)
    torch.testing.assert_close(
        native_logits[:, native_candidate : full_gemm_candidate + 1],
        expected_native_pair,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        full_gemm_logits[:, native_candidate : full_gemm_candidate + 1],
        expected_full_pair,
        rtol=0.0,
        atol=0.0,
    )

    native_scores = torch.sigmoid(native_logits)
    full_gemm_scores = torch.sigmoid(full_gemm_logits)
    native_margin = (
        native_scores[0, native_candidate]
        - native_scores[0, full_gemm_candidate]
    )
    full_gemm_margin = (
        full_gemm_scores[0, native_candidate]
        - full_gemm_scores[0, full_gemm_candidate]
    )
    assert native_margin > full_gemm_margin

    bias = torch.full((num_routes,), -10.0, device=device, dtype=torch.float32)
    bias[: top_k - 1] = 10.0
    bias[native_candidate] = 0.0
    bias[full_gemm_candidate] = (native_margin + full_gemm_margin) / 2
    return hidden, weight, bias, native_logits, full_gemm_logits


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_xllm_router_gemm_matches_native_mp2_rounding_reference(device):
    hidden = (
        torch.sin(torch.arange(24, device=device, dtype=torch.float32) * 0.173)
        .view(3, 8)
        .to(torch.bfloat16)
    )
    weight = (
        torch.cos(torch.arange(80, device=device, dtype=torch.float32) * 0.097)
        .view(10, 8)
        .to(torch.bfloat16)
    )

    actual = _xllm_router_gemm(hidden, weight, source_partitions=2)
    expected = _native_mp2_router_gemm_reference(hidden, weight)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_xllm_router_gemm_explicit_mp1_returns_fp32_native_logits(device):
    hidden = torch.randn(3, 8, device=device, dtype=torch.bfloat16)
    weight = torch.randn(10, 8, device=device, dtype=torch.bfloat16)

    actual = _xllm_router_gemm(hidden, weight, source_partitions=1)
    expected = F.linear(hidden, weight).float()

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_xllm_router_gemm_missing_provenance_preserves_legacy_behavior():
    hidden = torch.randn(3, 8, dtype=torch.bfloat16)
    weight = torch.randn(10, 8, dtype=torch.bfloat16)

    actual = _xllm_router_gemm(hidden, weight, source_partitions=None)
    expected = F.linear(hidden, weight)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


_MISSING_ROUTER_PROVENANCE = object()


@pytest.mark.parametrize(
    "partitions,expected",
    [
        pytest.param(_MISSING_ROUTER_PROVENANCE, None, id="missing-legacy"),
        pytest.param(1, 1, id="explicit-mp1"),
        pytest.param(2, 2, id="explicit-mp2"),
    ],
)
def test_xllm_source_router_partition_config_roundtrip(partitions, expected):
    kwargs = {"hidden_size": 8}
    if partitions is not _MISSING_ROUTER_PROVENANCE:
        kwargs[_XLLM_SOURCE_ROUTER_PARTITIONS_CONFIG_KEY] = partitions
    config = PretrainedConfig(**kwargs)
    restored = PretrainedConfig.from_dict(config.to_dict())

    assert _get_xllm_source_router_gemm_partitions(restored) == expected
    assert (
        _XLLM_SOURCE_ROUTER_PARTITIONS_CONFIG_KEY in restored.to_dict()
    ) == (partitions is not _MISSING_ROUTER_PROVENANCE)


@pytest.mark.parametrize("partitions", [None, True, 0, -1, 3, 1.5, "2"])
def test_xllm_source_router_partition_config_rejects_bad_values(partitions):
    config = SimpleNamespace(
        hidden_size=8,
        xllm_source_router_gemm_partitions=partitions,
    )

    with pytest.raises(
        ValueError,
        match="source_router_gemm_partitions.*Omit the key",
    ):
        _get_xllm_source_router_gemm_partitions(config)


def test_xllm_source_router_partition_config_rejects_odd_hidden_size():
    config = SimpleNamespace(
        hidden_size=7,
        xllm_source_router_gemm_partitions=2,
    )

    with pytest.raises(ValueError, match="requires hidden_size divisible"):
        _get_xllm_source_router_gemm_partitions(config)


@pytest.mark.parametrize(
    "hidden,weight,partitions,error",
    [
        (
            torch.randn(2, 7, dtype=torch.bfloat16),
            torch.randn(4, 7, dtype=torch.bfloat16),
            2,
            "divisible",
        ),
        (torch.randn(2, 8), torch.randn(4, 8), 1, "BF16"),
        (torch.randn(2, 8), torch.randn(4, 8), 2, "BF16"),
        (
            torch.randn(2, 8, dtype=torch.bfloat16),
            torch.randn(4, 6, dtype=torch.bfloat16),
            2,
            "differ",
        ),
    ],
)
def test_xllm_router_gemm_rejects_invalid_runtime_contract(
    hidden, weight, partitions, error
):
    with pytest.raises(ValueError, match=error):
        _xllm_router_gemm(hidden, weight, source_partitions=partitions)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_ffn_top8_routes_match_native_mp2_at_real_shape_boundary(device):
    top_k = 8
    scaling_factor = 2.5
    hidden, weight, bias, native_logits, full_gemm_logits = (
        _real_shape_boundary_router_case(
            num_routes=100,
            top_k=top_k,
            device=device,
        )
    )
    config = SimpleNamespace(
        hidden_size=2560,
        num_experts=100,
        num_experts_per_tok=top_k,
        moe_gate_bias=True,
        xllm_source_router_gemm_partitions=2,
    )
    gate = XllmMoEGate(config).to(device=device)
    gate.weight.data = weight
    with torch.no_grad():
        gate.bias.copy_(bias)
    topk = TopK(
        top_k=top_k,
        renormalize=True,
        scoring_func="sigmoid",
        correction_bias=gate.bias,
    )

    actual_logits = gate(hidden)
    actual_topk = topk.forward_native(hidden, actual_logits)
    actual_weights = actual_topk.topk_weights * scaling_factor
    expected_weights, expected_ids = _native_router_topk_reference(
        native_logits,
        bias,
        top_k=top_k,
        scaling_factor=scaling_factor,
    )
    _, full_gemm_ids = _native_router_topk_reference(
        full_gemm_logits,
        bias,
        top_k=top_k,
        scaling_factor=scaling_factor,
    )

    torch.testing.assert_close(actual_logits, native_logits, rtol=0.0, atol=0.0)
    actual_weights, actual_ids = _canonical_routes(
        actual_weights, actual_topk.topk_ids
    )
    expected_weights, expected_ids = _canonical_routes(
        expected_weights, expected_ids
    )
    _, full_gemm_ids = _canonical_routes(expected_weights, full_gemm_ids)
    assert actual_ids.tolist() == [list(range(8))]
    assert full_gemm_ids.tolist() == [[0, 1, 2, 3, 4, 5, 6, 8]]
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_value_top4_routes_match_native_mp2_at_real_shape_boundary(device):
    top_k = 4
    scaling_factor = 2.5
    hidden, weight, bias, native_logits, full_gemm_logits = (
        _real_shape_boundary_router_case(
            num_routes=64,
            top_k=top_k,
            device=device,
        )
    )
    attention = object.__new__(XllmMoVAAttention)
    torch.nn.Module.__init__(attention)
    attention.source_router_gemm_partitions = 2
    attention.router_score_func = "sigmoid"
    attention.router_scaling_factor = scaling_factor
    attention.renormalize = True
    attention.num_values_per_tok = top_k
    attention.v_router = torch.nn.Linear(
        2560, 64, bias=False, device=device, dtype=torch.bfloat16
    )
    attention.v_router.bias = torch.nn.Parameter(
        bias.clone(), requires_grad=False
    )
    with torch.no_grad():
        attention.v_router.weight.copy_(weight)

    seen = {}

    class RecordingValueExperts(torch.nn.Module):
        def forward(self, hidden_states, routing_weights, selected_values):
            seen["weights"] = routing_weights
            seen["ids"] = selected_values
            # The real CPU and CUDA value-expert paths cast these coefficients
            # to the BF16 projected activation immediately before multiplying.
            seen["native_weights"] = routing_weights.to(hidden_states.dtype)
            return hidden_states

    attention.v_experts = RecordingValueExperts()
    output = attention._project_value(hidden)
    expected_weights, expected_ids = _native_router_topk_reference(
        native_logits,
        bias,
        top_k=top_k,
        scaling_factor=scaling_factor,
        output_dtype=torch.bfloat16,
    )
    _, full_gemm_ids = _native_router_topk_reference(
        full_gemm_logits,
        bias,
        top_k=top_k,
        scaling_factor=scaling_factor,
    )

    assert seen["weights"].dtype == torch.float32
    actual_weights, actual_ids = _canonical_routes(
        seen["native_weights"], seen["ids"]
    )
    expected_weights, expected_ids = _canonical_routes(
        expected_weights, expected_ids
    )
    _, full_gemm_ids = _canonical_routes(expected_weights, full_gemm_ids)
    assert actual_ids.tolist() == [[0, 1, 2, 3]]
    assert full_gemm_ids.tolist() == [[0, 1, 2, 4]]
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(output, hidden)


def test_mova_router_bias_changes_selection_not_weight():
    logits = torch.tensor([[2.0, 1.0, -1.0]], dtype=torch.float32)
    bias = torch.tensor([0.0, 0.0, 10.0], dtype=torch.float32)

    weights, selected = mova_router_topk(
        logits,
        bias,
        score_func="sigmoid",
        top_k=1,
        scaling_factor=2.5,
    )

    assert selected.tolist() == [[2]]
    expected = torch.sigmoid(logits)[0, 2] * 2.5
    torch.testing.assert_close(weights[0, 0], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("top_k", [1, 4])
def test_fused_mova_router_matches_selection_only_reference(top_k):
    torch.manual_seed(5)
    logits = torch.randn(19, 64, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(64, device="cuda", dtype=torch.float32)

    weights, selected = mova_router_topk(
        logits,
        bias,
        score_func="sigmoid",
        top_k=top_k,
        scaling_factor=2.5,
    )
    scores = torch.sigmoid(logits.float())
    expected_selected = torch.topk(scores + bias, top_k, dim=-1).indices
    expected_weights = torch.gather(scores, 1, expected_selected)
    if top_k > 1:
        expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
    expected_weights = (expected_weights * 2.5).to(logits.dtype)

    torch.testing.assert_close(selected.long(), expected_selected)
    torch.testing.assert_close(weights, expected_weights, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_routed_linear_reference_matches_explicit_mixture(top_k):
    torch.manual_seed(7)
    hidden = torch.randn(5, 6)
    experts = torch.randn(4, 3, 6)
    selected = torch.stack([torch.randperm(4)[:top_k] for _ in range(hidden.shape[0])])
    routing = torch.rand(hidden.shape[0], top_k)

    actual = routed_linear_reference(hidden, experts, routing, selected)
    expected = torch.zeros_like(actual)
    for token in range(hidden.shape[0]):
        for slot in range(top_k):
            expert = selected[token, slot]
            expected[token] += routing[token, slot] * F.silu(
                F.linear(hidden[token], experts[expert])
            )

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 17, 257])
def test_fused_routed_linear_matches_reference(num_tokens):
    torch.manual_seed(11)
    device = torch.device("cuda")
    hidden = torch.randn(num_tokens, 64, device=device, dtype=torch.bfloat16)
    experts = torch.randn(8, 32, 64, device=device, dtype=torch.bfloat16)
    # Include a hot expert while leaving at least one expert empty.
    selected = torch.randint(0, 7, (num_tokens, 4), device=device, dtype=torch.int32)
    selected[:, 0] = 0
    routing = torch.rand(num_tokens, 4, device=device, dtype=torch.bfloat16)

    expected = routed_linear_reference(hidden, experts, routing, selected)
    actual = routed_linear(hidden, experts, routing, selected)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_routed_linear_matches_36b_tp8_shape():
    torch.manual_seed(17)
    hidden = torch.randn(9, 2560, device="cuda", dtype=torch.bfloat16)
    experts = torch.randn(64, 128, 2560, device="cuda", dtype=torch.bfloat16)
    selected = torch.randint(0, 64, (9, 4), device="cuda", dtype=torch.int32)
    routing = torch.rand(9, 4, device="cuda", dtype=torch.bfloat16)

    expected = routed_linear_reference(hidden, experts, routing, selected)
    actual = routed_linear(hidden, experts, routing, selected)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 128])
def test_fused_routed_linear_cuda_graph_capture_and_replay(num_tokens):
    torch.manual_seed(23)
    hidden = torch.randn(num_tokens, 64, device="cuda", dtype=torch.bfloat16)
    experts = torch.randn(8, 32, 64, device="cuda", dtype=torch.bfloat16)
    selected = torch.randint(0, 8, (num_tokens, 4), device="cuda", dtype=torch.int32)
    routing = torch.rand(num_tokens, 4, device="cuda", dtype=torch.bfloat16)

    # Warm up Triton compilation and allocator state outside capture.
    routed_linear(hidden, experts, routing, selected)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = routed_linear(hidden, experts, routing, selected)

    hidden.copy_(torch.randn_like(hidden))
    selected.copy_(torch.randint_like(selected, 0, 8))
    routing.copy_(torch.rand_like(routing))
    expected = routed_linear_reference(hidden, experts, routing, selected)
    graph.replay()
    torch.testing.assert_close(captured, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 128])
def test_fused_routed_linear_torch_compile_fullgraph(num_tokens):
    torch.manual_seed(29)
    hidden = torch.randn(num_tokens, 64, device="cuda", dtype=torch.bfloat16)
    experts = torch.randn(8, 32, 64, device="cuda", dtype=torch.bfloat16)
    selected = torch.randint(0, 8, (num_tokens, 4), device="cuda", dtype=torch.int32)
    routing = torch.rand(num_tokens, 4, device="cuda", dtype=torch.bfloat16)
    compiled = torch.compile(routed_linear, backend="eager", fullgraph=True)

    expected = routed_linear_reference(hidden, experts, routing, selected)
    actual = compiled(hidden, experts, routing, selected)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_value_experts_load_output_shards_and_forward():
    layer = RoutedValueExperts(
        num_experts=3,
        input_size=4,
        output_size=6,
        tp_rank=1,
        tp_size=2,
    )
    full = torch.arange(3 * 6 * 4, dtype=torch.float32).view(3, 6, 4)
    layer.weight_loader(layer.weight, full)
    torch.testing.assert_close(layer.weight, full[:, 3:])

    replacement = torch.full((6, 4), -2.0)
    layer.weight_loader(layer.weight, replacement, 1)
    torch.testing.assert_close(layer.weight[1], replacement[3:])

    hidden = torch.randn(2, 4)
    selected = torch.tensor([[0, 1], [2, 1]])
    routing = torch.tensor([[0.25, 0.75], [0.6, 0.4]])
    torch.testing.assert_close(
        layer(hidden, routing, selected),
        routed_linear_reference(hidden, layer.weight, routing, selected),
    )


def test_qkg_loader_packs_one_local_gqa_group_and_interleaves_qk():
    hidden_size = 3
    num_heads = 4
    num_kv_heads = 2
    head_dim = 4
    qkg = XllmQKGParallelLinear(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        tp_rank=1,
        tp_size=2,
    )
    q = torch.arange(num_heads * head_dim * hidden_size, dtype=torch.float32).view(
        num_heads * head_dim, hidden_size
    )
    gate = q + 1000
    k = (
        torch.arange(num_kv_heads * head_dim * hidden_size, dtype=torch.float32).view(
            num_kv_heads * head_dim, hidden_size
        )
        + 2000
    )

    qkg.weight_loader(qkg.weight, q, "q")
    qkg.weight_loader(qkg.weight, gate, "gate")
    qkg.weight_loader(qkg.weight, k, "k")

    packed = qkg.weight.view(1, 5, head_dim, hidden_size)
    expected_q = _interleave_rope_weight(q, num_heads).view(
        num_kv_heads, 2, head_dim, hidden_size
    )[1]
    expected_gate = gate.view(num_kv_heads, 2, head_dim, hidden_size)[1]
    expected_k = _interleave_rope_weight(k, num_kv_heads).view(
        num_kv_heads, 1, head_dim, hidden_size
    )[1]
    torch.testing.assert_close(packed[0, :2], expected_q)
    torch.testing.assert_close(packed[0, 2:4], expected_gate)
    torch.testing.assert_close(packed[0, 4:], expected_k)

    x = torch.randn(2, hidden_size)
    q_out, k_out, gate_out = qkg(x)
    torch.testing.assert_close(q_out, F.linear(x, expected_q.reshape(-1, hidden_size)))
    torch.testing.assert_close(k_out, F.linear(x, expected_k.reshape(-1, hidden_size)))
    torch.testing.assert_close(
        gate_out, F.linear(x, expected_gate.reshape(-1, hidden_size))
    )


def test_zero_centered_group_norm_uses_weight_plus_one():
    norm = XllmGroupRMSNorm(4, n_groups=2, eps=0.0, zero_centered=True)
    x = torch.tensor([[3.0, 4.0, 5.0, 12.0]])
    expected = torch.tensor([[3.0, 4.0, 5.0, 12.0]]) / torch.tensor(
        [[12.5**0.5, 12.5**0.5, 84.5**0.5, 84.5**0.5]]
    )
    torch.testing.assert_close(norm(x), expected)

    norm.weight.data.fill_(1.0)
    torch.testing.assert_close(norm(x), 2 * expected)


def test_group_norm_preserves_legacy_direct_weight_semantics():
    norm = XllmGroupRMSNorm(4, n_groups=2, eps=0.0, zero_centered=False)
    norm.weight.data.fill_(2.0)
    x = torch.tensor([[3.0, 4.0, 5.0, 12.0]])
    expected = (
        2
        * torch.tensor([[3.0, 4.0, 5.0, 12.0]])
        / torch.tensor([[12.5**0.5, 12.5**0.5, 84.5**0.5, 84.5**0.5]])
    )
    torch.testing.assert_close(norm(x), expected)


def test_mova_config_rejects_misaligned_attention_and_ffn_layout(monkeypatch):
    config = _valid_mova_config(mlp_only_layers=[0, 1])
    monkeypatch.setattr(
        "sglang.srt.models.xllm.torch.get_default_dtype", lambda: torch.bfloat16
    )
    monkeypatch.setattr("sglang.srt.models.xllm.get_attention_tp_size", lambda: 1)
    with pytest.raises(ValueError, match="mlp_only_layers"):
        _validate_mova_config(config, quant_config=None)


def test_mova_config_accepts_36b_contract(monkeypatch):
    monkeypatch.setattr(
        "sglang.srt.models.xllm.torch.get_default_dtype", lambda: torch.bfloat16
    )
    monkeypatch.setattr("sglang.srt.models.xllm.get_attention_tp_size", lambda: 8)
    _validate_mova_config(_valid_mova_config(), quant_config=None)


@pytest.mark.parametrize(
    "partitions",
    [
        pytest.param(_MISSING_ROUTER_PROVENANCE, id="missing-legacy"),
        pytest.param(1, id="explicit-mp1"),
        pytest.param(2, id="explicit-mp2"),
    ],
)
def test_mova_config_accepts_supported_source_router_modes(monkeypatch, partitions):
    config = _valid_mova_config()
    if partitions is _MISSING_ROUTER_PROVENANCE:
        delattr(config, _XLLM_SOURCE_ROUTER_PARTITIONS_CONFIG_KEY)
    else:
        setattr(config, _XLLM_SOURCE_ROUTER_PARTITIONS_CONFIG_KEY, partitions)
    monkeypatch.setattr(
        "sglang.srt.models.xllm.torch.get_default_dtype", lambda: torch.bfloat16
    )
    monkeypatch.setattr("sglang.srt.models.xllm.get_attention_tp_size", lambda: 8)

    _validate_mova_config(config, quant_config=None)


@pytest.mark.parametrize(
    "rope_scaling",
    [
        {"rope_type": "default", "rope_theta": 10_000_000.0},
        {"type": "default"},
    ],
)
def test_mova_config_accepts_transformers_normalized_default_rope(
    monkeypatch, rope_scaling
):
    monkeypatch.setattr(
        "sglang.srt.models.xllm.torch.get_default_dtype", lambda: torch.bfloat16
    )
    monkeypatch.setattr("sglang.srt.models.xllm.get_attention_tp_size", lambda: 8)
    _validate_mova_config(
        _valid_mova_config(rope_scaling=rope_scaling), quant_config=None
    )


def test_mova_config_rejects_actual_rope_scaling(monkeypatch):
    monkeypatch.setattr(
        "sglang.srt.models.xllm.torch.get_default_dtype", lambda: torch.bfloat16
    )
    monkeypatch.setattr("sglang.srt.models.xllm.get_attention_tp_size", lambda: 8)
    with pytest.raises(ValueError, match="non-default RoPE scaling"):
        _validate_mova_config(
            _valid_mova_config(
                rope_scaling={"rope_type": "linear", "factor": 2.0}
            ),
            quant_config=None,
        )


def test_mova_config_rejects_accidental_auto_fp16(monkeypatch):
    monkeypatch.setattr(
        "sglang.srt.models.xllm.torch.get_default_dtype", lambda: torch.float16
    )
    with pytest.raises(ValueError, match="--dtype bfloat16"):
        _validate_mova_config(_valid_mova_config(), quant_config=None)


def test_softplus_attention_gate_uses_ln2_beta():
    gate = torch.tensor([-2.0, 0.0, 2.0])
    module = SimpleNamespace(attn_gate_func="softplus")
    actual = _XllmMoVAAttentionBase._activate_gate(module, gate)
    expected = F.softplus(gate, beta=math.log(2))
    torch.testing.assert_close(actual, expected)


def test_interleaved_rope_matches_hf_neox_after_weight_permutation():
    torch.manual_seed(13)
    head_dim = 8
    hidden_size = 5
    positions = torch.tensor([1, 7, 31], dtype=torch.long)
    hidden = torch.randn(3, hidden_size)
    hf_weight = torch.randn(head_dim, hidden_size)
    hf_projection = F.linear(hidden, hf_weight)
    native_projection = F.linear(
        hidden, _interleave_rope_weight(hf_weight, num_heads=1)
    )

    hf_rope = get_rope(
        head_dim,
        rotary_dim=head_dim,
        max_position=64,
        base=10000,
        is_neox_style=True,
    )
    native_rope = get_rope(
        head_dim,
        rotary_dim=head_dim,
        max_position=64,
        base=10000,
        is_neox_style=False,
    )
    hf_rotated, _ = hf_rope.forward_native(positions, hf_projection, hf_projection)
    native_rotated, _ = native_rope.forward_native(
        positions, native_projection, native_projection
    )
    torch.testing.assert_close(
        native_rotated,
        _interleave_rope_weight(hf_rotated.transpose(0, 1), num_heads=1).transpose(
            0, 1
        ),
    )


def test_mova_parameter_mapper_stages_qkg_and_all_value_experts():
    config = SimpleNamespace(num_values=64)
    model = SimpleNamespace(
        stacked_params_mapping=_xllm_stacked_params_mapping(config),
        expert_params_mapping=[],
    )
    mapper = ParameterMapper.from_model(model)

    q = mapper.map("model.layers.3.self_attn.q_proj.weight")
    assert q.sglang_name == "model.layers.3.self_attn.qkg_proj.weight"
    assert q.shard_id == "q"
    assert q.num_shards == 3

    gate = mapper.map("model.layers.3.self_attn.attn_gate_proj.weight")
    assert gate.sglang_name == "model.layers.3.self_attn.qkg_proj.weight"
    assert gate.shard_id == "gate"
    assert gate.num_shards == 3

    value = mapper.map("model.layers.3.self_attn.v_experts.63.weight")
    assert value.sglang_name == "model.layers.3.self_attn.v_experts.weight"
    assert value.shard_id == 63
    assert value.num_shards == 64

    router = mapper.map("model.layers.3.self_attn.v_router.bias")
    assert router.sglang_name == "model.layers.3.self_attn.v_router.bias"
    assert router.num_shards == 1


def test_legacy_xllm_mapping_is_unchanged():
    assert _xllm_stacked_params_mapping(SimpleNamespace(num_values=0)) == [
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]
