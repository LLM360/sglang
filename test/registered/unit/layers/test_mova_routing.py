from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.entrypoints.openai.protocol import SglExt
from sglang.srt.entrypoints.openai.utils import process_routed_experts_from_ret
from sglang.srt.layers.moe import routed_experts_capturer as routing
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.srt.sampling.sampling_params import SamplingParams


@pytest.mark.parametrize("can_run_graph", [False, True])
def test_value_capture_uses_local_tokens_and_sparse_layers(monkeypatch, can_run_graph):
    config = SimpleNamespace(
        num_hidden_layers=3,
        num_dense_layers=1,
        num_experts_per_tok=2,
        num_values=8,
        num_values_per_tok=1,
    )
    monkeypatch.setattr(
        routing,
        "get_global_server_args",
        lambda: SimpleNamespace(chunked_prefill_size=4, dp_size=2),
    )
    monkeypatch.setattr(
        routing, "get_moe_a2a_backend", lambda: SimpleNamespace(is_deepep=lambda: False)
    )
    monkeypatch.setattr(routing, "is_dp_attention_enabled", lambda: True)
    monkeypatch.setattr(routing, "get_attention_dp_rank", lambda: 1)
    monkeypatch.setattr(routing, "get_dp_local_info", lambda batch: (3, 2))
    zeros = torch.zeros
    # Pinned allocation requires a CUDA driver; the cache logic also runs on CPU.
    monkeypatch.setattr(
        torch, "zeros", lambda *a, **kw: zeros(*a, **{**kw, "pin_memory": False})
    )
    capturer = routing.RoutedExpertsCapturer.create(
        True, SimpleNamespace(hf_text_config=config), 1, 16, 4, "cpu"
    )
    assert capturer.value_host_cache.buffer.shape == (16, 2, 1)
    assert capturer.value_device_cache.buffer.shape[1:] == (2, 1)
    ffn = torch.tensor([[1, 2, 99], [3, 4, 99]], dtype=torch.int32)
    values = torch.tensor([[5], [6]], dtype=torch.int32)
    # FFN tokens include other DP ranks; attention tokens are already local.
    start = 4 if can_run_graph else 3
    global_ffn = torch.zeros(start + 2, 3, dtype=torch.int32)
    global_ffn[start:] = ffn
    for layer in (1, 2):
        capturer.capture(layer, global_ffn)
        capturer.capture(layer, values + layer - 1, is_value=True)
    batch = SimpleNamespace(out_cache_loc=torch.tensor([7, 2]))
    capturer.on_forward_end(batch, can_run_graph, 4)

    # Decode adds one token without disturbing the cached prefill prefix.
    batch.out_cache_loc = torch.tensor([9])
    monkeypatch.setattr(routing, "get_dp_local_info", lambda batch: (3, 1))
    for layer in (1, 2):
        capturer.capture(layer, torch.tensor([[4]], dtype=torch.int32), is_value=True)
    capturer.on_forward_end(batch, can_run_graph, 4)
    pool = SimpleNamespace(req_to_token=torch.tensor([[7, 2, 9, 0]]))
    result = capturer.get_routed_experts(0, 4, pool, is_value=True)
    assert result.tolist() == [[[5], [6]], [[6], [7]], [[4], [4]]]
    assert capturer.get_routed_experts(0, 3, pool)[:, 1].tolist() == [[1, 2], [3, 4]]


@pytest.mark.parametrize("opt_in", [(False, True), (True, False), (False, False)])
def test_routing_output_keeps_mixed_requests_aligned(opt_in):
    scheduler = SchedulerOutputProcessorMixin()
    scheduler.get_load = lambda: None
    scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    scheduler.server_args = SimpleNamespace(enable_request_time_stats_logging=False)
    scheduler.model_config = SimpleNamespace(is_multimodal_gen=False)
    scheduler.attn_tp_rank = scheduler.dp_rank = 0
    scheduler.send_to_detokenizer = Mock()
    reqs = []
    for i, enabled in enumerate(opt_in):
        req = Req(str(i), "", [1, 2], SamplingParams(), return_routed_experts=enabled)
        req.output_ids = [3]
        req.finished_reason = FINISH_LENGTH(1)
        req.routed_experts = torch.tensor([[[i]]], dtype=torch.int32)
        req.routed_value_experts = torch.tensor([[[i + 2]]], dtype=torch.int32)
        reqs.append(req)
    scheduler.stream_output_generation(reqs, return_logprob=False)
    output = scheduler.send_to_detokenizer.send_output.call_args.args[0]
    assert output.rids == ["0", "1"]
    for field in ("routed_experts", "routed_value_experts"):
        routes = getattr(output, field)
        if not any(opt_in):
            assert routes is None
        else:
            assert len(routes) == 2
            for req, actual in zip(reqs, routes):
                assert actual is (getattr(req, field) if req.return_routed_experts else None)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("values", [None, "value_routes"])
def test_openai_routing_fields(enabled, values):
    meta = {"routed_experts": "ffn_routes", "routed_value_experts": values}
    actual = process_routed_experts_from_ret(
        {"meta_info": meta}, SimpleNamespace(return_routed_experts=enabled)
    )
    expected = {k: v for k, v in meta.items() if enabled and v is not None}
    assert actual == expected
    assert SglExt(**actual).model_dump() == expected
