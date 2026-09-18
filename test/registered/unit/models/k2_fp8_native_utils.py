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

"""CPU metadata for native K2 FP8 storage tests."""

from contextlib import contextmanager
from types import SimpleNamespace

import torch

from sglang.srt.layers import dp_attention
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a8_fp8,
)
from sglang.srt.layers.rotary_embedding import base, factory
from sglang.srt.models import xllm
from sglang.srt.runtime_context import get_context, get_flags, get_parallel


def _no_fp8_kernel(*args, **kwargs):
    raise AssertionError("A native storage test must not call an FP8 kernel")


@contextmanager
def native_cpu_context(monkeypatch, *, tp=1, rank=0):
    """Supply group and hardware metadata without a collective or kernel call."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with (
            monkeypatch.context() as patch,
            get_context().override_server_args(
                enable_eplb=False,
                init_expert_location="trivial",
                ep_num_redundant_experts=0,
                enable_two_batch_overlap=False,
                moe_runner_backend="triton",
                moe_a2a_backend="none",
                enable_dp_lm_head=True,
            ),
            get_parallel().override(
                tp_size=tp,
                tp_rank=rank,
                attn_tp_size=tp,
                attn_tp_rank=rank,
                moe_tp_size=tp,
                moe_tp_rank=rank,
                moe_ep_size=1,
                moe_ep_rank=0,
                attn_dp_size=1,
                attn_dp_rank=0,
                attn_cp_size=1,
                attn_cp_rank=0,
                moe_dp_size=1,
            ),
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.TRITON,
                a2a_backend=MoeA2ABackend.NONE,
            ),
        ):
            group = SimpleNamespace(
                world_size=1, rank_in_group=0, is_first_rank=True, is_last_rank=True
            )
            patch.setattr(xllm, "get_pp_group", lambda: group)
            patch.setattr(
                dp_attention, "_get_moe_dp_group", lambda: SimpleNamespace(world_size=1)
            )
            patch.setattr(base, "_is_cpu", True)
            patch.setattr(factory, "_ROPE_DICT", {})
            patch.setattr(torch.cuda, "get_device_capability", lambda *args: (9, 0))
            patch.setattr(
                compressed_tensors_w8a8_fp8,
                "dispatch_w8a8_block_fp8_linear",
                lambda: _no_fp8_kernel,
            )
            yield
    finally:
        torch.set_default_dtype(previous)
