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

"""Reject another source stream after a native MoVA FP8 restore."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_xllm_mova_fp8 as mova_tests
import torch

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.layers.quantization import fp8
from sglang.srt.model_loader import loader as model_loader
from sglang.srt.model_loader.loader import PreshardedModelLoader
from sglang.srt.models import xllm
from sglang.test.ci.ci_register import register_cpu_ci

pytest_plugins = ("test_xllm_mova_fp8",)
register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_mova_presharded_restore_rejects_second_load_before_iterator(
    native_loader_model, monkeypatch, tmp_path
):
    print(
        "K2 MoVA restore sources: "
        + json.dumps(
            {
                module.__name__: hashlib.sha256(
                    Path(module.__file__).read_bytes()
                ).hexdigest()
                for module in (xllm, model_loader, mova_tests, fp8)
            }
        )
    )
    assert fp8._is_cpu is False
    assert fp8._is_fp8_fnuz is False
    assert fp8._use_aiter is False
    donor, tensors = native_loader_model()
    target, _ = native_loader_model()
    assert type(donor) is xllm.XllmForCausalLM
    assert type(target) is xllm.XllmForCausalLM
    assert donor.config.xllm_source_router_gemm_partitions == 2
    assert donor.config.num_values == 4
    assert donor.quant_config.get_name() == "fp8"
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
        model_path=str(tmp_path), is_draft_model=False, dtype=torch.bfloat16
    )
    shard_config = {"tp": 1}
    models = iter((donor, target))
    monkeypatch.setattr(loader, "_collect_shard_config", lambda config: shard_config)
    monkeypatch.setattr(
        model_loader, "_get_quantization_config", lambda *args: donor.quant_config
    )
    monkeypatch.setattr(model_loader, "_initialize_model", lambda *args: next(models))

    def source_weights(config, model):
        assert model is donor
        return iter(tensors.items())

    monkeypatch.setattr(loader, "_get_all_weights", source_weights)
    initial = loader.load_model(model_config=config, device_config=DeviceConfig("cpu"))
    assert initial is donor
    expected_state = {
        name: value.detach().clone() for name, value in donor.state_dict().items()
    }
    expected_extras = {
        name: value.detach().clone()
        for name, value in loader._collect_extra_tensors(donor).items()
    }
    folder = Path(loader._presharded_dir(config, shard_config))
    assert (folder / loader.READY_FILENAME).is_file()
    assert list(folder.glob("*.safetensor"))
    checks = []
    verify = loader._verify_rank_checksum

    def observe_checksum(*args):
        verify(*args)
        checks.append("rank")

    monkeypatch.setattr(loader, "_verify_rank_checksum", observe_checksum)
    restored = loader.load_model(model_config=config, device_config=DeviceConfig("cpu"))
    assert restored is target
    assert restored.training is False
    assert checks == ["rank"]
    for expected, actual in (
        (expected_state, restored.state_dict()),
        (expected_extras, loader._collect_extra_tensors(restored)),
    ):
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            assert actual[name].shape == value.shape
            assert actual[name].dtype == value.dtype
            torch.testing.assert_close(
                actual[name].detach().contiguous().reshape(-1).view(torch.uint8),
                value.contiguous().reshape(-1).view(torch.uint8),
                rtol=0,
                atol=0,
            )
    accessed = []

    def forbidden_source():
        accessed.append("read")
        yield "model.norm.weight", torch.zeros_like(tensors["model.norm.weight"])

    with pytest.raises(
        ValueError, match="^MoVA FP8 requires a fresh model for every initial load$"
    ):
        restored.load_weights(forbidden_source())
    assert accessed == []


if __name__ == "__main__":
    import sys

    args = [arg for arg in sys.argv[1:] if arg != "-f"]
    sys.exit(pytest.main([__file__, "-v", "-s", *args]))
