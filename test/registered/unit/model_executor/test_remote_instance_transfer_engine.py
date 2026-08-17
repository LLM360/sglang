import sys
import types
import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.model_runner import ModelRunner

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class TestRemoteInstanceTransferEngine(CustomTestCase):
    def test_initializes_with_detected_local_ip(self):
        class FakeTransferEngine:
            def __init__(self):
                self.initialize_args = None

            def initialize(self, *args):
                self.initialize_args = args

            def get_rpc_port(self):
                return 12345

        mooncake = types.ModuleType("mooncake")
        mooncake.__path__ = []
        mooncake_engine = types.ModuleType("mooncake.engine")
        mooncake_engine.TransferEngine = FakeTransferEngine

        runner = ModelRunner.__new__(ModelRunner)
        with patch.dict(
            sys.modules,
            {"mooncake": mooncake, "mooncake.engine": mooncake_engine},
        ), patch(
            "sglang.srt.model_executor.model_runner.get_local_ip_auto",
            return_value="10.20.30.40",
        ), patch(
            "sglang.srt.model_executor.model_runner.envs.MOONCAKE_DEVICE.get",
            return_value="mlx5_0",
        ):
            runner.remote_instance_init_transfer_engine()

        self.assertEqual(
            runner.remote_instance_transfer_engine.initialize_args,
            ("10.20.30.40", "P2PHANDSHAKE", "rdma", "mlx5_0"),
        )
        self.assertEqual(
            runner.remote_instance_transfer_engine_session_id,
            "10.20.30.40:12345",
        )

    def test_registers_initialized_engine_info_with_bootstrap(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.server_args = types.SimpleNamespace(
            dist_init_addr="10.20.30.1:17503",
            engine_info_bootstrap_port=17502,
        )
        runner.tp_rank = 3
        runner.remote_instance_transfer_engine_session_id = "10.20.30.40:12345"
        runner.remote_instance_transfer_engine_weight_info = {
            "model.weight": (4096, 128, 2)
        }

        with patch("requests.put") as put:
            put.return_value.status_code = 200
            runner._register_to_engine_info_bootstrap()

        put.assert_called_once_with(
            "http://10.20.30.1:17502/register_transfer_engine_info",
            json={
                "tp_rank": 3,
                "transfer_engine_info": {
                    "session_id": "10.20.30.40:12345",
                    "weights_info_dict": {"model.weight": (4096, 128, 2)},
                },
            },
            timeout=5,
        )


if __name__ == "__main__":
    unittest.main()
