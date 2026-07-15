"""Tests for the call-site helper that builds extend-token positions while honoring
per-request cached non-contiguous positions.

This is the bridge between ScheduleBatch state (per-request cached_positions on
cache hits) and ForwardBatch's positions tensor.
"""

import unittest

import torch

from sglang.srt.model_executor.forward_batch_info import build_extend_positions

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="stage-b-test-1-gpu-small")


class TestBuildExtendPositions(unittest.TestCase):
    def test_legacy_path_when_no_cached_positions(self):
        """When cached_positions_per_req is None (or all None entries), positions are
        contiguous starting from extend_prefix_lens[i] — i.e. unchanged from today."""
        positions, _ = build_extend_positions(
            attn_backend="torch_native",  # forces torch path (not triton-supported)
            extend_prefix_lens=torch.tensor([2, 4], dtype=torch.int64),
            extend_seq_lens=torch.tensor([3, 1], dtype=torch.int64),
            extend_num_tokens=4,
            extend_prefix_lens_cpu=[2, 4],
            cached_positions_per_req=None,
            device="cpu",
        )
        # Req 0: arange(2, 5) = [2,3,4]. Req 1: arange(4, 5) = [4].
        self.assertEqual(positions.tolist(), [2, 3, 4, 4])

    def test_cached_positions_override_extends_from_max_plus_one(self):
        """When a request has cached non-contiguous positions ending at p, its extend
        tokens start at p+1, not at extend_prefix_lens. Other requests fall back to
        extend_prefix_lens (legacy)."""
        cached_positions_per_req = [
            torch.tensor([0, 1, 6, 7, 8], dtype=torch.int64),  # max 8 -> extend starts at 9
            None,  # legacy fallback -> extend starts at 4
        ]
        positions, _ = build_extend_positions(
            attn_backend="torch_native",
            extend_prefix_lens=torch.tensor([5, 4], dtype=torch.int64),
            extend_seq_lens=torch.tensor([3, 1], dtype=torch.int64),
            extend_num_tokens=4,
            extend_prefix_lens_cpu=[5, 4],
            cached_positions_per_req=cached_positions_per_req,
            device="cpu",
        )
        # Req 0: arange(9, 12) = [9,10,11]. Req 1: arange(4, 5) = [4].
        self.assertEqual(positions.tolist(), [9, 10, 11, 4])


if __name__ == "__main__":
    unittest.main()
