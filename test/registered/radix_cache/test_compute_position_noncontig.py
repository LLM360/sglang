"""Tests for non-contiguous extend-positions in forward_batch_info.compute_position_torch.

When a request hits a cached entry with non-contiguous original positions (e.g.
[0, 1, 6, 7, 8] — gap where thoughts used to live), the new extend tokens must
continue from max(cached_positions) + 1, not from len(cached). This test pins
down the extended API that supports that case.
"""

import unittest

import torch

from sglang.srt.model_executor.forward_batch_info import (
    compute_position,
    compute_position_torch,
)


class TestComputePositionNonContiguous(unittest.TestCase):
    def test_extend_position_start_overrides_prefix_len(self):
        """When extend_position_start is provided, positions for each request's
        extend tokens start at extend_position_start[i] rather than extend_prefix_lens[i]."""
        # Single request: cached 5 tokens at positions [0, 1, 6, 7, 8], now extending by 3.
        # Standard behavior would put extend positions at [5, 6, 7]; with the override,
        # they should be at [9, 10, 11].
        extend_prefix_lens = torch.tensor([5], dtype=torch.int64)
        extend_seq_lens = torch.tensor([3], dtype=torch.int64)
        extend_position_start = torch.tensor([9], dtype=torch.int64)

        positions, _ = compute_position_torch(
            extend_prefix_lens, extend_seq_lens, extend_position_start
        )
        self.assertEqual(positions.tolist(), [9, 10, 11])

    def test_none_override_preserves_legacy_behavior(self):
        """When extend_position_start is None, positions start at extend_prefix_lens (unchanged)."""
        extend_prefix_lens = torch.tensor([2, 4], dtype=torch.int64)
        extend_seq_lens = torch.tensor([3, 1], dtype=torch.int64)

        positions, _ = compute_position_torch(extend_prefix_lens, extend_seq_lens)
        # Request 0: starts at 2 -> [2, 3, 4]. Request 1: starts at 4 -> [4].
        self.assertEqual(positions.tolist(), [2, 3, 4, 4])

    def test_compute_position_wrapper_forwards_override(self):
        """compute_position(...) (the wrapper) must forward extend_position_start to the
        underlying torch / triton implementation."""
        extend_prefix_lens = torch.tensor([5], dtype=torch.int64)
        extend_seq_lens = torch.tensor([3], dtype=torch.int64)
        extend_position_start = torch.tensor([9], dtype=torch.int64)

        # Use the non-triton backend name to force the torch path; the wrapper still
        # routes to compute_position_triton when support_triton(attn_backend) is True
        # on CUDA hosts. The torch path is the unambiguous behavioral test.
        positions, _ = compute_position(
            attn_backend="aiter",  # not triton-supported -> takes torch path
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_sum=3,
            extend_position_start=extend_position_start,
        )
        self.assertEqual(positions.tolist(), [9, 10, 11])


if __name__ == "__main__":
    unittest.main()
