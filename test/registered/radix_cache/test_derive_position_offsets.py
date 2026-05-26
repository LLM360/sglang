"""Tests for the helper that computes per-request RoPE position offsets so decode
positions continue from where the non-contiguous prefill cache hit left off.
"""

import unittest

import torch

from sglang.srt.mem_cache.common import derive_position_offsets


class TestDerivePositionOffsets(unittest.TestCase):
    def test_returns_none_when_no_cached_positions(self):
        out = derive_position_offsets(
            extend_prefix_lens=[3, 5],
            cached_positions_per_req=[None, None],
        )
        self.assertIsNone(out)

    def test_offset_equals_max_minus_prefix_minus_one_plus_one(self):
        """Per-req offset = max(cached_positions) - (prefix_len - 1).

        Example: prefix_len=5 (cached 5 tokens), cached_positions=[0,1,6,7,8]
        -> last cached position is 8, legacy max for 5 tokens is 4, offset is 4.
        """
        out = derive_position_offsets(
            extend_prefix_lens=[5, 3],
            cached_positions_per_req=[
                torch.tensor([0, 1, 6, 7, 8], dtype=torch.int64),  # max 8 -> offset 4
                None,  # no cache positions -> offset 0
            ],
        )
        self.assertEqual(out, [4, 0])


if __name__ == "__main__":
    unittest.main()
