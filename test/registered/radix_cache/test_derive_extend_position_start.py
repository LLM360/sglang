"""Tests for the helper that derives per-request extend_position_start from cached
positions returned by cache hits.

This helper is what the scheduler / ForwardBatch call site uses to bridge between
the radix tree (which returns non-contiguous original_positions on cache hits) and
compute_position's extend_position_start parameter.
"""

import unittest

import torch

from sglang.srt.mem_cache.common import derive_extend_position_start


class TestDeriveExtendPositionStart(unittest.TestCase):
    def test_returns_none_when_all_requests_lack_cached_positions(self):
        """When no request has cached positions (e.g. flag off, or no cache hit), the
        helper returns None — signaling that compute_position should use the legacy
        contiguous behavior."""
        out = derive_extend_position_start(
            extend_prefix_lens=[3, 5],
            cached_positions_per_req=[None, None],
        )
        self.assertIsNone(out)

    def test_uses_max_plus_one_for_cached_request(self):
        """A request with cached non-contiguous positions returns max(positions) + 1;
        a request without cached positions falls back to extend_prefix_lens (legacy)."""
        out = derive_extend_position_start(
            extend_prefix_lens=[5, 3],
            cached_positions_per_req=[
                torch.tensor([0, 1, 6, 7, 8], dtype=torch.int64),  # max 8 -> start 9
                None,  # legacy fallback -> start 3
            ],
        )
        self.assertEqual(out, [9, 3])


if __name__ == "__main__":
    unittest.main()
