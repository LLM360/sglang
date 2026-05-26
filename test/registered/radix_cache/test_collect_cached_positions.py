"""Tests for the helper that aggregates Req.cached_positions into a per-request list
suitable for ForwardBatch.init_new to consume via build_extend_positions.
"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.schedule_batch import collect_cached_positions


class TestCollectCachedPositions(unittest.TestCase):
    def test_returns_none_when_no_req_has_cached_positions(self):
        reqs = [MagicMock(cached_positions=None), MagicMock(cached_positions=None)]
        self.assertIsNone(collect_cached_positions(reqs))

    def test_returns_list_when_any_req_has_cached_positions(self):
        r1 = MagicMock(cached_positions=torch.tensor([0, 1, 6, 7, 8], dtype=torch.int64))
        r2 = MagicMock(cached_positions=None)
        out = collect_cached_positions([r1, r2])
        self.assertIsNotNone(out)
        self.assertEqual(out[0].tolist(), [0, 1, 6, 7, 8])
        self.assertIsNone(out[1])


if __name__ == "__main__":
    unittest.main()
