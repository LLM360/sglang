"""Tests that Req captures the cached non-contiguous positions returned by match_prefix.

When the prefix cache has an entry with original_positions set (e.g. because a prior
turn was inserted via the split path), a future request that hits that entry must
record those positions on the Req so the scheduler can build the right
extend_position_start for compute_position.
"""

import unittest

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams


class TestReqCachedPositions(unittest.TestCase):
    def test_match_with_non_contiguous_positions_stored_on_req(self):
        # Seed the radix tree with [A, B, X, Y, Z] at positions [0, 1, 6, 7, 8] —
        # simulating a prior turn that was inserted via the split path.
        tree = RadixCache.create_simulated()
        tree.insert(
            InsertParams(
                key=RadixKey(token_ids=[10, 11, 30, 31, 32], extra_key=None),
                value=torch.tensor([100, 101, 102, 103, 104], dtype=torch.int64),
                original_positions=torch.tensor([0, 1, 6, 7, 8], dtype=torch.int64),
            )
        )

        # New request whose input matches the cached entry.
        req = Req(
            rid="r1",
            origin_input_text="...",
            origin_input_ids=[10, 11, 30, 31, 32, 99],  # +1 trailing token so we don't truncate
            sampling_params=SamplingParams(),
        )
        req.init_next_round_input(tree_cache=tree)

        self.assertIsNotNone(req.cached_positions)
        # The first 5 tokens should hit the cached prefix; positions reflect the original
        # non-contiguous layout. (Match may stop one token short to enable logprob compute.)
        self.assertEqual(req.cached_positions.tolist()[:5], [0, 1, 6, 7, 8])


if __name__ == "__main__":
    unittest.main()
