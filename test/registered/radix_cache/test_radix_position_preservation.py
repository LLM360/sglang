"""Tests for radix-tree round-tripping of per-token original RoPE positions.

The radix tree must accept original_positions on insert and return them on
match_prefix, so callers can preserve non-contiguous positions (e.g. when a
generated thought slice was excluded from the cached entry) across cache hits.

These tests run without a GPU or model — they use RadixCache.create_simulated().
"""

import unittest

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, suite="stage-b-test-1-gpu-small")


class TestRadixPositionPreservation(CustomTestCase):
    """Radix tree must carry per-token original_positions through insert and match."""

    def setUp(self):
        self.tree = RadixCache.create_simulated()

    def test_insert_accepts_original_positions(self):
        """InsertParams must accept an original_positions tensor matching key length."""
        token_ids = [10, 11, 12, 13, 14]
        positions = torch.tensor([0, 1, 6, 7, 8], dtype=torch.int64)
        kv_indices = torch.tensor([100, 101, 102, 103, 104], dtype=torch.int64)

        result = self.tree.insert(
            InsertParams(
                key=RadixKey(token_ids=token_ids, extra_key=None),
                value=kv_indices,
                original_positions=positions,
            )
        )
        # Insert is expected to succeed; prefix_len reflects pre-existing tree overlap (here, 0).
        self.assertEqual(result.prefix_len, 0)

    def test_match_returns_non_contiguous_positions(self):
        """After inserting with non-contiguous positions, match_prefix must return them."""
        token_ids = [10, 11, 12, 13, 14]
        positions = torch.tensor([0, 1, 6, 7, 8], dtype=torch.int64)
        kv_indices = torch.tensor([100, 101, 102, 103, 104], dtype=torch.int64)

        self.tree.insert(
            InsertParams(
                key=RadixKey(token_ids=token_ids, extra_key=None),
                value=kv_indices,
                original_positions=positions,
            )
        )

        match = self.tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=token_ids, extra_key=None))
        )

        self.assertIsNotNone(match.original_positions)
        self.assertEqual(match.original_positions.tolist(), [0, 1, 6, 7, 8])
        # device_indices must still match the inserted kv_indices.
        self.assertEqual(match.device_indices.tolist(), [100, 101, 102, 103, 104])

    def test_match_returns_none_positions_for_legacy_insert(self):
        """Backwards-compat: insert without original_positions returns None on match."""
        token_ids = [20, 21, 22]
        kv_indices = torch.tensor([200, 201, 202], dtype=torch.int64)

        self.tree.insert(
            InsertParams(
                key=RadixKey(token_ids=token_ids, extra_key=None),
                value=kv_indices,
            )
        )

        match = self.tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=token_ids, extra_key=None))
        )

        self.assertIsNone(match.original_positions)
        self.assertEqual(match.device_indices.tolist(), [200, 201, 202])

    def test_partial_match_returns_position_prefix(self):
        """If only a prefix of the cached entry matches, returned positions cover that prefix."""
        token_ids = [10, 11, 12, 13, 14]
        positions = torch.tensor([0, 1, 6, 7, 8], dtype=torch.int64)
        kv_indices = torch.tensor([100, 101, 102, 103, 104], dtype=torch.int64)

        self.tree.insert(
            InsertParams(
                key=RadixKey(token_ids=token_ids, extra_key=None),
                value=kv_indices,
                original_positions=positions,
            )
        )

        # Query with only the first 3 tokens; expect positions [0, 1, 6]
        match = self.tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=[10, 11, 12], extra_key=None))
        )

        self.assertIsNotNone(match.original_positions)
        self.assertEqual(match.original_positions.tolist(), [0, 1, 6])
        self.assertEqual(match.device_indices.tolist(), [100, 101, 102])

    def test_extend_existing_path_with_positions(self):
        """Inserting a longer sequence with positions extends an existing prefix path."""
        # First, insert the contiguous prompt [A, B] at positions [0, 1].
        self.tree.insert(
            InsertParams(
                key=RadixKey(token_ids=[1, 2], extra_key=None),
                value=torch.tensor([100, 101], dtype=torch.int64),
                original_positions=torch.tensor([0, 1], dtype=torch.int64),
            )
        )

        # Then, insert [A, B, X, Y, Z] at positions [0, 1, 6, 7, 8] (gap where thoughts were).
        self.tree.insert(
            InsertParams(
                key=RadixKey(token_ids=[1, 2, 3, 4, 5], extra_key=None),
                value=torch.tensor([100, 101, 102, 103, 104], dtype=torch.int64),
                original_positions=torch.tensor([0, 1, 6, 7, 8], dtype=torch.int64),
            )
        )

        match = self.tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=[1, 2, 3, 4, 5], extra_key=None))
        )
        self.assertIsNotNone(match.original_positions)
        self.assertEqual(match.original_positions.tolist(), [0, 1, 6, 7, 8])


if __name__ == "__main__":
    unittest.main()
