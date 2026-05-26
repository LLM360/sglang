"""Tests for the --no-cache-thoughts split-insertion helper.

When a reasoning request finishes with --no-cache-thoughts enabled, the request's
KV must be split: the input + post-</think> answer is inserted into the radix tree
with original positions preserved; the thought-slice KV pages are freed directly.

This test pins down the helper function's contract: given a finished Req's metadata
and its full per-request KV slot, produce the virtual token list / kv_indices /
positions that should be inserted, plus the kv_indices that should be freed.

Tests run without a GPU or model.
"""

import unittest

import torch

from sglang.srt.mem_cache.common import split_kv_for_no_cache_thoughts
from sglang.test.test_utils import CustomTestCase


class TestNoCacheThoughtsSplit(CustomTestCase):
    """Validate the split-insertion helper for --no-cache-thoughts."""

    def test_split_basic_case(self):
        """
        Setup mirrors a typical reasoning request:
          positions:  0   1   2       3   4   5         6   7   8
          tokens:     A   B   <think> T1  T2  </think>  X   Y   Z
                      └─prompt─┘     └────thoughts────┘ └─answer─┘
        answer_start_position = 6 (position right after </think>)
        kv_indices in the per-request slot: [100..108], one per token.
        """
        origin_input_ids = [101, 102]  # A, B
        output_ids = [201, 202, 203, 204, 301, 302, 303]  # think, T1, T2, end, X, Y, Z
        req_to_token_slot = torch.tensor(
            [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008], dtype=torch.int64
        )
        answer_start_position = 6

        result = split_kv_for_no_cache_thoughts(
            origin_input_ids=origin_input_ids,
            output_ids=output_ids,
            req_to_token_slot=req_to_token_slot,
            answer_start_position=answer_start_position,
        )

        # Virtual token list: [A, B, X, Y, Z]
        self.assertEqual(result.virtual_token_ids, [101, 102, 301, 302, 303])
        # Virtual kv_indices: pointers for slots [0, 1, 6, 7, 8]
        self.assertEqual(
            result.virtual_kv_indices.tolist(), [1000, 1001, 1006, 1007, 1008]
        )
        # Virtual positions: prompt positions + answer original positions
        self.assertEqual(
            result.virtual_positions.tolist(), [0, 1, 6, 7, 8]
        )
        # Thought kv_indices to free: slots [2, 3, 4, 5]
        self.assertEqual(
            result.thought_kv_indices_to_free.tolist(), [1002, 1003, 1004, 1005]
        )

    def test_split_no_answer_yet(self):
        """If answer_start_position == total_len (i.e. </think> emitted last, no answer
        tokens generated yet), virtual sequence is just the prompt and there's no answer
        slice."""
        origin_input_ids = [101, 102]
        output_ids = [201, 202, 203, 204]  # think, T1, T2, end_think — no answer yet
        req_to_token_slot = torch.tensor(
            [1000, 1001, 1002, 1003, 1004, 1005], dtype=torch.int64
        )
        # </think> at position 5; answer starts at 6, but seq ends at 5.
        answer_start_position = 6

        result = split_kv_for_no_cache_thoughts(
            origin_input_ids=origin_input_ids,
            output_ids=output_ids,
            req_to_token_slot=req_to_token_slot,
            answer_start_position=answer_start_position,
        )

        # Virtual list contains only the input.
        self.assertEqual(result.virtual_token_ids, [101, 102])
        self.assertEqual(result.virtual_kv_indices.tolist(), [1000, 1001])
        self.assertEqual(result.virtual_positions.tolist(), [0, 1])
        # All output tokens are thoughts to free.
        self.assertEqual(
            result.thought_kv_indices_to_free.tolist(), [1002, 1003, 1004, 1005]
        )

    def test_split_long_answer(self):
        """Multi-token answer with a longer thought slice."""
        origin_input_ids = [10, 11, 12]  # 3-token prompt at positions 0-2
        # 5-token thoughts at positions 3-7, then 4-token answer at positions 8-11
        output_ids = [20, 21, 22, 23, 24, 30, 31, 32, 33]
        req_to_token_slot = torch.tensor(
            [500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511],
            dtype=torch.int64,
        )
        answer_start_position = 8

        result = split_kv_for_no_cache_thoughts(
            origin_input_ids=origin_input_ids,
            output_ids=output_ids,
            req_to_token_slot=req_to_token_slot,
            answer_start_position=answer_start_position,
        )

        # Virtual list: [10, 11, 12, 30, 31, 32, 33]
        self.assertEqual(result.virtual_token_ids, [10, 11, 12, 30, 31, 32, 33])
        # Virtual kv_indices: slots [0, 1, 2, 8, 9, 10, 11]
        self.assertEqual(
            result.virtual_kv_indices.tolist(),
            [500, 501, 502, 508, 509, 510, 511],
        )
        # Virtual positions: prompt [0, 1, 2] + answer [8, 9, 10, 11]
        self.assertEqual(
            result.virtual_positions.tolist(), [0, 1, 2, 8, 9, 10, 11]
        )
        # Thoughts: slots [3, 4, 5, 6, 7]
        self.assertEqual(
            result.thought_kv_indices_to_free.tolist(),
            [503, 504, 505, 506, 507],
        )


if __name__ == "__main__":
    unittest.main()
