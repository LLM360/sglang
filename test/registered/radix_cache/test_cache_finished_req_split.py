"""Tests that RadixCache.cache_finished_req accepts a pre-computed split, bypassing
the per-req KV-pool lookup and inserting [prompt + post-</think> answer] with original
positions preserved.

The split is constructed by split_kv_for_no_cache_thoughts in the caller; this test
pins down the cache-side contract that consumes it.
"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.common import split_kv_for_no_cache_thoughts
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey


class TestCacheFinishedReqSplit(unittest.TestCase):
    def test_split_inserts_virtual_slice_with_positions(self):
        # Mock allocator records free() calls so we can assert the thought slice was freed.
        mock_allocator = MagicMock()
        tree = RadixCache.create_simulated(mock_allocator=mock_allocator)

        # Synthetic finished reasoning request:
        #   positions:  0   1   2   3   4   5    6   7   8
        #   tokens:     A   B  <t> T1  T2 </t>   X   Y   Z
        #   prompt=[A,B] thoughts=[<t>,T1,T2,</t>] answer=[X,Y,Z]; answer_start_position=6
        split = split_kv_for_no_cache_thoughts(
            origin_input_ids=[101, 102],
            output_ids=[201, 202, 203, 204, 301, 302, 303],
            req_to_token_slot=torch.tensor(
                [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
                dtype=torch.int64,
            ),
            answer_start_position=6,
        )

        # Build a minimal Req-like stub: cache_finished_req(req, ..., split=split) should
        # use split.virtual_* directly and ignore req.req_to_token_pool/req_pool_idx.
        req_stub = MagicMock()
        req_stub.extra_key = None
        req_stub.priority = 0
        req_stub.pop_committed_kv_cache.return_value = 0  # bookkeeping no-op
        req_stub.last_node = tree.root_node
        req_stub.cache_protected_len = 0

        tree.cache_finished_req(req_stub, is_insert=True, split=split)

        # The radix tree should now contain a path matching the virtual token ids
        # (skipping the thought slice).
        match = tree.match_prefix(
            MatchPrefixParams(
                key=RadixKey(token_ids=[101, 102, 301, 302, 303], extra_key=None)
            )
        )
        self.assertEqual(match.device_indices.tolist(), [1000, 1001, 1006, 1007, 1008])
        self.assertIsNotNone(match.original_positions)
        self.assertEqual(match.original_positions.tolist(), [0, 1, 6, 7, 8])

        # The thought-slice kv_indices ([1002, 1003, 1004, 1005]) should have been freed.
        freed_calls = mock_allocator.free.call_args_list
        freed_indices = []
        for call in freed_calls:
            arg = call.args[0] if call.args else call.kwargs.get("indices")
            if isinstance(arg, torch.Tensor):
                freed_indices.extend(arg.tolist())
        self.assertEqual(
            sorted(set(freed_indices) & {1002, 1003, 1004, 1005}),
            [1002, 1003, 1004, 1005],
        )


if __name__ == "__main__":
    unittest.main()
