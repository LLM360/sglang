"""ChunkCache.cache_finished_req must accept the split kwarg used by the
--no-cache-thoughts code path. ChunkCache doesn't do prefix caching, so the
behavior is to ignore split entirely and fall back to its default cleanup.
"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.common import NoCacheThoughtsSplit


class TestChunkCacheSplitKwarg(unittest.TestCase):
    def test_cache_finished_req_accepts_split(self):
        cache = ChunkCache.__new__(ChunkCache)
        cache.req_to_token_pool = MagicMock()
        cache.req_to_token_pool.req_to_token = torch.tensor(
            [[10, 11, 12]], dtype=torch.int64
        )
        cache.token_to_kv_pool_allocator = MagicMock()

        req = MagicMock()
        req.pop_committed_kv_cache.return_value = 3
        req.req_pool_idx = 0

        split = NoCacheThoughtsSplit(
            virtual_token_ids=[1, 2],
            virtual_kv_indices=torch.tensor([10, 12], dtype=torch.int64),
            virtual_positions=torch.tensor([0, 5], dtype=torch.int64),
            thought_kv_indices_to_free=torch.tensor([11], dtype=torch.int64),
        )

        # Must not raise TypeError on the new kwarg.
        cache.cache_finished_req(req, is_insert=True, split=split)


if __name__ == "__main__":
    unittest.main()
