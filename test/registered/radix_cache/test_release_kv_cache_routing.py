"""Tests that release_kv_cache routes through the split helper when --no-cache-thoughts
is enabled and the request has a recorded answer_start_position.

The test mocks the tree_cache and req objects narrowly enough to observe the routing
decision without needing a real KV pool. The next-cycle test will exercise the
non-flag path to ensure no regression.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.common import (
    NoCacheThoughtsSplit,
    release_kv_cache,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="stage-b-test-1-gpu-small")


class TestReleaseKvCacheRouting(unittest.TestCase):
    def _make_tree_cache_mock(self):
        tree = MagicMock()
        tree.supports_mamba.return_value = False
        # Per-req KV slot used by the split helper.
        tree.req_to_token_pool.req_to_token = torch.tensor(
            [[1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]],
            dtype=torch.int64,
        )
        # The split helper builds its slot indexing from this tensor.
        return tree

    def _make_req_mock(self):
        req = MagicMock()
        req.req_pool_idx = 0
        req.require_reasoning = True
        req.answer_start_position = 6
        req.origin_input_ids = [101, 102]
        req.output_ids = [201, 202, 203, 204, 301, 302, 303]
        req.pop_overallocated_kv_cache.return_value = (0, 0)
        req.mamba_pool_idx = None
        return req

    def test_routes_through_split_when_flag_on(self):
        tree = self._make_tree_cache_mock()
        req = self._make_req_mock()

        fake_server_args = MagicMock()
        fake_server_args.no_cache_thoughts = True
        fake_server_args.page_size = 1
        fake_server_args.speculative_algorithm = None

        with patch(
            "sglang.srt.mem_cache.common.get_global_server_args",
            return_value=fake_server_args,
        ):
            release_kv_cache(req, tree, is_insert=True)

        # cache_finished_req must have been called with a split kwarg.
        call = tree.cache_finished_req.call_args
        self.assertIsNotNone(call, "cache_finished_req was not called")
        split = call.kwargs.get("split")
        self.assertIsNotNone(split, "split kwarg was not passed")
        self.assertIsInstance(split, NoCacheThoughtsSplit)
        # Sanity-check the split contents match the synthetic Req.
        self.assertEqual(split.virtual_token_ids, [101, 102, 301, 302, 303])
        self.assertEqual(split.virtual_positions.tolist(), [0, 1, 6, 7, 8])

    def test_no_split_when_flag_off(self):
        tree = self._make_tree_cache_mock()
        req = self._make_req_mock()  # has require_reasoning + answer_start_position set

        fake_server_args = MagicMock()
        fake_server_args.no_cache_thoughts = False  # flag off
        fake_server_args.page_size = 1
        fake_server_args.speculative_algorithm = None

        with patch(
            "sglang.srt.mem_cache.common.get_global_server_args",
            return_value=fake_server_args,
        ):
            release_kv_cache(req, tree, is_insert=True)

        call = tree.cache_finished_req.call_args
        self.assertIsNotNone(call)
        self.assertIsNone(
            call.kwargs.get("split"),
            "split kwarg should not be passed when --no-cache-thoughts is off",
        )


if __name__ == "__main__":
    unittest.main()
