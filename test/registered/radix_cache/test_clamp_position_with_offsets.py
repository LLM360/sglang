"""Tests for clamp_position honoring per-request position offsets so decode tokens
after a non-contiguous prefill cache hit get the right RoPE positions.

At decode step N the next token's RoPE position should be:
   (seq_lens[i] - 1) + position_offsets[i]
where position_offsets[i] accounts for the gap in RoPE-space caused by thought
tokens that were skipped from the cached entry.
"""

import unittest

import torch

from sglang.srt.model_executor.forward_batch_info import _clamp_position_native

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="stage-b-test-1-gpu-small")


class TestClampPositionWithOffsets(unittest.TestCase):
    def test_offset_shifts_position(self):
        # Single req, seq_len=6 (so legacy position is 5), with a 4-position gap
        # in RoPE space from skipped thoughts -> next decode position should be 9.
        seq_lens = torch.tensor([6], dtype=torch.int64)
        offsets = torch.tensor([4], dtype=torch.int64)

        positions = _clamp_position_native(seq_lens, position_offsets=offsets)
        self.assertEqual(positions.tolist(), [9])

    def test_legacy_behavior_when_offsets_none(self):
        # Without offsets, behavior must match today's clamp(seq_lens - 1, min=0).
        seq_lens = torch.tensor([5, 0, 3], dtype=torch.int64)
        positions = _clamp_position_native(seq_lens)
        self.assertEqual(positions.tolist(), [4, 0, 2])


if __name__ == "__main__":
    unittest.main()
