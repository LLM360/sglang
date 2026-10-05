"""Tests for Req.answer_start_position tracking via update_reasoning_tokens."""

import unittest

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="stage-b-test-1-gpu-small")


class TestReqAnswerStartPosition(unittest.TestCase):
    def test_set_when_think_end_detected(self):
        """When update_reasoning_tokens sees the </think> token, answer_start_position
        is set to len(input) + reasoning_tokens, i.e. the position right after </think>."""
        # Prompt is 2 tokens [10, 11] at positions 0, 1.
        # Thoughts are 4 tokens [20, 21, 22, 99] at positions 2, 3, 4, 5 — 99 is </think>.
        # Answer should start at position 6.
        req = Req(rid="r1", origin_input_text="hi", origin_input_ids=[10, 11],
                  sampling_params=SamplingParams(), require_reasoning=True)
        think_end_id = 99
        # Feed thought tokens one at a time; not yet the </think> id.
        req.update_reasoning_tokens(20, think_end_id)
        req.update_reasoning_tokens(21, think_end_id)
        req.update_reasoning_tokens(22, think_end_id)
        self.assertIsNone(req.answer_start_position)
        # Feed the </think> token.
        req.update_reasoning_tokens(99, think_end_id)
        self.assertTrue(req._is_reasoning_over)
        self.assertEqual(req.answer_start_position, 6)


if __name__ == "__main__":
    unittest.main()
