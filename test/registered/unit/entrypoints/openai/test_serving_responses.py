"""Unit tests for non-Harmony Responses streaming."""

import unittest
from unittest.mock import AsyncMock, Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.context import SimpleContext
from sglang.srt.entrypoints.openai.protocol import (
    RequestResponseMetadata,
    ResponsesRequest,
    ResponsesResponse,
    UsageInfo,
)
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.srt.utils import get_or_create_event_loop

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestSimpleResponsesStreaming(CustomTestCase):
    def _collect_events(self, model_deltas, *, reasoning=None):
        serving = object.__new__(OpenAIServingResponses)
        serving.reasoning_parser = "k2_v3"
        request = ResponsesRequest(
            input="test",
            stream=True,
            store=False,
            reasoning=reasoning,
        )
        context = SimpleContext()

        async def result_generator():
            cumulative_text = ""
            for delta_text, finish_type in model_deltas:
                cumulative_text += delta_text
                context.append_output(
                    {
                        "text": cumulative_text,
                        "meta_info": {
                            "finish_reason": (
                                {"type": finish_type} if finish_type else None
                            )
                        },
                    }
                )
                yield context

        async def collect():
            return [
                event
                async for event in serving._process_simple_streaming_events(
                    request=request,
                    result_generator=result_generator(),
                )
            ]

        return get_or_create_event_loop().run_until_complete(collect())

    def test_simple_streaming_skips_empty_reasoning_boundary(self):
        events = self._collect_events(
            [("</ifm|think>", None), ("The answer is 42.", "stop")]
        )

        self.assertFalse(any("reasoning" in event.type for event in events))
        self.assertEqual(
            [
                event.delta
                for event in events
                if event.type == "response.output_text.delta"
            ],
            ["The answer is 42."],
        )

    def test_simple_streaming_treats_combined_empty_reasoning_as_content(self):
        events = self._collect_events([("</ifm|think>The answer is 42.", "stop")])

        self.assertFalse(any("reasoning" in event.type for event in events))
        text_delta = next(
            event for event in events if event.type == "response.output_text.delta"
        )
        self.assertEqual(text_delta.delta, "The answer is 42.")

    def test_simple_streaming_preserves_nonempty_reasoning_before_boundary(self):
        events = self._collect_events(
            [("Need to calculate.</ifm|think>The answer is 42.", "stop")]
        )

        self.assertEqual(
            [
                event.delta
                for event in events
                if event.type == "response.reasoning_text.delta"
            ],
            ["Need to calculate."],
        )
        reasoning_done = next(
            event for event in events if event.type == "response.reasoning_text.done"
        )
        self.assertEqual(reasoning_done.text, "Need to calculate.")
        self.assertEqual(
            [
                event.delta
                for event in events
                if event.type == "response.output_text.delta"
            ],
            ["The answer is 42."],
        )

    def test_simple_streaming_terminal_finalization_preserves_held_content(self):
        wrapper = "<ifm|tool_calls>held tool markup</ifm|tool_calls>"
        events = self._collect_events([(f"Need lookup.{wrapper}", "stop")])

        self.assertEqual(
            [
                event.delta
                for event in events
                if event.type == "response.reasoning_text.delta"
            ],
            ["Need lookup."],
        )
        self.assertEqual(
            [
                event.delta
                for event in events
                if event.type == "response.output_text.delta"
            ],
            [wrapper],
        )

    def test_simple_streaming_plain_unclosed_output_finalizes_as_content(self):
        answer = "Plain answer without a reasoning close."
        events = self._collect_events([(answer, "stop")])

        self.assertFalse(any("reasoning" in event.type for event in events))
        self.assertEqual(
            [
                event.delta
                for event in events
                if event.type == "response.output_text.delta"
            ],
            [answer],
        )

    def test_simple_streaming_combined_delta_keeps_reasoning_and_content(self):
        events = self._collect_events(
            [("Held reasoning</ifm|think>Final answer", "stop")]
        )

        self.assertEqual(
            [
                event.delta
                for event in events
                if event.type == "response.reasoning_text.delta"
            ],
            ["Held reasoning"],
        )
        self.assertEqual(
            [
                event.delta
                for event in events
                if event.type == "response.output_text.delta"
            ],
            ["Final answer"],
        )

    def test_simple_streaming_uses_responses_reasoning_effort(self):
        events = self._collect_events(
            [("Fast reasoning</ifm|think_fast>Final answer", "stop")],
            reasoning={"effort": "medium"},
        )

        self.assertEqual(
            [
                event.delta
                for event in events
                if event.type == "response.reasoning_text.delta"
            ],
            ["Fast reasoning"],
        )
        self.assertEqual(
            [
                event.delta
                for event in events
                if event.type == "response.output_text.delta"
            ],
            ["Final answer"],
        )

    def test_make_request_forwards_responses_reasoning_effort(self):
        serving = object.__new__(OpenAIServingResponses)
        serving.tokenizer_manager = Mock()
        serving.tokenizer_manager.model_config.is_multimodal = False
        serving._process_messages = Mock(
            return_value=Mock(prompt_ids=[1, 2], prompt=None)
        )
        request = ResponsesRequest(
            model="model",
            input="test",
            stream=True,
            store=False,
            reasoning={"effort": "low"},
        )

        get_or_create_event_loop().run_until_complete(
            serving._make_request(request, prev_response=None, tokenizer=Mock())
        )

        chat_request = serving._process_messages.call_args.args[0]
        self.assertEqual(chat_request.reasoning_effort, "low")

    def test_simple_streaming_keeps_content_index_stable_across_deltas(self):
        events = self._collect_events(
            [
                ("</ifm|think>First", None),
                (" second", "stop"),
            ]
        )

        text_deltas = [
            event for event in events if event.type == "response.output_text.delta"
        ]
        self.assertEqual([event.delta for event in text_deltas], ["First", " second"])
        self.assertTrue(
            all(
                event.content_index == 0
                for event in events
                if hasattr(event, "content_index")
            )
        )

    def test_simple_streaming_abort_does_not_finalize_held_content(self):
        events = self._collect_events(
            [("<ifm|tool_calls>held</ifm|tool_calls>", "abort")]
        )

        self.assertEqual(events, [])

    def test_completed_event_uses_shared_finalization_and_maps_usage(self):
        serving = object.__new__(OpenAIServingResponses)
        request = ResponsesRequest(input="test", stream=True, store=False)
        response = ResponsesResponse.from_request(
            request=request,
            sampling_params={},
            model_name="model",
            created_time=123,
            output=[],
            status="completed",
            usage=UsageInfo(
                prompt_tokens=3,
                completion_tokens=5,
                reasoning_tokens=2,
                total_tokens=8,
            ),
        )
        serving.responses_full_generator = AsyncMock(return_value=response)

        event = get_or_create_event_loop().run_until_complete(
            serving._create_streaming_completed_event(
                request=request,
                sampling_params={},
                context=SimpleContext(),
                model_name="model",
                tokenizer=None,
                request_metadata=RequestResponseMetadata(request_id="request"),
                created_time=123,
            )
        )

        self.assertEqual(event.type, "response.completed")
        self.assertEqual(event.response.usage.input_tokens, 3)
        self.assertEqual(event.response.usage.output_tokens, 5)
        self.assertEqual(event.response.usage.total_tokens, 8)
        serving.responses_full_generator.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
