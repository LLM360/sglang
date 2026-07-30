"""K2-V3 reasoning-boundary regression tests from LLM360/vllm PR #12."""

import unittest

from sglang.srt.parser.reasoning_parser import (
    K2V3Detector,
    K2V3DetectorLegacy,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

EFFORT_TOKENS = {
    "high": ("<ifm|think>", "</ifm|think>"),
    "medium": ("<ifm|think_fast>", "</ifm|think_fast>"),
    "low": ("<ifm|think_faster>", "</ifm|think_faster>"),
}
EFFORTS = tuple(EFFORT_TOKENS)


def _tool_call(name: str) -> str:
    return (
        f"<ifm|tool_call>{name}"
        "<ifm|arg_key>city</ifm|arg_key>"
        "<ifm|arg_value>Tokyo</ifm|arg_value>"
        "</ifm|tool_call>"
    )


TOOL_CALL = _tool_call("get_weather")
SECOND_TOOL_CALL = _tool_call("get_time")
GROUPED_TOOL_CALL = f"<ifm|tool_calls>{TOOL_CALL}</ifm|tool_calls>"
GROUPED_TOOL_CALLS = f"<ifm|tool_calls>{TOOL_CALL}{SECOND_TOOL_CALL}</ifm|tool_calls>"


def _stream(detector: K2V3Detector, deltas: list[str]):
    return [detector.parse_streaming_increment(delta) for delta in deltas]


class TestK2V3DetectorTokenSelection(CustomTestCase):
    def test_effort_tokens(self):
        for effort, expected_tokens in EFFORT_TOKENS.items():
            with self.subTest(effort=effort):
                detector = K2V3Detector(reasoning_effort=effort)
                self.assertEqual(
                    (detector.think_start_token, detector.think_end_token),
                    expected_tokens,
                )

    def test_default_none_and_unknown_effort_use_high(self):
        for effort in (None, "none", "ultra"):
            with self.subTest(effort=effort):
                kwargs = {} if effort is None else {"reasoning_effort": effort}
                detector = K2V3Detector(**kwargs)
                self.assertEqual(detector.think_start_token, "<ifm|think>")
                self.assertEqual(detector.think_end_token, "</ifm|think>")

    def test_force_reasoning_false_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires force_reasoning=True"):
            K2V3Detector(force_reasoning=False)


class TestK2V3NonStreaming(CustomTestCase):
    def test_nonstreaming_without_boundary_returns_content(self):
        for effort in EFFORTS:
            for output in ("", "The answer is 42."):
                with self.subTest(effort=effort, output=output):
                    result = K2V3Detector(reasoning_effort=effort).detect_and_parse(
                        output
                    )
                    self.assertEqual(result.reasoning_text, "")
                    self.assertEqual(result.normal_text, output)

    def test_nonstreaming_tool_calls_wrapper_implicitly_ends_unclosed_reasoning(self):
        for effort in EFFORTS:
            for reasoning in ("", "Need lookup. "):
                for tool_section in (GROUPED_TOOL_CALL, GROUPED_TOOL_CALLS):
                    with self.subTest(
                        effort=effort,
                        reasoning=reasoning,
                        tool_section=tool_section,
                    ):
                        result = K2V3Detector(reasoning_effort=effort).detect_and_parse(
                            reasoning + tool_section
                        )
                        self.assertEqual(result.reasoning_text, reasoning)
                        self.assertEqual(result.normal_text, tool_section)

    def test_nonstreaming_generated_start_with_unclosed_tool_call(self):
        for effort, (start, _) in EFFORT_TOKENS.items():
            with self.subTest(effort=effort):
                result = K2V3Detector(reasoning_effort=effort).detect_and_parse(
                    f"{start}Need lookup. {GROUPED_TOOL_CALL}"
                )
                self.assertEqual(result.reasoning_text, "Need lookup. ")
                self.assertEqual(result.normal_text, GROUPED_TOOL_CALL)

    def test_nonstreaming_incomplete_tool_calls_wrapper_is_implicit_boundary(self):
        incomplete_tools = (
            "<ifm|tool_calls>",
            "<ifm|tool_calls><ifm|tool_call>get_weather",
            f"<ifm|tool_calls>{TOOL_CALL}",
        )
        for effort in EFFORTS:
            for incomplete_tool in incomplete_tools:
                with self.subTest(effort=effort, incomplete_tool=incomplete_tool):
                    result = K2V3Detector(reasoning_effort=effort).detect_and_parse(
                        f"Need lookup. {incomplete_tool}"
                    )
                    self.assertEqual(result.reasoning_text, "Need lookup. ")
                    self.assertEqual(result.normal_text, incomplete_tool)

    def test_nonstreaming_singular_tool_tag_is_not_an_implicit_boundary(self):
        output = f"Need lookup. {TOOL_CALL}"
        for effort in EFFORTS:
            with self.subTest(effort=effort):
                result = K2V3Detector(reasoning_effort=effort).detect_and_parse(output)
                self.assertEqual(result.reasoning_text, "")
                self.assertEqual(result.normal_text, output)

    def test_nonstreaming_explicit_close_requires_complete_plural_tool_wrapper(self):
        incomplete_plural = f"<ifm|tool_calls>{TOOL_CALL}"
        for effort, (_, end) in EFFORT_TOKENS.items():
            for tool_markup in (TOOL_CALL, incomplete_plural):
                with self.subTest(effort=effort, tool_markup=tool_markup):
                    result = K2V3Detector(reasoning_effort=effort).detect_and_parse(
                        f"Need lookup.{end}{tool_markup}"
                    )
                    self.assertEqual(result.reasoning_text, "Need lookup.")
                    self.assertEqual(result.normal_text, tool_markup)

    def test_nonstreaming_explicit_close_response_matrix(self):
        cases = (
            ("", "", "", ""),
            ("Need lookup.", "", "Need lookup.", ""),
            ("", "The answer is 42.", "", "The answer is 42."),
            ("Need lookup.", GROUPED_TOOL_CALL, "Need lookup.", GROUPED_TOOL_CALL),
            (
                "Need lookup.",
                f"Calling the tool.\n{GROUPED_TOOL_CALL}",
                "Need lookup.",
                f"Calling the tool.\n{GROUPED_TOOL_CALL}",
            ),
        )
        for effort, (_, end) in EFFORT_TOKENS.items():
            for reasoning, tail, expected_reasoning, expected_content in cases:
                with self.subTest(effort=effort, reasoning=reasoning, tail=tail):
                    result = K2V3Detector(reasoning_effort=effort).detect_and_parse(
                        reasoning + end + tail
                    )
                    self.assertEqual(result.reasoning_text, expected_reasoning)
                    self.assertEqual(result.normal_text, expected_content)

    def test_nonstreaming_explicit_close_takes_precedence_over_tool_marker(self):
        reasoning_tool = _tool_call("consider_weather")
        expected_reasoning = f"Maybe call this tool: {reasoning_tool}"
        for effort, (_, end) in EFFORT_TOKENS.items():
            with self.subTest(effort=effort):
                result = K2V3Detector(reasoning_effort=effort).detect_and_parse(
                    f"{expected_reasoning}{end}{GROUPED_TOOL_CALL}"
                )
                self.assertEqual(result.reasoning_text, expected_reasoning)
                self.assertEqual(result.normal_text, GROUPED_TOOL_CALL)

    def test_nonstreaming_multiple_close_tokens_preserve_extra_close_as_content(self):
        for effort, (_, end) in EFFORT_TOKENS.items():
            with self.subTest(effort=effort):
                result = K2V3Detector(reasoning_effort=effort).detect_and_parse(
                    f"Need lookup.{end}{end}The answer is 42."
                )
                self.assertEqual(result.reasoning_text, "Need lookup.")
                self.assertEqual(result.normal_text, f"{end}The answer is 42.")


class TestK2V3Streaming(CustomTestCase):
    def test_streaming_standalone_end_token_emits_empty_reasoning(self):
        for effort, (_, end) in EFFORT_TOKENS.items():
            with self.subTest(effort=effort):
                result = K2V3Detector(
                    reasoning_effort=effort
                ).parse_streaming_increment(end)
                self.assertTrue(result.has_reasoning_text)
                self.assertEqual(result.reasoning_text, "")
                self.assertFalse(result.has_normal_text)

    def test_streaming_end_token_routes_following_tool_call_to_content(self):
        for effort, (_, end) in EFFORT_TOKENS.items():
            for reasoning in ("", "Need lookup"):
                with self.subTest(effort=effort, reasoning=reasoning):
                    detector = K2V3Detector(reasoning_effort=effort)
                    first, second = _stream(detector, [reasoning + end, TOOL_CALL])
                    self.assertTrue(first.has_reasoning_text)
                    self.assertEqual(first.reasoning_text, reasoning)
                    self.assertEqual(second.normal_text, TOOL_CALL)

    def test_streaming_missing_close_quarantines_plural_wrapper_until_finalization(
        self,
    ):
        for effort in EFFORTS:
            for tool_section in (GROUPED_TOOL_CALL, GROUPED_TOOL_CALLS):
                with self.subTest(effort=effort, tool_section=tool_section):
                    detector = K2V3Detector(reasoning_effort=effort)
                    marker = "<ifm|tool_calls>"
                    emitted = _stream(
                        detector,
                        [
                            "Need lookup. ",
                            marker[:8],
                            marker[8:] + tool_section[len(marker) :],
                        ],
                    )
                    self.assertTrue(
                        all(not result.has_normal_text for result in emitted)
                    )
                    finalization = detector.finalize_reasoning_streaming()
                    self.assertIsNotNone(finalization)
                    result = finalization.result
                    self.assertEqual(result.reasoning_text, "Need lookup. ")
                    self.assertEqual(result.normal_text, tool_section)

    def test_streaming_failed_plural_marker_candidate_is_released_as_content(self):
        for effort in EFFORTS:
            with self.subTest(effort=effort):
                detector = K2V3Detector(reasoning_effort=effort)
                _stream(detector, ["Need ", "<ifm|tool_call", ">not grouped"])
                result = detector.finalize_reasoning_streaming().result
                self.assertEqual(result.reasoning_text, "")
                self.assertEqual(result.normal_text, "Need <ifm|tool_call>not grouped")

    def test_streaming_explicit_close_releases_quarantined_wrapper_as_reasoning(self):
        for effort, (_, end) in EFFORT_TOKENS.items():
            with self.subTest(effort=effort):
                detector = K2V3Detector(reasoning_effort=effort)
                results = _stream(
                    detector,
                    ["Maybe ", GROUPED_TOOL_CALL, f" reconsider{end}Answer"],
                )
                emitted = [result for result in results if result.has_reasoning_text]
                self.assertEqual(len(emitted), 1)
                self.assertEqual(
                    emitted[0].reasoning_text,
                    f"Maybe {GROUPED_TOOL_CALL} reconsider",
                )
                self.assertEqual(emitted[0].normal_text, "Answer")
                self.assertIsNone(detector.finalize_reasoning_streaming())

    def test_streaming_only_post_close_plural_wrapper_becomes_content(self):
        for effort, (_, end) in EFFORT_TOKENS.items():
            with self.subTest(effort=effort):
                detector = K2V3Detector(reasoning_effort=effort)
                results = _stream(
                    detector,
                    [GROUPED_TOOL_CALL, end, GROUPED_TOOL_CALLS],
                )
                self.assertEqual(results[1].reasoning_text, GROUPED_TOOL_CALL)
                self.assertEqual(results[2].normal_text, GROUPED_TOOL_CALLS)

    def test_streaming_incomplete_plural_wrapper_is_released_at_finalization(self):
        incomplete_tools = (
            "<ifm|tool_calls>",
            "<ifm|tool_calls><ifm|tool_call>get_weather",
        )
        for effort in EFFORTS:
            for incomplete_tool in incomplete_tools:
                with self.subTest(effort=effort, incomplete_tool=incomplete_tool):
                    detector = K2V3Detector(reasoning_effort=effort)
                    result = detector.parse_streaming_increment(
                        f"Need lookup. {incomplete_tool}"
                    )
                    self.assertFalse(result.has_normal_text)
                    final = detector.finalize_reasoning_streaming().result
                    self.assertEqual(final.reasoning_text, "Need lookup. ")
                    self.assertEqual(final.normal_text, incomplete_tool)

    def test_streaming_without_boundary_finalizes_as_content(self):
        for effort in EFFORTS:
            with self.subTest(effort=effort):
                detector = K2V3Detector(reasoning_effort=effort)
                _stream(detector, ["Answer ", "without a close token."])
                final = detector.finalize_reasoning_streaming().result
                self.assertEqual(final.reasoning_text, "")
                self.assertEqual(final.normal_text, "Answer without a close token.")

    def test_streaming_optional_generated_start_is_not_emitted(self):
        for effort, (start, _) in EFFORT_TOKENS.items():
            for deltas in ([start, "Reasoning"], [start + "Reasoning"]):
                with self.subTest(effort=effort, deltas=deltas):
                    detector = K2V3Detector(reasoning_effort=effort)
                    results = _stream(detector, deltas)
                    self.assertTrue(
                        all(not result.has_reasoning_text for result in results)
                    )
                    final = detector.finalize_reasoning_streaming().result
                    self.assertEqual(final.reasoning_text, "")
                    self.assertEqual(final.normal_text, "Reasoning")

    def test_streaming_multiple_close_tokens_use_first_boundary(self):
        for effort, (_, end) in EFFORT_TOKENS.items():
            with self.subTest(effort=effort):
                result = K2V3Detector(
                    reasoning_effort=effort
                ).parse_streaming_increment(f"Reasoning{end}{end}Answer")
                self.assertEqual(result.reasoning_text, "Reasoning")
                self.assertEqual(result.normal_text, f"{end}Answer")

    def test_streaming_terminal_partial_plural_marker_is_finalized_as_content(self):
        for effort in EFFORTS:
            with self.subTest(effort=effort):
                detector = K2V3Detector(reasoning_effort=effort)
                detector.parse_streaming_increment("Need lookup. <ifm|tool_")
                final = detector.finalize_reasoning_streaming().result
                self.assertEqual(final.reasoning_text, "")
                self.assertEqual(final.normal_text, "Need lookup. <ifm|tool_")


class TestK2V3Legacy(CustomTestCase):
    def test_legacy_parser_keeps_eager_streaming_behavior(self):
        detector = K2V3DetectorLegacy(reasoning_effort="medium")
        first = detector.parse_streaming_increment("partial reasoning")
        second = detector.parse_streaming_increment("</think_fast>answer")
        self.assertEqual(first.reasoning_text, "partial reasoning")
        self.assertEqual(second.normal_text, "answer")


if __name__ == "__main__":
    unittest.main()
