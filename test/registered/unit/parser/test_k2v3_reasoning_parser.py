"""Unit tests for K2-v3 reasoning detectors (canonical IFM + legacy)."""

import unittest

from sglang.srt.parser.reasoning_parser import (
    K2V3Detector,
    K2V3DetectorLegacy,
    ReasoningParser,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestK2V3DetectorTokenSelection(CustomTestCase):
    """K2-v3 selects its IFM token pair based on reasoning_effort."""

    def test_high_effort_uses_think_tokens(self):
        detector = K2V3Detector(reasoning_effort="high")
        self.assertEqual(detector.think_start_token, "<ifm|think>")
        self.assertEqual(detector.think_end_token, "</ifm|think>")

    def test_medium_effort_uses_think_fast_tokens(self):
        detector = K2V3Detector(reasoning_effort="medium")
        self.assertEqual(detector.think_start_token, "<ifm|think_fast>")
        self.assertEqual(detector.think_end_token, "</ifm|think_fast>")

    def test_low_effort_uses_think_faster_tokens(self):
        detector = K2V3Detector(reasoning_effort="low")
        self.assertEqual(detector.think_start_token, "<ifm|think_faster>")
        self.assertEqual(detector.think_end_token, "</ifm|think_faster>")

    def test_none_effort_maps_to_high(self):
        detector = K2V3Detector(reasoning_effort="none")
        self.assertEqual(detector.think_start_token, "<ifm|think>")

    def test_unknown_effort_maps_to_high(self):
        detector = K2V3Detector(reasoning_effort="not-a-thing")
        self.assertEqual(detector.think_start_token, "<ifm|think>")

    def test_default_effort_is_high(self):
        detector = K2V3Detector()
        self.assertEqual(detector.think_start_token, "<ifm|think>")

    def test_tool_start_token_is_ifm(self):
        self.assertEqual(K2V3Detector().tool_start_token, "<ifm|tool_call>")

    def test_force_reasoning_false_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires force_reasoning=True"):
            K2V3Detector(force_reasoning=False)


class TestK2V3DetectorLegacyTokenSelection(CustomTestCase):
    """The legacy detector selects the pre-IFM token pair based on effort."""

    def test_high_effort_uses_legacy_think_tokens(self):
        detector = K2V3DetectorLegacy(reasoning_effort="high")
        self.assertEqual(detector.think_start_token, "<think>")
        self.assertEqual(detector.think_end_token, "</think>")

    def test_medium_effort_uses_legacy_think_fast_tokens(self):
        detector = K2V3DetectorLegacy(reasoning_effort="medium")
        self.assertEqual(detector.think_start_token, "<think_fast>")
        self.assertEqual(detector.think_end_token, "</think_fast>")

    def test_low_effort_uses_legacy_think_faster_tokens(self):
        detector = K2V3DetectorLegacy(reasoning_effort="low")
        self.assertEqual(detector.think_start_token, "<think_faster>")
        self.assertEqual(detector.think_end_token, "</think_faster>")

    def test_none_effort_maps_to_high(self):
        detector = K2V3DetectorLegacy(reasoning_effort="none")
        self.assertEqual(detector.think_start_token, "<think>")

    def test_default_effort_is_high(self):
        detector = K2V3DetectorLegacy()
        self.assertEqual(detector.think_start_token, "<think>")

    def test_tool_start_token_is_legacy(self):
        self.assertEqual(K2V3DetectorLegacy().tool_start_token, "<tool_call>")

    def test_force_reasoning_false_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires force_reasoning=True"):
            K2V3DetectorLegacy(force_reasoning=False)


class TestK2V3DetectorToolCallSplit(CustomTestCase):
    """K2-v3 exits reasoning mode when <ifm|tool_call> appears without an end token."""

    def test_tool_call_split_detect_and_parse(self):
        for effort in ["high", "medium", "low"]:
            with self.subTest(effort=effort):
                detector = K2V3Detector(reasoning_effort=effort)
                text = (
                    "I'll check the file.\n"
                    "<ifm|tool_call>read<ifm|arg_key>filePath</ifm|arg_key>"
                    "<ifm|arg_value>/tmp/f</ifm|arg_value></ifm|tool_call>"
                )
                result = detector.detect_and_parse(text)
                self.assertEqual(result.reasoning_text, "I'll check the file.\n")
                self.assertTrue(result.normal_text.startswith("<ifm|tool_call>"))

    def test_tool_call_split_streaming(self):
        for effort in ["high", "medium", "low"]:
            with self.subTest(effort=effort):
                detector = K2V3Detector(reasoning_effort=effort)
                r1 = detector.parse_streaming_increment("I'll check the file.\n")
                self.assertEqual(r1.reasoning_text, "I'll check the file.\n")
                self.assertEqual(r1.normal_text, "")
                r2 = detector.parse_streaming_increment(
                    "<ifm|tool_call>read</ifm|tool_call>"
                )
                self.assertEqual(r2.normal_text, "<ifm|tool_call>read</ifm|tool_call>")
                self.assertEqual(r2.reasoning_text, "")


class TestK2V3DetectorLegacyToolCallSplit(CustomTestCase):
    """The legacy detector exits reasoning on the legacy <tool_call> boundary."""

    def test_tool_call_split_detect_and_parse(self):
        for effort in ["high", "medium", "low"]:
            with self.subTest(effort=effort):
                detector = K2V3DetectorLegacy(reasoning_effort=effort)
                text = (
                    "I'll check the file.\n"
                    '<tool_call>\n{"name": "read"}\n</tool_call>'
                )
                result = detector.detect_and_parse(text)
                self.assertEqual(result.reasoning_text, "I'll check the file.\n")
                self.assertTrue(result.normal_text.startswith("<tool_call>"))

    def test_tool_call_split_streaming(self):
        detector = K2V3DetectorLegacy(reasoning_effort="high")
        r1 = detector.parse_streaming_increment("I'll check the file.\n")
        self.assertEqual(r1.reasoning_text, "I'll check the file.\n")
        r2 = detector.parse_streaming_increment('<tool_call>\n{"name": "read"}\n</tool_call>')
        self.assertEqual(r2.normal_text, '<tool_call>\n{"name": "read"}\n</tool_call>')
        self.assertEqual(r2.reasoning_text, "")


class TestK2V3DetectorParsing(CustomTestCase):
    """K2-v3 inherits the standard <think>...</think> state machine (IFM tokens)."""

    def test_high_effort_preserves_newline_prefixed_think_start(self):
        detector = K2V3Detector(reasoning_effort="high")
        text = "\n<ifm|think>reasoning here</ifm|think>final answer"
        result = detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "\nreasoning here")
        self.assertEqual(result.normal_text, "final answer")

    def test_high_effort_falls_back_to_bare_think_start(self):
        detector = K2V3Detector(reasoning_effort="high")
        text = "<ifm|think>reasoning here</ifm|think>final answer"
        result = detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "reasoning here")
        self.assertEqual(result.normal_text, "final answer")

    def test_preserves_reasoning_and_post_think_newlines(self):
        detector = K2V3Detector(reasoning_effort="high")
        text = (
            "<ifm|think>\n</ifm|think>\n"
            "<ifm|tool_calls><ifm|tool_call>get_weather</ifm|tool_call>"
            "</ifm|tool_calls>"
        )
        result = detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "\n")
        self.assertTrue(result.normal_text.startswith("\n<ifm|tool_calls>"))

    def test_streaming_preserves_newline_prefixed_think_start(self):
        detector = K2V3Detector(reasoning_effort="high")
        result = detector.parse_streaming_increment("\n<ifm|think>reasoning")
        self.assertEqual(result.reasoning_text, "\nreasoning")
        self.assertEqual(result.normal_text, "")

    def test_medium_effort_parses_end_only_output(self):
        detector = K2V3Detector(reasoning_effort="medium")
        text = "reasoning here</ifm|think_fast>final answer"
        result = detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "reasoning here")
        self.assertEqual(result.normal_text, "final answer")

    def test_medium_effort_parses_think_fast_block(self):
        detector = K2V3Detector(reasoning_effort="medium")
        text = "<ifm|think_fast>reasoning here</ifm|think_fast>final answer"
        result = detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "reasoning here")
        self.assertEqual(result.normal_text, "final answer")

    def test_low_effort_parses_think_faster_block(self):
        detector = K2V3Detector(reasoning_effort="low")
        text = "<ifm|think_faster>r</ifm|think_faster>a"
        result = detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "r")
        self.assertEqual(result.normal_text, "a")

    def test_streaming_medium_effort(self):
        detector = K2V3Detector(reasoning_effort="medium", force_reasoning=True)
        r1 = detector.parse_streaming_increment("partial reason")
        self.assertEqual(r1.reasoning_text, "partial reason")
        r2 = detector.parse_streaming_increment("ing</ifm|think_fast>answer")
        self.assertEqual(r2.reasoning_text, "ing")
        self.assertEqual(r2.normal_text, "answer")

    def test_ignores_legacy_think_tokens(self):
        """The canonical detector does not treat bare <think> as a boundary."""
        detector = K2V3Detector(reasoning_effort="high")
        result = detector.detect_and_parse("reasoning</think>still reasoning")
        # No </ifm|think> end token -> everything stays reasoning.
        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.reasoning_text, "reasoning</think>still reasoning")


class TestK2V3DetectorLegacyParsing(CustomTestCase):
    """The legacy detector parses the pre-IFM <think>...</think> tokens natively."""

    def test_high_effort_parses_legacy_think_block(self):
        detector = K2V3DetectorLegacy(reasoning_effort="high")
        text = "<think>reasoning here</think>final answer"
        result = detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "reasoning here")
        self.assertEqual(result.normal_text, "final answer")

    def test_high_effort_parses_legacy_end_only_output(self):
        detector = K2V3DetectorLegacy(reasoning_effort="high")
        text = "reasoning here</think>final answer"
        result = detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "reasoning here")
        self.assertEqual(result.normal_text, "final answer")

    def test_medium_effort_parses_legacy_think_fast_block(self):
        detector = K2V3DetectorLegacy(reasoning_effort="medium")
        text = "<think_fast>reasoning here</think_fast>final answer"
        result = detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "reasoning here")
        self.assertEqual(result.normal_text, "final answer")

    def test_low_effort_parses_legacy_think_faster_block(self):
        detector = K2V3DetectorLegacy(reasoning_effort="low")
        text = "<think_faster>r</think_faster>a"
        result = detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "r")
        self.assertEqual(result.normal_text, "a")

    def test_legacy_tokens_stream_natively(self):
        """Unlike the old normalize-on-parse design, legacy streaming now works."""
        detector = K2V3DetectorLegacy(reasoning_effort="medium", force_reasoning=True)
        r1 = detector.parse_streaming_increment("partial reason")
        self.assertEqual(r1.reasoning_text, "partial reason")
        r2 = detector.parse_streaming_increment("ing</think_fast>answer")
        self.assertEqual(r2.reasoning_text, "ing")
        self.assertEqual(r2.normal_text, "answer")

    def test_legacy_and_canonical_are_equivalent_modulo_tokens(self):
        for effort in ["high", "medium", "low"]:
            with self.subTest(effort=effort):
                legacy = K2V3DetectorLegacy(reasoning_effort=effort)
                canonical = K2V3Detector(reasoning_effort=effort)
                legacy_text = "r" + legacy.think_end_token + "a"
                canonical_text = "r" + canonical.think_end_token + "a"
                self.assertEqual(
                    legacy.detect_and_parse(legacy_text).reasoning_text,
                    canonical.detect_and_parse(canonical_text).reasoning_text,
                )
                self.assertEqual(
                    legacy.detect_and_parse(legacy_text).normal_text,
                    canonical.detect_and_parse(canonical_text).normal_text,
                )


class TestK2V3ParserIntegration(CustomTestCase):
    """ReasoningParser wires the k2_v3 / k2_v3_legacy model types to their detectors."""

    def test_parser_routes_to_k2v3_detector(self):
        parser = ReasoningParser(model_type="k2_v3")
        self.assertIsInstance(parser.detector, K2V3Detector)
        self.assertEqual(parser.detector.think_start_token, "<ifm|think>")

    def test_parser_routes_to_k2v3_legacy_detector(self):
        parser = ReasoningParser(model_type="k2_v3_legacy")
        self.assertIsInstance(parser.detector, K2V3DetectorLegacy)
        self.assertEqual(parser.detector.think_start_token, "<think>")
        self.assertEqual(parser.detector.tool_start_token, "<tool_call>")

    def test_k2v3_and_legacy_are_both_registered_for_cli(self):
        keys = ReasoningParser.DetectorMap.keys()
        self.assertIn("k2_v3", keys)
        self.assertIn("k2_v3_legacy", keys)

    def test_parser_rejects_force_reasoning_false(self):
        with self.assertRaisesRegex(ValueError, "requires force_reasoning=True"):
            ReasoningParser(model_type="k2_v3", force_reasoning=False)

    def test_legacy_parser_rejects_force_reasoning_false(self):
        with self.assertRaisesRegex(ValueError, "requires force_reasoning=True"):
            ReasoningParser(model_type="k2_v3_legacy", force_reasoning=False)

    def test_parser_allows_force_reasoning_false_for_non_k2v3(self):
        parser = ReasoningParser(model_type="qwen3", force_reasoning=False)
        self.assertEqual(parser.detector.think_start_token, "<think>")

    def test_parser_forwards_reasoning_effort_medium(self):
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="k2-v3",
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"reasoning_effort": "medium"},
        )
        parser = ReasoningParser(model_type="k2_v3", request=req)
        self.assertEqual(parser.detector.think_start_token, "<ifm|think_fast>")
        self.assertEqual(parser.detector.think_end_token, "</ifm|think_fast>")

    def test_parser_forwards_reasoning_effort_low(self):
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="k2-v3",
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"reasoning_effort": "low"},
        )
        parser = ReasoningParser(model_type="k2_v3", request=req)
        self.assertEqual(parser.detector.think_start_token, "<ifm|think_faster>")

    def test_legacy_parser_forwards_reasoning_effort_medium(self):
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="k2-v3",
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"reasoning_effort": "medium"},
        )
        parser = ReasoningParser(model_type="k2_v3_legacy", request=req)
        self.assertEqual(parser.detector.think_start_token, "<think_fast>")
        self.assertEqual(parser.detector.think_end_token, "</think_fast>")

    def test_parser_reads_top_level_reasoning_effort(self):
        """serving_chat.py pops reasoning_effort out of chat_template_kwargs
        and moves it to request.reasoning_effort before reaching the parser.
        The parser must read from the top-level field, not just the kwargs."""
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="k2-v3",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="medium",
        )
        parser = ReasoningParser(model_type="k2_v3", request=req)
        self.assertEqual(parser.detector.think_start_token, "<ifm|think_fast>")
        self.assertEqual(parser.detector.think_end_token, "</ifm|think_fast>")

    def test_parser_ignores_reasoning_effort_for_non_k2v3(self):
        """reasoning_effort kwarg must NOT be forwarded to Qwen3Detector.

        Qwen3Detector.__init__ does not accept reasoning_effort. If the
        ReasoningParser guard is ever dropped, building this parser would
        raise TypeError. Test the absence-of-leak property explicitly, not
        just the resulting token value.
        """
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="qwen3",
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"reasoning_effort": "medium"},
        )
        try:
            parser = ReasoningParser(model_type="qwen3", request=req)
        except TypeError as e:
            self.fail(
                "reasoning_effort was incorrectly forwarded to "
                f"Qwen3Detector: {e}"
            )
        self.assertEqual(parser.detector.think_start_token, "<think>")


if __name__ == "__main__":
    unittest.main()
