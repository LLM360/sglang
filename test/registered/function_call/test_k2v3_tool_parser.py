"""Unit tests for K2V3Detector (BBQ 0518 IFM tool-call format).

Ported from vLLM's K2V3ToolParser tests
(tests/entrypoints/openai/tool_parsers/test_multi_format_tool_parser.py).
"""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.multi_format_detector import (
    K2V3Detector,
    K2V3DetectorTracking,
    MultiFormatDetector,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(5, "stage-a-test-cpu")


def _make_tool(name, parameters=None):
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=f"{name} tool",
            parameters=parameters
            or {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        ),
    )


class TestK2V3DetectorConstruction(unittest.TestCase):
    """K2V3Detector defaults to the IFM 'xml' dialect with no delegate."""

    def test_default_dialect_is_xml(self):
        det = K2V3Detector()
        self.assertEqual(det.tool_format, "xml")
        self.assertIsNone(det._delegate)

    def test_subclass_of_multi_format(self):
        self.assertIsInstance(K2V3Detector(), MultiFormatDetector)

    def test_tool_call_format_kwarg_selects_dialect(self):
        det = K2V3Detector(chat_template_kwargs={"tool_call_format": "xml_typed"})
        self.assertEqual(det.tool_format, "xml_typed")

    def test_tool_format_kwarg_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "Unsupported argument: tool_format"
        ):
            K2V3Detector(chat_template_kwargs={"tool_format": "json"})

    def test_tool_calling_format_kwarg_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "Unsupported argument: tool_calling_format"
        ):
            K2V3Detector(chat_template_kwargs={"tool_calling_format": "xml_typed"})

    def test_json_dialect_via_positional(self):
        det = K2V3Detector(tool_format="json")
        self.assertEqual(det.tool_format, "json")

    def test_unknown_dialect_errors(self):
        with self.assertRaisesRegex(ValueError, "Unsupported tool_format"):
            K2V3Detector(tool_format="not-a-dialect")


class TestK2V3XmlExtraction(unittest.TestCase):
    def setUp(self):
        self.tools = [
            _make_tool(
                "get_weather",
                {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                    },
                },
            )
        ]
        self.det = K2V3Detector()

    def test_has_tool_call(self):
        text = (
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        self.assertTrue(self.det.has_tool_call(text))
        self.assertFalse(self.det.has_tool_call("no markers here"))

    def test_single_call(self):
        text = (
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[0].tool_index, 0)
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Tokyo"})
        self.assertIsNone(result.normal_text)

    def test_schema_typed_value_coercion(self):
        # 'days' is declared integer in the schema -> deserialized to int 3.
        text = (
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>days</ifm|arg_key><ifm|arg_value>3</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(json.loads(result.calls[0].parameters), {"days": 3})

    def test_schema_string_preserves_argument_whitespace(self):
        text = (
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key>"
            "<ifm|arg_value>\nTokyo\n</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "\nTokyo\n"})

    def test_schema_integer_strips_outer_whitespace_before_coercion(self):
        text = (
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>days</ifm|arg_key><ifm|arg_value>\n3\n</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(json.loads(result.calls[0].parameters), {"days": 3})

    def test_no_args(self):
        text = "<ifm|tool_call>get_weather</ifm|tool_call>"
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {})

    def test_no_markers_returns_full_text(self):
        result = self.det.detect_and_parse("plain text", self.tools)
        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, "plain text")

    def test_unknown_tool_is_forwarded(self):
        text = (
            "<ifm|tool_call>not_registered"
            "<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].tool_index, -1)
        self.assertEqual(result.calls[0].name, "not_registered")


class TestK2V3XmlTyped(unittest.TestCase):
    """The <ifm|arg_type> hint forces a value to stay a string."""

    def test_arg_type_string_keeps_numeric_as_string(self):
        det = K2V3Detector(tool_format="xml_typed")
        # No schema type for user_id; the inline <ifm|arg_type>string</...> wins.
        text = (
            "<ifm|tool_call>study_args"
            "<ifm|arg_key>user_id</ifm|arg_key>"
            "<ifm|arg_type>string</ifm|arg_type>"
            "<ifm|arg_value>12345</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        result = det.detect_and_parse(text, [_make_tool("study_args")])
        self.assertEqual(json.loads(result.calls[0].parameters), {"user_id": "12345"})

    def test_arg_type_any_preserves_argument_whitespace(self):
        det = K2V3Detector(tool_format="xml_typed")
        text = (
            "<ifm|tool_call>study_args"
            "<ifm|arg_key>notes</ifm|arg_key>"
            "<ifm|arg_type>any</ifm|arg_type>"
            "<ifm|arg_value>\nkeep me\n</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        result = det.detect_and_parse(text, [_make_tool("study_args")])
        self.assertEqual(
            json.loads(result.calls[0].parameters), {"notes": "\nkeep me\n"}
        )

    def test_boolean_arg_type_strips_outer_whitespace_before_coercion(self):
        det = K2V3Detector(tool_format="xml_typed")
        text = (
            "<ifm|tool_call>study_args"
            "<ifm|arg_key>enabled</ifm|arg_key>"
            "<ifm|arg_type>boolean</ifm|arg_type>"
            "<ifm|arg_value>\ntrue\n</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        result = det.detect_and_parse(text, [_make_tool("study_args")])
        self.assertEqual(json.loads(result.calls[0].parameters), {"enabled": True})


class TestK2V3ReasoningPrefix(unittest.TestCase):
    def setUp(self):
        self.tools = [_make_tool("get_weather")]
        self.det = K2V3Detector()

    def test_ifm_reasoning_prefix_is_stripped(self):
        text = (
            "<ifm|think>need lookup</ifm|think>\n"
            "<ifm|tool_calls>\n"
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>\n"
            "</ifm|tool_calls>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "\n")
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Tokyo"})

    def test_tool_only_preserves_newline_after_reasoning_prefix(self):
        text = (
            "<ifm|think>\n</ifm|think>\n"
            "<ifm|tool_calls>\n"
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>\n"
            "</ifm|tool_calls>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "\n")
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Tokyo"})

    def test_tool_only_preserves_whitespace_prefix_before_tool_calls(self):
        text = (
            "\n"
            "<ifm|tool_calls>\n"
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>\n"
            "</ifm|tool_calls>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "\n")
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Tokyo"})

    def test_all_three_effort_blocks_are_stripped(self):
        for think in ("think", "think_fast", "think_faster"):
            with self.subTest(think=think):
                text = (
                    f"<ifm|{think}>reasoning</ifm|{think}>"
                    "<ifm|tool_call>get_weather"
                    "<ifm|arg_key>city</ifm|arg_key>"
                    "<ifm|arg_value>Tokyo</ifm|arg_value>"
                    "</ifm|tool_call>"
                )
                result = self.det.detect_and_parse(text, self.tools)
                self.assertIsNone(result.normal_text)
                self.assertEqual(result.calls[0].name, "get_weather")

    def test_stripped_ifm_reasoning_prefix_empty_content_returns_none(self):
        text = (
            "<ifm|think>need lookup</ifm|think>"
            "<ifm|tool_calls>"
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>"
            "</ifm|tool_calls>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertIsNone(result.normal_text)
        self.assertEqual(result.calls[0].name, "get_weather")

    def test_legacy_think_prefix_is_not_stripped(self):
        text = (
            "<think>legacy reasoning</think>\n"
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "<think>legacy reasoning</think>\n")
        self.assertEqual(result.calls[0].name, "get_weather")


class TestK2V3JsonDialect(unittest.TestCase):
    def setUp(self):
        self.tools = [_make_tool("get_weather")]
        self.det = K2V3Detector(tool_format="json")

    def test_json_object_tool_call(self):
        text = (
            "<ifm|tool_call>"
            '{"name": "get_weather", "arguments": {"city": "Tokyo"}}'
            "</ifm|tool_call>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertIsNone(result.normal_text)
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Tokyo"})

    def test_json_arguments_as_string(self):
        text = (
            "<ifm|tool_call>"
            '{"name": "get_weather", "arguments": "{\\"city\\": \\"Tokyo\\"}"}'
            "</ifm|tool_call>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Tokyo"})

    def test_json_list_of_tool_calls(self):
        text = (
            "<ifm|tool_call>"
            '[{"name": "get_weather", "arguments": {"city": "Tokyo"}},'
            ' {"name": "get_weather", "arguments": {"city": "Osaka"}}]'
            "</ifm|tool_call>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual([c.name for c in result.calls], ["get_weather", "get_weather"])
        self.assertEqual(json.loads(result.calls[1].parameters), {"city": "Osaka"})


def _collect_stream(detector, tools, chunks):
    """Feed ``chunks`` to a detector's streaming parser and reassemble the
    per-tool name + parameters, plus the concatenated normal text. Mirrors the
    GLM detector streaming tests' collection helper."""
    by_index = {}
    order = []
    normal_text = ""
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, tools)
        normal_text += result.normal_text or ""
        for item in result.calls:
            idx = item.tool_index
            if idx not in by_index:
                by_index[idx] = {"name": None, "parameters": ""}
                order.append(idx)
            if item.name:
                by_index[idx]["name"] = item.name
            if item.parameters:
                by_index[idx]["parameters"] += item.parameters
    return normal_text, [by_index[i] for i in order]


def _event_types(detector):
    return [event["type"] for event in detector.fallback_events]


class TestK2V3DetectorTracking(unittest.TestCase):
    """Opt-in K2-v3 detector that records selected non-stream fallbacks."""

    def setUp(self):
        self.tools = [
            _make_tool(
                "study_args",
                {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "flag": {"type": "boolean"},
                        "payload": {"type": "object"},
                        "days": {"type": "integer"},
                        "notes": {"type": "any"},
                        "user_id": {"type": "string"},
                    },
                },
            )
        ]

    @staticmethod
    def _call_tuple(call):
        return call.tool_index, call.name, call.parameters

    def _assert_parse_equal(self, text, *, dialect="xml", tools=None):
        tools = tools or self.tools
        base = K2V3Detector(tool_format=dialect).detect_and_parse(text, tools)
        tracked_detector = K2V3DetectorTracking(tool_format=dialect)
        tracked = tracked_detector.detect_and_parse(text, tools)

        self.assertEqual(tracked.normal_text, base.normal_text)
        self.assertEqual(len(tracked.calls), len(base.calls))
        self.assertEqual(
            [self._call_tuple(call) for call in tracked.calls],
            [self._call_tuple(call) for call in base.calls],
        )
        return tracked_detector, tracked

    def _assert_stream_equal(self, chunks, *, dialect="xml", tools=None):
        tools = tools or self.tools
        base = K2V3Detector(tool_format=dialect)
        tracked = K2V3DetectorTracking(tool_format=dialect)
        for chunk in chunks:
            with self.subTest(dialect=dialect, chunk=chunk):
                base_result = base.parse_streaming_increment(chunk, tools)
                tracked_result = tracked.parse_streaming_increment(chunk, tools)
                self.assertEqual(tracked_result.normal_text, base_result.normal_text)
                self.assertEqual(
                    [self._call_tuple(call) for call in tracked_result.calls],
                    [self._call_tuple(call) for call in base_result.calls],
                )
        self.assertEqual(tracked.fallback_events, [])

    def test_non_stream_output_matches_base_detector(self):
        cases = [
            (
                "direct_xml",
                "xml",
                "<ifm|tool_call>study_args"
                "<ifm|arg_key>city</ifm|arg_key>"
                "<ifm|arg_value>Tokyo</ifm|arg_value>"
                "</ifm|tool_call>",
            ),
            ("xml_no_args", "xml", "<ifm|tool_call>study_args</ifm|tool_call>"),
            ("unknown_tool", "xml", "<ifm|tool_call>unknown</ifm|tool_call>"),
            (
                "ordinary_prefix",
                "xml",
                "Sure.<ifm|tool_call>study_args</ifm|tool_call>",
            ),
            (
                "whitespace_prefix",
                "xml",
                "\n<ifm|tool_call>study_args</ifm|tool_call>",
            ),
            (
                "ifm_reasoning_no_residue",
                "xml",
                "<ifm|think>need lookup</ifm|think>"
                "<ifm|tool_call>study_args</ifm|tool_call>",
            ),
            (
                "ifm_reasoning_newline_residue",
                "xml",
                "<ifm|think>need lookup</ifm|think>\n"
                "<ifm|tool_call>study_args</ifm|tool_call>",
            ),
            (
                "wrapper",
                "xml",
                "<ifm|tool_calls>\n"
                "<ifm|tool_call>study_args</ifm|tool_call>"
                "</ifm|tool_calls>",
            ),
            (
                "multiple_xml_calls",
                "xml",
                "<ifm|tool_call>study_args"
                "<ifm|arg_key>city</ifm|arg_key>"
                "<ifm|arg_value>Tokyo</ifm|arg_value>"
                "</ifm|tool_call>"
                "<ifm|tool_call>study_args"
                "<ifm|arg_key>city</ifm|arg_key>"
                "<ifm|arg_value>Osaka</ifm|arg_value>"
                "</ifm|tool_call>",
            ),
            (
                "schema_string_whitespace",
                "xml",
                "<ifm|tool_call>study_args"
                "<ifm|arg_key>city</ifm|arg_key>"
                "<ifm|arg_value>\nTokyo\n</ifm|arg_value>"
                "</ifm|tool_call>",
            ),
            (
                "schema_integer",
                "xml",
                "<ifm|tool_call>study_args"
                "<ifm|arg_key>days</ifm|arg_key>"
                "<ifm|arg_value>3</ifm|arg_value>"
                "</ifm|tool_call>",
            ),
            (
                "schema_boolean_true",
                "xml",
                "<ifm|tool_call>study_args"
                "<ifm|arg_key>flag</ifm|arg_key>"
                "<ifm|arg_value>True</ifm|arg_value>"
                "</ifm|tool_call>",
            ),
            (
                "raw_string_fallback",
                "xml",
                "<ifm|tool_call>study_args"
                "<ifm|arg_key>payload</ifm|arg_key>"
                "<ifm|arg_value>abc</ifm|arg_value>"
                "</ifm|tool_call>",
            ),
            (
                "python_literal_current_behavior",
                "xml",
                "<ifm|tool_call>study_args"
                "<ifm|arg_key>payload</ifm|arg_key>"
                "<ifm|arg_value>{1, 2}</ifm|arg_value>"
                "</ifm|tool_call>",
            ),
            (
                "xml_typed_string",
                "xml_typed",
                "<ifm|tool_call>study_args"
                "<ifm|arg_key>user_id</ifm|arg_key>"
                "<ifm|arg_type>string</ifm|arg_type>"
                "<ifm|arg_value>12345</ifm|arg_value>"
                "</ifm|tool_call>",
            ),
            (
                "xml_typed_any_whitespace",
                "xml_typed",
                "<ifm|tool_call>study_args"
                "<ifm|arg_key>notes</ifm|arg_key>"
                "<ifm|arg_type>any</ifm|arg_type>"
                "<ifm|arg_value>\nkeep me\n</ifm|arg_value>"
                "</ifm|tool_call>",
            ),
            (
                "xml_typed_boolean",
                "xml_typed",
                "<ifm|tool_call>study_args"
                "<ifm|arg_key>flag</ifm|arg_key>"
                "<ifm|arg_type>boolean</ifm|arg_type>"
                "<ifm|arg_value>True</ifm|arg_value>"
                "</ifm|tool_call>",
            ),
            (
                "schema_wins_over_inline",
                "xml_typed",
                "<ifm|tool_call>study_args"
                "<ifm|arg_key>days</ifm|arg_key>"
                "<ifm|arg_type>string</ifm|arg_type>"
                "<ifm|arg_value>3</ifm|arg_value>"
                "</ifm|tool_call>",
            ),
            (
                "json_object_args",
                "json",
                "<ifm|tool_call>"
                '{"name": "study_args", "arguments": {"city": "Tokyo"}}'
                "</ifm|tool_call>",
            ),
            (
                "json_string_args",
                "json",
                "<ifm|tool_call>"
                '{"name": "study_args", "arguments": "{\\"city\\": \\"Tokyo\\"}"}'
                "</ifm|tool_call>",
            ),
            (
                "json_call_list",
                "json",
                "<ifm|tool_call>"
                '[{"name": "study_args", "arguments": {"city": "Tokyo"}},'
                ' {"name": "study_args", "arguments": {"city": "Osaka"}}]'
                "</ifm|tool_call>",
            ),
            ("no_tool_call_text", "xml", "No tool call here."),
        ]

        for name, dialect, text in cases:
            with self.subTest(name=name, dialect=dialect):
                self._assert_parse_equal(text, dialect=dialect)

    def test_tracks_json_failure_before_ast_literal_eval(self):
        text = (
            "<ifm|tool_call>study_args"
            "<ifm|arg_key>flag</ifm|arg_key>"
            "<ifm|arg_value>True</ifm|arg_value>"
            "</ifm|tool_call>"
        )

        detector, result = self._assert_parse_equal(text)

        self.assertEqual(json.loads(result.calls[0].parameters), {"flag": True})
        self.assertEqual(
            _event_types(detector), ["json_loads_failed_ast_literal_eval"]
        )
        self.assertEqual(detector.fallback_events[0]["phase"], "coercion")

    def test_tracks_raw_string_after_json_and_ast_fail(self):
        text = (
            "<ifm|tool_call>study_args"
            "<ifm|arg_key>payload</ifm|arg_key>"
            "<ifm|arg_value>abc</ifm|arg_value>"
            "</ifm|tool_call>"
        )

        detector, result = self._assert_parse_equal(text)

        self.assertEqual(json.loads(result.calls[0].parameters), {"payload": "abc"})
        self.assertEqual(
            _event_types(detector),
            [
                "json_loads_failed_ast_literal_eval",
                "ast_literal_eval_failed_raw_string",
            ],
        )

    def test_string_schema_does_not_track_deserialization_fallbacks(self):
        text = (
            "<ifm|tool_call>study_args"
            "<ifm|arg_key>user_id</ifm|arg_key>"
            "<ifm|arg_value>12345</ifm|arg_value>"
            "</ifm|tool_call>"
        )

        detector, result = self._assert_parse_equal(text)

        self.assertEqual(
            json.loads(result.calls[0].parameters), {"user_id": "12345"}
        )
        self.assertEqual(detector.fallback_events, [])

    def test_tracks_ifm_reasoning_prefix_stripped(self):
        cases = [
            (
                "<ifm|think>need lookup</ifm|think>"
                "<ifm|tool_call>study_args</ifm|tool_call>"
            ),
            (
                "<ifm|think>need lookup</ifm|think>\n"
                "<ifm|tool_call>study_args</ifm|tool_call>"
            ),
        ]
        for text in cases:
            with self.subTest(text=text):
                detector, _ = self._assert_parse_equal(text)
                self.assertEqual(
                    _event_types(detector), ["ifm_reasoning_prefix_stripped"]
                )
                self.assertEqual(detector.fallback_events[0]["phase"], "non_stream")

    def test_clean_parse_does_not_track_events(self):
        text = (
            "<ifm|tool_call>study_args"
            "<ifm|arg_key>city</ifm|arg_key>"
            "<ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>"
        )

        detector, _ = self._assert_parse_equal(text)

        self.assertEqual(detector.fallback_events, [])

    def test_reused_detector_clears_events_on_non_stream_parse(self):
        detector = K2V3DetectorTracking()
        fallback_text = (
            "<ifm|tool_call>study_args"
            "<ifm|arg_key>payload</ifm|arg_key>"
            "<ifm|arg_value>abc</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        clean_text = (
            "<ifm|tool_call>study_args"
            "<ifm|arg_key>city</ifm|arg_key>"
            "<ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>"
        )

        detector.detect_and_parse(fallback_text, self.tools)
        self.assertTrue(detector.fallback_events)
        detector.detect_and_parse(clean_text, self.tools)

        self.assertEqual(detector.fallback_events, [])

    def test_clear_fallback_events(self):
        detector = K2V3DetectorTracking()
        detector._record_fallback("json_loads_failed_ast_literal_eval", "coercion")

        detector.clear_fallback_events()

        self.assertEqual(detector.fallback_events, [])

    def test_streaming_output_matches_base_detector_without_events(self):
        cases = [
            (
                "xml_boolean_true",
                "xml",
                list(
                    "<ifm|tool_call>study_args"
                    "<ifm|arg_key>flag</ifm|arg_key>"
                    "<ifm|arg_value>True</ifm|arg_value>"
                    "</ifm|tool_call>"
                ),
            ),
            (
                "xml_raw_string_payload",
                "xml",
                list(
                    "<ifm|tool_call>study_args"
                    "<ifm|arg_key>payload</ifm|arg_key>"
                    "<ifm|arg_value>abc</ifm|arg_value>"
                    "</ifm|tool_call>"
                ),
            ),
            (
                "xml_typed_any_whitespace",
                "xml_typed",
                list(
                    "<ifm|tool_call>study_args"
                    "<ifm|arg_key>notes</ifm|arg_key>"
                    "<ifm|arg_type>any</ifm|arg_type>"
                    "<ifm|arg_value>\nkeep me\n</ifm|arg_value>"
                    "</ifm|tool_call>"
                ),
            ),
            (
                "json_streaming",
                "json",
                list(
                    "<ifm|tool_call>"
                    '{"name": "study_args", "arguments": {"city": "Tokyo"}}'
                    "</ifm|tool_call>"
                ),
            ),
            (
                "reasoning_prefix_wrapper",
                "xml",
                list(
                    "<ifm|think>need lookup</ifm|think>\n"
                    "<ifm|tool_calls>\n"
                    "<ifm|tool_call>study_args"
                    "<ifm|arg_key>city</ifm|arg_key>"
                    "<ifm|arg_value>Tokyo</ifm|arg_value>"
                    "</ifm|tool_call>\n"
                    "</ifm|tool_calls>"
                ),
            ),
        ]
        for name, dialect, chunks in cases:
            with self.subTest(name=name, dialect=dialect):
                self._assert_stream_equal(chunks, dialect=dialect)


class TestK2V3XmlStreaming(unittest.TestCase):
    """The IFM xml/xml_typed dialects stream incrementally (name then args)."""

    def setUp(self):
        self.tools = [
            _make_tool(
                "get_weather",
                {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                    },
                },
            )
        ]
        self.block = (
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>"
        )

    def _assert_single_weather(self, normal_text, calls, expected_args):
        self.assertEqual(normal_text, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["parameters"]), expected_args)
        # Streaming must reassemble to the same arguments the non-stream parser
        # produces.
        nonstream = K2V3Detector().detect_and_parse(self.block, self.tools)
        self.assertEqual(
            json.loads(calls[0]["parameters"]),
            json.loads(nonstream.calls[0].parameters),
        )

    def test_whole_block_one_chunk(self):
        normal, calls = _collect_stream(K2V3Detector(), self.tools, [self.block])
        self._assert_single_weather(normal, calls, {"city": "Tokyo"})

    def test_char_by_char(self):
        chunks = list(self.block)  # one character per chunk
        normal, calls = _collect_stream(K2V3Detector(), self.tools, chunks)
        self._assert_single_weather(normal, calls, {"city": "Tokyo"})

    def test_first_param_item_is_name_only(self):
        det = K2V3Detector()
        # Send the name-bearing prefix, then the rest.
        det.parse_streaming_increment("<ifm|tool_call>get_weather", self.tools)
        first = det.parse_streaming_increment(
            "<ifm|arg_key>city</ifm|arg_key>", self.tools
        )
        # The name item is emitted with empty parameters before any args.
        name_items = [c for c in first.calls if c.name]
        self.assertTrue(any(c.name == "get_weather" for c in name_items))
        self.assertTrue(all(c.parameters == "" for c in name_items))

    def test_split_at_awkward_tag_boundaries(self):
        chunks = [
            "<ifm|tool_call>get_wea",
            "ther<ifm|arg_",
            "key>city</ifm|arg_key><ifm|arg_value>To",
            "kyo</ifm|arg_value></ifm|tool_call>",
        ]
        normal, calls = _collect_stream(K2V3Detector(), self.tools, chunks)
        self._assert_single_weather(normal, calls, {"city": "Tokyo"})

    def test_schema_typed_integer(self):
        block = (
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>days</ifm|arg_key><ifm|arg_value>3</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        normal, calls = _collect_stream(K2V3Detector(), self.tools, list(block))
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0]["parameters"]), {"days": 3})

    def test_no_args(self):
        block = "<ifm|tool_call>get_weather</ifm|tool_call>"
        normal, calls = _collect_stream(K2V3Detector(), self.tools, list(block))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["parameters"]), {})

    def test_empty_value_object_typed_stays_valid_json(self):
        # An empty <ifm|arg_value></ifm|arg_value> on a non-string (array) type is
        # malformed input; streaming must still reassemble to parseable JSON.
        tools = [_make_tool("todo", {"type": "object", "properties": {"items": {"type": "array"}}})]
        block = (
            "<ifm|tool_call>todo"
            "<ifm|arg_key>items</ifm|arg_key><ifm|arg_value></ifm|arg_value>"
            "</ifm|tool_call>"
        )
        normal, calls = _collect_stream(K2V3Detector(), tools, list(block))
        self.assertEqual(len(calls), 1)
        # Reassembles to valid JSON (does not produce '{"items": }').
        self.assertEqual(json.loads(calls[0]["parameters"]), {"items": ""})

    def test_multiple_calls_separate_chunks(self):
        chunk_a = (
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        chunk_b = (
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>Osaka</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        normal, calls = _collect_stream(
            K2V3Detector(), self.tools, [chunk_a, chunk_b]
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(json.loads(calls[0]["parameters"]), {"city": "Tokyo"})
        self.assertEqual(json.loads(calls[1]["parameters"]), {"city": "Osaka"})

    def test_multiple_calls_same_chunk(self):
        combined = (
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>"
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>Osaka</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        normal, calls = _collect_stream(K2V3Detector(), self.tools, [combined])
        self.assertEqual(normal, "")
        self.assertEqual(len(calls), 2)
        self.assertEqual(json.loads(calls[0]["parameters"]), {"city": "Tokyo"})
        self.assertEqual(json.loads(calls[1]["parameters"]), {"city": "Osaka"})

    def test_multiple_calls_char_by_char(self):
        combined = (
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>"
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>Osaka</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        normal, calls = _collect_stream(K2V3Detector(), self.tools, list(combined))
        self.assertEqual([c["name"] for c in calls], ["get_weather", "get_weather"])
        self.assertEqual(json.loads(calls[0]["parameters"]), {"city": "Tokyo"})
        self.assertEqual(json.loads(calls[1]["parameters"]), {"city": "Osaka"})

    def test_reasoning_prefix_and_wrapper_stripped(self):
        wire = (
            "<ifm|think>need lookup</ifm|think>\n"
            "<ifm|tool_calls>\n"
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>\n"
            "</ifm|tool_calls>"
        )
        normal, calls = _collect_stream(K2V3Detector(), self.tools, list(wire))
        # No reasoning text or structural tokens leak; only incidental
        # boundary whitespace may pass through when fed character-by-character.
        self.assertEqual(normal.strip(), "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["parameters"]), {"city": "Tokyo"})

    def test_normal_text_before_tool_call_is_emitted(self):
        wire = "Sure, let me check.<ifm|tool_call>get_weather</ifm|tool_call>"
        normal, calls = _collect_stream(K2V3Detector(), self.tools, list(wire))
        self.assertEqual(normal, "Sure, let me check.")
        self.assertEqual(calls[0]["name"], "get_weather")

    def test_whitespace_before_tool_call_is_emitted(self):
        wire = "\n<ifm|tool_call>get_weather</ifm|tool_call>"
        normal, calls = _collect_stream(K2V3Detector(), self.tools, list(wire))
        self.assertEqual(normal, "\n")
        self.assertEqual(calls[0]["name"], "get_weather")

    def test_normal_text_after_tool_call_is_buffered(self):
        det = K2V3Detector()
        first = det.parse_streaming_increment(
            f"Before {self.block}After", self.tools
        )
        self.assertEqual(first.normal_text, "Before ")
        self.assertTrue(any(call.name == "get_weather" for call in first.calls))

        second = det.parse_streaming_increment(" later", self.tools)
        self.assertEqual(second.normal_text, "After later")
        self.assertEqual(second.calls, [])

    def test_normal_text_between_tool_calls_splits_results(self):
        det = K2V3Detector()
        block_b = (
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>Osaka</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        first = det.parse_streaming_increment(
            f"{self.block}Between{block_b}", self.tools
        )
        self.assertEqual(first.normal_text, "")
        self.assertEqual(
            [call.name for call in first.calls if call.name], ["get_weather"]
        )

        second = det.parse_streaming_increment("Tail", self.tools)
        self.assertEqual(second.normal_text, "Between")
        self.assertEqual(
            [call.name for call in second.calls if call.name], ["get_weather"]
        )

        third = det.parse_streaming_increment(" done", self.tools)
        self.assertEqual(third.normal_text, "Tail done")
        self.assertEqual(third.calls, [])


class TestK2V3XmlTypedStreaming(unittest.TestCase):
    """The inline <ifm|arg_type> hint forces a value's type while streaming."""

    def test_inline_string_keeps_numeric_as_string(self):
        det = K2V3Detector(tool_format="xml_typed")
        tools = [_make_tool("study_args")]
        block = (
            "<ifm|tool_call>study_args"
            "<ifm|arg_key>user_id</ifm|arg_key>"
            "<ifm|arg_type>string</ifm|arg_type>"
            "<ifm|arg_value>12345</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        normal, calls = _collect_stream(det, tools, list(block))
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0]["parameters"]), {"user_id": "12345"})
        # Matches the non-stream coercion.
        nonstream = K2V3Detector(tool_format="xml_typed").detect_and_parse(
            block, tools
        )
        self.assertEqual(
            json.loads(calls[0]["parameters"]),
            json.loads(nonstream.calls[0].parameters),
        )

    def test_inline_any_preserves_argument_whitespace(self):
        det = K2V3Detector(tool_format="xml_typed")
        tools = [_make_tool("study_args")]
        block = (
            "<ifm|tool_call>study_args"
            "<ifm|arg_key>notes</ifm|arg_key>"
            "<ifm|arg_type>any</ifm|arg_type>"
            "<ifm|arg_value>\nkeep me\n</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        normal, calls = _collect_stream(det, tools, list(block))
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            json.loads(calls[0]["parameters"]), {"notes": "\nkeep me\n"}
        )


class TestK2V3JsonStreaming(unittest.TestCase):
    """The IFM json dialect streams at tool-call-block granularity."""

    def setUp(self):
        self.tools = [_make_tool("get_weather")]

    def test_single_object_block(self):
        det = K2V3Detector(tool_format="json")
        block = (
            "<ifm|tool_call>"
            '{"name": "get_weather", "arguments": {"city": "Tokyo"}}'
            "</ifm|tool_call>"
        )
        normal, calls = _collect_stream(det, self.tools, list(block))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["parameters"]), {"city": "Tokyo"})

    def test_emits_only_after_block_completes(self):
        det = K2V3Detector(tool_format="json")
        # Partial block: nothing emitted yet.
        partial = det.parse_streaming_increment(
            '<ifm|tool_call>{"name": "get_weather", "argum', self.tools
        )
        self.assertEqual(partial.calls, [])
        # Completing the block emits the call.
        rest = det.parse_streaming_increment(
            'ents": {"city": "Tokyo"}}</ifm|tool_call>', self.tools
        )
        names = [c.name for c in rest.calls if c.name]
        self.assertIn("get_weather", names)

    def test_list_of_tool_calls(self):
        det = K2V3Detector(tool_format="json")
        block = (
            "<ifm|tool_call>"
            '[{"name": "get_weather", "arguments": {"city": "Tokyo"}},'
            ' {"name": "get_weather", "arguments": {"city": "Osaka"}}]'
            "</ifm|tool_call>"
        )
        normal, calls = _collect_stream(det, self.tools, list(block))
        self.assertEqual([c["name"] for c in calls], ["get_weather", "get_weather"])
        self.assertEqual(json.loads(calls[0]["parameters"]), {"city": "Tokyo"})
        self.assertEqual(json.loads(calls[1]["parameters"]), {"city": "Osaka"})


class TestK2V3StreamingRegistry(unittest.TestCase):
    """FunctionCallParser('k2_v3') streams tool calls end-to-end."""

    def test_parse_stream_chunk_streams(self):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        tools = [_make_tool("get_weather")]
        parser = FunctionCallParser(
            tools=tools,
            tool_call_parser="k2_v3",
            chat_template_kwargs={"tool_call_format": "xml"},
        )
        block = (
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        name = None
        params = ""
        for ch in list(block):
            _, calls = parser.parse_stream_chunk(ch)
            for c in calls:
                if c.name:
                    name = c.name
                if c.parameters:
                    params += c.parameters
        self.assertEqual(name, "get_weather")
        self.assertEqual(json.loads(params), {"city": "Tokyo"})


class TestK2V3RegistryWiring(unittest.TestCase):
    """FunctionCallParser resolves 'k2_v3' to K2V3Detector end-to-end."""

    def test_registry_builds_k2v3_detector(self):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        tools = [_make_tool("get_weather")]
        parser = FunctionCallParser(
            tools=tools,
            tool_call_parser="k2_v3",
            chat_template_kwargs={"tool_call_format": "xml"},
        )
        self.assertIsInstance(parser.detector, K2V3Detector)
        self.assertEqual(parser.detector.tool_format, "xml")

    def test_registry_builds_k2v3_tracking_detector(self):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        tools = [_make_tool("get_weather")]
        parser = FunctionCallParser(
            tools=tools,
            tool_call_parser="k2_v3_tracking",
            chat_template_kwargs={"tool_call_format": "xml"},
        )

        self.assertIsInstance(parser.detector, K2V3DetectorTracking)
        self.assertEqual(parser.detector.tool_format, "xml")
        self.assertEqual(parser.detector.fallback_events, [])

    def test_registry_uses_tool_call_format_dialects_from_template(self):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        tools = [_make_tool("get_weather")]
        cases = {
            "json": (
                '<ifm|tool_call>{"name": "get_weather", '
                '"arguments": {"city": "Tokyo"}}</ifm|tool_call>'
            ),
            "xml_typed": (
                "<ifm|tool_call>get_weather"
                "<ifm|arg_key>city</ifm|arg_key>"
                "<ifm|arg_type>string</ifm|arg_type>"
                "<ifm|arg_value>Tokyo</ifm|arg_value>"
                "</ifm|tool_call>"
            ),
        }

        for dialect, wire_output in cases.items():
            with self.subTest(dialect=dialect):
                parser = FunctionCallParser(
                    tools=tools,
                    tool_call_parser="k2_v3",
                    chat_template_kwargs={"tool_call_format": dialect},
                )
                self.assertIsInstance(parser.detector, K2V3Detector)
                self.assertEqual(parser.detector.tool_format, dialect)

                normal_text, calls = parser.parse_non_stream(wire_output)
                self.assertIsNone(normal_text)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0].name, "get_weather")
                self.assertEqual(json.loads(calls[0].parameters), {"city": "Tokyo"})

    def test_registry_rejects_alias_format_kwargs(self):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        tools = [_make_tool("get_weather")]
        for key in ("tool_format", "tool_calling_format"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(
                    ValueError, f"Unsupported argument: {key}"
                ):
                    FunctionCallParser(
                        tools=tools,
                        tool_call_parser="k2_v3",
                        chat_template_kwargs={key: "json"},
                    )

    def test_full_pipeline_non_stream(self):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        tools = [_make_tool("get_weather")]
        parser = FunctionCallParser(tools=tools, tool_call_parser="k2_v3")
        wire_output = (
            "<ifm|think>looking it up</ifm|think>"
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        normal_text, calls = parser.parse_non_stream(wire_output)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(json.loads(calls[0].parameters), {"city": "Tokyo"})
        self.assertIsNone(normal_text)


if __name__ == "__main__":
    unittest.main()
