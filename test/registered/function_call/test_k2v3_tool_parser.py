"""K2-V3 and multi-format tool parser regression tests.

Ported from LLM360/vllm PR #12.
"""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.multi_format_detector import (
    K2V3Detector,
    MultiFormatDetector,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(5, "stage-a-test-cpu")


def _make_tool(name: str, parameters: dict | None = None) -> Tool:
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


TOOLS = [_make_tool("get_weather"), _make_tool("get_time")]
CALL_1 = (
    "<ifm|tool_call>get_weather"
    "<ifm|arg_key>city</ifm|arg_key>"
    "<ifm|arg_value>Tokyo</ifm|arg_value>"
    "</ifm|tool_call>"
)
CALL_2 = (
    "<ifm|tool_call>get_time"
    "<ifm|arg_key>city</ifm|arg_key>"
    "<ifm|arg_value>Seoul</ifm|arg_value>"
    "</ifm|tool_call>"
)
GROUPED_CALL_1 = f"<ifm|tool_calls>{CALL_1}</ifm|tool_calls>"
GROUPED_CALLS = f"<ifm|tool_calls>{CALL_1}{CALL_2}</ifm|tool_calls>"


def _collect_stream(detector, chunks, *, finalize=False):
    normal_text = ""
    calls = []
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, TOOLS)
        normal_text += result.normal_text or ""
        calls.extend(result.calls)
    if finalize:
        result = detector.finalize_streaming(TOOLS)
        normal_text += result.normal_text or ""
        calls.extend(result.calls)
    return normal_text, calls


class TestConstruction(CustomTestCase):
    def test_k2_defaults_to_xml(self):
        detector = K2V3Detector()
        self.assertEqual(detector.tool_format, "xml")
        self.assertIsNone(detector._delegate)

    def test_tool_call_format_selects_dialect(self):
        detector = K2V3Detector(chat_template_kwargs={"tool_call_format": "xml_typed"})
        self.assertEqual(detector.tool_format, "xml_typed")

    def test_legacy_format_kwargs_are_rejected(self):
        for name in ("tool_format", "tool_calling_format"):
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, f"Unsupported argument: {name}"
            ):
                K2V3Detector(chat_template_kwargs={name: "xml"})


class TestMultiFormatStreaming(CustomTestCase):
    def test_streaming_emits_leading_content_and_complete_ifm_call_together(self):
        detector = MultiFormatDetector(tool_format="xml")

        result = detector.parse_streaming_increment("\n" + CALL_1, TOOLS)

        self.assertEqual(result.normal_text, "\n")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Tokyo"})

    def test_streaming_emits_each_complete_ifm_call_once(self):
        detector = MultiFormatDetector(tool_format="xml")

        _, calls = _collect_stream(
            detector,
            ["<ifm|tool_calls>" + CALL_1, CALL_2 + "</ifm|tool_calls>"],
        )

        self.assertEqual([call.name for call in calls], ["get_weather", "get_time"])
        self.assertEqual(
            [json.loads(call.parameters) for call in calls],
            [{"city": "Tokyo"}, {"city": "Seoul"}],
        )
        self.assertEqual([call.tool_index for call in calls], [0, 1])

    def test_streaming_records_structured_arguments_for_finish_processing(self):
        detector = MultiFormatDetector(tool_format="xml")

        detector.parse_streaming_increment(CALL_1, TOOLS)

        self.assertEqual(
            detector.prev_tool_call_arr,
            [{"name": "get_weather", "arguments": {"city": "Tokyo"}}],
        )
        self.assertEqual(detector.streamed_args_for_tool, ['{"city": "Tokyo"}'])

    def test_ifm_streaming_does_not_treat_generic_tool_call_as_marker(self):
        detector = MultiFormatDetector(tool_format="xml")
        content = "Use <tool_call> literally in the documentation."

        normal_text, calls = _collect_stream(detector, [content], finalize=True)

        self.assertEqual(normal_text, content)
        self.assertEqual(calls, [])


class TestK2GroupedToolCalls(CustomTestCase):
    def test_k2_v3_streaming_requires_grouped_ifm_tool_calls(self):
        detector = K2V3Detector()

        normal_text, calls = _collect_stream(detector, [CALL_1], finalize=True)

        self.assertEqual(normal_text, CALL_1)
        self.assertEqual(calls, [])

    def test_k2_v3_nonstreaming_requires_grouped_ifm_tool_calls(self):
        result = K2V3Detector().detect_and_parse(CALL_1, TOOLS)

        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, CALL_1)

    def test_k2_v3_nonstreaming_requires_closed_grouped_ifm_tool_calls(self):
        incomplete_group = "<ifm|tool_calls>" + CALL_1

        result = K2V3Detector().detect_and_parse(incomplete_group, TOOLS)

        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, incomplete_group)

    def test_k2_v3_nonstreaming_invalid_closed_group_preserves_full_content(self):
        wrapped_contents = ("", "<ifm|tool_call>get_weather")
        for wrapped_content in wrapped_contents:
            with self.subTest(wrapped_content=wrapped_content):
                model_output = (
                    "Content prefix. "
                    f"<ifm|tool_calls>{wrapped_content}</ifm|tool_calls>"
                    " Content suffix."
                )

                result = K2V3Detector().detect_and_parse(model_output, TOOLS)

                self.assertEqual(result.calls, [])
                self.assertEqual(result.normal_text, model_output)

    def test_k2_v3_nonstreaming_parses_only_closed_grouped_ifm_tool_calls(self):
        cases = (
            (GROUPED_CALL_1, ["get_weather"]),
            (GROUPED_CALLS, ["get_weather", "get_time"]),
        )
        for grouped_calls, expected_names in cases:
            with self.subTest(expected_names=expected_names):
                result = K2V3Detector().detect_and_parse(grouped_calls, TOOLS)

                self.assertEqual(result.normal_text, "")
                self.assertEqual([call.name for call in result.calls], expected_names)

    def test_multi_format_nonstreaming_keeps_singular_ifm_compatibility(self):
        result = MultiFormatDetector(tool_format="xml").detect_and_parse(CALL_1, TOOLS)

        self.assertEqual([call.name for call in result.calls], ["get_weather"])

    def test_k2_v3_streaming_parses_grouped_ifm_tool_calls(self):
        normal_text, calls = _collect_stream(K2V3Detector(), [GROUPED_CALL_1])

        self.assertEqual(normal_text, "")
        self.assertEqual([call.name for call in calls], ["get_weather"])

    def test_k2_v3_streaming_waits_for_complete_group_before_emitting_calls(self):
        detector = K2V3Detector()
        incomplete_group = "<ifm|tool_calls>" + CALL_1

        result = detector.parse_streaming_increment(incomplete_group, TOOLS)

        self.assertEqual(result.calls, [])
        self.assertTrue(detector.has_pending_streaming_output())
        final_result = detector.finalize_streaming(TOOLS)
        self.assertEqual(final_result.normal_text, incomplete_group)
        self.assertEqual(final_result.calls, [])
        self.assertFalse(detector.has_pending_streaming_output())

    def test_k2_v3_streaming_preserves_multiple_complete_calls(self):
        _, calls = _collect_stream(K2V3Detector(), [GROUPED_CALLS])

        self.assertEqual([call.name for call in calls], ["get_weather", "get_time"])
        self.assertEqual([call.tool_index for call in calls], [0, 1])

    def test_grouped_json_dialect_uses_schema_coercion(self):
        output = (
            "<ifm|tool_calls><ifm|tool_call>"
            '{"name":"get_weather","arguments":{"city":12345}}'
            "</ifm|tool_call></ifm|tool_calls>"
        )

        result = K2V3Detector(tool_format="json").detect_and_parse(output, TOOLS)

        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "12345"})


class TestK2RegressionCoverage(CustomTestCase):
    """Retain parser coverage that predates the grouped-call regressions."""

    @staticmethod
    def _group(*calls: str) -> str:
        return f"<ifm|tool_calls>{''.join(calls)}</ifm|tool_calls>"

    def test_construction_and_unknown_dialect(self):
        self.assertIsInstance(K2V3Detector(), MultiFormatDetector)
        self.assertEqual(K2V3Detector(tool_format="json").tool_format, "json")
        with self.assertRaisesRegex(ValueError, "Unsupported tool_format"):
            K2V3Detector(tool_format="not-a-dialect")

    def test_xml_schema_coercion_no_args_and_unknown_tools(self):
        tools = [
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
        cases = (
            (
                "<ifm|tool_call>get_weather"
                "<ifm|arg_key>city</ifm|arg_key>"
                "<ifm|arg_value>\nTokyo\n</ifm|arg_value>"
                "</ifm|tool_call>",
                {"city": "\nTokyo\n"},
                0,
            ),
            (
                "<ifm|tool_call>get_weather"
                "<ifm|arg_key>days</ifm|arg_key>"
                "<ifm|arg_value>\n3\n</ifm|arg_value>"
                "</ifm|tool_call>",
                {"days": 3},
                0,
            ),
            ("<ifm|tool_call>get_weather</ifm|tool_call>", {}, 0),
            (
                "<ifm|tool_call>not_registered"
                "<ifm|arg_key>city</ifm|arg_key>"
                "<ifm|arg_value>Tokyo</ifm|arg_value>"
                "</ifm|tool_call>",
                {"city": "Tokyo"},
                -1,
            ),
        )

        for call, expected_arguments, expected_index in cases:
            with self.subTest(expected_arguments=expected_arguments):
                result = K2V3Detector().detect_and_parse(self._group(call), tools)
                self.assertEqual(len(result.calls), 1)
                self.assertEqual(
                    json.loads(result.calls[0].parameters), expected_arguments
                )
                self.assertEqual(result.calls[0].tool_index, expected_index)

    def test_xml_typed_inline_coercion(self):
        tools = [_make_tool("study_args")]
        cases = (
            (
                "user_id",
                "string",
                "12345",
                {"user_id": "12345"},
            ),
            (
                "notes",
                "any",
                "\nkeep me\n",
                {"notes": "\nkeep me\n"},
            ),
            (
                "enabled",
                "boolean",
                "\ntrue\n",
                {"enabled": True},
            ),
        )

        for key, arg_type, value, expected_arguments in cases:
            with self.subTest(arg_type=arg_type):
                call = (
                    "<ifm|tool_call>study_args"
                    f"<ifm|arg_key>{key}</ifm|arg_key>"
                    f"<ifm|arg_type>{arg_type}</ifm|arg_type>"
                    f"<ifm|arg_value>{value}</ifm|arg_value>"
                    "</ifm|tool_call>"
                )
                result = K2V3Detector(tool_format="xml_typed").detect_and_parse(
                    self._group(call), tools
                )
                self.assertEqual(
                    json.loads(result.calls[0].parameters), expected_arguments
                )

    def test_json_argument_shapes(self):
        detector = K2V3Detector(tool_format="json")
        cases = (
            (
                '{"name":"get_weather","arguments":{"city":"Tokyo"}}',
                [{"city": "Tokyo"}],
            ),
            (
                '{"name":"get_weather","arguments":"{\\"city\\":\\"Tokyo\\"}"}',
                [{"city": "Tokyo"}],
            ),
            (
                '[{"name":"get_weather","arguments":{"city":"Tokyo"}},'
                '{"name":"get_weather","arguments":{"city":"Osaka"}}]',
                [{"city": "Tokyo"}, {"city": "Osaka"}],
            ),
        )

        for payload, expected_arguments in cases:
            with self.subTest(expected_arguments=expected_arguments):
                call = f"<ifm|tool_call>{payload}</ifm|tool_call>"
                result = detector.detect_and_parse(self._group(call), TOOLS)
                self.assertEqual(
                    [json.loads(item.parameters) for item in result.calls],
                    expected_arguments,
                )

    def test_reasoning_prefix_and_whitespace_handling(self):
        for effort in ("think", "think_fast", "think_faster"):
            with self.subTest(effort=effort):
                output = f"<ifm|{effort}>need lookup</ifm|{effort}>" + GROUPED_CALL_1
                result = K2V3Detector().detect_and_parse(output, TOOLS)
                self.assertEqual(result.normal_text, "")

        output = "<ifm|think>need lookup</ifm|think>\n" + GROUPED_CALL_1
        result = K2V3Detector().detect_and_parse(output, TOOLS)
        self.assertEqual(result.normal_text, "")

        legacy_prefix = "<think>legacy reasoning</think>\n"
        result = K2V3Detector().detect_and_parse(legacy_prefix + GROUPED_CALL_1, TOOLS)
        self.assertEqual(result.normal_text, legacy_prefix)

    def test_grouped_streaming_handles_character_and_awkward_splits(self):
        split_points = (
            list(GROUPED_CALL_1),
            [
                "<ifm|tool_cal",
                "ls><ifm|tool_call>get_wea",
                "ther<ifm|arg_key>city</ifm|arg_key><ifm|arg_value>To",
                "kyo</ifm|arg_value></ifm|tool_call></ifm|tool_calls>",
            ],
        )

        for chunks in split_points:
            with self.subTest(chunk_count=len(chunks)):
                normal_text, calls = _collect_stream(K2V3Detector(), chunks)
                self.assertEqual(normal_text, "")
                self.assertEqual([call.name for call in calls], ["get_weather"])
                self.assertEqual(json.loads(calls[0].parameters), {"city": "Tokyo"})

    def test_grouped_streaming_no_args_duplicate_calls_and_malformed_value(self):
        no_args = "<ifm|tool_call>get_weather</ifm|tool_call>"
        duplicate = CALL_1 + CALL_1.replace("Tokyo", "Osaka")
        array_tools = [
            _make_tool(
                "todo",
                {
                    "type": "object",
                    "properties": {"items": {"type": "array"}},
                },
            )
        ]
        empty_array = (
            "<ifm|tool_call>todo"
            "<ifm|arg_key>items</ifm|arg_key>"
            "<ifm|arg_value></ifm|arg_value>"
            "</ifm|tool_call>"
        )

        _, calls = _collect_stream(K2V3Detector(), [self._group(no_args)])
        self.assertEqual(json.loads(calls[0].parameters), {})

        _, calls = _collect_stream(K2V3Detector(), [self._group(duplicate)])
        self.assertEqual([call.tool_index for call in calls], [0, 1])
        self.assertEqual(
            [json.loads(call.parameters) for call in calls],
            [{"city": "Tokyo"}, {"city": "Osaka"}],
        )

        result = K2V3Detector().detect_and_parse(self._group(empty_array), array_tools)
        self.assertEqual(json.loads(result.calls[0].parameters), {"items": ""})

    def test_grouped_streaming_preserves_prefix_and_schema_coercion(self):
        tools = [
            _make_tool(
                "get_weather",
                {
                    "type": "object",
                    "properties": {"days": {"type": "integer"}},
                },
            )
        ]
        call = (
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>days</ifm|arg_key>"
            "<ifm|arg_value>3</ifm|arg_value>"
            "</ifm|tool_call>"
        )

        normal_text = ""
        calls = []
        detector = K2V3Detector()
        for chunk in "I will check. " + self._group(call):
            result = detector.parse_streaming_increment(chunk, tools)
            normal_text += result.normal_text or ""
            calls.extend(result.calls)

        self.assertEqual(normal_text, "I will check. ")
        self.assertEqual(json.loads(calls[0].parameters), {"days": 3})

    def test_grouped_json_streaming_waits_for_complete_wrapper(self):
        payload = (
            '<ifm|tool_call>{"name":"get_weather",'
            '"arguments":{"city":"Tokyo"}}</ifm|tool_call>'
        )
        grouped = self._group(payload)
        detector = K2V3Detector(tool_format="json")

        partial = detector.parse_streaming_increment(grouped[:-1], TOOLS)
        self.assertEqual(partial.calls, [])
        completed = detector.parse_streaming_increment(grouped[-1], TOOLS)
        self.assertEqual([call.name for call in completed.calls], ["get_weather"])
        self.assertEqual(json.loads(completed.calls[0].parameters), {"city": "Tokyo"})

    def test_function_call_parser_registry_and_streaming(self):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        parser = FunctionCallParser(
            tools=TOOLS,
            tool_call_parser="k2_v3",
            chat_template_kwargs={"tool_call_format": "xml"},
        )
        self.assertIsInstance(parser.detector, K2V3Detector)

        streamed_calls = []
        for chunk in GROUPED_CALL_1:
            _, calls = parser.parse_stream_chunk(chunk)
            streamed_calls.extend(calls)
        self.assertEqual([call.name for call in streamed_calls], ["get_weather"])
        self.assertEqual(json.loads(streamed_calls[0].parameters), {"city": "Tokyo"})

        parser = FunctionCallParser(tools=TOOLS, tool_call_parser="k2_v3")
        normal_text, calls = parser.parse_non_stream(GROUPED_CALL_1)
        self.assertEqual(normal_text, "")
        self.assertEqual([call.name for call in calls], ["get_weather"])

    def test_function_call_parser_dialect_and_alias_validation(self):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        parser = FunctionCallParser(
            tools=TOOLS,
            tool_call_parser="k2_v3",
            chat_template_kwargs={"tool_call_format": "json"},
        )
        self.assertEqual(parser.detector.tool_format, "json")

        for alias in ("tool_format", "tool_calling_format"):
            with self.subTest(alias=alias), self.assertRaisesRegex(
                ValueError, f"Unsupported argument: {alias}"
            ):
                FunctionCallParser(
                    tools=TOOLS,
                    tool_call_parser="k2_v3",
                    chat_template_kwargs={alias: "json"},
                )


if __name__ == "__main__":
    unittest.main()
