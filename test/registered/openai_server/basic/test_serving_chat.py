"""
Unit-tests for OpenAIServingChat — rewritten to use only the std-lib 'unittest'.
Run with either:
    python tests/test_serving_chat_unit.py -v
or
    python -m unittest discover -s tests -p "test_*unit.py" -v
"""

import json
import unittest
import uuid
from http import HTTPStatus
from typing import Optional
from unittest.mock import Mock, patch

from fastapi import Request

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    MessageProcessingResult,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.utils import get_or_create_event_loop
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


class _MockTokenizerManager:
    """Minimal mock that satisfies OpenAIServingChat."""

    def __init__(self):
        self.model_config = Mock(is_multimodal=False)
        self.server_args = Mock(
            enable_cache_report=False,
            tool_call_parser="hermes",
            reasoning_parser=None,
            stream_response_default_include_usage=False,
        )
        # Mock hf_config for _use_dpsk_v32_encoding check
        mock_hf_config = Mock()
        mock_hf_config.architectures = ["LlamaForCausalLM"]
        self.model_config.hf_config = mock_hf_config

        self.chat_template_name: Optional[str] = "llama-3"

        # tokenizer stub
        self.tokenizer = Mock()
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.tokenizer.decode.return_value = "Test response"
        self.tokenizer.chat_template = None
        self.tokenizer.bos_token_id = 1

        # async generator stub for generate_request
        async def _mock_generate():
            yield {
                "text": "Test response",
                "meta_info": {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": [(0.1, 1, "Test"), (0.2, 2, "response")],
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.generate_request = Mock(return_value=_mock_generate())
        self.create_abort_task = Mock()


class _MockTemplateManager:
    """Minimal mock for TemplateManager."""

    def __init__(self):
        self.chat_template_name: Optional[str] = "llama-3"
        self.jinja_template_content_format: Optional[str] = None
        self.completion_template_name: Optional[str] = None


class ServingChatTestCase(unittest.TestCase):
    # ------------- common fixtures -------------
    def setUp(self):
        self.tm = _MockTokenizerManager()
        self.template_manager = _MockTemplateManager()
        self.chat = OpenAIServingChat(self.tm, self.template_manager)

        # frequently reused requests
        self.basic_req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            temperature=0.7,
            max_tokens=100,
            stream=False,
        )
        self.stream_req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            temperature=0.7,
            max_tokens=100,
            stream=True,
        )

        self.fastapi_request = Mock(spec=Request)
        self.fastapi_request.headers = {}

    # ------------- conversion tests -------------
    def test_convert_to_internal_request_single(self):
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock, patch.object(self.chat, "_process_messages") as proc_mock:
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_ins.image_data = conv_ins.audio_data = None
            conv_ins.modalities = []
            conv_ins.stop_str = ["</s>"]
            conv_mock.return_value = conv_ins

            proc_mock.return_value = MessageProcessingResult(
                "Test prompt",
                [1, 2, 3],
                None,
                None,
                [],
                ["</s>"],
                None,
            )

            adapted, processed = self.chat._convert_to_internal_request(self.basic_req)
            self.assertIsInstance(adapted, GenerateReqInput)
            self.assertFalse(adapted.stream)
            self.assertEqual(processed, self.basic_req)

    def test_jinja_uses_openai_tool_schema_first(self):
        """Ensure Jinja chat templates receive OpenAI-shaped tools by default."""
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "description": "Add two numbers.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
        )

        self.chat._process_messages(req, is_multimodal=False)

        expected_tools = [tool.model_dump() for tool in req.tools]
        kwargs = self.tm.tokenizer.apply_chat_template.call_args.kwargs
        self.assertEqual(kwargs["tools"], expected_tools)

    def test_jinja_tool_schema_fallback_to_flat_function(self):
        """Fallback to function-only schema when template rejects OpenAI wrapper."""
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "description": "Add two numbers.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
        )

        self.tm.tokenizer.apply_chat_template.side_effect = [
            RuntimeError("template expects flat tools format"),
            [1, 2, 3],
        ]

        self.chat._process_messages(req, is_multimodal=False)

        first_tools = self.tm.tokenizer.apply_chat_template.call_args_list[0].kwargs[
            "tools"
        ]
        second_tools = self.tm.tokenizer.apply_chat_template.call_args_list[1].kwargs[
            "tools"
        ]
        self.assertEqual(first_tools, [tool.model_dump() for tool in req.tools])
        self.assertEqual(
            second_tools, [tool.function.model_dump() for tool in req.tools]
        )

    def test_k2v3_combined_preserves_reasoning_and_content_newlines(self):
        self.chat.reasoning_parser = "k2_v3"
        self.chat.tool_call_parser = "k2_v3"
        self.template_manager.force_reasoning = False
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Weather?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "date": {"type": "string"},
                            },
                        },
                    },
                }
            ],
        )
        ret = [
            {
                "text": (
                    "<ifm|think>\n</ifm|think>\n"
                    "<ifm|tool_calls>\n"
                    "<ifm|tool_call>get_weather"
                    "<ifm|arg_key>location</ifm|arg_key>"
                    "<ifm|arg_value>Boston, MA</ifm|arg_value>"
                    "<ifm|arg_key>date</ifm|arg_key>"
                    "<ifm|arg_value>tomorrow</ifm|arg_value>"
                    "</ifm|tool_call>\n"
                    "</ifm|tool_calls>"
                ),
                "meta_info": {
                    "id": "chatcmpl-test",
                    "prompt_tokens": 5,
                    "completion_tokens": 9,
                    "reasoning_tokens": 0,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "weight_version": "test",
                },
                "index": 0,
            }
        ]

        response = self.chat._build_chat_response(req, ret, created=0)

        choice = response.choices[0]
        self.assertEqual(choice.message.reasoning_content, "\n")
        self.assertEqual(choice.message.content, "\n")
        self.assertEqual(choice.finish_reason, "tool_calls")
        self.assertEqual(choice.message.tool_calls[0].function.name, "get_weather")
        self.assertEqual(
            json.loads(choice.message.tool_calls[0].function.arguments),
            {"location": "Boston, MA", "date": "tomorrow"},
        )

    def test_k2v3_combined_stream_preserves_content_newline(self):
        self.chat.reasoning_parser = "k2_v3"
        self.chat.tool_call_parser = "k2_v3"
        self.template_manager.force_reasoning = False
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Weather?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "date": {"type": "string"},
                            },
                        },
                    },
                }
            ],
            stream=True,
        )
        raw_delta = (
            "<ifm|think>\n</ifm|think>\n"
            "<ifm|tool_calls>\n"
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>location</ifm|arg_key>"
            "<ifm|arg_value>Boston, MA</ifm|arg_value>"
            "<ifm|arg_key>date</ifm|arg_key>"
            "<ifm|arg_value>tomorrow</ifm|arg_value>"
            "</ifm|tool_call>\n"
            "</ifm|tool_calls>"
        )
        content = {"meta_info": {"id": "chatcmpl-test"}}
        reasoning_text, tool_delta = self.chat._process_reasoning_stream(
            0, raw_delta, {}, content, req
        )
        self.assertEqual(reasoning_text, "\n")
        self.assertTrue(tool_delta.startswith("\n<ifm|tool_calls>"))

        async def collect_chunks():
            chunks = []
            gen = self.chat._process_tool_call_stream(
                index=0,
                delta=tool_delta,
                parser_dict={},
                content=content,
                request=req,
                has_tool_calls={},
            )
            async for emitted in gen:
                chunks.append(emitted)
            return chunks

        loop = get_or_create_event_loop()
        chunks = loop.run_until_complete(collect_chunks())
        payloads = [json.loads(chunk[len("data: ") :]) for chunk in chunks]

        first_delta = payloads[0]["choices"][0]["delta"]
        self.assertEqual(first_delta.get("content"), "\n")
        tool_deltas = [
            payload["choices"][0]["delta"]["tool_calls"]
            for payload in payloads
            if payload["choices"][0]["delta"].get("tool_calls")
        ]
        self.assertTrue(tool_deltas)
        self.assertEqual(tool_deltas[0][0]["function"]["name"], "get_weather")

    def test_k2v3_nonstreaming_requires_a_valid_closed_plural_wrapper(self):
        self.chat.tool_call_parser = "k2_v3"
        tools = ChatCompletionRequest(
            model="x",
            messages=[],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        ).tools
        singular = "<ifm|tool_call>get_weather</ifm|tool_call>"
        invalid_group = (
            "Content <ifm|tool_calls><ifm|tool_call>get_weather"
            "</ifm|tool_calls> suffix"
        )
        for output in (singular, invalid_group):
            with self.subTest(output=output):
                result = self.chat._process_tool_calls(
                    output,
                    tools,
                    {"type": "stop", "matched": None},
                    tool_choice="auto",
                    chat_template_kwargs={"tool_call_format": "xml"},
                )
                self.assertFalse(result.tool_calls)
                self.assertEqual(result.remaining_text, output)
                self.assertEqual(result.finish_reason["type"], "stop")

        grouped = f"<ifm|tool_calls>{singular}</ifm|tool_calls>"
        result = self.chat._process_tool_calls(
            grouped,
            tools,
            {"type": "stop", "matched": None},
            tool_choice="auto",
            chat_template_kwargs={"tool_call_format": "xml"},
        )
        self.assertEqual(result.remaining_text, "")
        self.assertEqual(result.finish_reason["type"], "tool_calls")
        self.assertEqual(result.tool_calls[0].function.name, "get_weather")

        result = self.chat._process_tool_calls(
            grouped,
            tools,
            {"type": "length", "matched": None},
            tool_choice="auto",
            chat_template_kwargs={"tool_call_format": "xml"},
        )
        self.assertEqual(result.remaining_text, "")
        self.assertEqual(result.finish_reason["type"], "tool_calls")
        self.assertEqual(result.tool_calls[0].function.name, "get_weather")

    def test_k2v3_streaming_reasoning_and_tool_boundary_matrix(self):
        """Port the serving-path matrix from LLM360/vllm PR #12."""
        tool_call = (
            "<ifm|tool_calls>"
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key>"
            "<ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>"
            "</ifm|tool_calls>"
        )
        singular_tool_call = tool_call.removeprefix("<ifm|tool_calls>").removesuffix(
            "</ifm|tool_calls>"
        )
        cases = [
            (
                "same_delta",
                [(f"</ifm|think>\n{tool_call}", "stop")],
                "\n",
                [""],
                "tool_calls",
            ),
            (
                "split_deltas",
                [("</ifm|think>", None), (f"\n{tool_call}", "stop")],
                "\n",
                [""],
                "tool_calls",
            ),
            (
                "missing_close_stop",
                [(tool_call, "stop")],
                "",
                [""],
                "tool_calls",
            ),
            (
                "missing_close_length",
                [(tool_call, "length")],
                "",
                [""],
                "tool_calls",
            ),
            (
                "missing_close_reasoning_and_tool",
                [(f"Need lookup. {tool_call}", "stop")],
                "",
                ["Need lookup. "],
                "tool_calls",
            ),
            (
                "plain_missing_close_stop",
                [("Plain answer ", None), ("without close.", "stop")],
                "Plain answer without close.",
                [""],
                "stop",
            ),
            (
                "plain_missing_close_length",
                [("Plain answer ", None), ("without close.", "length")],
                "Plain answer without close.",
                [""],
                "length",
            ),
            (
                "missing_close_incomplete_length",
                [
                    (
                        "Need lookup. <ifm|tool_calls>" "<ifm|tool_call>get_weather",
                        "length",
                    )
                ],
                "<ifm|tool_calls><ifm|tool_call>get_weather",
                ["Need lookup. "],
                "length",
            ),
            (
                "explicit_close_incomplete_tool_is_content",
                [
                    (
                        "Need lookup.</ifm|think><ifm|tool_calls>"
                        "<ifm|tool_call>get_weather",
                        "stop",
                    )
                ],
                "<ifm|tool_calls><ifm|tool_call>get_weather",
                ["Need lookup."],
                "stop",
            ),
            (
                "delayed_explicit_close",
                [
                    (f"Maybe {tool_call}", None),
                    (f" reconsider</ifm|think>{tool_call}", "stop"),
                ],
                "",
                [f"Maybe {tool_call} reconsider"],
                "tool_calls",
            ),
            (
                "multiple_close_tokens",
                [("Need lookup.</ifm|think></ifm|think>Answer", "stop")],
                "</ifm|think>Answer",
                ["Need lookup."],
                "stop",
            ),
            (
                "singular_tool_tag_remains_content",
                [(f"Need lookup. {singular_tool_call}", "stop")],
                f"Need lookup. {singular_tool_call}",
                [""],
                "stop",
            ),
        ]

        for (
            name,
            model_deltas,
            expected_content,
            expected_reasoning,
            expected_finish,
        ) in cases:
            with self.subTest(name=name):
                self.chat.reasoning_parser = "k2_v3"
                self.chat.tool_call_parser = "k2_v3"
                self.template_manager.force_reasoning = False
                request = ChatCompletionRequest(
                    model="x",
                    messages=[{"role": "user", "content": "Weather?"}],
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}},
                                },
                            },
                        }
                    ],
                    tool_choice="auto",
                    stream=True,
                    chat_template_kwargs={"tool_call_format": "xml"},
                )

                async def generate():
                    cumulative_text = ""
                    for delta_text, finish_type in model_deltas:
                        cumulative_text += delta_text
                        yield {
                            "text": cumulative_text,
                            "meta_info": {
                                "id": "chatcmpl-test",
                                "prompt_tokens": 1,
                                "completion_tokens": len(cumulative_text),
                                "reasoning_tokens": 0,
                                "cached_tokens": 0,
                                "finish_reason": (
                                    {"type": finish_type, "matched": None}
                                    if finish_type
                                    else None
                                ),
                            },
                            "index": 0,
                        }

                self.tm.generate_request.return_value = generate()

                async def collect_chunks():
                    return [
                        chunk
                        async for chunk in self.chat._generate_chat_stream(
                            GenerateReqInput(input_ids=[1], stream=True),
                            request,
                            self.fastapi_request,
                        )
                    ]

                chunks = get_or_create_event_loop().run_until_complete(collect_chunks())
                payloads = [
                    json.loads(chunk[len("data: ") :])
                    for chunk in chunks
                    if chunk.startswith("data: {")
                ]
                choices = [
                    payload["choices"][0] for payload in payloads if payload["choices"]
                ]
                reasoning = [
                    choice["delta"]["reasoning_content"]
                    for choice in choices
                    if choice["delta"].get("reasoning_content") is not None
                ]
                content = "".join(
                    choice["delta"].get("content") or "" for choice in choices
                )
                finish = next(
                    choice["finish_reason"]
                    for choice in choices
                    if choice["finish_reason"] is not None
                )
                tool_calls = [
                    tool
                    for choice in choices
                    for tool in choice["delta"].get("tool_calls") or []
                ]

                self.assertEqual(reasoning, expected_reasoning)
                self.assertEqual(content, expected_content)
                self.assertEqual(finish, expected_finish)
                if expected_finish == "tool_calls":
                    self.assertEqual(len(tool_calls), 1)
                    self.assertEqual(tool_calls[0]["function"]["name"], "get_weather")
                    self.assertEqual(
                        json.loads(tool_calls[0]["function"]["arguments"]),
                        {"city": "Tokyo"},
                    )
                else:
                    self.assertEqual(tool_calls, [])

    def test_k2v3_streaming_abort_does_not_finalize_held_output(self):
        self.chat.reasoning_parser = "k2_v3"
        self.chat.tool_call_parser = "k2_v3"
        self.template_manager.force_reasoning = False
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Weather?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            stream=True,
            chat_template_kwargs={"tool_call_format": "xml"},
        )

        async def generate():
            yield {
                "text": "<ifm|tool_calls><ifm|tool_call>get_weather",
                "meta_info": {
                    "id": "chatcmpl-test",
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "reasoning_tokens": 0,
                    "cached_tokens": 0,
                    "finish_reason": {
                        "type": "abort",
                        "status_code": HTTPStatus.INTERNAL_SERVER_ERROR,
                        "message": "aborted",
                    },
                },
                "index": 0,
            }

        self.tm.generate_request.return_value = generate()

        async def collect_chunks():
            return [
                chunk
                async for chunk in self.chat._generate_chat_stream(
                    GenerateReqInput(input_ids=[1], stream=True),
                    request,
                    self.fastapi_request,
                )
            ]

        chunks = get_or_create_event_loop().run_until_complete(collect_chunks())
        payloads = [
            json.loads(chunk[len("data: ") :])
            for chunk in chunks
            if chunk.startswith("data: {")
        ]
        self.assertFalse(
            any(
                choice["delta"].get("reasoning_content")
                or choice["delta"].get("content")
                or choice["delta"].get("tool_calls")
                for payload in payloads
                for choice in payload.get("choices", [])
            )
        )

    def test_k2v3_forced_tool_streaming_drains_terminal_json(self):
        """Required and named tools must emit complete arguments at the boundary."""
        forced_output_cases = [
            (
                "required_same_delta",
                "required",
                [
                    (
                        "</ifm|think>",
                        '[{"name":"get_weather","parameters":{"city":"Tokyo"}}]',
                        "stop",
                    )
                ],
                {"city": "Tokyo"},
            ),
            (
                "required_split_deltas",
                "required",
                [
                    ("</ifm|think>", "", None),
                    (
                        "",
                        '[{"name":"get_weather","parameters":{"city":"Tokyo"}}]',
                        "stop",
                    ),
                ],
                {"city": "Tokyo"},
            ),
            (
                "named_same_delta",
                {"type": "function", "function": {"name": "get_weather"}},
                [
                    (
                        "</ifm|think>",
                        '[{"name":"get_weather","parameters":{"city":"Tokyo"}}]',
                        "stop",
                    )
                ],
                {"city": "Tokyo"},
            ),
            (
                "named_split_deltas",
                {"type": "function", "function": {"name": "get_weather"}},
                [
                    ("</ifm|think>", "", None),
                    (
                        "",
                        '[{"name":"get_weather","parameters":{"city":"Tokyo"}}]',
                        "stop",
                    ),
                ],
                {"city": "Tokyo"},
            ),
            (
                "required_empty_arguments",
                "required",
                [("</ifm|think>", '[{"name":"get_weather","parameters":{}}]', "stop")],
                {},
            ),
        ]

        for name, tool_choice, model_deltas, expected_arguments in forced_output_cases:
            with self.subTest(name=name):
                self.chat.reasoning_parser = "k2_v3"
                self.chat.tool_call_parser = "k2_v3"
                self.template_manager.force_reasoning = False
                request = ChatCompletionRequest(
                    model="x",
                    messages=[{"role": "user", "content": "Weather?"}],
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}},
                                },
                            },
                        }
                    ],
                    tool_choice=tool_choice,
                    stream=True,
                )

                async def generate():
                    cumulative_text = ""
                    for reasoning_delta, tool_delta, finish_type in model_deltas:
                        cumulative_text += reasoning_delta + tool_delta
                        yield {
                            "text": cumulative_text,
                            "meta_info": {
                                "id": "chatcmpl-test",
                                "prompt_tokens": 1,
                                "completion_tokens": len(cumulative_text),
                                "reasoning_tokens": 0,
                                "cached_tokens": 0,
                                "finish_reason": (
                                    {"type": finish_type, "matched": None}
                                    if finish_type
                                    else None
                                ),
                            },
                            "index": 0,
                        }

                self.tm.generate_request.return_value = generate()

                async def collect_chunks():
                    return [
                        chunk
                        async for chunk in self.chat._generate_chat_stream(
                            GenerateReqInput(input_ids=[1], stream=True),
                            request,
                            self.fastapi_request,
                        )
                    ]

                chunks = get_or_create_event_loop().run_until_complete(collect_chunks())
                payloads = [
                    json.loads(chunk[len("data: ") :])
                    for chunk in chunks
                    if chunk.startswith("data: {")
                ]
                choices = [
                    payload["choices"][0] for payload in payloads if payload["choices"]
                ]

                streamed_calls = {}
                for choice in choices:
                    for tool_call in choice["delta"].get("tool_calls") or []:
                        call = streamed_calls.setdefault(
                            tool_call["index"], {"name": None, "arguments": ""}
                        )
                        function = tool_call["function"]
                        if function.get("name") is not None:
                            call["name"] = function["name"]
                        call["arguments"] += function.get("arguments") or ""

                self.assertEqual(list(streamed_calls), [0])
                self.assertEqual(streamed_calls[0]["name"], "get_weather")
                self.assertEqual(
                    json.loads(streamed_calls[0]["arguments"]), expected_arguments
                )
                self.assertEqual(
                    next(
                        choice["finish_reason"]
                        for choice in choices
                        if choice["finish_reason"] is not None
                    ),
                    "tool_calls",
                )

    def test_stop_str_isolation_between_requests(self):
        """Test that stop strings from one request don't affect subsequent requests.

        This tests the fix for the bug where conv.stop_str was being mutated globally,
        causing stop strings from one request to persist in subsequent requests.
        """
        # Mock conversation template with initial stop_str
        initial_stop_str = ["\n"]

        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock:
            # Create a mock conversation object that will be returned by generate_chat_conv
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_ins.image_data = None
            conv_ins.audio_data = None
            conv_ins.modalities = []
            conv_ins.stop_str = (
                initial_stop_str.copy()
            )  # Template's default stop strings
            conv_mock.return_value = conv_ins

            # First request with additional stop string
            req1 = ChatCompletionRequest(
                model="x",
                messages=[{"role": "user", "content": "First request"}],
                stop=["CUSTOM_STOP"],
            )

            # Call the actual _apply_conversation_template method (not mocked)
            result1 = self.chat._apply_conversation_template(req1, is_multimodal=False)

            # Verify first request has both stop strings
            expected_stop1 = initial_stop_str + ["CUSTOM_STOP"]
            self.assertEqual(result1.stop, expected_stop1)

            # Verify the original template's stop_str wasn't mutated after first request
            self.assertEqual(conv_ins.stop_str, initial_stop_str)

            # Second request without additional stop string
            req2 = ChatCompletionRequest(
                model="x",
                messages=[{"role": "user", "content": "Second request"}],
                # No custom stop strings
            )
            result2 = self.chat._apply_conversation_template(req2, is_multimodal=False)

            # Verify second request only has original stop strings (no CUSTOM_STOP from req1)
            self.assertEqual(result2.stop, initial_stop_str)
            self.assertNotIn("CUSTOM_STOP", result2.stop)
            self.assertEqual(conv_ins.stop_str, initial_stop_str)

    def test_unstreamed_tool_args_completion(self):
        """Test that remaining tool call arguments are sent when generation finishes."""

        # Mock FunctionCallParser with detector that has partial tool call data
        mock_parser = Mock()
        mock_detector = Mock()

        # Simulate a tool call that was partially streamed
        mock_detector.prev_tool_call_arr = [
            {
                "name": "get_weather",
                "arguments": {"location": "San Francisco", "unit": "celsius"},
            }
        ]
        mock_detector.streamed_args_for_tool = [
            '{"location": "San Francisco"'  # Partial arguments streamed so far
        ]
        mock_parser.detector = mock_detector

        content = {
            "meta_info": {
                "id": "chatcmpl-test123",
            }
        }

        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        # Test the completion method
        result = self.chat._check_for_unstreamed_tool_args(
            parser=mock_parser,
            content=content,
            request=request,
            index=0,
        )

        # Should return a chunk with remaining arguments
        self.assertIsNotNone(result, "Should return chunk with remaining arguments")

        # Parse the result to verify content
        self.assertTrue(result.startswith("data: "))
        chunk = json.loads(result[6:])
        tool_calls = chunk["choices"][0]["delta"]["tool_calls"]
        self.assertEqual(len(tool_calls), 1)
        arguments = tool_calls[0]["function"]["arguments"]
        self.assertIn(', "unit": "celsius"}', arguments)

        self.assertIn(
            '"finish_reason":null',
            result,
            "Should not include finish_reason in completion chunk",
        )

    def test_unstreamed_tool_args_no_completion_needed(self):
        """Test that no completion chunk is sent when all arguments were already streamed."""

        # Mock FunctionCallParser with detector that has complete tool call data
        mock_parser = Mock()
        mock_detector = Mock()

        # Simulate a tool call that was completely streamed
        mock_detector.prev_tool_call_arr = [
            {"name": "get_weather", "arguments": {"location": "San Francisco"}}
        ]
        mock_detector.streamed_args_for_tool = [
            '{"location": "San Francisco"}'  # All arguments already streamed
        ]
        mock_parser.detector = mock_detector

        content = {
            "meta_info": {
                "id": "chatcmpl-test123",
            }
        }

        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        # Test the completion method
        result = self.chat._check_for_unstreamed_tool_args(
            parser=mock_parser,
            content=content,
            request=request,
            index=0,
        )

        # Should return None since no completion is needed
        self.assertIsNone(result, "Should return None when no completion is needed")

    def test_unstreamed_tool_args_no_parser_data(self):
        """Test that no completion chunk is sent when parser has no tool call data."""

        # Mock FunctionCallParser with empty detector
        mock_parser = Mock()
        mock_detector = Mock()
        mock_detector.prev_tool_call_arr = []
        mock_detector.streamed_args_for_tool = []
        mock_parser.detector = mock_detector

        content = {
            "meta_info": {
                "id": "chatcmpl-test123",
            }
        }

        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        # Test the completion method
        result = self.chat._check_for_unstreamed_tool_args(
            parser=mock_parser,
            content=content,
            request=request,
            index=0,
        )

        # Should return None since there's no parser data
        self.assertIsNone(
            result, "Should return None when parser has no tool call data"
        )

    # ------------- kimi_k2 tool_call_id formatting -------------
    def test_kimi_k2_non_streaming_tool_call_id_format(self):
        """Ensure non-streaming tool_call.id matches functions.{name}:{index} for kimi_k2 parser."""

        # Force kimi_k2 parser
        self.chat.tool_call_parser = "kimi_k2"

        # Mock FunctionCallParser.parse_non_stream to return one tool call
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.FunctionCallParser"
        ) as ParserMock:
            parser_instance = ParserMock.return_value

            # Build a mock ToolCallItem-like object
            call_info = Mock()
            call_info.name = "get_weather"
            call_info.parameters = '{"city":"Paris"}'
            call_info.tool_index = 0

            parser_instance.has_tool_call.return_value = True
            parser_instance.parse_non_stream.return_value = ("", [call_info])

            finish_reason = {"type": "stop", "matched": None}
            tools = [
                {"type": "function", "function": {"name": "get_weather"}},
            ]

            tool_calls, remaining_text, finish_reason = self.chat._process_tool_calls(
                text="<|tool_calls_section_begin|>...",
                tools=tools,
                finish_reason=finish_reason,
            )

            self.assertIsNotNone(tool_calls)
            self.assertEqual(len(tool_calls), 1)
            self.assertEqual(tool_calls[0].id, "functions.get_weather:0")
            self.assertEqual(tool_calls[0].function.name, "get_weather")

    def test_kimi_k2_streaming_tool_call_id_format(self):
        """Ensure streaming first chunk tool_call.id matches functions.{name}:{index} for kimi_k2 parser."""

        # Force kimi_k2 parser
        self.chat.tool_call_parser = "kimi_k2"

        # Prepare request with tools
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            stream=True,
        )

        # Patch FunctionCallParser used inside _process_tool_call_stream
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.FunctionCallParser"
        ) as ParserMock:
            parser_instance = ParserMock.return_value

            # First call returns one ToolCallItem-like chunk (with name)
            first_chunk_call = Mock()
            first_chunk_call.tool_index = 0
            first_chunk_call.name = "get_weather"
            first_chunk_call.parameters = ""
            parser_instance.parse_stream_chunk.side_effect = [
                ("", [first_chunk_call]),
                ("", []),
            ]

            async def collect_first_tool_chunk():
                gen = self.chat._process_tool_call_stream(
                    index=0,
                    delta="irrelevant",
                    parser_dict={},
                    content={"meta_info": {"id": "chatcmpl-test"}},
                    request=req,
                    has_tool_calls={},
                )
                # Get first yielded SSE line
                line = None
                async for emitted in gen:
                    line = emitted
                    break
                return line

            loop = get_or_create_event_loop()
            line = loop.run_until_complete(collect_first_tool_chunk())
            self.assertIsNotNone(line)
            self.assertTrue(line.startswith("data: "))

            payload = json.loads(line[len("data: ") :])
            tool_calls = payload["choices"][0]["delta"]["tool_calls"]
            self.assertEqual(tool_calls[0]["id"], "functions.get_weather:0")

    def test_kimi_k2_non_streaming_tool_call_id_with_history(self):
        """Ensure non-streaming tool_call.id increase with tool calls history for kimi_k2 parser."""

        # Force kimi_k2 parser
        self.chat.tool_call_parser = "kimi_k2"

        # Prepare request with tool calls history
        req = ChatCompletionRequest(
            model="x",
            messages=[
                {"role": "user", "content": "What's the weather today in paris?"},
                {
                    "role": "assistant",
                    "content": "Let me do some search first.",
                    "tool_calls": [
                        {
                            "id": "functions.get_weather:0",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "It's rainy in paris now.",
                    "tool_call_id": "functions.get_weather:0",
                },
                {
                    "role": "assistant",
                    "content": "It's rainy now.",
                },
                {
                    "role": "user",
                    "content": "What about LA and Tokyo?",
                },
            ],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            stream=False,
        )

        # Mock FunctionCallParser.parse_non_stream to return one tool call
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.FunctionCallParser"
        ) as ParserMock:
            parser_instance = ParserMock.return_value

            # Build a mock ToolCallItem-like object
            call_info = Mock()
            call_info.name = "get_weather"
            call_info.parameters = '{"city":"Loa Angeles"}'
            # Kimi-K2 series models might generate fixed number tool_indx,
            # ignoring the tool calls history and mess up all the following tool calls
            call_info.tool_index = 0

            call_info2 = Mock()
            call_info2.name = "get_weather"
            call_info2.parameters = '{"city":"Tokyo"}'
            call_info2.tool_index = 1

            parser_instance.has_tool_call.return_value = True
            parser_instance.parse_non_stream.return_value = (
                "",
                [call_info, call_info2],
            )

            finish_reason = {"type": "stop", "matched": None}
            tools = [
                {"type": "function", "function": {"name": "get_weather"}},
            ]

            history_tool_calls_cnt = self.chat._get_history_tool_calls_cnt(req)
            tool_calls, remaining_text, _ = self.chat._process_tool_calls(
                text="<|tool_calls_section_begin|>...",
                tools=tools,
                finish_reason=finish_reason,
                history_tool_calls_cnt=history_tool_calls_cnt,
            )

            self.assertEqual(history_tool_calls_cnt, 1)
            self.assertIsNotNone(tool_calls)
            self.assertEqual(len(tool_calls), 2)
            self.assertEqual(tool_calls[0].id, "functions.get_weather:1")
            self.assertEqual(tool_calls[0].function.name, "get_weather")
            self.assertEqual(tool_calls[1].id, "functions.get_weather:2")
            self.assertEqual(tool_calls[1].function.name, "get_weather")

    def test_kimi_k2_streaming_tool_call_id_with_history(self):
        """Ensure streaming first chunk tool_call.id increase with tool calls history for kimi_k2 parser."""

        # Force kimi_k2 parser
        self.chat.tool_call_parser = "kimi_k2"

        # Prepare request with tool calls history
        req = ChatCompletionRequest(
            model="x",
            messages=[
                {"role": "user", "content": "What's the weather today in paris?"},
                {
                    "role": "assistant",
                    "content": "Let me do some search first.",
                    "tool_calls": [
                        {
                            "id": "functions.get_weather:0",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "It's rainy in paris now.",
                    "tool_call_id": "functions.get_weather:0",
                },
                {
                    "role": "assistant",
                    "content": "It's rainy now.",
                },
                {
                    "role": "user",
                    "content": "What about LA?",
                },
            ],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            stream=True,
        )

        # Patch FunctionCallParser used inside _process_tool_call_stream
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.FunctionCallParser"
        ) as ParserMock:
            parser_instance = ParserMock.return_value

            # First call returns one ToolCallItem-like chunk (with name)
            first_chunk_call = Mock()
            # Kimi-K2 series models might generate fixed number tool_indx,
            # ignoring the tool calls history and mess up all the following tool calls
            first_chunk_call.tool_index = 0
            first_chunk_call.name = "get_weather"
            first_chunk_call.parameters = ""
            parser_instance.parse_stream_chunk.side_effect = [
                ("", [first_chunk_call]),
                ("", []),
            ]

            async def collect_first_tool_chunk():
                gen = self.chat._process_tool_call_stream(
                    index=0,
                    delta="irrelevant",
                    parser_dict={},
                    content={"meta_info": {"id": "chatcmpl-test"}},
                    request=req,
                    has_tool_calls={},
                )
                # Get first yielded SSE line
                line = None
                async for emitted in gen:
                    line = emitted
                    break
                return line

            loop = get_or_create_event_loop()
            line = loop.run_until_complete(collect_first_tool_chunk())
            self.assertIsNotNone(line)
            self.assertTrue(line.startswith("data: "))

            payload = json.loads(line[len("data: ") :])
            tool_calls = payload["choices"][0]["delta"]["tool_calls"]
            self.assertEqual(tool_calls[0]["id"], "functions.get_weather:1")

    def test_dpsk_v32_encoding_path(self):
        """Test DeepSeek V3.2 encoding path detection and application."""
        from sglang.srt.managers.template_manager import TemplateManager
        from sglang.srt.server_args import PortArgs, ServerArgs

        server_args = ServerArgs(model_path="deepseek-ai/DeepSeek-V3.2")
        port_args = PortArgs.init_new(server_args)

        # Use mocks for TokenizerManager components to avoid full initialization
        with patch(
            "sglang.srt.managers.tokenizer_manager.TokenizerManager"
        ) as MockTokenizerManager:
            tokenizer_manager = MockTokenizerManager(server_args, port_args)
            tokenizer_manager.server_args = server_args
            tokenizer_manager.model_config = Mock()
            tokenizer_manager.model_config.get_default_sampling_params.return_value = (
                None
            )

            # Mock hf_config
            mock_hf_config = Mock()
            mock_hf_config.architectures = ["DeepseekV32ForCausalLM"]

            tokenizer_manager.model_config.hf_config = mock_hf_config

            # Case 1: No chat template in tokenizer -> should use dpsk encoding
            tokenizer_manager.tokenizer = Mock()
            tokenizer_manager.tokenizer.chat_template = None

            serving_chat = OpenAIServingChat(tokenizer_manager, TemplateManager())
            self.assertTrue(serving_chat.use_dpsk_v32_encoding)

            # Case 2: Chat template exists -> should NOT use dpsk encoding
            tokenizer_manager.tokenizer.chat_template = "some template"
            serving_chat = OpenAIServingChat(tokenizer_manager, TemplateManager())
            self.assertFalse(serving_chat.use_dpsk_v32_encoding)

            # Case 3: Not DeepSeek V3.2 architecture -> should NOT use dpsk encoding
            tokenizer_manager.tokenizer.chat_template = None
            mock_hf_config.architectures = ["LlamaForCausalLM"]
            serving_chat = OpenAIServingChat(tokenizer_manager, TemplateManager())
            self.assertFalse(serving_chat.use_dpsk_v32_encoding)

    def test_streaming_abort_yields_error(self):
        """Test that an abort finish reason during streaming correctly yields an error and stops."""
        err_msg = "Aborted by scheduler"
        err_code = HTTPStatus.INTERNAL_SERVER_ERROR

        async def _mock_generate_abort():
            yield {
                "text": "Partial ",
                "meta_info": {
                    "id": "chatcmpl-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "cached_tokens": 0,
                    "finish_reason": {
                        "type": "abort",
                        "status_code": err_code,
                        "message": err_msg,
                    },
                    "output_token_logprobs": None,
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.tm.generate_request.return_value = _mock_generate_abort()

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            temperature=0.7,
            max_tokens=100,
            stream=True,
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock:
            # Create a mock conversation object
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_mock.return_value = conv_ins

            adapted_request, _ = self.chat._convert_to_internal_request(
                req, self.fastapi_request
            )

            async def run_stream():
                chunks = []
                try:
                    async for chunk in self.chat._generate_chat_stream(
                        adapted_request, req, self.fastapi_request
                    ):
                        chunks.append(chunk)
                except Exception as e:
                    print(f"Error during stream iteration: {e}")
                return chunks

        loop = get_or_create_event_loop()
        chunks = loop.run_until_complete(run_stream())

        error_chunk_data = None
        for c in chunks:
            if "error" in c:
                error_chunk_data = json.loads(c[len("data: ") :])
                break
        self.assertIsNotNone(error_chunk_data, "Error chunk not found in stream")
        self.assertEqual(error_chunk_data["error"]["message"], err_msg)
        self.assertEqual(error_chunk_data["error"]["code"], err_code.value)

        # Ensure the stream stops after the abort error
        # The last chunk should be "data: [DONE]\n\n"
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")

        # Check that there is an error chunk and a DONE chunk
        self.assertEqual(len(chunks), 2)
        self.assertIn("error", chunks[0])

    # ------------- X-Data-Parallel-Rank header tests -------------
    def test_extract_routed_dp_rank_from_header_no_header(self):
        """Test that None is returned when no header is present."""
        self.fastapi_request.headers = {}
        result = self.chat.extract_routed_dp_rank_from_header(
            self.fastapi_request, body_routed_dp_rank=None
        )
        self.assertIsNone(result)

    def test_extract_routed_dp_rank_from_header_with_header(self):
        """Test that header value is extracted correctly."""
        self.fastapi_request.headers = {"x-data-parallel-rank": "2"}
        result = self.chat.extract_routed_dp_rank_from_header(
            self.fastapi_request, body_routed_dp_rank=None
        )
        self.assertEqual(result, 2)

    def test_extract_routed_dp_rank_header_overrides_body(self):
        """Test that header value has higher priority than body."""
        self.fastapi_request.headers = {"x-data-parallel-rank": "3"}
        result = self.chat.extract_routed_dp_rank_from_header(
            self.fastapi_request, body_routed_dp_rank=1
        )
        self.assertEqual(result, 3)  # header wins

    def test_extract_routed_dp_rank_from_header_invalid(self):
        """Test that invalid header value raises HTTPException."""
        from fastapi import HTTPException

        self.fastapi_request.headers = {"x-data-parallel-rank": "abc"}
        with self.assertRaises(HTTPException) as context:
            self.chat.extract_routed_dp_rank_from_header(
                self.fastapi_request, body_routed_dp_rank=None
            )
        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("must be an integer", context.exception.detail)


if __name__ == "__main__":
    unittest.main(verbosity=2)
