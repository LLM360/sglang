from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StreamingParseResult


class JsonArrayParser(BaseFormatDetector):
    """
    Parser for JSON array tool calls when JSON schema constraints are active.

    This parser is used when tool_choice="required" or a specific tool is named,
    bypassing model-specific parsers in favor of direct JSON array parsing.
    """

    def __init__(self):
        super().__init__()
        # Configure for JSON array parsing
        self.bot_token = "["
        self.eot_token = "]"
        self.tool_call_separator = ","

    def has_tool_call(self, text: str) -> bool:
        """
        Check if the given text contains a JSON tool call (array or single object).
        """
        return "[" in text or "{" in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Parse JSON tool calls using the base class implementation.
        """
        raise NotImplementedError(
            "Detect and parse not supported for JSON schema constraints."
        )

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing with tool validation.
        """
        return super().parse_streaming_increment(new_text, tools)

    def finalize_streaming(self, tools: List[Tool]) -> StreamingParseResult:
        """Drain complete JSON calls still buffered in the terminal engine delta."""
        normal_text = ""
        calls = []

        # BaseFormatDetector emits a call's name and arguments on separate
        # invocations. A terminal engine delta can contain the entire JSON
        # array, so keep parsing the existing buffer until no state changes.
        max_steps = 2 * (self._buffer.count(self.tool_call_separator) + 1) + 2
        for _ in range(max_steps):
            before = (
                self._buffer,
                self.current_tool_id,
                self.current_tool_name_sent,
                tuple(self.streamed_args_for_tool),
                repr(self.prev_tool_call_arr),
            )
            result = super().parse_streaming_increment("", tools)
            normal_text += result.normal_text or ""
            calls.extend(result.calls)
            after = (
                self._buffer,
                self.current_tool_id,
                self.current_tool_name_sent,
                tuple(self.streamed_args_for_tool),
                repr(self.prev_tool_call_arr),
            )
            if after == before:
                break

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def structure_info(self) -> callable:
        """
        Return a function that creates StructureInfo for constrained generation.
        This is not used for JSON schema constraints as they are handled
        by the constraint backends directly.
        """
        raise NotImplementedError("structure_info not used for JSON schema constraints")
