"""Tests for AnthropicAdapter message conversion.

All tests are pure data-transformation — no API key or network required.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent.llm.anthropic import AnthropicAdapter
from agent.llm.base import LLMResponse
from agent.types import Message, MessageRole, ToolCall, ToolResult


@pytest.fixture()
def adapter() -> AnthropicAdapter:
    return AnthropicAdapter()


# ---------------------------------------------------------------------------
# to_provider_messages
# ---------------------------------------------------------------------------

class TestToProviderMessages:
    def test_user_message(self, adapter):
        msgs = [Message.user("Hello")]
        result = adapter.to_provider_messages(msgs)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_assistant_message_no_tools(self, adapter):
        msgs = [Message.assistant("Hi there")]
        result = adapter.to_provider_messages(msgs)
        assert result == [{"role": "assistant", "content": "Hi there"}]

    def test_assistant_message_with_tool_calls_no_text(self, adapter):
        tc = ToolCall(id="tc_1", name="calculator", arguments={"expression": "2+2"})
        msgs = [Message.assistant("", tool_calls=[tc])]
        result = adapter.to_provider_messages(msgs)

        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert isinstance(msg["content"], list)
        # No leading text block
        assert len(msg["content"]) == 1
        block = msg["content"][0]
        assert block["type"] == "tool_use"
        assert block["id"] == "tc_1"
        assert block["name"] == "calculator"
        assert block["input"] == {"expression": "2+2"}

    def test_assistant_message_with_tool_calls_and_text(self, adapter):
        tc = ToolCall(id="tc_2", name="calculator", arguments={"expression": "3*3"})
        msgs = [Message.assistant("Let me compute.", tool_calls=[tc])]
        result = adapter.to_provider_messages(msgs)

        content = result[0]["content"]
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "Let me compute."}
        assert content[1]["type"] == "tool_use"

    def test_tool_result_message(self, adapter):
        tr = ToolResult(tool_call_id="tc_1", content="4")
        msgs = [Message.tool_result_msg([tr])]
        result = adapter.to_provider_messages(msgs)

        assert result[0]["role"] == "user"
        content = result[0]["content"]
        assert len(content) == 1
        assert content[0] == {
            "type": "tool_result",
            "tool_use_id": "tc_1",
            "content": "4",
        }

    def test_tool_result_message_with_error(self, adapter):
        tr = ToolResult(tool_call_id="tc_err", content="Something failed", error=True)
        msgs = [Message.tool_result_msg([tr])]
        result = adapter.to_provider_messages(msgs)

        block = result[0]["content"][0]
        assert block["is_error"] is True
        assert block["content"] == "Something failed"

    def test_tool_result_success_has_no_is_error_key(self, adapter):
        tr = ToolResult(tool_call_id="tc_ok", content="ok")
        msgs = [Message.tool_result_msg([tr])]
        result = adapter.to_provider_messages(msgs)

        block = result[0]["content"][0]
        assert "is_error" not in block

    def test_multiple_tool_results_in_one_message(self, adapter):
        trs = [
            ToolResult(tool_call_id="a", content="res_a"),
            ToolResult(tool_call_id="b", content="res_b"),
        ]
        msgs = [Message.tool_result_msg(trs)]
        result = adapter.to_provider_messages(msgs)

        content = result[0]["content"]
        assert len(content) == 2
        assert content[0]["tool_use_id"] == "a"
        assert content[1]["tool_use_id"] == "b"

    def test_full_conversation_ordering(self, adapter):
        """Verify a complete user→assistant(tool)→tool_result→assistant sequence."""
        tc = ToolCall(id="tc_x", name="calculator", arguments={"expression": "1+1"})
        tr = ToolResult(tool_call_id="tc_x", content="2")
        msgs = [
            Message.user("What is 1+1?"),
            Message.assistant("", tool_calls=[tc]),
            Message.tool_result_msg([tr]),
            Message.assistant("It is 2."),
        ]
        result = adapter.to_provider_messages(msgs)

        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"   # tool_result maps to user role
        assert result[3]["role"] == "assistant"

    def test_unknown_role_raises(self, adapter):
        msg = Message(role="bad_role")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown message role"):
            adapter.to_provider_messages([msg])


# ---------------------------------------------------------------------------
# to_provider_tools
# ---------------------------------------------------------------------------

class TestToProviderTools:
    def test_parameters_renamed_to_input_schema(self, adapter):
        internal_tools = [
            {
                "name": "calculator",
                "description": "Evaluates expressions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "The expression"},
                    },
                    "required": ["expression"],
                },
            }
        ]
        result = adapter.to_provider_tools(internal_tools)

        assert len(result) == 1
        t = result[0]
        assert t["name"] == "calculator"
        assert t["description"] == "Evaluates expressions."
        assert "input_schema" in t
        assert "parameters" not in t
        assert t["input_schema"]["properties"]["expression"]["type"] == "string"

    def test_empty_tool_list(self, adapter):
        assert adapter.to_provider_tools([]) == []

    def test_multiple_tools_converted(self, adapter):
        tools = [
            {"name": "a", "description": "A", "parameters": {"type": "object", "properties": {}, "required": []}},
            {"name": "b", "description": "B", "parameters": {"type": "object", "properties": {}, "required": []}},
        ]
        result = adapter.to_provider_tools(tools)
        assert [t["name"] for t in result] == ["a", "b"]


# ---------------------------------------------------------------------------
# from_provider_response — using SimpleNamespace to fake SDK objects
# ---------------------------------------------------------------------------

def _make_block(**kwargs: Any) -> SimpleNamespace:
    """Create a fake Anthropic content block."""
    return SimpleNamespace(**kwargs)


def _make_response(blocks: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(content=blocks)


class TestFromProviderResponse:
    def test_text_only_response(self, adapter):
        raw = _make_response([_make_block(type="text", text="Hello!")])
        resp = adapter.from_provider_response(raw)
        assert isinstance(resp, LLMResponse)
        assert resp.content == "Hello!"
        assert resp.tool_calls == []

    def test_tool_use_only_response(self, adapter):
        raw = _make_response([
            _make_block(type="tool_use", id="tc_1", name="calculator", input={"expression": "6*7"})
        ])
        resp = adapter.from_provider_response(raw)
        assert resp.content == ""
        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc.id == "tc_1"
        assert tc.name == "calculator"
        assert tc.arguments == {"expression": "6*7"}

    def test_mixed_text_and_tool_use(self, adapter):
        raw = _make_response([
            _make_block(type="text",     text="Let me compute."),
            _make_block(type="tool_use", id="tc_2", name="calculator", input={"expression": "3+3"}),
        ])
        resp = adapter.from_provider_response(raw)
        assert resp.content == "Let me compute."
        assert len(resp.tool_calls) == 1

    def test_multiple_tool_calls(self, adapter):
        raw = _make_response([
            _make_block(type="tool_use", id="tc_a", name="tool_a", input={"x": 1}),
            _make_block(type="tool_use", id="tc_b", name="tool_b", input={"y": 2}),
        ])
        resp = adapter.from_provider_response(raw)
        assert len(resp.tool_calls) == 2
        assert resp.tool_calls[0].name == "tool_a"
        assert resp.tool_calls[1].name == "tool_b"

    def test_tool_input_as_json_string(self, adapter):
        """Anthropic may (rarely) return tool input as a JSON string."""
        import json
        raw = _make_response([
            _make_block(type="tool_use", id="tc_s", name="calc", input=json.dumps({"val": 99}))
        ])
        resp = adapter.from_provider_response(raw)
        assert resp.tool_calls[0].arguments == {"val": 99}

    def test_multiple_text_blocks_joined(self, adapter):
        raw = _make_response([
            _make_block(type="text", text="First."),
            _make_block(type="text", text="Second."),
        ])
        resp = adapter.from_provider_response(raw)
        assert resp.content == "First. Second."

    def test_unknown_block_type_ignored(self, adapter):
        """Unknown block types should not raise; they are silently skipped."""
        raw = _make_response([
            _make_block(type="unknown_future_type", data="whatever"),
            _make_block(type="text", text="Visible."),
        ])
        resp = adapter.from_provider_response(raw)
        assert resp.content == "Visible."

    def test_empty_response(self, adapter):
        raw = _make_response([])
        resp = adapter.from_provider_response(raw)
        assert resp.content == ""
        assert resp.tool_calls == []

    def test_response_blocks_as_dicts(self, adapter):
        """Adapter should handle dict-style blocks in addition to objects."""
        raw = SimpleNamespace(content=[
            {"type": "text", "text": "dict block"},
        ])
        resp = adapter.from_provider_response(raw)
        assert resp.content == "dict block"


# ---------------------------------------------------------------------------
# Round-trip sanity: internal → wire format preserves all IDs
# ---------------------------------------------------------------------------

class TestRoundTripIds:
    def test_tool_call_id_preserved_in_wire_format(self, adapter):
        tc = ToolCall(id="unique-id-123", name="foo", arguments={})
        msg = Message.assistant("", tool_calls=[tc])
        wire = adapter.to_provider_messages([msg])
        block = wire[0]["content"][0]
        assert block["id"] == "unique-id-123"

    def test_tool_result_use_id_matches_call_id(self, adapter):
        tr = ToolResult(tool_call_id="unique-id-123", content="result")
        msg = Message.tool_result_msg([tr])
        wire = adapter.to_provider_messages([msg])
        block = wire[0]["content"][0]
        assert block["tool_use_id"] == "unique-id-123"
