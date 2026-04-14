"""Tests for OpenAIAdapter message conversion.

All tests are pure data-transformation — no API key or network required.
Structure mirrors test_anthropic_adapter.py.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from agent.llm.base import LLMResponse
from agent.llm.openai import OpenAIAdapter
from agent.types import Message, MessageRole, ToolCall, ToolResult


@pytest.fixture()
def adapter() -> OpenAIAdapter:
    return OpenAIAdapter()


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
        tc = ToolCall(id="call_1", name="calculator", arguments={"expression": "2+2"})
        msgs = [Message.assistant("", tool_calls=[tc])]
        result = adapter.to_provider_messages(msgs)

        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert msg["content"] is None  # absent text → None
        assert len(msg["tool_calls"]) == 1
        tc_wire = msg["tool_calls"][0]
        assert tc_wire["id"] == "call_1"
        assert tc_wire["type"] == "function"
        assert tc_wire["function"]["name"] == "calculator"
        assert json.loads(tc_wire["function"]["arguments"]) == {"expression": "2+2"}

    def test_assistant_message_with_tool_calls_and_text(self, adapter):
        tc = ToolCall(id="call_2", name="calculator", arguments={"expression": "3*3"})
        msgs = [Message.assistant("Let me compute.", tool_calls=[tc])]
        result = adapter.to_provider_messages(msgs)

        msg = result[0]
        assert msg["content"] == "Let me compute."
        assert len(msg["tool_calls"]) == 1

    def test_tool_call_arguments_serialised_as_json_string(self, adapter):
        """OpenAI requires arguments to be a JSON-encoded string, not a dict."""
        tc = ToolCall(id="call_3", name="foo", arguments={"a": 1, "b": "x"})
        msgs = [Message.assistant("", tool_calls=[tc])]
        result = adapter.to_provider_messages(msgs)

        raw_args = result[0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(raw_args, str)
        assert json.loads(raw_args) == {"a": 1, "b": "x"}

    def test_tool_result_expands_to_one_message_per_result(self, adapter):
        """A single internal tool_result Message becomes N OpenAI tool messages."""
        trs = [
            ToolResult(tool_call_id="call_a", content="res_a"),
            ToolResult(tool_call_id="call_b", content="res_b"),
        ]
        msgs = [Message.tool_result_msg(trs)]
        result = adapter.to_provider_messages(msgs)

        assert len(result) == 2
        assert result[0] == {"role": "tool", "tool_call_id": "call_a", "content": "res_a"}
        assert result[1] == {"role": "tool", "tool_call_id": "call_b", "content": "res_b"}

    def test_tool_result_single(self, adapter):
        tr = ToolResult(tool_call_id="call_1", content="42")
        msgs = [Message.tool_result_msg([tr])]
        result = adapter.to_provider_messages(msgs)

        assert len(result) == 1
        assert result[0] == {"role": "tool", "tool_call_id": "call_1", "content": "42"}

    def test_tool_result_error_content_still_forwarded(self, adapter):
        """OpenAI has no is_error flag; error text is passed as content."""
        tr = ToolResult(tool_call_id="call_e", content="Something failed", error=True)
        msgs = [Message.tool_result_msg([tr])]
        result = adapter.to_provider_messages(msgs)

        assert result[0]["content"] == "Something failed"
        assert "is_error" not in result[0]

    def test_full_conversation_ordering(self, adapter):
        """
        Internal: user → assistant(tool) → tool_result → assistant
        OpenAI:   user → assistant(tool) → tool(x1)     → assistant
        """
        tc = ToolCall(id="call_x", name="calculator", arguments={"expression": "1+1"})
        tr = ToolResult(tool_call_id="call_x", content="2")
        msgs = [
            Message.user("What is 1+1?"),
            Message.assistant("", tool_calls=[tc]),
            Message.tool_result_msg([tr]),
            Message.assistant("It is 2."),
        ]
        result = adapter.to_provider_messages(msgs)

        assert len(result) == 4
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "tool"
        assert result[3]["role"] == "assistant"

    def test_multi_tool_call_expands_result_messages(self, adapter):
        """Two parallel tool calls produce two separate tool result messages."""
        tc1 = ToolCall(id="call_1", name="tool_a", arguments={})
        tc2 = ToolCall(id="call_2", name="tool_b", arguments={})
        tr1 = ToolResult(tool_call_id="call_1", content="r1")
        tr2 = ToolResult(tool_call_id="call_2", content="r2")
        msgs = [
            Message.assistant("", tool_calls=[tc1, tc2]),
            Message.tool_result_msg([tr1, tr2]),
        ]
        result = adapter.to_provider_messages(msgs)

        # 1 assistant + 2 tool result messages
        assert len(result) == 3
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "tool"
        assert result[2]["role"] == "tool"

    def test_unknown_role_raises(self, adapter):
        msg = Message(role="bad_role")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown message role"):
            adapter.to_provider_messages([msg])


# ---------------------------------------------------------------------------
# to_provider_tools
# ---------------------------------------------------------------------------

class TestToProviderTools:
    def test_tools_wrapped_in_function_type(self, adapter):
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
        assert t["type"] == "function"
        fn = t["function"]
        assert fn["name"] == "calculator"
        assert fn["description"] == "Evaluates expressions."
        # parameters key is preserved as-is
        assert fn["parameters"]["properties"]["expression"]["type"] == "string"

    def test_parameters_key_unchanged(self, adapter):
        """OpenAI uses 'parameters' — no renaming needed."""
        tools = [{"name": "t", "description": "d", "parameters": {"type": "object", "properties": {}, "required": []}}]
        result = adapter.to_provider_tools(tools)
        assert "parameters" in result[0]["function"]
        assert "input_schema" not in result[0]["function"]

    def test_empty_tool_list(self, adapter):
        assert adapter.to_provider_tools([]) == []

    def test_multiple_tools_converted(self, adapter):
        tools = [
            {"name": "a", "description": "A", "parameters": {"type": "object", "properties": {}, "required": []}},
            {"name": "b", "description": "B", "parameters": {"type": "object", "properties": {}, "required": []}},
        ]
        result = adapter.to_provider_tools(tools)
        assert [t["function"]["name"] for t in result] == ["a", "b"]
        assert all(t["type"] == "function" for t in result)


# ---------------------------------------------------------------------------
# from_provider_response — fake SDK objects via SimpleNamespace / dicts
# ---------------------------------------------------------------------------

def _make_function(name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, arguments=arguments)


def _make_tool_call(id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(id=id, type="function", function=_make_function(name, arguments))


def _make_message(content: str | None, tool_calls: list | None) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _make_choice(message: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(message=message)


def _make_response(choices: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(choices=choices)


class TestFromProviderResponse:
    def test_text_only_response(self, adapter):
        raw = _make_response([_make_choice(_make_message("Hello!", None))])
        resp = adapter.from_provider_response(raw)
        assert isinstance(resp, LLMResponse)
        assert resp.content == "Hello!"
        assert resp.tool_calls == []

    def test_tool_call_only_response(self, adapter):
        tc = _make_tool_call("call_1", "calculator", json.dumps({"expression": "6*7"}))
        raw = _make_response([_make_choice(_make_message(None, [tc]))])
        resp = adapter.from_provider_response(raw)

        assert resp.content == ""
        assert len(resp.tool_calls) == 1
        tc_out = resp.tool_calls[0]
        assert tc_out.id == "call_1"
        assert tc_out.name == "calculator"
        assert tc_out.arguments == {"expression": "6*7"}

    def test_mixed_content_and_tool_call(self, adapter):
        tc = _make_tool_call("call_2", "calculator", json.dumps({"expression": "3+3"}))
        raw = _make_response([_make_choice(_make_message("Let me compute.", [tc]))])
        resp = adapter.from_provider_response(raw)

        assert resp.content == "Let me compute."
        assert len(resp.tool_calls) == 1

    def test_multiple_tool_calls(self, adapter):
        tcs = [
            _make_tool_call("call_a", "tool_a", json.dumps({"x": 1})),
            _make_tool_call("call_b", "tool_b", json.dumps({"y": 2})),
        ]
        raw = _make_response([_make_choice(_make_message(None, tcs))])
        resp = adapter.from_provider_response(raw)

        assert len(resp.tool_calls) == 2
        assert resp.tool_calls[0].name == "tool_a"
        assert resp.tool_calls[1].name == "tool_b"

    def test_arguments_parsed_from_json_string(self, adapter):
        tc = _make_tool_call("call_s", "calc", '{"val": 99}')
        raw = _make_response([_make_choice(_make_message(None, [tc]))])
        resp = adapter.from_provider_response(raw)
        assert resp.tool_calls[0].arguments == {"val": 99}

    def test_arguments_already_dict_accepted(self, adapter):
        """If arguments arrives as a dict (non-standard), it is used as-is."""
        tc = _make_tool_call("call_d", "calc", {"val": 42})  # type: ignore[arg-type]
        raw = _make_response([_make_choice(_make_message(None, [tc]))])
        resp = adapter.from_provider_response(raw)
        assert resp.tool_calls[0].arguments == {"val": 42}

    def test_none_content_becomes_empty_string(self, adapter):
        raw = _make_response([_make_choice(_make_message(None, None))])
        resp = adapter.from_provider_response(raw)
        assert resp.content == ""

    def test_empty_tool_calls_list(self, adapter):
        raw = _make_response([_make_choice(_make_message("Done.", []))])
        resp = adapter.from_provider_response(raw)
        assert resp.tool_calls == []

    def test_response_as_dicts(self, adapter):
        """Adapter should handle fully dict-based responses."""
        raw = {
            "choices": [
                {
                    "message": {
                        "content": "dict response",
                        "tool_calls": None,
                    }
                }
            ]
        }
        resp = adapter.from_provider_response(raw)
        assert resp.content == "dict response"


# ---------------------------------------------------------------------------
# Round-trip sanity: internal → wire format preserves all IDs
# ---------------------------------------------------------------------------

class TestRoundTripIds:
    def test_tool_call_id_preserved_in_wire_format(self, adapter):
        tc = ToolCall(id="unique-id-456", name="foo", arguments={})
        msg = Message.assistant("", tool_calls=[tc])
        wire = adapter.to_provider_messages([msg])
        assert wire[0]["tool_calls"][0]["id"] == "unique-id-456"

    def test_tool_result_call_id_preserved(self, adapter):
        tr = ToolResult(tool_call_id="unique-id-456", content="result")
        msg = Message.tool_result_msg([tr])
        wire = adapter.to_provider_messages([msg])
        assert wire[0]["tool_call_id"] == "unique-id-456"

    def test_call_id_links_assistant_to_result(self, adapter):
        """The id in the assistant tool_call must match tool_call_id in tool message."""
        tc = ToolCall(id="link-id-789", name="calc", arguments={})
        tr = ToolResult(tool_call_id="link-id-789", content="ok")
        wire = adapter.to_provider_messages([
            Message.assistant("", tool_calls=[tc]),
            Message.tool_result_msg([tr]),
        ])
        call_id   = wire[0]["tool_calls"][0]["id"]
        result_id = wire[1]["tool_call_id"]
        assert call_id == result_id
