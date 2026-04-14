"""End-to-end test: one full ReAct loop with a tool call."""
from __future__ import annotations

import pytest

from agent.engine import AgentEngine
from agent.llm.base import LLMResponse
from agent.llm.mock import MockProvider
from agent.storage import InMemoryStore
from agent.tools import ToolRegistry
from agent.types import MessageRole, SessionState, ToolCall
from examples.example_tools import make_calculator, make_get_current_time


def _build_engine(responses: list[LLMResponse]) -> tuple[AgentEngine, str]:
    registry = ToolRegistry()
    registry.register(make_calculator())
    registry.register(make_get_current_time())

    engine = AgentEngine(
        store=InMemoryStore(),
        llm=MockProvider(responses),
        tools=registry,
    )
    session = engine.create_session()
    return engine, session.id


# ---------------------------------------------------------------------------
# Happy path: one tool call then a final answer
# ---------------------------------------------------------------------------

def test_single_tool_call_then_answer():
    responses = [
        LLMResponse(
            content="",
            tool_calls=[ToolCall.new("calculator", {"expression": "6 * 7"})],
        ),
        LLMResponse(content="The answer is 42."),
    ]
    engine, sid = _build_engine(responses)

    result = engine.run(sid, "What is 6 times 7?")

    assert result == "The answer is 42."

    session = engine.store.get(sid)
    assert session.state == SessionState.waiting_for_user

    # Verify message ordering: user → assistant(tool_calls) → tool_result → assistant
    msgs = session.messages
    assert msgs[0].role == MessageRole.user
    assert msgs[1].role == MessageRole.assistant
    assert msgs[1].tool_calls[0].name == "calculator"
    assert msgs[2].role == MessageRole.tool_result
    assert msgs[2].tool_results[0].content == "42"
    assert msgs[3].role == MessageRole.assistant
    assert msgs[3].content == "The answer is 42."


def test_tool_result_follows_tool_call_immediately():
    """tool_result must be the very next message after an assistant with tool_calls."""
    responses = [
        LLMResponse(
            content="",
            tool_calls=[ToolCall.new("calculator", {"expression": "1 + 1"})],
        ),
        LLMResponse(content="Done."),
    ]
    engine, sid = _build_engine(responses)
    engine.run(sid, "hi")

    msgs = engine.store.get(sid).messages
    for i, msg in enumerate(msgs):
        if msg.role == MessageRole.assistant and msg.tool_calls:
            assert i + 1 < len(msgs), "No message after tool_calls"
            assert msgs[i + 1].role == MessageRole.tool_result, (
                f"Expected tool_result at index {i+1}, got {msgs[i+1].role}"
            )


# ---------------------------------------------------------------------------
# No tool call: direct answer
# ---------------------------------------------------------------------------

def test_no_tool_call_direct_answer():
    responses = [LLMResponse(content="Hello, world!")]
    engine, sid = _build_engine(responses)

    result = engine.run(sid, "Say hello.")

    assert result == "Hello, world!"
    msgs = engine.store.get(sid).messages
    assert len(msgs) == 2  # user + assistant
    assert msgs[1].role == MessageRole.assistant


# ---------------------------------------------------------------------------
# Multiple turns in the same session
# ---------------------------------------------------------------------------

def test_multi_turn_session():
    responses = [
        LLMResponse(content="Turn one response."),
        LLMResponse(content="Turn two response."),
    ]
    engine, sid = _build_engine(responses)

    r1 = engine.run(sid, "First message.")
    r2 = engine.run(sid, "Second message.")

    assert r1 == "Turn one response."
    assert r2 == "Turn two response."

    msgs = engine.store.get(sid).messages
    assert len(msgs) == 4  # user, assistant, user, assistant


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

def test_cancellation_stops_loop():
    """Cancelling mid-stream should stop before exhausting the script."""

    call_count = 0

    class CancellingLLM(MockProvider):
        def complete(self, messages, tools):
            nonlocal call_count
            call_count += 1
            # Signal cancel before returning so the next iteration stops.
            engine.cancel(sid)
            return super().complete(messages, tools)

    responses = [
        LLMResponse(
            content="",
            tool_calls=[ToolCall.new("calculator", {"expression": "1+1"})],
        ),
        LLMResponse(content="Should not reach here."),
    ]
    registry = ToolRegistry()
    registry.register(make_calculator())
    store = InMemoryStore()

    # We need access to engine before it's fully built — use a list trick.
    engine_holder: list[AgentEngine] = []
    llm = CancellingLLM(responses)
    engine = AgentEngine(store=store, llm=llm, tools=registry)
    engine_holder.append(engine)

    session = engine.create_session()
    sid = session.id

    # Run; the engine should stop after one tool execution because cancelled=True.
    events = list(engine.stream_run(sid, "go"))
    session = engine.store.get(sid)
    assert session.state == SessionState.finished
    assert session.cancelled is True
    assert call_count == 1


# ---------------------------------------------------------------------------
# Tool error handling
# ---------------------------------------------------------------------------

def test_tool_error_is_captured_not_raised():
    responses = [
        LLMResponse(
            content="",
            tool_calls=[ToolCall.new("calculator", {"expression": "1/0"})],
        ),
        LLMResponse(content="It errored."),
    ]
    engine, sid = _build_engine(responses)

    result = engine.run(sid, "Divide by zero.")

    assert result == "It errored."
    msgs = engine.store.get(sid).messages
    tr_msg = msgs[2]
    assert tr_msg.role == MessageRole.tool_result
    assert tr_msg.tool_results[0].error is True


# ---------------------------------------------------------------------------
# Unknown tool
# ---------------------------------------------------------------------------

def test_unknown_tool_raises_in_result():
    responses = [
        LLMResponse(
            content="",
            tool_calls=[ToolCall.new("nonexistent_tool", {})],
        ),
        LLMResponse(content="Handled."),
    ]
    engine, sid = _build_engine(responses)

    result = engine.run(sid, "Use a missing tool.")

    assert result == "Handled."
    msgs = engine.store.get(sid).messages
    tr = msgs[2].tool_results[0]
    assert tr.error is True
    assert "Unknown tool" in tr.content


# ---------------------------------------------------------------------------
# Session state after error
# ---------------------------------------------------------------------------

def test_engine_error_sets_error_state():
    """Exhausting MockProvider raises RuntimeError → session enters error state."""
    responses: list[LLMResponse] = []  # empty script
    engine, sid = _build_engine(responses)

    with pytest.raises(RuntimeError):
        engine.run(sid, "anything")

    session = engine.store.get(sid)
    assert session.state == SessionState.error
    assert session.error_message != ""
