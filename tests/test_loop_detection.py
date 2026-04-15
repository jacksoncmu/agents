"""Tests for loop detection and safe intervention message handling.

Tests are grouped by:
  - LoopDetectorConfig       config validation
  - LoopDetection            standalone detection logic
  - InterventionQueue        queue mechanics
  - ProtocolOrdering         intervention never breaks assistant→tool_result
  - EngineIntegration        end-to-end with the engine loop
  - Telemetry                events emitted at the right points
"""
from __future__ import annotations

import pytest

from agent.engine import AgentEngine
from agent.llm.base import LLMResponse
from agent.llm.mock import MockProvider
from agent.loop_detector import (
    InterventionQueue,
    LoopDetection,
    LoopDetector,
    LoopDetectorConfig,
)
from agent.storage import InMemoryStore
from agent.telemetry import InMemoryBackend, NullBackend, Outcome, set_backend
from agent.tools import ToolDefinition, ToolParam, ToolRegistry
from agent.types import Message, MessageRole, ToolCall, ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assistant_with_tool(name: str = "t", args: dict | None = None, call_id: str = "id0") -> Message:
    return Message.assistant("", tool_calls=[
        ToolCall(id=call_id, name=name, arguments=args or {}),
    ])


def _tool_result_msg(content: str = "ok", call_id: str = "id0") -> Message:
    return Message.tool_result_msg([ToolResult(tool_call_id=call_id, content=content)])


def _build_repeated_pattern(
    tool_name: str = "search",
    args: dict | None = None,
    count: int = 3,
) -> list[Message]:
    """Build a message list with ``count`` identical tool-call/result pairs."""
    msgs: list[Message] = [Message.user("do the thing")]
    for i in range(count):
        cid = f"cid-{i}"
        msgs.append(_assistant_with_tool(tool_name, args or {"q": "same"}, call_id=cid))
        msgs.append(_tool_result_msg("result", call_id=cid))
    return msgs


@pytest.fixture(autouse=True)
def telemetry_backend():
    backend = InMemoryBackend()
    set_backend(backend)
    yield backend
    set_backend(NullBackend())


# ===========================================================================
# LoopDetectorConfig validation
# ===========================================================================

class TestLoopDetectorConfig:
    def test_valid_defaults(self):
        cfg = LoopDetectorConfig()
        assert cfg.window_size == 10
        assert cfg.repeat_threshold == 3
        assert cfg.enabled is True

    def test_window_size_zero_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            LoopDetectorConfig(window_size=0)

    def test_repeat_threshold_one_raises(self):
        with pytest.raises(ValueError, match="repeat_threshold"):
            LoopDetectorConfig(repeat_threshold=1)

    def test_custom_values(self):
        cfg = LoopDetectorConfig(window_size=5, repeat_threshold=2)
        assert cfg.window_size == 5
        assert cfg.repeat_threshold == 2


# ===========================================================================
# Standalone detection logic
# ===========================================================================

class TestLoopDetection:
    def test_no_loop_with_diverse_calls(self):
        msgs = [Message.user("hi")]
        for i in range(5):
            cid = f"cid-{i}"
            msgs.append(_assistant_with_tool(f"tool_{i}", {"x": i}, call_id=cid))
            msgs.append(_tool_result_msg("ok", call_id=cid))

        detector = LoopDetector(LoopDetectorConfig(repeat_threshold=3))
        result = detector.check(msgs)
        assert not result.detected

    def test_detects_repeated_identical_calls(self):
        msgs = _build_repeated_pattern("search", {"q": "same"}, count=4)
        detector = LoopDetector(LoopDetectorConfig(repeat_threshold=3))
        result = detector.check(msgs)
        assert result.detected
        assert result.count >= 3
        assert "search" in result.pattern

    def test_threshold_exact_boundary(self):
        """Exactly threshold repetitions should trigger."""
        msgs = _build_repeated_pattern("t", {"a": 1}, count=3)
        detector = LoopDetector(LoopDetectorConfig(repeat_threshold=3))
        result = detector.check(msgs)
        assert result.detected
        assert result.count == 3

    def test_below_threshold_no_detection(self):
        """One fewer than threshold should NOT trigger."""
        msgs = _build_repeated_pattern("t", {"a": 1}, count=2)
        detector = LoopDetector(LoopDetectorConfig(repeat_threshold=3))
        result = detector.check(msgs)
        assert not result.detected

    def test_window_limits_scope(self):
        """Old repetitions outside the window shouldn't count."""
        msgs = _build_repeated_pattern("old_tool", {"x": 1}, count=5)
        # Add diverse calls to push old ones outside window
        for i in range(6):
            cid = f"new-{i}"
            msgs.append(_assistant_with_tool(f"new_{i}", {"y": i}, call_id=cid))
            msgs.append(_tool_result_msg("ok", call_id=cid))

        detector = LoopDetector(LoopDetectorConfig(window_size=5, repeat_threshold=3))
        result = detector.check(msgs)
        assert not result.detected

    def test_disabled_detector_always_false(self):
        msgs = _build_repeated_pattern("t", count=10)
        detector = LoopDetector(LoopDetectorConfig(enabled=False))
        result = detector.check(msgs)
        assert not result.detected

    def test_empty_messages(self):
        detector = LoopDetector()
        result = detector.check([])
        assert not result.detected

    def test_no_tool_calls_in_messages(self):
        msgs = [Message.user("hi"), Message.assistant("hello")]
        detector = LoopDetector()
        result = detector.check(msgs)
        assert not result.detected

    def test_different_args_not_same_pattern(self):
        """Same tool name but different arguments should not be a loop."""
        msgs = [Message.user("start")]
        for i in range(5):
            cid = f"cid-{i}"
            msgs.append(_assistant_with_tool("search", {"q": f"query_{i}"}, call_id=cid))
            msgs.append(_tool_result_msg("ok", call_id=cid))

        detector = LoopDetector(LoopDetectorConfig(repeat_threshold=3))
        result = detector.check(msgs)
        assert not result.detected

    def test_multi_tool_calls_in_single_message(self):
        """Pattern includes all tool calls in one assistant message."""
        msgs = [Message.user("start")]
        for i in range(3):
            msg = Message.assistant("", tool_calls=[
                ToolCall(id=f"a-{i}", name="read", arguments={"file": "x"}),
                ToolCall(id=f"b-{i}", name="write", arguments={"file": "y"}),
            ])
            msgs.append(msg)
            msgs.append(Message.tool_result_msg([
                ToolResult(tool_call_id=f"a-{i}", content="ok"),
                ToolResult(tool_call_id=f"b-{i}", content="ok"),
            ]))

        detector = LoopDetector(LoopDetectorConfig(repeat_threshold=3))
        result = detector.check(msgs)
        assert result.detected
        assert "read" in result.pattern
        assert "write" in result.pattern

    def test_intervention_message_content(self):
        detection = LoopDetection(
            detected=True,
            pattern="search:[('q', 'same')]",
            count=4,
            reason="test reason",
        )
        msg = LoopDetector.make_intervention_message(detection)
        assert msg.role == MessageRole.user
        assert "[LOOP DETECTED]" in msg.content
        assert "4 times" in msg.content
        assert "different approach" in msg.content


# ===========================================================================
# Intervention queue
# ===========================================================================

class TestInterventionQueue:
    def test_empty_queue(self):
        q = InterventionQueue()
        assert q.is_empty
        assert q.pending == 0
        assert q.flush() == []

    def test_enqueue_and_flush(self):
        q = InterventionQueue()
        msg = Message.user("intervention")
        q.enqueue(msg)
        assert q.pending == 1
        assert not q.is_empty

        flushed = q.flush()
        assert len(flushed) == 1
        assert flushed[0].content == "intervention"
        assert q.is_empty

    def test_flush_clears_queue(self):
        q = InterventionQueue()
        q.enqueue(Message.user("a"))
        q.enqueue(Message.user("b"))
        first = q.flush()
        assert len(first) == 2

        second = q.flush()
        assert len(second) == 0

    def test_multiple_enqueues_preserve_order(self):
        q = InterventionQueue()
        q.enqueue(Message.user("first"))
        q.enqueue(Message.user("second"))
        q.enqueue(Message.user("third"))

        flushed = q.flush()
        assert [m.content for m in flushed] == ["first", "second", "third"]


# ===========================================================================
# Protocol ordering — intervention never breaks assistant→tool_result
# ===========================================================================

class TestProtocolOrdering:
    """Verify that intervention messages never appear between
    an assistant message (with tool_calls) and the next tool_result message."""

    def _check_protocol(self, messages: list[Message]) -> None:
        """Assert that every assistant msg with tool_calls is immediately
        followed by a tool_result msg."""
        for i, msg in enumerate(messages):
            if msg.role == MessageRole.assistant and msg.tool_calls:
                assert i + 1 < len(messages), (
                    f"Assistant message at index {i} has tool_calls but "
                    "is the last message"
                )
                next_msg = messages[i + 1]
                assert next_msg.role == MessageRole.tool_result, (
                    f"Message at index {i+1} after assistant(tool_calls) "
                    f"should be tool_result, got {next_msg.role}. "
                    f"Content: {next_msg.content[:80]!r}"
                )

    def test_intervention_not_between_assistant_and_tool_result(self):
        """Run the engine with loop detection active; verify protocol
        ordering in the final message list."""
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            "echo", "echoes input",
            [ToolParam("msg", "string", "message")],
            lambda msg: f"echo: {msg}",
        ))

        tc = ToolCall.new("echo", {"msg": "hello"})
        # 3 identical tool calls → triggers loop detection at threshold=3
        # Then one more call (model sees intervention), then final answer
        responses = [
            LLMResponse(content="", tool_calls=[tc]),
            LLMResponse(content="", tool_calls=[ToolCall.new("echo", {"msg": "hello"})]),
            LLMResponse(content="", tool_calls=[ToolCall.new("echo", {"msg": "hello"})]),
            # After loop detection, model gets intervention and answers
            LLMResponse(content="I'll try something different."),
        ]

        store = InMemoryStore()
        detector = LoopDetector(LoopDetectorConfig(repeat_threshold=3))
        engine = AgentEngine(
            store=store,
            llm=MockProvider(responses),
            tools=registry,
            loop_detector=detector,
        )
        sid = engine.create_session().id
        engine.run(sid, "keep echoing")

        session = store.get(sid)
        self._check_protocol(session.messages)

    def test_intervention_appears_before_llm_call_not_mid_protocol(self):
        """The intervention message should appear after a tool_result,
        never between an assistant and tool_result."""
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            "noop", "does nothing", [], lambda: "done",
        ))

        tc1 = ToolCall.new("noop", {})
        tc2 = ToolCall.new("noop", {})
        tc3 = ToolCall.new("noop", {})

        responses = [
            LLMResponse(content="", tool_calls=[tc1]),
            LLMResponse(content="", tool_calls=[tc2]),
            LLMResponse(content="", tool_calls=[tc3]),
            LLMResponse(content="Changed approach."),
        ]

        store = InMemoryStore()
        detector = LoopDetector(LoopDetectorConfig(repeat_threshold=3))
        engine = AgentEngine(
            store=store,
            llm=MockProvider(responses),
            tools=registry,
            loop_detector=detector,
        )
        sid = engine.create_session().id
        engine.run(sid, "go")

        session = store.get(sid)
        self._check_protocol(session.messages)

        # Verify the intervention message exists somewhere
        intervention_msgs = [
            m for m in session.messages
            if m.role == MessageRole.user and "[LOOP DETECTED]" in m.content
        ]
        assert intervention_msgs, "Expected an intervention message in the session"

    def test_intervention_flush_position_in_messages(self):
        """After loop detection, the intervention message should appear
        after the last tool_result and before the next assistant message."""
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            "ping", "pings", [], lambda: "pong",
        ))

        responses = [
            LLMResponse(content="", tool_calls=[ToolCall.new("ping", {})]),
            LLMResponse(content="", tool_calls=[ToolCall.new("ping", {})]),
            LLMResponse(content="", tool_calls=[ToolCall.new("ping", {})]),
            LLMResponse(content="Done."),
        ]

        store = InMemoryStore()
        detector = LoopDetector(LoopDetectorConfig(repeat_threshold=3))
        engine = AgentEngine(
            store=store,
            llm=MockProvider(responses),
            tools=registry,
            loop_detector=detector,
        )
        sid = engine.create_session().id
        engine.run(sid, "start")

        session = store.get(sid)
        # Find the intervention message
        for i, m in enumerate(session.messages):
            if m.role == MessageRole.user and "[LOOP DETECTED]" in m.content:
                # Previous message should be tool_result
                assert i > 0
                assert session.messages[i - 1].role == MessageRole.tool_result
                # Next message (if any) should be assistant
                if i + 1 < len(session.messages):
                    assert session.messages[i + 1].role == MessageRole.assistant
                break
        else:
            pytest.fail("Intervention message not found")


# ===========================================================================
# Engine integration
# ===========================================================================

class TestEngineIntegration:
    def test_loop_detected_event_in_stream(self):
        """stream_run should yield a [loop_detected] event."""
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            "echo", "echoes", [ToolParam("x", "string", "input")],
            lambda x: f"echo:{x}",
        ))

        responses = [
            LLMResponse(content="", tool_calls=[ToolCall.new("echo", {"x": "a"})]),
            LLMResponse(content="", tool_calls=[ToolCall.new("echo", {"x": "a"})]),
            LLMResponse(content="", tool_calls=[ToolCall.new("echo", {"x": "a"})]),
            LLMResponse(content="Final."),
        ]

        store = InMemoryStore()
        detector = LoopDetector(LoopDetectorConfig(repeat_threshold=3))
        engine = AgentEngine(
            store=store,
            llm=MockProvider(responses),
            tools=registry,
            loop_detector=detector,
        )
        sid = engine.create_session().id
        events = list(engine.stream_run(sid, "go"))

        loop_events = [e for e in events if e.startswith("[loop_detected]")]
        assert loop_events, "Expected [loop_detected] in stream events"
        assert "Final." in events[-1]

    def test_no_loop_detector_normal_operation(self):
        """Without a loop detector, the engine runs normally."""
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            "echo", "echoes", [ToolParam("x", "string", "input")],
            lambda x: f"echo:{x}",
        ))

        tc = ToolCall.new("echo", {"x": "hi"})
        responses = [
            LLMResponse(content="", tool_calls=[tc]),
            LLMResponse(content="Done."),
        ]

        store = InMemoryStore()
        engine = AgentEngine(
            store=store,
            llm=MockProvider(responses),
            tools=registry,
            loop_detector=None,
        )
        sid = engine.create_session().id
        result = engine.run(sid, "hi")
        assert result == "Done."

    def test_loop_detection_with_below_threshold(self, telemetry_backend):
        """Below threshold, no intervention is injected."""
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            "echo", "echoes", [ToolParam("x", "string", "input")],
            lambda x: f"echo:{x}",
        ))

        responses = [
            LLMResponse(content="", tool_calls=[ToolCall.new("echo", {"x": "a"})]),
            LLMResponse(content="", tool_calls=[ToolCall.new("echo", {"x": "a"})]),
            # Only 2 repeats, threshold=3 → no loop
            LLMResponse(content="Done."),
        ]

        store = InMemoryStore()
        detector = LoopDetector(LoopDetectorConfig(repeat_threshold=3))
        engine = AgentEngine(
            store=store,
            llm=MockProvider(responses),
            tools=registry,
            loop_detector=detector,
        )
        sid = engine.create_session().id
        result = engine.run(sid, "go")
        assert result == "Done."

        session = store.get(sid)
        intervention_msgs = [
            m for m in session.messages
            if m.role == MessageRole.user and "[LOOP DETECTED]" in m.content
        ]
        assert not intervention_msgs

    def test_session_state_correct_after_loop_intervention(self):
        """Session should end in waiting_for_user after loop + final answer."""
        from agent.types import SessionState

        registry = ToolRegistry()
        registry.register(ToolDefinition(
            "t", "test", [], lambda: "ok",
        ))

        responses = [
            LLMResponse(content="", tool_calls=[ToolCall.new("t", {})]),
            LLMResponse(content="", tool_calls=[ToolCall.new("t", {})]),
            LLMResponse(content="", tool_calls=[ToolCall.new("t", {})]),
            LLMResponse(content="Changed my mind."),
        ]

        store = InMemoryStore()
        detector = LoopDetector(LoopDetectorConfig(repeat_threshold=3))
        engine = AgentEngine(
            store=store,
            llm=MockProvider(responses),
            tools=registry,
            loop_detector=detector,
        )
        sid = engine.create_session().id
        result = engine.run(sid, "go")
        assert result == "Changed my mind."

        session = store.get(sid)
        assert session.state == SessionState.waiting_for_user

    def test_intervention_message_visible_to_llm(self):
        """After loop detection, the LLM should receive the intervention
        message in its input on the next call."""
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            "t", "test", [], lambda: "ok",
        ))

        # Track what messages the LLM receives
        received_messages: list[list[Message]] = []

        class TrackingProvider(MockProvider):
            def complete(self, messages, tools):
                received_messages.append(list(messages))
                return super().complete(messages, tools)

        responses = [
            LLMResponse(content="", tool_calls=[ToolCall.new("t", {})]),
            LLMResponse(content="", tool_calls=[ToolCall.new("t", {})]),
            LLMResponse(content="", tool_calls=[ToolCall.new("t", {})]),
            LLMResponse(content="Ok, changed."),
        ]

        store = InMemoryStore()
        detector = LoopDetector(LoopDetectorConfig(repeat_threshold=3))
        engine = AgentEngine(
            store=store,
            llm=TrackingProvider(responses),
            tools=registry,
            loop_detector=detector,
        )
        sid = engine.create_session().id
        engine.run(sid, "go")

        # The 4th LLM call (index 3) should have the intervention message
        assert len(received_messages) == 4
        fourth_call_msgs = received_messages[3]
        has_intervention = any(
            m.role == MessageRole.user and "[LOOP DETECTED]" in m.content
            for m in fourth_call_msgs
        )
        assert has_intervention, (
            "LLM's 4th call should include the intervention message"
        )


# ===========================================================================
# Telemetry
# ===========================================================================

class TestLoopDetectionTelemetry:
    def test_loop_detected_event(self, telemetry_backend):
        msgs = _build_repeated_pattern("search", {"q": "same"}, count=4)
        detector = LoopDetector(LoopDetectorConfig(repeat_threshold=3))
        detector.check(msgs, session_id="s1", iteration=5)

        events = telemetry_backend.by_name("loop.detected")
        assert len(events) == 1
        e = events[0]
        assert e.outcome == Outcome.blocked
        assert e.session_id == "s1"
        assert e.iteration == 5
        assert "pattern" in e.data
        assert e.data["count"] >= 3

    def test_no_loop_event(self, telemetry_backend):
        msgs = [Message.user("hi"), _assistant_with_tool("t1"), _tool_result_msg()]
        detector = LoopDetector(LoopDetectorConfig(repeat_threshold=3))
        detector.check(msgs, session_id="s2")

        events = telemetry_backend.by_name("loop.not_detected")
        assert len(events) == 1
        assert events[0].outcome == Outcome.condition_unsatisfied

    def test_disabled_emits_no_events(self, telemetry_backend):
        msgs = _build_repeated_pattern("t", count=5)
        detector = LoopDetector(LoopDetectorConfig(enabled=False))
        detector.check(msgs)

        assert not telemetry_backend.by_name("loop.detected")
        assert not telemetry_backend.by_name("loop.not_detected")

    def test_intervention_flushed_event(self, telemetry_backend):
        """Engine should emit loop.intervention_flushed when queue is drained."""
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            "t", "test", [], lambda: "ok",
        ))

        responses = [
            LLMResponse(content="", tool_calls=[ToolCall.new("t", {})]),
            LLMResponse(content="", tool_calls=[ToolCall.new("t", {})]),
            LLMResponse(content="", tool_calls=[ToolCall.new("t", {})]),
            LLMResponse(content="Done."),
        ]

        store = InMemoryStore()
        detector = LoopDetector(LoopDetectorConfig(repeat_threshold=3))
        engine = AgentEngine(
            store=store,
            llm=MockProvider(responses),
            tools=registry,
            loop_detector=detector,
        )
        sid = engine.create_session().id
        engine.run(sid, "go")

        flushed = telemetry_backend.by_name("loop.intervention_flushed")
        assert flushed, "Expected loop.intervention_flushed event"
        assert flushed[0].outcome == Outcome.executed
        assert flushed[0].data["count"] == 1

    def test_loop_detected_event_in_engine_context(self, telemetry_backend):
        """loop.detected should fire during engine run with correct session_id."""
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            "t", "test", [], lambda: "ok",
        ))

        responses = [
            LLMResponse(content="", tool_calls=[ToolCall.new("t", {})]),
            LLMResponse(content="", tool_calls=[ToolCall.new("t", {})]),
            LLMResponse(content="", tool_calls=[ToolCall.new("t", {})]),
            LLMResponse(content="Done."),
        ]

        store = InMemoryStore()
        detector = LoopDetector(LoopDetectorConfig(repeat_threshold=3))
        engine = AgentEngine(
            store=store,
            llm=MockProvider(responses),
            tools=registry,
            loop_detector=detector,
        )
        sid = engine.create_session().id
        engine.run(sid, "go")

        detected = telemetry_backend.by_name("loop.detected")
        assert detected
        assert detected[0].session_id == sid
