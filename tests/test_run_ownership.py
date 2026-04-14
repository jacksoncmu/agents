"""Tests for run ownership, lock discipline, and cancellation behaviour."""
from __future__ import annotations

import threading
import time
from typing import Iterator

import pytest

from agent.engine import AgentEngine, ConcurrentRunError
from agent.llm.base import LLMResponse
from agent.llm.mock import MockProvider
from agent.storage import InMemoryStore
from agent.tools import ExecutionResult, ToolDefinition, ToolParam, ToolRegistry
from agent.types import SessionState, ToolCall
from examples.example_tools import make_calculator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine(responses: list[LLMResponse], tools: ToolRegistry | None = None) -> tuple[AgentEngine, str]:
    if tools is None:
        tools = ToolRegistry()
    store = InMemoryStore()
    engine = AgentEngine(store=store, llm=MockProvider(responses), tools=tools)
    sid = engine.create_session().id
    return engine, sid


def _slow_tool(delay: float = 0.05) -> ToolDefinition:
    """A tool that sleeps, giving us a window to try concurrent operations."""
    def handler(x: str) -> str:
        time.sleep(delay)
        return f"done: {x}"

    return ToolDefinition(
        "slow_tool", "sleeps briefly",
        [ToolParam("x", "string", "input")],
        handler,
    )


# ---------------------------------------------------------------------------
# ConcurrentRunError
# ---------------------------------------------------------------------------

class TestConcurrentRunRejection:
    def test_second_run_raises_concurrent_run_error(self):
        """While one run is active, a second run() on the same session must raise."""
        registry = ToolRegistry()
        registry.register(_slow_tool(0.1))
        tc = ToolCall.new("slow_tool", {"x": "hello"})
        store = InMemoryStore()
        engine = AgentEngine(
            store=store,
            llm=MockProvider([
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="Done."),
            ]),
            tools=registry,
        )
        sid = engine.create_session().id

        errors: list[Exception] = []
        results: list[str] = []

        def first_run():
            results.append(engine.run(sid, "go"))

        def second_run():
            time.sleep(0.02)  # let first_run acquire ownership
            try:
                engine.run(sid, "also go")
            except ConcurrentRunError as exc:
                errors.append(exc)

        t1 = threading.Thread(target=first_run)
        t2 = threading.Thread(target=second_run)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert len(errors) == 1
        assert isinstance(errors[0], ConcurrentRunError)
        assert len(results) == 1

    def test_stream_run_raises_concurrent_run_error(self):
        """stream_run also rejects a concurrent attempt."""
        registry = ToolRegistry()
        registry.register(_slow_tool(0.1))
        tc = ToolCall.new("slow_tool", {"x": "x"})
        store = InMemoryStore()
        engine = AgentEngine(
            store=store,
            llm=MockProvider([
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="Done."),
            ]),
            tools=registry,
        )
        sid = engine.create_session().id

        errors: list[Exception] = []

        def drain_stream():
            for _ in engine.stream_run(sid, "go"):
                pass

        def concurrent_attempt():
            time.sleep(0.02)
            try:
                for _ in engine.stream_run(sid, "also go"):
                    pass
            except ConcurrentRunError as exc:
                errors.append(exc)

        t1 = threading.Thread(target=drain_stream)
        t2 = threading.Thread(target=concurrent_attempt)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert len(errors) == 1


# ---------------------------------------------------------------------------
# Lock released on normal completion
# ---------------------------------------------------------------------------

class TestLockReleased:
    def test_lock_released_after_successful_run(self):
        engine, sid = _engine([LLMResponse(content="hello")])
        engine.run(sid, "hi")
        assert not engine.store.is_run_owned(sid)

    def test_lock_released_after_error(self):
        """If the LLM raises, ownership must still be released."""
        store = InMemoryStore()

        class BoomLLM(MockProvider):
            def complete(self, messages, tools):
                raise RuntimeError("boom")

        engine = AgentEngine(store=store, llm=BoomLLM([]), tools=ToolRegistry())
        sid = engine.create_session().id

        with pytest.raises(RuntimeError, match="boom"):
            engine.run(sid, "hi")

        assert not store.is_run_owned(sid)

    def test_lock_released_after_cancellation(self):
        """Cancellation sets state=finished; ownership must be released."""
        registry = ToolRegistry()
        registry.register(_slow_tool(0.05))
        tc = ToolCall.new("slow_tool", {"x": "x"})
        store = InMemoryStore()
        engine = AgentEngine(
            store=store,
            llm=MockProvider([
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="Done."),
            ]),
            tools=registry,
        )
        sid = engine.create_session().id

        def run_it():
            engine.run(sid, "go")

        t = threading.Thread(target=run_it)
        t.start()
        time.sleep(0.01)  # let it start
        engine.cancel(sid)
        t.join()

        assert not store.is_run_owned(sid)

    def test_lock_released_on_stream_run_success(self):
        engine, sid = _engine([LLMResponse(content="hello")])
        for _ in engine.stream_run(sid, "hi"):
            pass
        assert not engine.store.is_run_owned(sid)

    def test_lock_released_on_stream_run_generator_abandoned(self):
        """If the caller drops the generator early, finally still fires."""
        engine, sid = _engine([LLMResponse(content="a long response")])
        gen = engine.stream_run(sid, "hi")
        next(gen)  # consume one event
        gen.close()  # abandon the generator
        assert not engine.store.is_run_owned(sid)

    def test_lock_released_on_stream_run_error(self):
        store = InMemoryStore()

        class BoomLLM(MockProvider):
            def complete(self, messages, tools):
                raise RuntimeError("stream boom")

        engine = AgentEngine(store=store, llm=BoomLLM([]), tools=ToolRegistry())
        sid = engine.create_session().id

        with pytest.raises(RuntimeError, match="stream boom"):
            for _ in engine.stream_run(sid, "hi"):
                pass

        assert not store.is_run_owned(sid)


# ---------------------------------------------------------------------------
# Lock released after confirmation pause
# ---------------------------------------------------------------------------

class TestLockReleasedOnConfirmation:
    def _confirmation_engine(self) -> tuple[AgentEngine, str]:
        from examples.example_tools import make_request_confirmation
        registry = ToolRegistry()
        registry.register(make_request_confirmation())
        store = InMemoryStore()
        tc = ToolCall.new("delete_file", {"filename": "f.txt"})
        engine = AgentEngine(
            store=store,
            llm=MockProvider([
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="Done."),
            ]),
            tools=registry,
        )
        sid = engine.create_session().id
        return engine, sid

    def test_lock_released_when_paused_for_confirmation(self):
        engine, sid = self._confirmation_engine()
        engine.run(sid, "delete it")
        session = engine.store.get(sid)
        assert session.state == SessionState.waiting_for_confirmation
        # Ownership must be released so the confirmation callback can re-acquire
        assert not engine.store.is_run_owned(sid)

    def test_resume_reacquires_and_releases(self):
        engine, sid = self._confirmation_engine()
        engine.run(sid, "delete it")
        engine.resume(sid, approved=True)
        assert not engine.store.is_run_owned(sid)

    def test_resume_rejected_releases(self):
        engine, sid = self._confirmation_engine()
        engine.run(sid, "delete it")
        engine.resume(sid, approved=False)
        assert not engine.store.is_run_owned(sid)


# ---------------------------------------------------------------------------
# Sequential re-use — run → run on same session after ownership released
# ---------------------------------------------------------------------------

class TestSequentialRuns:
    def test_second_run_succeeds_after_first_completes(self):
        engine, sid = _engine([
            LLMResponse(content="first"),
            LLMResponse(content="second"),
        ])
        r1 = engine.run(sid, "one")
        r2 = engine.run(sid, "two")
        assert r1 == "first"
        assert r2 == "second"

    def test_continue_run_reacquires_and_releases(self):
        """continue_run with a healthy paused-but-not-running session acquires/releases."""
        store = InMemoryStore()
        engine = AgentEngine(
            store=store,
            llm=MockProvider([
                LLMResponse(content="first"),
                LLMResponse(content="continued"),
            ]),
            tools=ToolRegistry(),
        )
        sid = engine.create_session().id
        engine.run(sid, "hello")
        # Manually put session back into waiting_for_user so continue_run is valid
        session = store.get(sid)
        assert session.state == SessionState.waiting_for_user

        result = engine.continue_run(sid)
        assert result == "continued"
        assert not store.is_run_owned(sid)


# ---------------------------------------------------------------------------
# Per-tool cancellation check
# ---------------------------------------------------------------------------

class TestPerToolCancellation:
    def test_cancellation_between_tools_stops_loop(self):
        """
        The session has two tool calls scheduled.  We cancel after the first
        tool executes.  The second tool must NOT execute.
        """
        calls: list[str] = []

        def first_handler(x: str) -> str:
            calls.append("first")
            return "first done"

        def second_handler(x: str) -> str:
            calls.append("second")
            return "second done"

        registry = ToolRegistry()
        registry.register(ToolDefinition(
            "first_tool", "first",
            [ToolParam("x", "string", "x")],
            first_handler,
        ))
        registry.register(ToolDefinition(
            "second_tool", "second",
            [ToolParam("x", "string", "x")],
            second_handler,
        ))

        tc1 = ToolCall.new("first_tool", {"x": "a"})
        tc2 = ToolCall.new("second_tool", {"x": "b"})

        store = InMemoryStore()
        engine = AgentEngine(
            store=store,
            llm=MockProvider([LLMResponse(content="", tool_calls=[tc1, tc2])]),
            tools=registry,
        )
        sid = engine.create_session().id

        # We'll cancel via a side-effecting first_handler replacement
        original_first = first_handler

        def cancelling_first(x: str) -> str:
            result = original_first(x)
            engine.cancel(sid)  # cancel after first tool
            return result

        registry._tools["first_tool"] = ToolDefinition(
            "first_tool", "first",
            [ToolParam("x", "string", "x")],
            cancelling_first,
        )

        engine.run(sid, "go")

        assert "first" in calls
        assert "second" not in calls

        session = store.get(sid)
        assert session.state == SessionState.finished
