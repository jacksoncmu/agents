"""Tests for telemetry event emission across tools, engine, and session transitions.

Each test installs an InMemoryBackend, exercises a specific code path, then
asserts that the expected events were emitted with the correct outcome.

Tests are organised by the component that owns each event:
  - ToolRegistry        → tool.validate, tool.confirmation_gate, tool.execute
  - AgentEngine (loop)  → loop.*, run.acquire, run.release, session.transition
"""
from __future__ import annotations

import pytest

from agent.engine import AgentEngine, ConcurrentRunError
from agent.llm.base import LLMResponse
from agent.llm.mock import MockProvider
from agent.storage import InMemoryStore
from agent.telemetry import InMemoryBackend, Outcome, set_backend
from agent.tools import (
    ExecutionResult,
    ToolDefinition,
    ToolParam,
    ToolRegistry,
)
from agent.types import SessionState, ToolCall
from examples.example_tools import make_calculator, make_request_confirmation


# ---------------------------------------------------------------------------
# Fixture: fresh backend installed before every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def telemetry_backend():
    """Install a fresh InMemoryBackend for every test; restore NullBackend after."""
    from agent.telemetry import NullBackend
    backend = InMemoryBackend()
    set_backend(backend)
    yield backend
    set_backend(NullBackend())


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _simple_registry(*tools):
    r = ToolRegistry()
    for t in tools:
        r.register(t)
    return r


def _engine(responses, tools=None):
    store = InMemoryStore()
    registry = tools or ToolRegistry()
    engine = AgentEngine(store=store, llm=MockProvider(responses), tools=registry)
    sid = engine.create_session().id
    return engine, sid


# ===========================================================================
# tool.validate
# ===========================================================================

class TestToolValidateEvents:
    def test_valid_args_emits_executed(self, telemetry_backend):
        r = _simple_registry(make_calculator())
        r.execute("calculator", {"expression": "1+1"})

        events = telemetry_backend.by_name("tool.validate")
        assert len(events) == 1
        assert events[0].outcome == Outcome.executed
        assert events[0].data["tool_name"] == "calculator"

    def test_missing_required_param_emits_error(self, telemetry_backend):
        r = _simple_registry(make_calculator())
        r.execute("calculator", {})

        events = telemetry_backend.by_name("tool.validate")
        assert len(events) == 1
        assert events[0].outcome == Outcome.error
        assert events[0].data["tool_name"] == "calculator"
        # errors list should identify the broken parameter
        errors = events[0].data["errors"]
        assert any(e["param"] == "expression" for e in errors)

    def test_wrong_type_emits_error_with_param_info(self, telemetry_backend):
        r = _simple_registry(make_calculator())
        r.execute("calculator", {"expression": 42})  # int, not str

        events = telemetry_backend.by_name("tool.validate")
        assert events[0].outcome == Outcome.error
        assert any(e["param"] == "expression" for e in events[0].data["errors"])

    def test_no_validate_event_emitted_for_unknown_tool(self, telemetry_backend):
        r = ToolRegistry()
        r.execute("ghost", {})

        assert telemetry_backend.by_name("tool.validate") == []

    def test_confirmed_path_also_emits_validate(self, telemetry_backend):
        r = _simple_registry(make_request_confirmation())
        r.execute_confirmed("delete_file", {"filename": "x.txt"})

        events = telemetry_backend.by_name("tool.validate")
        assert len(events) == 1
        assert events[0].outcome == Outcome.executed
        assert events[0].data.get("confirmed") is True


# ===========================================================================
# tool.confirmation_gate
# ===========================================================================

class TestToolConfirmationGateEvents:
    def test_confirmation_gate_emits_blocked(self, telemetry_backend):
        r = _simple_registry(make_request_confirmation())
        result = r.execute("delete_file", {"filename": "report.txt"})

        assert result.requires_confirmation is True
        events = telemetry_backend.by_name("tool.confirmation_gate")
        assert len(events) == 1
        assert events[0].outcome == Outcome.blocked
        assert events[0].data["tool_name"] == "delete_file"

    def test_non_confirmation_tool_emits_no_gate_event(self, telemetry_backend):
        r = _simple_registry(make_calculator())
        r.execute("calculator", {"expression": "2+2"})

        assert telemetry_backend.by_name("tool.confirmation_gate") == []


# ===========================================================================
# tool.execute
# ===========================================================================

class TestToolExecuteEvents:
    def test_successful_execution_emits_executed(self, telemetry_backend):
        r = _simple_registry(make_calculator())
        r.execute("calculator", {"expression": "3*3"})

        events = telemetry_backend.by_name("tool.execute")
        assert len(events) == 1
        assert events[0].outcome == Outcome.executed
        assert events[0].data["tool_name"] == "calculator"
        assert events[0].data.get("confirmed") is False

    def test_unknown_tool_emits_error(self, telemetry_backend):
        r = ToolRegistry()
        r.execute("nonexistent", {})

        events = telemetry_backend.by_name("tool.execute")
        assert len(events) == 1
        assert events[0].outcome == Outcome.error
        assert events[0].data["tool_name"] == "nonexistent"
        assert events[0].data["reason"] == "unknown_tool"

    def test_handler_exception_emits_error(self, telemetry_backend):
        def boom(x: str) -> str:
            raise RuntimeError("kaboom")

        r = _simple_registry(ToolDefinition("bad", "d", [ToolParam("x", "string", "x")], boom))
        r.execute("bad", {"x": "hi"})

        events = telemetry_backend.by_name("tool.execute")
        assert events[0].outcome == Outcome.error
        assert "kaboom" in events[0].data["error"]

    def test_validation_failure_emits_no_execute_event(self, telemetry_backend):
        r = _simple_registry(make_calculator())
        r.execute("calculator", {})  # missing 'expression'

        # validate error → early return; handler never runs
        assert telemetry_backend.by_name("tool.execute") == []

    def test_confirmation_required_emits_no_execute_event(self, telemetry_backend):
        r = _simple_registry(make_request_confirmation())
        r.execute("delete_file", {"filename": "f.txt"})

        # gate intercepts before handler; no execute event
        assert telemetry_backend.by_name("tool.execute") == []

    def test_confirmed_execute_sets_confirmed_flag(self, telemetry_backend):
        r = _simple_registry(make_request_confirmation())
        r.execute_confirmed("delete_file", {"filename": "f.txt"})

        events = telemetry_backend.by_name("tool.execute")
        assert len(events) == 1
        assert events[0].outcome == Outcome.executed
        assert events[0].data.get("confirmed") is True


# ===========================================================================
# run.acquire / run.release
# ===========================================================================

class TestRunOwnershipEvents:
    def test_successful_run_emits_acquire_then_release(self, telemetry_backend):
        engine, sid = _engine([LLMResponse(content="hello")])
        engine.run(sid, "hi")

        acquires = telemetry_backend.by_name("run.acquire")
        releases = telemetry_backend.by_name("run.release")

        assert len(acquires) == 1
        assert acquires[0].outcome == Outcome.executed
        assert acquires[0].session_id == sid

        assert len(releases) == 1
        assert releases[0].outcome == Outcome.executed
        assert releases[0].session_id == sid

    def test_concurrent_run_emits_blocked_acquire(self, telemetry_backend):
        store = InMemoryStore()
        engine = AgentEngine(
            store=store,
            llm=MockProvider([LLMResponse(content="hi")]),
            tools=ToolRegistry(),
        )
        sid = engine.create_session().id

        # Manually grab ownership so the next call is rejected
        store.try_acquire_run(sid)
        with pytest.raises(ConcurrentRunError):
            engine.run(sid, "hi")

        acquires = telemetry_backend.by_name("run.acquire")
        # The _acquire_run helper emits blocked
        blocked = [e for e in acquires if e.outcome == Outcome.blocked]
        assert len(blocked) == 1
        assert blocked[0].session_id == sid

    def test_stream_run_emits_acquire_and_release(self, telemetry_backend):
        engine, sid = _engine([LLMResponse(content="streamed")])
        for _ in engine.stream_run(sid, "hi"):
            pass

        assert any(e.outcome == Outcome.executed for e in telemetry_backend.by_name("run.acquire"))
        assert any(e.outcome == Outcome.executed for e in telemetry_backend.by_name("run.release"))

    def test_release_emitted_even_on_error(self, telemetry_backend):
        class BoomLLM(MockProvider):
            def complete(self, messages, tools):
                raise RuntimeError("fail")

        store = InMemoryStore()
        engine = AgentEngine(store=store, llm=BoomLLM([]), tools=ToolRegistry())
        sid = engine.create_session().id

        with pytest.raises(RuntimeError):
            engine.run(sid, "hi")

        assert telemetry_backend.by_name("run.release")


# ===========================================================================
# session.transition
# ===========================================================================

class TestSessionTransitionEvents:
    def test_transitions_emitted_on_normal_run(self, telemetry_backend):
        engine, sid = _engine([LLMResponse(content="done")])
        engine.run(sid, "go")

        transitions = telemetry_backend.by_name("session.transition")
        states = [(e.data["from_state"], e.data["to_state"]) for e in transitions]

        # waiting_for_user → running → waiting_for_user
        assert ("waiting_for_user", "running") in states
        assert ("running", "waiting_for_user") in states

    def test_transition_to_error_emitted_on_llm_failure(self, telemetry_backend):
        class BoomLLM(MockProvider):
            def complete(self, messages, tools):
                raise RuntimeError("fail")

        store = InMemoryStore()
        engine = AgentEngine(store=store, llm=BoomLLM([]), tools=ToolRegistry())
        sid = engine.create_session().id

        with pytest.raises(RuntimeError):
            engine.run(sid, "hi")

        transitions = telemetry_backend.by_name("session.transition")
        to_error = [e for e in transitions if e.data["to_state"] == "error"]
        assert to_error, "Expected a transition to error state"
        assert to_error[0].session_id == sid

    def test_transition_to_finished_on_cancellation(self, telemetry_backend):
        # Run once so the session is in waiting_for_user, then cancel and
        # run again — the loop sees the flag on the first iteration.
        engine, sid = _engine([
            LLMResponse(content="first"),
            LLMResponse(content="never"),
        ])
        engine.run(sid, "hello")           # session → waiting_for_user
        engine.store.set_cancellation(sid, True)
        engine.run(sid, "second message")  # loop cancels immediately

        transitions = telemetry_backend.by_name("session.transition")
        to_finished = [e for e in transitions if e.data["to_state"] == "finished"]
        assert to_finished

    def test_transition_to_waiting_for_confirmation(self, telemetry_backend):
        tc = ToolCall.new("delete_file", {"filename": "f.txt"})
        registry = _simple_registry(make_request_confirmation())
        store = InMemoryStore()
        engine = AgentEngine(
            store=store,
            llm=MockProvider([LLMResponse(content="", tool_calls=[tc]),
                               LLMResponse(content="done")]),
            tools=registry,
        )
        sid = engine.create_session().id
        engine.run(sid, "go")

        transitions = telemetry_backend.by_name("session.transition")
        to_confirm = [e for e in transitions if e.data["to_state"] == "waiting_for_confirmation"]
        assert to_confirm


# ===========================================================================
# loop.iteration.start
# ===========================================================================

class TestLoopIterationStartEvents:
    def test_single_iteration_emits_one_start(self, telemetry_backend):
        engine, sid = _engine([LLMResponse(content="answer")])
        engine.run(sid, "question")

        starts = telemetry_backend.by_name("loop.iteration.start")
        assert len(starts) == 1
        assert starts[0].outcome == Outcome.executed
        assert starts[0].session_id == sid
        assert starts[0].iteration == 0

    def test_two_tool_iterations_emit_two_starts(self, telemetry_backend):
        registry = _simple_registry(make_calculator())
        tc = ToolCall.new("calculator", {"expression": "1+1"})
        engine = AgentEngine(
            store=InMemoryStore(),
            llm=MockProvider([
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="final"),
            ]),
            tools=registry,
        )
        sid = engine.create_session().id
        engine.run(sid, "calculate")

        starts = telemetry_backend.by_name("loop.iteration.start")
        assert len(starts) == 2
        assert [e.iteration for e in starts] == [0, 1]


# ===========================================================================
# loop.cancelled
# ===========================================================================

class TestLoopCancelledEvents:
    def test_cancelled_at_iteration_boundary_emits_blocked(self, telemetry_backend):
        # Pre-cancel, then start a fresh run — the loop sees it at iteration 0.
        engine, sid = _engine([LLMResponse(content="never reached")])
        engine.store.set_cancellation(sid, True)
        engine.run(sid, "go")

        events = telemetry_backend.by_name("loop.cancelled")
        assert len(events) == 1
        assert events[0].outcome == Outcome.blocked
        assert events[0].session_id == sid
        assert events[0].iteration == 0

    def test_no_cancelled_event_on_normal_run(self, telemetry_backend):
        engine, sid = _engine([LLMResponse(content="fine")])
        engine.run(sid, "go")
        assert telemetry_backend.by_name("loop.cancelled") == []


# ===========================================================================
# loop.cancelled_between_tools
# ===========================================================================

class TestLoopCancelledBetweenToolsEvents:
    def test_emits_blocked_with_pending_tool_name(self, telemetry_backend):
        calls = []

        def first(x: str) -> str:
            calls.append("first")
            return "ok"

        def second(x: str) -> str:
            calls.append("second")
            return "ok"

        registry = ToolRegistry()
        registry.register(ToolDefinition(
            "first_tool", "first", [ToolParam("x", "string", "x")], first))
        registry.register(ToolDefinition(
            "second_tool", "second", [ToolParam("x", "string", "x")], second))

        tc1 = ToolCall.new("first_tool", {"x": "a"})
        tc2 = ToolCall.new("second_tool", {"x": "b"})
        store = InMemoryStore()

        # Cancel after first tool runs
        def cancelling_first(x: str) -> str:
            result = first(x)
            store.set_cancellation(sid, True)
            return result

        registry._tools["first_tool"] = ToolDefinition(
            "first_tool", "first", [ToolParam("x", "string", "x")], cancelling_first)

        engine = AgentEngine(
            store=store,
            llm=MockProvider([LLMResponse(content="", tool_calls=[tc1, tc2])]),
            tools=registry,
        )
        sid = engine.create_session().id
        engine.run(sid, "go")

        events = telemetry_backend.by_name("loop.cancelled_between_tools")
        assert len(events) == 1
        assert events[0].outcome == Outcome.blocked
        assert events[0].data["pending_tool"] == "second_tool"
        assert "second" not in calls  # second tool must not have run


# ===========================================================================
# loop.final_answer
# ===========================================================================

class TestLoopFinalAnswerEvents:
    def test_final_answer_emits_condition_unsatisfied(self, telemetry_backend):
        engine, sid = _engine([LLMResponse(content="the answer is 42")])
        engine.run(sid, "what is the answer?")

        events = telemetry_backend.by_name("loop.final_answer")
        assert len(events) == 1
        assert events[0].outcome == Outcome.condition_unsatisfied
        assert events[0].session_id == sid
        assert events[0].iteration == 0
        # content_length lets observers see how long the final answer was
        assert events[0].data["content_length"] == len("the answer is 42")

    def test_no_final_answer_event_when_tools_called(self, telemetry_backend):
        registry = _simple_registry(make_calculator())
        tc = ToolCall.new("calculator", {"expression": "1+1"})
        engine = AgentEngine(
            store=InMemoryStore(),
            llm=MockProvider([
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="done"),
            ]),
            tools=registry,
        )
        sid = engine.create_session().id
        engine.run(sid, "calc")

        # Only iteration 1 emits final_answer (iteration 0 had tool calls)
        events = telemetry_backend.by_name("loop.final_answer")
        assert len(events) == 1
        assert events[0].iteration == 1


# ===========================================================================
# loop.confirmation_pause
# ===========================================================================

class TestLoopConfirmationPauseEvents:
    def test_confirmation_pause_emits_blocked(self, telemetry_backend):
        tc = ToolCall.new("delete_file", {"filename": "data.csv"})
        registry = _simple_registry(make_request_confirmation())
        engine = AgentEngine(
            store=InMemoryStore(),
            llm=MockProvider([
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="deleted"),
            ]),
            tools=registry,
        )
        sid = engine.create_session().id
        engine.run(sid, "delete data.csv")

        events = telemetry_backend.by_name("loop.confirmation_pause")
        assert len(events) == 1
        assert events[0].outcome == Outcome.blocked
        assert events[0].data["pending_count"] == 1
        assert "delete_file" in events[0].data["pending_tools"]

    def test_no_confirmation_pause_on_auto_approved_tool(self, telemetry_backend):
        engine, sid = _engine(
            [LLMResponse(content="", tool_calls=[ToolCall.new("calculator", {"expression": "1"})]),
             LLMResponse(content="ok")],
            tools=_simple_registry(make_calculator()),
        )
        engine.run(sid, "go")
        assert telemetry_backend.by_name("loop.confirmation_pause") == []


# ===========================================================================
# loop.max_iterations_exceeded
# ===========================================================================

class TestLoopMaxIterationsEvents:
    def test_max_iterations_emits_error(self, telemetry_backend, monkeypatch):
        # Override MAX_ITERATIONS to 2 so the test runs quickly
        import agent.engine as engine_mod
        monkeypatch.setattr(engine_mod, "MAX_ITERATIONS", 2)

        registry = _simple_registry(make_calculator())
        # LLM always returns a tool call → loop never terminates
        tc = ToolCall.new("calculator", {"expression": "1"})
        engine = AgentEngine(
            store=InMemoryStore(),
            llm=MockProvider([LLMResponse(content="", tool_calls=[tc])] * 10),
            tools=registry,
        )
        sid = engine.create_session().id

        with pytest.raises(RuntimeError, match="MAX_ITERATIONS"):
            engine.run(sid, "loop forever")

        events = telemetry_backend.by_name("loop.max_iterations_exceeded")
        assert len(events) == 1
        assert events[0].outcome == Outcome.error
        assert events[0].data["max_iterations"] == 2


# ===========================================================================
# Event ordering: key invariants across a full run
# ===========================================================================

class TestEventOrdering:
    def test_full_run_event_sequence(self, telemetry_backend):
        """
        For a single-turn run with one tool call:
          run.acquire → loop.iteration.start → tool.validate →
          tool.execute → loop.final_answer → run.release
        """
        registry = _simple_registry(make_calculator())
        tc = ToolCall.new("calculator", {"expression": "7*6"})
        engine = AgentEngine(
            store=InMemoryStore(),
            llm=MockProvider([
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="42"),
            ]),
            tools=registry,
        )
        sid = engine.create_session().id
        engine.run(sid, "what is 7*6?")

        names = [e.name for e in telemetry_backend.events]

        # These must appear, in this relative order
        def pos(name):
            return next(i for i, n in enumerate(names) if n == name)

        assert pos("run.acquire") < pos("loop.iteration.start")
        assert pos("loop.iteration.start") < pos("tool.validate")
        assert pos("tool.validate") < pos("tool.execute")
        assert pos("loop.final_answer") < pos("run.release")

    def test_all_session_ids_match(self, telemetry_backend):
        """Every event that carries session_id should match the session we created."""
        engine, sid = _engine([LLMResponse(content="hi")])
        engine.run(sid, "hello")

        for event in telemetry_backend.events:
            if event.session_id is not None:
                assert event.session_id == sid, (
                    f"Event {event.name} has session_id={event.session_id!r}, expected {sid!r}"
                )
