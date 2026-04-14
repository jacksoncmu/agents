"""Tests for session storage, message persistence, checkpoints, and recovery.

Each test class is parameterized over InMemoryStore and FileStore so both
backends are verified to behave identically.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from agent.engine import AgentEngine
from agent.llm.base import LLMResponse
from agent.llm.mock import MockProvider
from agent.storage import (
    LABEL_POST_LLM,
    LABEL_POST_TOOLS,
    Checkpoint,
    FileStore,
    InMemoryStore,
    RunStatus,
    SessionStore,
    SessionSummary,
)
from agent.tools import ToolRegistry
from agent.types import (
    Message,
    MessageRole,
    Session,
    SessionState,
    ToolCall,
    ToolResult,
)
from examples.example_tools import make_calculator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=["memory", "file"])
def store(request, tmp_path) -> SessionStore:
    if request.param == "memory":
        return InMemoryStore()
    return FileStore(tmp_path / "store")


@pytest.fixture()
def memory_store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture()
def file_store(tmp_path) -> FileStore:
    return FileStore(tmp_path / "store")


# ---------------------------------------------------------------------------
# Session creation
# ---------------------------------------------------------------------------

class TestSessionCreation:
    def test_create_returns_session(self, store):
        s = store.create()
        assert isinstance(s, Session)
        assert s.id

    def test_created_session_has_timestamps(self, store):
        s = store.create()
        assert isinstance(s.created_at, datetime)
        assert isinstance(s.updated_at, datetime)

    def test_created_session_is_waiting_for_user(self, store):
        s = store.create()
        assert s.state == SessionState.waiting_for_user

    def test_created_session_is_retrievable(self, store):
        s = store.create()
        fetched = store.get(s.id)
        assert fetched is not None
        assert fetched.id == s.id

    def test_get_unknown_returns_none(self, store):
        assert store.get("nonexistent") is None

    def test_list_ids_contains_created(self, store):
        s = store.create()
        assert s.id in store.list_ids()

    def test_delete_removes_session(self, store):
        s = store.create()
        store.delete(s.id)
        assert store.get(s.id) is None
        assert s.id not in store.list_ids()

    def test_multiple_sessions_independent(self, store):
        s1 = store.create()
        s2 = store.create()
        assert s1.id != s2.id
        assert len(store.list_ids()) == 2


# ---------------------------------------------------------------------------
# Message persistence
# ---------------------------------------------------------------------------

class TestMessagePersistence:
    def test_messages_survive_save_get_round_trip(self, store):
        s = store.create()
        s.messages.append(Message.user("hello"))
        s.messages.append(Message.assistant("world"))
        store.save(s)

        fetched = store.get(s.id)
        assert len(fetched.messages) == 2
        assert fetched.messages[0].content == "hello"
        assert fetched.messages[0].role == MessageRole.user
        assert fetched.messages[1].content == "world"
        assert fetched.messages[1].role == MessageRole.assistant

    def test_tool_call_message_round_trips(self, store):
        s = store.create()
        tc = ToolCall(id="tc_1", name="calculator", arguments={"expression": "1+1"})
        s.messages.append(Message.assistant("", tool_calls=[tc]))
        store.save(s)

        fetched = store.get(s.id)
        tc_back = fetched.messages[0].tool_calls[0]
        assert tc_back.id == "tc_1"
        assert tc_back.name == "calculator"
        assert tc_back.arguments == {"expression": "1+1"}

    def test_tool_result_message_round_trips(self, store):
        s = store.create()
        tr = ToolResult(tool_call_id="tc_1", content="2", error=False)
        s.messages.append(Message.tool_result_msg([tr]))
        store.save(s)

        fetched = store.get(s.id)
        tr_back = fetched.messages[0].tool_results[0]
        assert tr_back.tool_call_id == "tc_1"
        assert tr_back.content == "2"
        assert tr_back.error is False

    def test_tool_result_error_flag_preserved(self, store):
        s = store.create()
        tr = ToolResult(tool_call_id="x", content="boom", error=True)
        s.messages.append(Message.tool_result_msg([tr]))
        store.save(s)

        fetched = store.get(s.id)
        assert fetched.messages[0].tool_results[0].error is True

    def test_appending_messages_accumulates(self, store):
        s = store.create()
        for i in range(5):
            s.messages.append(Message.user(f"msg {i}"))
            store.save(s)

        fetched = store.get(s.id)
        assert len(fetched.messages) == 5
        assert fetched.messages[4].content == "msg 4"

    def test_updated_at_changes_on_save(self, store):
        s = store.create()
        t0 = s.updated_at
        import time; time.sleep(0.01)
        s.messages.append(Message.user("hi"))
        store.save(s)
        fetched = store.get(s.id)
        assert fetched.updated_at >= t0

    def test_state_persists(self, store):
        s = store.create()
        s.state = SessionState.running
        store.save(s)
        assert store.get(s.id).state == SessionState.running

    def test_error_message_persists(self, store):
        s = store.create()
        s.error_message = "something broke"
        store.save(s)
        assert store.get(s.id).error_message == "something broke"

    def test_pending_confirmation_persists(self, store):
        s = store.create()
        s.pending_confirmation = [ToolCall(id="x", name="foo", arguments={"a": 1})]
        store.save(s)
        fetched = store.get(s.id)
        assert len(fetched.pending_confirmation) == 1
        assert fetched.pending_confirmation[0].name == "foo"


# ---------------------------------------------------------------------------
# Checkpoint persistence
# ---------------------------------------------------------------------------

class TestCheckpointPersistence:
    def _cp(self, session_id: str, iteration: int = 0, label: str = LABEL_POST_TOOLS, msg_count: int = 2) -> Checkpoint:
        return Checkpoint.new(
            session_id=session_id,
            iteration=iteration,
            message_count=msg_count,
            state=SessionState.running,
            label=label,
        )

    def test_save_and_retrieve_checkpoint(self, store):
        s = store.create()
        cp = self._cp(s.id)
        store.save_checkpoint(cp)

        fetched = store.latest_checkpoint(s.id)
        assert fetched is not None
        assert fetched.id == cp.id
        assert fetched.session_id == s.id

    def test_latest_checkpoint_returns_most_recent(self, store):
        s = store.create()
        cp1 = self._cp(s.id, iteration=0)
        cp2 = self._cp(s.id, iteration=1)
        cp3 = self._cp(s.id, iteration=2)
        store.save_checkpoint(cp1)
        store.save_checkpoint(cp2)
        store.save_checkpoint(cp3)

        latest = store.latest_checkpoint(s.id)
        assert latest.id == cp3.id

    def test_list_checkpoints_ordered(self, store):
        s = store.create()
        for i in range(4):
            store.save_checkpoint(self._cp(s.id, iteration=i))

        cps = store.list_checkpoints(s.id)
        assert len(cps) == 4
        assert [cp.iteration for cp in cps] == [0, 1, 2, 3]

    def test_latest_checkpoint_none_when_empty(self, store):
        s = store.create()
        assert store.latest_checkpoint(s.id) is None

    def test_list_checkpoints_empty_when_none(self, store):
        s = store.create()
        assert store.list_checkpoints(s.id) == []

    def test_checkpoint_label_preserved(self, store):
        s = store.create()
        cp = self._cp(s.id, label=LABEL_POST_LLM)
        store.save_checkpoint(cp)
        assert store.latest_checkpoint(s.id).label == LABEL_POST_LLM

    def test_checkpoint_message_count_preserved(self, store):
        s = store.create()
        cp = self._cp(s.id, msg_count=7)
        store.save_checkpoint(cp)
        assert store.latest_checkpoint(s.id).message_count == 7

    def test_delete_checkpoints_clears_all(self, store):
        s = store.create()
        for i in range(3):
            store.save_checkpoint(self._cp(s.id, iteration=i))
        store.delete_checkpoints(s.id)
        assert store.list_checkpoints(s.id) == []
        assert store.latest_checkpoint(s.id) is None

    def test_checkpoints_isolated_between_sessions(self, store):
        s1 = store.create()
        s2 = store.create()
        store.save_checkpoint(self._cp(s1.id))
        assert store.latest_checkpoint(s2.id) is None

    def test_checkpoint_has_timestamp(self, store):
        s = store.create()
        cp = self._cp(s.id)
        store.save_checkpoint(cp)
        fetched = store.latest_checkpoint(s.id)
        assert isinstance(fetched.created_at, datetime)

    def test_engine_writes_post_llm_checkpoint(self, memory_store):
        """Engine writes LABEL_POST_LLM checkpoints during the loop."""
        registry = ToolRegistry()
        registry.register(make_calculator())
        llm = MockProvider([LLMResponse(content="Done.")])
        engine = AgentEngine(store=memory_store, llm=llm, tools=registry)
        sid = engine.create_session().id

        engine.run(sid, "hi")

        cps = memory_store.list_checkpoints(sid)
        labels = [cp.label for cp in cps]
        assert LABEL_POST_LLM in labels

    def test_engine_writes_post_tools_checkpoint(self, memory_store):
        """Engine writes LABEL_POST_TOOLS checkpoint after tool execution."""
        registry = ToolRegistry()
        registry.register(make_calculator())
        tc = ToolCall.new("calculator", {"expression": "2+2"})
        llm = MockProvider([
            LLMResponse(content="", tool_calls=[tc]),
            LLMResponse(content="The answer is 4."),
        ])
        engine = AgentEngine(store=memory_store, llm=llm, tools=registry)
        sid = engine.create_session().id

        engine.run(sid, "What is 2+2?")

        cps = memory_store.list_checkpoints(sid)
        labels = [cp.label for cp in cps]
        assert LABEL_POST_TOOLS in labels

    def test_post_tools_checkpoint_message_count_is_correct(self, memory_store):
        """The post_tools checkpoint's message_count matches the session."""
        registry = ToolRegistry()
        registry.register(make_calculator())
        tc = ToolCall.new("calculator", {"expression": "1+1"})
        llm = MockProvider([
            LLMResponse(content="", tool_calls=[tc]),
            LLMResponse(content="Done."),
        ])
        engine = AgentEngine(store=memory_store, llm=llm, tools=registry)
        sid = engine.create_session().id

        engine.run(sid, "Calculate.")

        # Messages: user(1) + assistant_with_tc(2) + tool_result(3) + final_assistant(4)
        session = memory_store.get(sid)
        post_tools_cps = [cp for cp in memory_store.list_checkpoints(sid) if cp.label == LABEL_POST_TOOLS]
        assert len(post_tools_cps) == 1
        # post_tools is written after tool_result appended, before final LLM call
        assert post_tools_cps[0].message_count == 3


# ---------------------------------------------------------------------------
# Cancellation flag persistence
# ---------------------------------------------------------------------------

class TestCancellationFlagPersistence:
    def test_set_cancellation_true(self, store):
        s = store.create()
        store.set_cancellation(s.id, True)
        assert store.get(s.id).cancelled is True

    def test_set_cancellation_false(self, store):
        s = store.create()
        s.cancelled = True
        store.save(s)
        store.set_cancellation(s.id, False)
        assert store.get(s.id).cancelled is False

    def test_set_cancellation_nonexistent_is_silent(self, store):
        store.set_cancellation("ghost", True)  # should not raise

    def test_cancellation_persists_across_save(self, store):
        s = store.create()
        store.set_cancellation(s.id, True)
        s = store.get(s.id)
        s.messages.append(Message.user("hi"))
        store.save(s)
        assert store.get(s.id).cancelled is True


# ---------------------------------------------------------------------------
# RunStatus / SessionSummary views
# ---------------------------------------------------------------------------

class TestStatusViews:
    def test_run_status_reflects_state(self, store):
        s = store.create()
        status = store.get_run_status(s.id)
        assert isinstance(status, RunStatus)
        assert status.session_id == s.id
        assert status.current_state == SessionState.waiting_for_user
        assert status.is_running is False
        assert status.cancel_requested is False

    def test_run_status_is_running_when_running(self, store):
        s = store.create()
        s.state = SessionState.running
        store.save(s)
        status = store.get_run_status(s.id)
        assert status.is_running is True

    def test_run_status_cancel_requested(self, store):
        s = store.create()
        store.set_cancellation(s.id, True)
        status = store.get_run_status(s.id)
        assert status.cancel_requested is True

    def test_run_status_unknown_returns_none(self, store):
        assert store.get_run_status("no-such-id") is None

    def test_list_sessions_returns_all(self, store):
        store.create()
        store.create()
        summaries = store.list_sessions()
        assert len(summaries) == 2
        assert all(isinstance(s, SessionSummary) for s in summaries)

    def test_session_summary_message_count(self, store):
        s = store.create()
        for i in range(5):
            s.messages.append(Message.user(f"msg {i}"))
        store.save(s)

        summaries = store.list_sessions()
        summary = next(x for x in summaries if x.session_id == s.id)
        assert summary.message_count == 5

    def test_session_summary_last_messages_limited(self, store):
        s = store.create()
        for i in range(10):
            s.messages.append(Message.user(f"msg {i}"))
        store.save(s)

        summaries = store.list_sessions(last_n_messages=3)
        summary = next(x for x in summaries if x.session_id == s.id)
        assert len(summary.last_messages) == 3
        assert summary.last_messages[-1].content == "msg 9"

    def test_session_summary_exposes_required_fields(self, store):
        s = store.create()
        summaries = store.list_sessions()
        summary = summaries[0]
        assert hasattr(summary, "session_id")
        assert hasattr(summary, "is_running")
        assert hasattr(summary, "current_state")
        assert hasattr(summary, "cancel_requested")
        assert hasattr(summary, "last_messages")


# ---------------------------------------------------------------------------
# Resume after interruption
# ---------------------------------------------------------------------------

class TestResumeAfterInterruption:
    def test_continue_run_resumes_from_post_tools_checkpoint(self, memory_store):
        """
        Simulate a crash mid-loop: session is left in 'running' with a
        post_tools checkpoint.  continue_run should re-enter and finish.
        """
        registry = ToolRegistry()
        registry.register(make_calculator())

        tc = ToolCall.new("calculator", {"expression": "3*3"})
        llm = MockProvider([
            LLMResponse(content="", tool_calls=[tc]),
            LLMResponse(content="The answer is 9."),
        ])
        engine = AgentEngine(store=memory_store, llm=llm, tools=registry)
        sid = engine.create_session().id

        # Run fully — checkpoints created
        result = engine.run(sid, "What is 3*3?")
        assert result == "The answer is 9."

        # Simulate crash: manually push state back to 'running'
        # and truncate messages to the post_tools checkpoint count
        session = memory_store.get(sid)
        post_tools_cp = next(
            cp for cp in reversed(memory_store.list_checkpoints(sid))
            if cp.label == LABEL_POST_TOOLS
        )
        session.messages = session.messages[: post_tools_cp.message_count]
        session.state = SessionState.running
        memory_store.save(session)

        # New LLM script for the resumed call (needs the final answer again)
        engine.llm = MockProvider([LLMResponse(content="The answer is 9.")])

        result = engine.continue_run(sid)
        assert result == "The answer is 9."
        assert memory_store.get(sid).state == SessionState.waiting_for_user

    def test_continue_run_no_checkpoint_resets_to_waiting(self, memory_store):
        """If there's no post_tools checkpoint, continue_run resets gracefully."""
        registry = ToolRegistry()
        llm = MockProvider([LLMResponse(content="done")])
        engine = AgentEngine(store=memory_store, llm=llm, tools=registry)
        sid = engine.create_session().id

        # Force into running with no checkpoints
        session = memory_store.get(sid)
        session.state = SessionState.running
        memory_store.save(session)
        # No checkpoints saved

        result = engine.continue_run(sid)
        assert result == ""
        assert memory_store.get(sid).state == SessionState.waiting_for_user

    def test_file_store_survives_process_restart(self, tmp_path):
        """Session created with one FileStore is readable by a new FileStore instance."""
        base = tmp_path / "store"
        store1 = FileStore(base)
        registry = ToolRegistry()
        registry.register(make_calculator())
        llm = MockProvider([LLMResponse(content="Hi.")])
        engine1 = AgentEngine(store=store1, llm=llm, tools=registry)

        sid = engine1.create_session().id
        engine1.run(sid, "Hello.")

        # "New process" — fresh store pointing to same directory
        store2 = FileStore(base)
        session = store2.get(sid)
        assert session is not None
        assert session.id == sid
        assert session.state == SessionState.waiting_for_user
        assert len(session.messages) > 0

    def test_file_store_checkpoints_survive_restart(self, tmp_path):
        base = tmp_path / "store"
        store1 = FileStore(base)
        registry = ToolRegistry()
        registry.register(make_calculator())
        tc = ToolCall.new("calculator", {"expression": "5+5"})
        llm = MockProvider([
            LLMResponse(content="", tool_calls=[tc]),
            LLMResponse(content="Ten."),
        ])
        engine1 = AgentEngine(store=store1, llm=llm, tools=registry)
        sid = engine1.create_session().id
        engine1.run(sid, "What is 5+5?")

        store2 = FileStore(base)
        cps = store2.list_checkpoints(sid)
        assert len(cps) > 0
        labels = [cp.label for cp in cps]
        assert LABEL_POST_TOOLS in labels

    def test_session_switching_reads_from_storage(self, memory_store):
        """
        Two sessions created and modified independently; reading either by ID
        returns the correct state.
        """
        registry = ToolRegistry()
        llm1 = MockProvider([LLMResponse(content="Session A done.")])
        llm2 = MockProvider([LLMResponse(content="Session B done.")])

        engine_a = AgentEngine(store=memory_store, llm=llm1, tools=registry)
        engine_b = AgentEngine(store=memory_store, llm=llm2, tools=registry)

        sid_a = engine_a.create_session().id
        sid_b = engine_b.create_session().id

        engine_a.run(sid_a, "Message to A.")
        engine_b.run(sid_b, "Message to B.")

        session_a = memory_store.get(sid_a)
        session_b = memory_store.get(sid_b)

        assert session_a.messages[-1].content == "Session A done."
        assert session_b.messages[-1].content == "Session B done."
        # Cross-check: reading the other session still works
        assert memory_store.get(sid_a).id == sid_a
        assert memory_store.get(sid_b).id == sid_b

    def test_cancellation_flag_checked_across_engine_instances(self, memory_store):
        """
        cancel() on one engine reference sets the flag; the loop in another
        engine sharing the same store respects it.
        """
        call_count = 0

        class CancelOnFirstCall(MockProvider):
            def complete(self, messages, tools):
                nonlocal call_count
                call_count += 1
                memory_store.set_cancellation(sid, True)
                return super().complete(messages, tools)

        registry = ToolRegistry()
        registry.register(make_calculator())
        tc = ToolCall.new("calculator", {"expression": "1+1"})
        llm = CancelOnFirstCall([
            LLMResponse(content="", tool_calls=[tc]),
            LLMResponse(content="Should not reach."),
        ])
        engine = AgentEngine(store=memory_store, llm=llm, tools=registry)
        sid = engine.create_session().id

        list(engine.stream_run(sid, "go"))

        session = memory_store.get(sid)
        assert session.cancelled is True
        assert session.state == SessionState.finished
        assert call_count == 1


# ---------------------------------------------------------------------------
# FileStore-specific: atomic writes and directory layout
# ---------------------------------------------------------------------------

class TestFileStoreStructure:
    def test_creates_sessions_directory(self, tmp_path):
        FileStore(tmp_path / "store")
        assert (tmp_path / "store" / "sessions").is_dir()

    def test_creates_checkpoints_directory(self, tmp_path):
        FileStore(tmp_path / "store")
        assert (tmp_path / "store" / "checkpoints").is_dir()

    def test_session_file_exists_after_create(self, tmp_path):
        store = FileStore(tmp_path / "store")
        s = store.create()
        assert (tmp_path / "store" / "sessions" / f"{s.id}.json").exists()

    def test_checkpoint_file_exists_after_save(self, tmp_path):
        store = FileStore(tmp_path / "store")
        s = store.create()
        cp = Checkpoint.new(s.id, 0, 1, SessionState.running, LABEL_POST_LLM)
        store.save_checkpoint(cp)
        assert (tmp_path / "store" / "checkpoints" / f"{s.id}.jsonl").exists()

    def test_session_file_deleted_on_delete(self, tmp_path):
        store = FileStore(tmp_path / "store")
        s = store.create()
        path = tmp_path / "store" / "sessions" / f"{s.id}.json"
        assert path.exists()
        store.delete(s.id)
        assert not path.exists()

    def test_session_json_is_valid(self, tmp_path):
        import json as _json
        store = FileStore(tmp_path / "store")
        s = store.create()
        s.messages.append(Message.user("test"))
        store.save(s)
        path = tmp_path / "store" / "sessions" / f"{s.id}.json"
        data = _json.loads(path.read_text())
        assert data["id"] == s.id
        assert len(data["messages"]) == 1

    def test_list_ids_only_returns_json_sessions(self, tmp_path):
        store = FileStore(tmp_path / "store")
        s1 = store.create()
        s2 = store.create()
        # Plant a non-session file in the sessions dir
        (tmp_path / "store" / "sessions" / "garbage.txt").write_text("noise")
        ids = store.list_ids()
        assert set(ids) == {s1.id, s2.id}
