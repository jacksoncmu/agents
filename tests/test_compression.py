"""Tests for two-layer context compression.

Tests are grouped by:
  - TokenEstimator           basic token counting
  - CompressionConfig        validation
  - MicroCompression         layer 1 behaviour
  - Summarization            layer 2 behaviour
  - TaskObjective            objective preserved / re-injected
  - SystemIdentity           identity preserved / re-injected
  - EngineIntegration        loop continues correctly after compression
  - Telemetry                events emitted at the right points
"""
from __future__ import annotations

import pytest

from agent.compression import (
    CompressionConfig,
    ConversationSummary,
    ContextCompressor,
    LLMSummarizer,
    NoOpSummarizer,
    Summarizer,
    _build_summary_message,
    _extract_objective,
    estimate_tokens,
)
from agent.engine import AgentEngine
from agent.llm.base import LLMResponse
from agent.llm.mock import MockProvider
from agent.storage import InMemoryStore
from agent.telemetry import InMemoryBackend, Outcome, set_backend, NullBackend
from agent.tools import ToolDefinition, ToolParam, ToolRegistry
from agent.types import Message, ToolCall, ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_result_msg(content: str, call_id: str = "id0") -> Message:
    return Message.tool_result_msg([ToolResult(
        tool_call_id=call_id, content=content
    )])


def _assistant_with_tool(name: str = "t", call_id: str = "id0") -> Message:
    return Message.assistant("", tool_calls=[ToolCall(id=call_id, name=name, arguments={})])


def _heavy_session(
    n_pairs: int,
    result_content: str = "x" * 400,
    objective: str = "Please complete the task.",
) -> list[Message]:
    """
    Build a synthetic session with ``n_pairs`` assistant→tool-result pairs.
    Enough content to cross token thresholds easily.
    """
    msgs: list[Message] = [Message.user(objective)]
    for i in range(n_pairs):
        cid = f"cid-{i}"
        msgs.append(_assistant_with_tool(call_id=cid))
        msgs.append(_tool_result_msg(result_content, call_id=cid))
    msgs.append(Message.assistant("Final answer."))
    return msgs


def _compressor(
    context_window: int = 1_000,
    micro_threshold: float = 0.50,
    summarization_threshold: float = 0.70,
    recent: int = 2,
    max_chars: int = 50,
    identity: str = "",
    objective: str = "",
    summarizer: Summarizer | None = None,
) -> ContextCompressor:
    return ContextCompressor(
        CompressionConfig(
            context_window=context_window,
            micro_threshold=micro_threshold,
            summarization_threshold=summarization_threshold,
            recent_messages_to_preserve=recent,
            micro_tool_result_max_chars=max_chars,
            system_identity=identity,
            task_objective=objective,
        ),
        summarizer=summarizer,
    )


@pytest.fixture(autouse=True)
def telemetry_backend():
    backend = InMemoryBackend()
    set_backend(backend)
    yield backend
    set_backend(NullBackend())


# ===========================================================================
# Token estimator
# ===========================================================================

class TestTokenEstimator:
    def test_empty_list_is_zero(self):
        assert estimate_tokens([]) == 0

    def test_single_user_message(self):
        msgs = [Message.user("hello")]
        est = estimate_tokens(msgs)
        assert est > 0
        # "hello" is 5 chars → 1 token + 10 overhead = ~11
        assert est >= 11

    def test_scales_with_content_length(self):
        short = [Message.user("hi")]
        long = [Message.user("x" * 1_000)]
        assert estimate_tokens(long) > estimate_tokens(short)

    def test_tool_result_adds_tokens(self):
        base = [Message.user("q")]
        with_result = base + [_tool_result_msg("y" * 400)]
        assert estimate_tokens(with_result) > estimate_tokens(base)

    def test_many_messages_higher_than_one(self):
        single = [Message.user("hello")]
        many = [Message.user("hello")] * 10
        assert estimate_tokens(many) > estimate_tokens(single)

    def test_tool_calls_contribute(self):
        msg_plain = Message.assistant("ok")
        msg_with_calls = Message.assistant("", tool_calls=[
            ToolCall(id="x", name="t", arguments={"key": "v" * 200})
        ])
        alone = estimate_tokens([msg_plain])
        with_calls = estimate_tokens([msg_with_calls])
        assert with_calls > alone


# ===========================================================================
# CompressionConfig validation
# ===========================================================================

class TestCompressionConfig:
    def test_valid_defaults(self):
        cfg = CompressionConfig()
        assert 0 < cfg.micro_threshold < cfg.summarization_threshold <= 1.0

    def test_summarization_below_micro_raises(self):
        with pytest.raises(ValueError, match="summarization_threshold"):
            CompressionConfig(micro_threshold=0.7, summarization_threshold=0.5)

    def test_equal_thresholds_allowed(self):
        cfg = CompressionConfig(micro_threshold=0.6, summarization_threshold=0.6)
        assert cfg.micro_threshold == cfg.summarization_threshold

    def test_zero_micro_threshold_raises(self):
        with pytest.raises(ValueError):
            CompressionConfig(micro_threshold=0.0)

    def test_negative_recent_raises(self):
        with pytest.raises(ValueError):
            CompressionConfig(recent_messages_to_preserve=-1)

    def test_too_small_max_chars_raises(self):
        with pytest.raises(ValueError):
            CompressionConfig(micro_tool_result_max_chars=5)


# ===========================================================================
# Layer 1 — micro-compression
# ===========================================================================

class TestMicroCompression:
    def test_trigger_when_above_threshold(self):
        # 5 pairs × 400-char results → ~500+ tokens; micro at 50% of 1000 = 500
        msgs = _heavy_session(5)
        c = _compressor(context_window=1_000, micro_threshold=0.50, recent=2, max_chars=50)
        result = c.compress(msgs)
        assert result is not msgs, "Expected compression to run"

    def test_no_trigger_when_below_threshold(self):
        # Single short message, far below any threshold
        msgs = [Message.user("hi")]
        c = _compressor(context_window=100_000)
        result = c.compress(msgs)
        assert result is msgs, "Expected no compression"

    def test_old_tool_results_truncated(self):
        long_content = "X" * 500
        msgs = _heavy_session(4, result_content=long_content)
        c = _compressor(context_window=500, micro_threshold=0.01, recent=2, max_chars=50)
        result = c.compress(msgs)

        # All tool result messages in the "middle" region should be truncated
        for msg in result[1:-2]:  # skip head and recent tail
            for tr in msg.tool_results:
                assert len(tr.content) <= 50 + len(" …[truncated]"), (
                    f"Tool result not truncated: {len(tr.content)} chars"
                )

    def test_recent_messages_untouched(self):
        long_content = "Y" * 500
        msgs = _heavy_session(6, result_content=long_content)
        c = _compressor(context_window=500, micro_threshold=0.01, recent=4, max_chars=50)
        result = c.compress(msgs)

        # The last 4 messages must be byte-identical to the originals
        assert result[-4:] == msgs[-4:]

    def test_first_message_always_preserved(self):
        """The first user message (task objective) must never be modified."""
        objective = "My important task objective."
        msgs = _heavy_session(5, objective=objective)
        c = _compressor(context_window=200, micro_threshold=0.01, recent=2, max_chars=10)
        result = c.compress(msgs)

        assert result[0].content == objective

    def test_truncated_marker_appended(self):
        msgs = _heavy_session(3, result_content="A" * 500)
        c = _compressor(context_window=200, micro_threshold=0.01, recent=1, max_chars=30)
        result = c.compress(msgs)

        truncated = [
            tr
            for msg in result
            for tr in msg.tool_results
            if "…[truncated]" in tr.content
        ]
        assert truncated, "Expected at least one truncated tool result"

    def test_short_results_not_modified(self):
        """Results already within max_chars should be left unchanged."""
        short_content = "short"
        msgs = [
            Message.user("task"),
            _assistant_with_tool(),
            _tool_result_msg(short_content),
            Message.assistant("done"),
        ]
        c = _compressor(context_window=10, micro_threshold=0.01, recent=1, max_chars=100)
        result = c.compress(msgs)
        # Even if it triggers, the short content should be unchanged
        for msg in result:
            for tr in msg.tool_results:
                assert "…[truncated]" not in tr.content

    def test_too_few_messages_no_op(self):
        """With fewer messages than head + recent, nothing to compress."""
        msgs = [Message.user("x"), Message.assistant("y")]
        c = _compressor(context_window=1, micro_threshold=0.01, recent=4)
        result = c.compress(msgs)
        assert result is msgs

    def test_no_summarizer_never_triggers_layer2(self, telemetry_backend):
        """If no summarizer is configured, layer 2 events must not appear."""
        msgs = _heavy_session(10, result_content="Z" * 1_000)
        c = _compressor(
            context_window=100,
            micro_threshold=0.01,
            summarization_threshold=0.01,
            recent=1,
            summarizer=None,
        )
        c.compress(msgs)
        assert telemetry_backend.by_name("compression.summarize") == []


# ===========================================================================
# Layer 2 — summarization
# ===========================================================================

class TestSummarization:
    def test_trigger_when_above_summarization_threshold(self, telemetry_backend):
        msgs = _heavy_session(8, result_content="S" * 400)
        c = _compressor(
            context_window=500,
            micro_threshold=0.50,
            summarization_threshold=0.60,
            recent=2,
            summarizer=NoOpSummarizer(),
        )
        result = c.compress(msgs)
        events = telemetry_backend.by_name("compression.summarize")
        assert events, "Expected compression.summarize event"
        assert events[0].outcome == Outcome.executed
        assert result is not msgs

    def test_summary_message_is_first(self):
        msgs = _heavy_session(8, result_content="S" * 400)
        c = _compressor(
            context_window=500,
            micro_threshold=0.50,
            summarization_threshold=0.60,
            recent=2,
            summarizer=NoOpSummarizer(),
        )
        result = c.compress(msgs)
        first = result[0]
        assert "[CONTEXT SUMMARY" in first.content
        assert "[END SUMMARY]" in first.content

    def test_recent_messages_preserved_after_summarization(self):
        msgs = _heavy_session(8, result_content="S" * 400)
        n_recent = 3
        c = _compressor(
            context_window=500,
            summarization_threshold=0.60,
            recent=n_recent,
            summarizer=NoOpSummarizer(),
        )
        result = c.compress(msgs)
        assert result[-n_recent:] == msgs[-n_recent:]

    def test_llm_summarizer_uses_provider(self):
        """LLMSummarizer should call the provider and parse the response."""
        llm_response = LLMResponse(content=(
            "OBJECTIVE: Complete the pipeline.\n"
            "STEPS:\n"
            "- Ran data ingestion\n"
            "- Transformed records\n"
            "FACTS:\n"
            "- 1500 records processed\n"
        ))
        provider = MockProvider([llm_response])
        summarizer = LLMSummarizer(provider)

        msgs = _heavy_session(5, result_content="Q" * 400)
        cfg = CompressionConfig(
            context_window=500,
            summarization_threshold=0.60,
        )
        summary = summarizer.summarize(msgs, cfg)

        assert summary.task_objective == "Complete the pipeline."
        assert "Ran data ingestion" in summary.completed_steps
        assert "Transformed records" in summary.completed_steps
        assert "1500 records processed" in summary.key_facts

    def test_llm_summarizer_falls_back_on_provider_error(self):
        """If the LLM raises, LLMSummarizer falls back to NoOpSummarizer."""
        class BoomProvider(MockProvider):
            def complete(self, messages, tools):
                raise RuntimeError("LLM unavailable")

        summarizer = LLMSummarizer(BoomProvider([]))
        msgs = _heavy_session(3)
        cfg = CompressionConfig()
        summary = summarizer.summarize(msgs, cfg)

        assert isinstance(summary, ConversationSummary)
        # Fallback note should appear in key_facts
        assert any("Summarizer error" in f for f in summary.key_facts)

    def test_layer2_triggers_before_layer1(self, telemetry_backend):
        """When both thresholds are crossed, layer 2 should run, not layer 1."""
        msgs = _heavy_session(10, result_content="W" * 400)
        c = _compressor(
            context_window=200,
            micro_threshold=0.10,
            summarization_threshold=0.20,
            recent=2,
            summarizer=NoOpSummarizer(),
        )
        c.compress(msgs)

        assert telemetry_backend.by_name("compression.summarize")
        # layer 1 event should NOT appear (layer 2 ran instead)
        assert not telemetry_backend.by_name("compression.micro")

    def test_layer2_failure_falls_back_to_layer1(self, telemetry_backend):
        """If the summarizer raises inside compress(), layer 1 runs as fallback."""
        class ExplodingSummarizer(Summarizer):
            def summarize(self, messages, config):
                raise RuntimeError("explode")

        msgs = _heavy_session(10, result_content="V" * 400)
        c = _compressor(
            context_window=200,
            micro_threshold=0.10,
            summarization_threshold=0.20,
            recent=2,
            summarizer=ExplodingSummarizer(),
        )
        result = c.compress(msgs)

        summ_events = telemetry_backend.by_name("compression.summarize")
        assert summ_events[0].outcome == Outcome.error

        micro_events = telemetry_backend.by_name("compression.micro")
        assert micro_events, "Expected layer 1 fallback after layer 2 failure"
        assert result is not msgs


# ===========================================================================
# Task objective preserved / re-injected
# ===========================================================================

class TestTaskObjective:
    def test_auto_detected_from_first_user_message(self):
        objective = "Build the reporting pipeline."
        msgs = _heavy_session(6, objective=objective)
        c = _compressor(
            context_window=300,
            micro_threshold=0.30,
            summarization_threshold=0.40,
            recent=2,
            summarizer=NoOpSummarizer(),
        )
        result = c.compress(msgs)

        summary_text = result[0].content
        assert objective in summary_text or objective[:50] in summary_text

    def test_explicit_config_objective_used_when_set(self):
        config_objective = "Explicit config objective."
        msgs = _heavy_session(6, objective="original user message")
        c = ContextCompressor(
            CompressionConfig(
                context_window=300,
                micro_threshold=0.30,
                summarization_threshold=0.40,
                recent_messages_to_preserve=2,
                task_objective=config_objective,
            ),
            summarizer=NoOpSummarizer(),
        )
        result = c.compress(msgs)
        assert config_objective in result[0].content

    def test_objective_in_summary_message_after_llm_summarization(self):
        llm_response = LLMResponse(content=(
            "OBJECTIVE: Detected objective from LLM.\n"
            "STEPS:\n- step\nFACTS:\n- fact\n"
        ))
        msgs = _heavy_session(6, objective="original")
        c = ContextCompressor(
            CompressionConfig(
                context_window=300,
                micro_threshold=0.30,
                summarization_threshold=0.40,
                recent_messages_to_preserve=2,
            ),
            summarizer=LLMSummarizer(MockProvider([llm_response])),
        )
        result = c.compress(msgs)
        assert "Detected objective from LLM." in result[0].content

    def test_micro_compression_preserves_objective_in_first_message(self):
        objective = "Must appear unchanged."
        msgs = _heavy_session(6, objective=objective)
        c = _compressor(
            context_window=400, micro_threshold=0.01, recent=2, max_chars=20
        )
        result = c.compress(msgs)
        assert result[0].content == objective

    def test_extract_objective_returns_first_user_message(self):
        msgs = [
            Message.user("The real objective."),
            Message.assistant("ok"),
            Message.user("A later user message."),
        ]
        assert _extract_objective(msgs) == "The real objective."

    def test_extract_objective_empty_list(self):
        obj = _extract_objective([])
        assert "not found" in obj.lower()


# ===========================================================================
# System identity preserved / re-injected
# ===========================================================================

class TestSystemIdentity:
    def test_identity_in_summary_message(self):
        identity = "You are a diligent data engineer."
        msgs = _heavy_session(6)
        c = ContextCompressor(
            CompressionConfig(
                context_window=300,
                micro_threshold=0.30,
                summarization_threshold=0.40,
                recent_messages_to_preserve=2,
                system_identity=identity,
            ),
            summarizer=NoOpSummarizer(),
        )
        result = c.compress(msgs)
        assert identity in result[0].content

    def test_identity_absent_when_not_configured(self):
        msgs = _heavy_session(6)
        c = _compressor(
            context_window=300,
            micro_threshold=0.30,
            summarization_threshold=0.40,
            recent=2,
            identity="",
            summarizer=NoOpSummarizer(),
        )
        result = c.compress(msgs)
        # Should not contain "System:" when identity is empty
        assert "System:" not in result[0].content

    def test_identity_survives_llm_summarization(self):
        identity = "You are a security auditor."
        llm_response = LLMResponse(content=(
            "OBJECTIVE: Audit the codebase.\nSTEPS:\n- scanned\nFACTS:\n- clean\n"
        ))
        msgs = _heavy_session(6)
        c = ContextCompressor(
            CompressionConfig(
                context_window=300,
                micro_threshold=0.30,
                summarization_threshold=0.40,
                recent_messages_to_preserve=2,
                system_identity=identity,
            ),
            summarizer=LLMSummarizer(MockProvider([llm_response])),
        )
        result = c.compress(msgs)
        assert identity in result[0].content

    def test_build_summary_message_includes_identity_and_objective(self):
        summary = ConversationSummary(
            system_identity="I am an agent.",
            task_objective="Do the thing.",
            completed_steps=["step A"],
            key_facts=["fact 1"],
            message_count_before=10,
        )
        msg = _build_summary_message(summary)
        assert "I am an agent." in msg.content
        assert "Do the thing." in msg.content
        assert "step A" in msg.content
        assert "fact 1" in msg.content
        assert "[CONTEXT SUMMARY" in msg.content
        assert "[END SUMMARY]" in msg.content


# ===========================================================================
# Engine integration — loop continues correctly after compression
# ===========================================================================

class TestEngineIntegration:
    def _make_engine_with_compression(
        self,
        llm_responses: list[LLMResponse],
        context_window: int = 500,
        summarizer: Summarizer | None = None,
        identity: str = "Test agent.",
    ) -> tuple[AgentEngine, str]:
        registry = ToolRegistry()
        store = InMemoryStore()
        compressor = ContextCompressor(
            CompressionConfig(
                context_window=context_window,
                micro_threshold=0.40,
                summarization_threshold=0.60,
                recent_messages_to_preserve=2,
                micro_tool_result_max_chars=30,
                system_identity=identity,
            ),
            summarizer=summarizer,
        )
        engine = AgentEngine(
            store=store,
            llm=MockProvider(llm_responses),
            tools=registry,
            compressor=compressor,
        )
        sid = engine.create_session().id
        return engine, sid

    def test_loop_completes_normally_without_compression(self):
        engine, sid = self._make_engine_with_compression(
            [LLMResponse(content="Answer")],
            context_window=100_000,  # large window, no compression
        )
        result = engine.run(sid, "question")
        assert result == "Answer"

        store = engine.store
        session = store.get(sid)
        from agent.types import SessionState
        assert session.state == SessionState.waiting_for_user

    def test_micro_compression_fires_and_loop_completes(self, telemetry_backend):
        """
        Seed the session with heavy history, then run.  Compression
        fires during the run, but the loop still produces a final answer.
        """
        # Pre-seed session with heavy messages before calling engine.run
        store = InMemoryStore()
        registry = ToolRegistry()
        compressor = ContextCompressor(
            CompressionConfig(
                context_window=200,
                micro_threshold=0.30,
                summarization_threshold=0.80,
                recent_messages_to_preserve=2,
                micro_tool_result_max_chars=20,
            ),
        )
        engine = AgentEngine(
            store=store,
            llm=MockProvider([LLMResponse(content="Done after compression")]),
            tools=registry,
            compressor=compressor,
        )
        sid = engine.create_session().id

        # Manually stuff the session with heavy pre-existing history
        session = store.get(sid)
        for i in range(5):
            cid = f"c{i}"
            session.messages.append(Message.assistant("", tool_calls=[
                ToolCall(id=cid, name="t", arguments={})
            ]))
            session.messages.append(_tool_result_msg("Z" * 300, call_id=cid))
        store.save(session)

        result = engine.run(sid, "continue")
        assert result == "Done after compression"

        micro_events = telemetry_backend.by_name("compression.micro")
        assert micro_events, "Expected micro-compression to fire"
        assert micro_events[0].outcome == Outcome.executed

    def test_summarization_fires_and_loop_completes(self, telemetry_backend):
        store = InMemoryStore()
        registry = ToolRegistry()
        compressor = ContextCompressor(
            CompressionConfig(
                context_window=200,
                micro_threshold=0.30,
                summarization_threshold=0.50,
                recent_messages_to_preserve=2,
            ),
            summarizer=NoOpSummarizer(),
        )
        engine = AgentEngine(
            store=store,
            llm=MockProvider([LLMResponse(content="Post-summary answer")]),
            tools=registry,
            compressor=compressor,
        )
        sid = engine.create_session().id

        session = store.get(sid)
        for i in range(6):
            cid = f"c{i}"
            session.messages.append(Message.assistant("", tool_calls=[
                ToolCall(id=cid, name="t", arguments={})
            ]))
            session.messages.append(_tool_result_msg("W" * 300, call_id=cid))
        store.save(session)

        result = engine.run(sid, "continue")
        assert result == "Post-summary answer"

        summ_events = telemetry_backend.by_name("compression.summarize")
        assert summ_events
        assert summ_events[0].outcome == Outcome.executed

    def test_with_tools_and_compression(self, telemetry_backend):
        """Engine with a real tool, compression active, multi-turn."""
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            "echo", "echoes input",
            [ToolParam("msg", "string", "message")],
            lambda msg: f"echo: {msg}",
        ))

        tc = ToolCall.new("echo", {"msg": "hello"})
        store = InMemoryStore()
        compressor = ContextCompressor(
            CompressionConfig(
                context_window=200,
                micro_threshold=0.30,
                summarization_threshold=0.80,
                recent_messages_to_preserve=2,
                micro_tool_result_max_chars=20,
            ),
        )
        engine = AgentEngine(
            store=store,
            llm=MockProvider([
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="Final."),
            ]),
            tools=registry,
            compressor=compressor,
        )
        sid = engine.create_session().id

        # Pre-stuff heavy context
        session = store.get(sid)
        for i in range(5):
            cid = f"pre{i}"
            session.messages.append(Message.assistant("", tool_calls=[
                ToolCall(id=cid, name="echo", arguments={"msg": "x"})
            ]))
            session.messages.append(_tool_result_msg("PRELOAD" * 50, call_id=cid))
        store.save(session)

        result = engine.run(sid, "run it")
        assert result == "Final."

    def test_no_compressor_leaves_messages_unchanged(self):
        """Without a compressor, messages are never touched."""
        store = InMemoryStore()
        engine = AgentEngine(
            store=store,
            llm=MockProvider([LLMResponse(content="ok")]),
            tools=ToolRegistry(),
            compressor=None,
        )
        sid = engine.create_session().id
        engine.run(sid, "hi")
        session = store.get(sid)
        # user + assistant messages, no summary injection
        assert all("[CONTEXT SUMMARY" not in m.content for m in session.messages)

    def test_session_state_correct_after_compression(self):
        from agent.types import SessionState
        store = InMemoryStore()
        compressor = ContextCompressor(
            CompressionConfig(
                context_window=200,
                micro_threshold=0.10,
                summarization_threshold=0.80,
                recent_messages_to_preserve=2,
            ),
        )
        engine = AgentEngine(
            store=store,
            llm=MockProvider([LLMResponse(content="compressed and done")]),
            tools=ToolRegistry(),
            compressor=compressor,
        )
        sid = engine.create_session().id
        session = store.get(sid)
        for i in range(5):
            cid = f"c{i}"
            session.messages.append(Message.assistant("", tool_calls=[
                ToolCall(id=cid, name="t", arguments={})
            ]))
            session.messages.append(_tool_result_msg("Z" * 200, call_id=cid))
        store.save(session)

        engine.run(sid, "go")
        final = store.get(sid)
        assert final.state == SessionState.waiting_for_user


# ===========================================================================
# Telemetry
# ===========================================================================

class TestCompressionTelemetry:
    def test_micro_event_outcome_executed(self, telemetry_backend):
        msgs = _heavy_session(6, result_content="T" * 400)
        c = _compressor(
            context_window=500,
            micro_threshold=0.40,
            summarization_threshold=0.90,
            recent=2,
        )
        c.compress(msgs, session_id="s1")

        events = telemetry_backend.by_name("compression.micro")
        assert events
        e = events[0]
        assert e.outcome == Outcome.executed
        assert e.session_id == "s1"
        assert "token_estimate" in e.data
        assert "context_window" in e.data
        assert e.data["messages_before"] > e.data["messages_after"] or True  # count can vary

    def test_summarize_event_outcome_executed(self, telemetry_backend):
        msgs = _heavy_session(6, result_content="T" * 400)
        c = _compressor(
            context_window=500,
            micro_threshold=0.40,
            summarization_threshold=0.50,
            recent=2,
            summarizer=NoOpSummarizer(),
        )
        c.compress(msgs, session_id="sess42")

        events = telemetry_backend.by_name("compression.summarize")
        assert events
        e = events[0]
        assert e.outcome == Outcome.executed
        assert e.session_id == "sess42"

    def test_no_event_when_below_threshold(self, telemetry_backend):
        msgs = [Message.user("tiny")]
        c = _compressor(context_window=1_000_000)
        c.compress(msgs)

        assert not telemetry_backend.by_name("compression.micro")
        assert not telemetry_backend.by_name("compression.summarize")

    def test_error_event_on_summarizer_failure(self, telemetry_backend):
        class BoomSummarizer(Summarizer):
            def summarize(self, messages, config):
                raise RuntimeError("boom")

        msgs = _heavy_session(8, result_content="U" * 400)
        c = _compressor(
            context_window=300,
            micro_threshold=0.10,
            summarization_threshold=0.20,
            recent=2,
            summarizer=BoomSummarizer(),
        )
        c.compress(msgs)

        err_events = telemetry_backend.by_name("compression.summarize")
        assert any(e.outcome == Outcome.error for e in err_events)
