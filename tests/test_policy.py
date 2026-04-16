"""Tests for the policy / rules engine.

Covers:
  - PolicyEngine.evaluate() unit tests (all verdicts, scoping, ordering)
  - Engine integration: block, require_confirmation, inject_reminder, allow
  - Protocol ordering: tool_result always follows assistant message
  - Telemetry events emitted correctly
  - No-policy-engine path is unchanged
"""
from __future__ import annotations

import pytest

from agent.engine import AgentEngine
from agent.llm.base import LLMResponse
from agent.llm.mock import MockProvider
from agent.policy import PolicyEngine, PolicyResult, PolicyRule, PolicyVerdict
from agent.storage import InMemoryStore
from agent.telemetry import InMemoryBackend, set_backend
from agent.tools import ToolContext, ToolDefinition, ToolParam, ToolRegistry
from agent.types import MessageRole, SessionState, ToolCall
from examples.example_tools import make_calculator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_allow() -> PolicyResult:
    return PolicyResult(verdict=PolicyVerdict.allow)


def _make_block(reason: str = "blocked", msg: str = "") -> PolicyResult:
    return PolicyResult(
        verdict=PolicyVerdict.block,
        reason=reason,
        block_message=msg or reason,
    )


def _make_confirmation(reason: str = "needs confirm") -> PolicyResult:
    return PolicyResult(verdict=PolicyVerdict.require_confirmation, reason=reason)


def _make_reminder(reminder: str, reason: str = "") -> PolicyResult:
    return PolicyResult(
        verdict=PolicyVerdict.inject_reminder,
        reason=reason or "reminder injected",
        reminder=reminder,
    )


def _tool_names_called(session) -> list[str]:
    return [tc.name for m in session.messages for tc in m.tool_calls]


def _tool_result_contents(session) -> list[str]:
    return [tr.content for m in session.messages for tr in m.tool_results]


def _make_spy_tool(name: str = "spy_tool") -> tuple[ToolDefinition, list[dict]]:
    """Returns (tool_definition, calls_log). calls_log is mutated on each execution."""
    calls: list[dict] = []

    def handler(*, ctx: ToolContext) -> str:
        calls.append({"executed": True})
        return "spy_result"

    tool = ToolDefinition(
        name=name,
        description="A spy tool that records executions.",
        params=[],
        handler=handler,
    )
    return tool, calls


def _build_engine(
    responses: list[LLMResponse],
    extra_tools: list[ToolDefinition] | None = None,
    policy: PolicyEngine | None = None,
) -> tuple[AgentEngine, str]:
    registry = ToolRegistry()
    registry.register(make_calculator())
    for t in extra_tools or []:
        registry.register(t)

    engine = AgentEngine(
        store=InMemoryStore(),
        llm=MockProvider(responses),
        tools=registry,
        policy=policy,
    )
    session = engine.create_session()
    return engine, session.id


# ===========================================================================
# Unit tests: PolicyEngine.evaluate()
# ===========================================================================

class TestPolicyEngineEvaluate:

    def test_no_rules_returns_allow(self):
        engine = PolicyEngine()
        tc = ToolCall.new("calculator", {"expression": "1+1"})
        result = engine.evaluate(tc, [])
        assert result.verdict == PolicyVerdict.allow

    def test_single_allow_rule_returns_allow(self):
        engine = PolicyEngine()
        engine.register(PolicyRule.for_tools(
            "calculator", name="r", handler=lambda tc, msgs: _make_allow(),
        ))
        tc = ToolCall.new("calculator", {"expression": "1+1"})
        result = engine.evaluate(tc, [])
        assert result.verdict == PolicyVerdict.allow

    def test_block_rule_returns_block(self):
        engine = PolicyEngine()
        engine.register(PolicyRule.for_tools(
            "calculator", name="r",
            handler=lambda tc, msgs: _make_block("not allowed"),
        ))
        tc = ToolCall.new("calculator", {})
        result = engine.evaluate(tc, [])
        assert result.verdict == PolicyVerdict.block
        assert result.reason == "not allowed"

    def test_require_confirmation_rule(self):
        engine = PolicyEngine()
        engine.register(PolicyRule.for_tools(
            "calculator", name="r",
            handler=lambda tc, msgs: _make_confirmation("confirm first"),
        ))
        tc = ToolCall.new("calculator", {})
        result = engine.evaluate(tc, [])
        assert result.verdict == PolicyVerdict.require_confirmation
        assert result.reason == "confirm first"

    def test_inject_reminder_rule(self):
        engine = PolicyEngine()
        engine.register(PolicyRule.for_tools(
            "calculator", name="r",
            handler=lambda tc, msgs: _make_reminder("check eligibility"),
        ))
        tc = ToolCall.new("calculator", {})
        result = engine.evaluate(tc, [])
        assert result.verdict == PolicyVerdict.inject_reminder
        assert result.reminder == "check eligibility"

    def test_rule_scoped_to_tool_fires_for_matching_tool(self):
        engine = PolicyEngine()
        engine.register(PolicyRule.for_tools(
            "target_tool", name="r",
            handler=lambda tc, msgs: _make_block("blocked"),
        ))
        tc = ToolCall.new("target_tool", {})
        result = engine.evaluate(tc, [])
        assert result.verdict == PolicyVerdict.block

    def test_rule_scoped_to_tool_does_not_fire_for_other_tool(self):
        engine = PolicyEngine()
        engine.register(PolicyRule.for_tools(
            "target_tool", name="r",
            handler=lambda tc, msgs: _make_block("blocked"),
        ))
        tc = ToolCall.new("other_tool", {})
        result = engine.evaluate(tc, [])
        assert result.verdict == PolicyVerdict.allow

    def test_wildcard_rule_fires_for_any_tool(self):
        engine = PolicyEngine()
        engine.register(PolicyRule.for_all_tools(
            name="global_block",
            handler=lambda tc, msgs: _make_block("all blocked"),
        ))
        for name in ("calculator", "any_tool", "xyz"):
            tc = ToolCall.new(name, {})
            result = engine.evaluate(tc, [])
            assert result.verdict == PolicyVerdict.block, f"Expected block for {name!r}"

    def test_first_non_allow_wins(self):
        engine = PolicyEngine()
        engine.register(PolicyRule.for_tools(
            "calculator", name="r1",
            handler=lambda tc, msgs: _make_allow(),
        ))
        engine.register(PolicyRule.for_tools(
            "calculator", name="r2",
            handler=lambda tc, msgs: _make_block("second rule blocks"),
        ))
        engine.register(PolicyRule.for_tools(
            "calculator", name="r3",
            handler=lambda tc, msgs: _make_reminder("third rule reminder"),
        ))
        tc = ToolCall.new("calculator", {})
        result = engine.evaluate(tc, [])
        assert result.verdict == PolicyVerdict.block
        assert result.reason == "second rule blocks"

    def test_multiple_allow_rules_all_pass_through(self):
        engine = PolicyEngine()
        for i in range(5):
            engine.register(PolicyRule.for_tools(
                "calculator", name=f"r{i}",
                handler=lambda tc, msgs: _make_allow(),
            ))
        tc = ToolCall.new("calculator", {})
        result = engine.evaluate(tc, [])
        assert result.verdict == PolicyVerdict.allow

    def test_rule_receives_tool_call_arguments(self):
        seen_args: list[dict] = []
        def handler(tc: ToolCall, msgs) -> PolicyResult:
            seen_args.append(dict(tc.arguments))
            return _make_allow()

        engine = PolicyEngine()
        engine.register(PolicyRule.for_tools("calculator", name="r", handler=handler))
        tc = ToolCall.new("calculator", {"expression": "2+2"})
        engine.evaluate(tc, [])
        assert seen_args == [{"expression": "2+2"}]

    def test_rule_receives_session_messages(self):
        from agent.types import Message
        seen_msg_counts: list[int] = []

        def handler(tc: ToolCall, msgs) -> PolicyResult:
            seen_msg_counts.append(len(msgs))
            return _make_allow()

        engine = PolicyEngine()
        engine.register(PolicyRule.for_all_tools(name="r", handler=handler))
        msgs = [Message.user("hi"), Message.user("there")]
        tc = ToolCall.new("any_tool", {})
        engine.evaluate(tc, msgs)
        assert seen_msg_counts == [2]

    def test_rules_property_returns_copy(self):
        engine = PolicyEngine()
        engine.register(PolicyRule.for_all_tools(
            name="r", handler=lambda tc, msgs: _make_allow(),
        ))
        rules = engine.rules
        rules.clear()
        assert len(engine.rules) == 1  # original unaffected


# ===========================================================================
# Integration tests: engine respects policy verdicts
# ===========================================================================

class TestPolicyBlock:

    def test_block_prevents_tool_handler_from_executing(self):
        spy, calls = _make_spy_tool("spy_tool")
        policy = PolicyEngine()
        policy.register(PolicyRule.for_tools(
            "spy_tool", name="r",
            handler=lambda tc, msgs: _make_block("no spy"),
        ))
        tc = ToolCall.new("spy_tool", {})
        engine, sid = _build_engine(
            responses=[
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="Blocked."),
            ],
            extra_tools=[spy],
            policy=policy,
        )
        engine.run(sid, "run spy")
        assert calls == [], "Tool handler must NOT be called when policy blocks"

    def test_block_injects_error_tool_result(self):
        spy, _ = _make_spy_tool("spy_tool")
        policy = PolicyEngine()
        policy.register(PolicyRule.for_tools(
            "spy_tool", name="r",
            handler=lambda tc, msgs: _make_block("blocked", msg="Operation not permitted."),
        ))
        tc = ToolCall.new("spy_tool", {})
        engine, sid = _build_engine(
            responses=[
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="Got blocked."),
            ],
            extra_tools=[spy],
            policy=policy,
        )
        engine.run(sid, "run spy")
        session = engine.store.get(sid)
        tr_contents = _tool_result_contents(session)
        assert len(tr_contents) == 1
        assert "Operation not permitted." in tr_contents[0]

    def test_block_tool_result_has_error_flag(self):
        spy, _ = _make_spy_tool("spy_tool")
        policy = PolicyEngine()
        policy.register(PolicyRule.for_tools(
            "spy_tool", name="r",
            handler=lambda tc, msgs: _make_block("no"),
        ))
        tc = ToolCall.new("spy_tool", {})
        engine, sid = _build_engine(
            responses=[
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="Done."),
            ],
            extra_tools=[spy],
            policy=policy,
        )
        engine.run(sid, "go")
        session = engine.store.get(sid)
        tr = session.messages[2].tool_results[0]
        assert tr.error is True

    def test_block_uses_block_message_over_reason(self):
        spy, _ = _make_spy_tool("spy_tool")
        policy = PolicyEngine()
        policy.register(PolicyRule.for_tools(
            "spy_tool", name="r",
            handler=lambda tc, msgs: PolicyResult(
                verdict=PolicyVerdict.block,
                reason="internal reason",
                block_message="user-facing message",
            ),
        ))
        tc = ToolCall.new("spy_tool", {})
        engine, sid = _build_engine(
            responses=[
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="Done."),
            ],
            extra_tools=[spy],
            policy=policy,
        )
        engine.run(sid, "go")
        session = engine.store.get(sid)
        content = session.messages[2].tool_results[0].content
        assert "user-facing message" in content
        assert "internal reason" not in content

    def test_block_falls_back_to_reason_when_no_block_message(self):
        spy, _ = _make_spy_tool("spy_tool")
        policy = PolicyEngine()
        policy.register(PolicyRule.for_tools(
            "spy_tool", name="r",
            handler=lambda tc, msgs: PolicyResult(
                verdict=PolicyVerdict.block,
                reason="reason only",
            ),
        ))
        tc = ToolCall.new("spy_tool", {})
        engine, sid = _build_engine(
            responses=[
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="Done."),
            ],
            extra_tools=[spy],
            policy=policy,
        )
        engine.run(sid, "go")
        session = engine.store.get(sid)
        content = session.messages[2].tool_results[0].content
        assert "reason only" in content

    def test_block_emits_telemetry_events(self):
        backend = InMemoryBackend()
        set_backend(backend)
        try:
            spy, _ = _make_spy_tool("spy_tool")
            policy = PolicyEngine()
            policy.register(PolicyRule.for_tools(
                "spy_tool", name="r",
                handler=lambda tc, msgs: _make_block("blocked"),
            ))
            tc = ToolCall.new("spy_tool", {})
            engine, sid = _build_engine(
                responses=[
                    LLMResponse(content="", tool_calls=[tc]),
                    LLMResponse(content="Done."),
                ],
                extra_tools=[spy],
                policy=policy,
            )
            engine.run(sid, "go")

            names = [e.name for e in backend.events]
            assert "policy.evaluated" in names
            assert "policy.blocked" in names
        finally:
            from agent.telemetry import NullBackend
            set_backend(NullBackend())


class TestPolicyRequireConfirmation:

    def test_require_confirmation_pauses_session(self):
        spy, _ = _make_spy_tool("spy_tool")
        policy = PolicyEngine()
        policy.register(PolicyRule.for_tools(
            "spy_tool", name="r",
            handler=lambda tc, msgs: _make_confirmation("confirm please"),
        ))
        tc = ToolCall.new("spy_tool", {})
        engine, sid = _build_engine(
            responses=[
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="Done."),
            ],
            extra_tools=[spy],
            policy=policy,
        )
        engine.run(sid, "go")
        session = engine.store.get(sid)
        assert session.state == SessionState.waiting_for_confirmation

    def test_require_confirmation_populates_pending_confirmation(self):
        spy, _ = _make_spy_tool("spy_tool")
        policy = PolicyEngine()
        policy.register(PolicyRule.for_tools(
            "spy_tool", name="r",
            handler=lambda tc, msgs: _make_confirmation("confirm"),
        ))
        tc = ToolCall.new("spy_tool", {})
        engine, sid = _build_engine(
            responses=[
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="Done."),
            ],
            extra_tools=[spy],
            policy=policy,
        )
        engine.run(sid, "go")
        session = engine.store.get(sid)
        assert len(session.pending_confirmation) == 1
        assert session.pending_confirmation[0].name == "spy_tool"

    def test_require_confirmation_tool_not_executed_before_resume(self):
        spy, calls = _make_spy_tool("spy_tool")
        policy = PolicyEngine()
        policy.register(PolicyRule.for_tools(
            "spy_tool", name="r",
            handler=lambda tc, msgs: _make_confirmation("confirm"),
        ))
        tc = ToolCall.new("spy_tool", {})
        engine, sid = _build_engine(
            responses=[
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="Done."),
            ],
            extra_tools=[spy],
            policy=policy,
        )
        engine.run(sid, "go")
        assert calls == [], "Tool handler must not run before confirmation"

    def test_require_confirmation_emits_telemetry(self):
        backend = InMemoryBackend()
        set_backend(backend)
        try:
            spy, _ = _make_spy_tool("spy_tool")
            policy = PolicyEngine()
            policy.register(PolicyRule.for_tools(
                "spy_tool", name="r",
                handler=lambda tc, msgs: _make_confirmation("confirm"),
            ))
            tc = ToolCall.new("spy_tool", {})
            engine, sid = _build_engine(
                responses=[
                    LLMResponse(content="", tool_calls=[tc]),
                    LLMResponse(content="Done."),
                ],
                extra_tools=[spy],
                policy=policy,
            )
            engine.run(sid, "go")

            names = [e.name for e in backend.events]
            assert "policy.evaluated" in names
            assert "policy.confirmation_required" in names
        finally:
            from agent.telemetry import NullBackend
            set_backend(NullBackend())

    def test_require_confirmation_approved_then_executes(self):
        spy, calls = _make_spy_tool("spy_tool")
        policy = PolicyEngine()
        policy.register(PolicyRule.for_tools(
            "spy_tool", name="r",
            handler=lambda tc, msgs: _make_confirmation("confirm"),
        ))
        tc = ToolCall.new("spy_tool", {})
        engine, sid = _build_engine(
            responses=[
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="Done after resume."),
            ],
            extra_tools=[spy],
            policy=policy,
        )
        engine.run(sid, "go")
        result = engine.resume(sid, approved=True)
        assert result == "Done after resume."
        assert len(calls) == 1, "Tool must execute after approval"


class TestPolicyInjectReminder:

    def test_inject_reminder_tool_still_executes(self):
        spy, calls = _make_spy_tool("spy_tool")
        policy = PolicyEngine()
        policy.register(PolicyRule.for_tools(
            "spy_tool", name="r",
            handler=lambda tc, msgs: _make_reminder("check before proceeding"),
        ))
        tc = ToolCall.new("spy_tool", {})
        engine, sid = _build_engine(
            responses=[
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="Seen reminder."),
            ],
            extra_tools=[spy],
            policy=policy,
        )
        engine.run(sid, "go")
        assert len(calls) == 1, "Tool handler must execute for inject_reminder"

    def test_inject_reminder_prepended_to_tool_result(self):
        spy, _ = _make_spy_tool("spy_tool")
        policy = PolicyEngine()
        policy.register(PolicyRule.for_tools(
            "spy_tool", name="r",
            handler=lambda tc, msgs: _make_reminder("IMPORTANT: verify eligibility"),
        ))
        tc = ToolCall.new("spy_tool", {})
        engine, sid = _build_engine(
            responses=[
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="Done."),
            ],
            extra_tools=[spy],
            policy=policy,
        )
        engine.run(sid, "go")
        session = engine.store.get(sid)
        content = _tool_result_contents(session)[0]
        assert "IMPORTANT: verify eligibility" in content
        assert "spy_result" in content  # original tool output also present

    def test_inject_reminder_reminder_appears_before_tool_output(self):
        spy, _ = _make_spy_tool("spy_tool")
        policy = PolicyEngine()
        policy.register(PolicyRule.for_tools(
            "spy_tool", name="r",
            handler=lambda tc, msgs: _make_reminder("reminder text"),
        ))
        tc = ToolCall.new("spy_tool", {})
        engine, sid = _build_engine(
            responses=[
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="Done."),
            ],
            extra_tools=[spy],
            policy=policy,
        )
        engine.run(sid, "go")
        session = engine.store.get(sid)
        content = _tool_result_contents(session)[0]
        reminder_pos = content.find("reminder text")
        output_pos   = content.find("spy_result")
        assert reminder_pos < output_pos

    def test_inject_reminder_no_reminder_text_does_not_alter_output(self):
        spy, _ = _make_spy_tool("spy_tool")
        policy = PolicyEngine()
        policy.register(PolicyRule.for_tools(
            "spy_tool", name="r",
            handler=lambda tc, msgs: PolicyResult(
                verdict=PolicyVerdict.inject_reminder,
                reason="reason",
                reminder="",  # empty reminder
            ),
        ))
        tc = ToolCall.new("spy_tool", {})
        engine, sid = _build_engine(
            responses=[
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="Done."),
            ],
            extra_tools=[spy],
            policy=policy,
        )
        engine.run(sid, "go")
        session = engine.store.get(sid)
        content = _tool_result_contents(session)[0]
        assert content == "spy_result"  # unmodified

    def test_inject_reminder_emits_telemetry(self):
        backend = InMemoryBackend()
        set_backend(backend)
        try:
            spy, _ = _make_spy_tool("spy_tool")
            policy = PolicyEngine()
            policy.register(PolicyRule.for_tools(
                "spy_tool", name="r",
                handler=lambda tc, msgs: _make_reminder("reminder"),
            ))
            tc = ToolCall.new("spy_tool", {})
            engine, sid = _build_engine(
                responses=[
                    LLMResponse(content="", tool_calls=[tc]),
                    LLMResponse(content="Done."),
                ],
                extra_tools=[spy],
                policy=policy,
            )
            engine.run(sid, "go")
            names = [e.name for e in backend.events]
            assert "policy.evaluated" in names
            assert "policy.reminder_injected" in names
        finally:
            from agent.telemetry import NullBackend
            set_backend(NullBackend())


class TestPolicyAllow:

    def test_allow_does_not_alter_tool_result(self):
        policy = PolicyEngine()
        policy.register(PolicyRule.for_tools(
            "calculator", name="r",
            handler=lambda tc, msgs: _make_allow(),
        ))
        tc = ToolCall.new("calculator", {"expression": "6 * 7"})
        engine, sid = _build_engine(
            responses=[
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="42."),
            ],
            policy=policy,
        )
        engine.run(sid, "go")
        session = engine.store.get(sid)
        assert "42" in _tool_result_contents(session)[0]

    def test_allow_does_not_emit_policy_block_event(self):
        backend = InMemoryBackend()
        set_backend(backend)
        try:
            policy = PolicyEngine()
            policy.register(PolicyRule.for_tools(
                "calculator", name="r",
                handler=lambda tc, msgs: _make_allow(),
            ))
            tc = ToolCall.new("calculator", {"expression": "1+1"})
            engine, sid = _build_engine(
                responses=[
                    LLMResponse(content="", tool_calls=[tc]),
                    LLMResponse(content="Done."),
                ],
                policy=policy,
            )
            engine.run(sid, "go")
            names = [e.name for e in backend.events]
            assert "policy.blocked" not in names
            assert "policy.reminder_injected" not in names
            assert "policy.confirmation_required" not in names
        finally:
            from agent.telemetry import NullBackend
            set_backend(NullBackend())


class TestNoPolicyEngine:

    def test_no_policy_engine_tool_executes_normally(self):
        tc = ToolCall.new("calculator", {"expression": "3 * 3"})
        engine, sid = _build_engine(
            responses=[
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="Nine."),
            ],
            policy=None,
        )
        result = engine.run(sid, "go")
        assert result == "Nine."
        session = engine.store.get(sid)
        assert "9" in _tool_result_contents(session)[0]

    def test_no_policy_engine_does_not_emit_policy_events(self):
        backend = InMemoryBackend()
        set_backend(backend)
        try:
            tc = ToolCall.new("calculator", {"expression": "1+1"})
            engine, sid = _build_engine(
                responses=[
                    LLMResponse(content="", tool_calls=[tc]),
                    LLMResponse(content="Done."),
                ],
                policy=None,
            )
            engine.run(sid, "go")
            names = [e.name for e in backend.events]
            assert not any(n.startswith("policy.") for n in names)
        finally:
            from agent.telemetry import NullBackend
            set_backend(NullBackend())


# ===========================================================================
# Protocol ordering tests
# ===========================================================================

class TestProtocolOrdering:

    def _assert_tool_result_follows_assistant(self, session) -> None:
        """Every assistant message with tool_calls must be immediately followed by tool_result."""
        msgs = session.messages
        for i, msg in enumerate(msgs):
            if msg.role == MessageRole.assistant and msg.tool_calls:
                assert i + 1 < len(msgs), "No message after tool_calls"
                assert msgs[i + 1].role == MessageRole.tool_result, (
                    f"Expected tool_result at index {i+1}, got {msgs[i+1].role}"
                )

    def test_protocol_preserved_after_block(self):
        spy, _ = _make_spy_tool("spy_tool")
        policy = PolicyEngine()
        policy.register(PolicyRule.for_tools(
            "spy_tool", name="r",
            handler=lambda tc, msgs: _make_block("blocked"),
        ))
        tc = ToolCall.new("spy_tool", {})
        engine, sid = _build_engine(
            responses=[
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="Done."),
            ],
            extra_tools=[spy],
            policy=policy,
        )
        engine.run(sid, "go")
        self._assert_tool_result_follows_assistant(engine.store.get(sid))

    def test_protocol_preserved_after_inject_reminder(self):
        spy, _ = _make_spy_tool("spy_tool")
        policy = PolicyEngine()
        policy.register(PolicyRule.for_tools(
            "spy_tool", name="r",
            handler=lambda tc, msgs: _make_reminder("note this"),
        ))
        tc = ToolCall.new("spy_tool", {})
        engine, sid = _build_engine(
            responses=[
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="Done."),
            ],
            extra_tools=[spy],
            policy=policy,
        )
        engine.run(sid, "go")
        self._assert_tool_result_follows_assistant(engine.store.get(sid))

    def test_protocol_preserved_mixed_block_and_allow(self):
        """Two tool calls in one turn: first blocked, second allowed."""
        spy, calls = _make_spy_tool("spy_tool")
        policy = PolicyEngine()
        policy.register(PolicyRule.for_tools(
            "spy_tool", name="r",
            handler=lambda tc, msgs: _make_block("blocked"),
        ))
        tc_calc = ToolCall.new("calculator", {"expression": "2+2"})
        tc_spy  = ToolCall.new("spy_tool", {})
        engine, sid = _build_engine(
            responses=[
                LLMResponse(content="", tool_calls=[tc_spy, tc_calc]),
                LLMResponse(content="Done."),
            ],
            extra_tools=[spy],
            policy=policy,
        )
        engine.run(sid, "go")
        session = engine.store.get(sid)
        self._assert_tool_result_follows_assistant(session)

        # spy_tool blocked, calculator executed
        tr_contents = _tool_result_contents(session)
        assert len(tr_contents) == 2
        assert "4" in tr_contents[1]  # calculator result
        assert calls == []  # spy_tool not executed

    def test_block_does_not_skip_other_tool_calls_in_same_turn(self):
        """If first of two tool calls is blocked, second still executes."""
        spy, calls = _make_spy_tool("spy_tool")
        policy = PolicyEngine()
        policy.register(PolicyRule.for_tools(
            "spy_tool", name="r",
            handler=lambda tc, msgs: _make_block("blocked"),
        ))
        tc_spy  = ToolCall.new("spy_tool", {})
        tc_calc = ToolCall.new("calculator", {"expression": "5+5"})
        engine, sid = _build_engine(
            responses=[
                LLMResponse(content="", tool_calls=[tc_spy, tc_calc]),
                LLMResponse(content="Done."),
            ],
            extra_tools=[spy],
            policy=policy,
        )
        engine.run(sid, "go")
        contents = _tool_result_contents(engine.store.get(sid))
        assert any("10" in c for c in contents), "calculator must still execute"


# ===========================================================================
# Scoping / correctness tests
# ===========================================================================

class TestPolicyScoping:

    def test_scoped_rule_does_not_affect_unrelated_tool(self):
        policy = PolicyEngine()
        policy.register(PolicyRule.for_tools(
            "some_other_tool", name="r",
            handler=lambda tc, msgs: _make_block("blocked"),
        ))
        tc = ToolCall.new("calculator", {"expression": "7*6"})
        engine, sid = _build_engine(
            responses=[
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="42."),
            ],
            policy=policy,
        )
        engine.run(sid, "go")
        session = engine.store.get(sid)
        assert session.state == SessionState.waiting_for_user
        assert "42" in _tool_result_contents(session)[0]

    def test_policy_evaluated_for_every_tool_in_multi_call_turn(self):
        """Each tool call in a multi-call turn is independently evaluated."""
        evaluated: list[str] = []

        def handler(tc: ToolCall, msgs) -> PolicyResult:
            evaluated.append(tc.name)
            return _make_allow()

        policy = PolicyEngine()
        policy.register(PolicyRule.for_all_tools(name="r", handler=handler))

        spy, _ = _make_spy_tool("spy_tool")
        tc1 = ToolCall.new("calculator", {"expression": "1+1"})
        tc2 = ToolCall.new("spy_tool", {})
        engine, sid = _build_engine(
            responses=[
                LLMResponse(content="", tool_calls=[tc1, tc2]),
                LLMResponse(content="Done."),
            ],
            extra_tools=[spy],
            policy=policy,
        )
        engine.run(sid, "go")
        assert "calculator" in evaluated
        assert "spy_tool" in evaluated

    def test_policy_evaluated_at_start_of_each_iteration(self):
        """Policy fires in iteration 1 and iteration 2 independently."""
        call_count: list[int] = [0]

        def handler(tc: ToolCall, msgs) -> PolicyResult:
            call_count[0] += 1
            return _make_allow()

        policy = PolicyEngine()
        policy.register(PolicyRule.for_all_tools(name="r", handler=handler))

        tc1 = ToolCall.new("calculator", {"expression": "1+1"})
        tc2 = ToolCall.new("calculator", {"expression": "2+2"})
        engine, sid = _build_engine(
            responses=[
                LLMResponse(content="", tool_calls=[tc1]),
                LLMResponse(content="", tool_calls=[tc2]),
                LLMResponse(content="Done."),
            ],
            policy=policy,
        )
        engine.run(sid, "go")
        assert call_count[0] == 2  # one evaluation per tool call, two iterations
