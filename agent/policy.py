"""Policy / rules engine — just-in-time structured injection.

The PolicyEngine is consulted before each tool call executes.  Rules are
registered per tool name (or as wildcard rules that match all tools).
The first non-allow verdict from a matching rule wins.

Verdicts
--------
allow               — no-op; tool executes normally.
block               — tool does NOT execute; a ToolResult error is injected
                      into the conversation with block_message as content.
require_confirmation— tool does NOT execute immediately; the session pauses
                      for human confirmation (same mechanism as the tool-level
                      requires_confirmation flag).
inject_reminder     — tool DOES execute; reminder text is prepended to the
                      ToolResult so the model sees policy context alongside
                      the actual output.

Usage
-----
    engine = PolicyEngine()

    # Block deletion outside business hours
    engine.register(PolicyRule.for_tools(
        "delete_record",
        name="no_delete_outside_hours",
        handler=lambda tc, msgs: PolicyResult(
            verdict=PolicyVerdict.block,
            reason="Deletions not allowed outside business hours.",
            block_message="This action is not permitted right now.",
        ),
    ))

    # Inject eligibility reminder before cancellation
    engine.register(PolicyRule.for_tools(
        "cancel_order",
        name="cancellation_eligibility",
        handler=lambda tc, msgs: PolicyResult(
            verdict=PolicyVerdict.inject_reminder,
            reason="Checking cancellation eligibility.",
            reminder="Verify order is within the 30-day cancellation window "
                     "and has not yet shipped before confirming.",
        ),
    ))

    agent_engine = AgentEngine(..., policy=engine)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from agent.types import Message, ToolCall


# ---------------------------------------------------------------------------
# Verdict and result types
# ---------------------------------------------------------------------------

class PolicyVerdict(str, Enum):
    allow                = "allow"
    block                = "block"
    require_confirmation = "require_confirmation"
    inject_reminder      = "inject_reminder"


@dataclass
class PolicyResult:
    """Outcome of evaluating one policy rule against a pending tool call."""
    verdict:       PolicyVerdict

    # Human-readable explanation — logged and emitted as telemetry regardless of verdict.
    reason:        str = ""

    # inject_reminder only: text prepended to the tool result visible to the model.
    reminder:      str = ""

    # block only: content of the injected ToolResult error message.
    # Falls back to reason if empty.
    block_message: str = ""


# ---------------------------------------------------------------------------
# Rule definition
# ---------------------------------------------------------------------------

@dataclass
class PolicyRule:
    """One named rule that maps a set of tool names to a handler function.

    handler receives (ToolCall, list[Message]) and returns PolicyResult.
    tool_names=frozenset() means the rule applies to every tool.
    """
    name:       str
    tool_names: frozenset[str]
    handler:    Callable[[ToolCall, list[Message]], PolicyResult]

    @staticmethod
    def for_tools(
        *tool_names: str,
        name: str = "",
        handler: Callable[[ToolCall, list[Message]], PolicyResult],
    ) -> "PolicyRule":
        """Create a rule scoped to one or more specific tool names."""
        return PolicyRule(
            name=name or "rule_for_" + "_".join(tool_names),
            tool_names=frozenset(tool_names),
            handler=handler,
        )

    @staticmethod
    def for_all_tools(
        name: str,
        handler: Callable[[ToolCall, list[Message]], PolicyResult],
    ) -> "PolicyRule":
        """Create a wildcard rule that applies to every tool call."""
        return PolicyRule(name=name, tool_names=frozenset(), handler=handler)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PolicyEngine:
    """Evaluates registered rules against a pending tool call.

    Rules are evaluated in registration order.  The first non-allow result
    is returned immediately.  If all rules allow (or no rules match), a
    synthetic allow result is returned.
    """

    def __init__(self) -> None:
        self._rules: list[PolicyRule] = []

    def register(self, rule: PolicyRule) -> None:
        """Append a rule to the evaluation chain."""
        self._rules.append(rule)

    @property
    def rules(self) -> list[PolicyRule]:
        return list(self._rules)

    def evaluate(
        self,
        tool_call: ToolCall,
        messages: list[Message],
    ) -> PolicyResult:
        """Return the first non-allow result from matching rules, or allow."""
        for rule in self._rules:
            if rule.tool_names and tool_call.name not in rule.tool_names:
                continue
            result = rule.handler(tool_call, messages)
            if result.verdict != PolicyVerdict.allow:
                return result
        return PolicyResult(verdict=PolicyVerdict.allow)
