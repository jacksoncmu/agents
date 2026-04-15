"""Eval case schema — defines what an evaluation task looks like.

An ``EvalCase`` fully specifies:
    - the user prompt that starts the agent
    - the scripted LLM responses (mock provider)
    - the tools available
    - pass/fail assertions on every dimension we care about

An ``EvalResult`` captures granular outcomes — not just pass/fail but
*which* checks passed and which failed, with details.

An ``EvalSuiteResult`` aggregates results across all cases and produces
a machine-readable summary.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from agent.llm.base import LLMResponse
from agent.tools import ToolDefinition
from agent.types import SessionState


# ---------------------------------------------------------------------------
# Check definitions — what we assert on
# ---------------------------------------------------------------------------

class CheckStatus(str, Enum):
    passed = "passed"
    failed = "failed"
    skipped = "skipped"


@dataclass
class Check:
    """One granular assertion within an eval."""
    name: str
    category: str   # e.g. "task_success", "tool_params", "state", "content", "process"
    status: CheckStatus
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "category": self.category,
            "status": self.status.value,
        }
        if self.detail:
            d["detail"] = self.detail
        return d


# ---------------------------------------------------------------------------
# Tool call expectation
# ---------------------------------------------------------------------------

@dataclass
class ExpectedToolCall:
    """What we expect one tool call to look like.

    ``args_subset`` — partial match: these keys and values must appear
    in the actual arguments (extra keys are allowed).
    ``args_exact``  — full match: arguments must equal this dict exactly.
    Use one or neither (not both).
    """
    name: str
    args_subset: dict[str, Any] | None = None
    args_exact: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# State expectation
# ---------------------------------------------------------------------------

@dataclass
class ExpectedState:
    """Assertions on the final session state."""
    session_state: SessionState = SessionState.waiting_for_user
    min_messages: int | None = None
    max_messages: int | None = None
    error_message_contains: str | None = None
    no_error: bool = True


# ---------------------------------------------------------------------------
# Eval case — one task to evaluate
# ---------------------------------------------------------------------------

@dataclass
class EvalCase:
    """A single evaluation task."""

    id: str
    name: str
    description: str

    # Input
    user_message: str
    llm_responses: list[LLMResponse]
    tools: list[ToolDefinition]

    # Expected outcomes
    expected_tool_sequence: list[ExpectedToolCall] = field(default_factory=list)
    expected_state: ExpectedState = field(default_factory=ExpectedState)
    required_response_substrings: list[str] = field(default_factory=list)
    forbidden_response_substrings: list[str] = field(default_factory=list)

    # Custom assertions — receive (session, final_text, checks_so_far)
    custom_checks: list[Callable[..., list[Check]]] = field(default_factory=list)

    # Tags for filtering
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Eval result — outcome for one case
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Granular result for a single eval case."""
    case_id: str
    case_name: str
    passed: bool
    checks: list[Check]
    final_text: str = ""
    error: str | None = None
    duration_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def checks_passed(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.passed)

    @property
    def checks_failed(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.failed)

    @property
    def checks_skipped(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.skipped)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "case_name": self.case_name,
            "passed": self.passed,
            "final_text": self.final_text,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
            "timestamp": self.timestamp,
            "checks_total": len(self.checks),
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "checks_skipped": self.checks_skipped,
            "checks": [c.to_dict() for c in self.checks],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Suite result — aggregates everything
# ---------------------------------------------------------------------------

@dataclass
class EvalSuiteResult:
    """Aggregated results for a full eval run."""
    results: list[EvalResult]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def total_checks(self) -> int:
        return sum(len(r.checks) for r in self.results)

    @property
    def total_checks_passed(self) -> int:
        return sum(r.checks_passed for r in self.results)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def summary_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cases_total": self.total,
            "cases_passed": self.passed,
            "cases_failed": self.failed,
            "pass_rate": round(self.pass_rate, 4),
            "checks_total": self.total_checks,
            "checks_passed": self.total_checks_passed,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.summary_dict(), indent=indent)

    def summary_text(self) -> str:
        """Human-readable summary report."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("EVAL SUITE REPORT")
        lines.append("=" * 60)
        lines.append(f"Timestamp:     {self.timestamp}")
        lines.append(f"Cases:         {self.passed}/{self.total} passed "
                     f"({self.pass_rate:.0%})")
        lines.append(f"Total checks:  {self.total_checks_passed}/{self.total_checks}")
        lines.append("-" * 60)

        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"\n[{status}] {r.case_id}: {r.case_name}")
            lines.append(f"       Duration: {r.duration_ms:.0f}ms | "
                        f"Checks: {r.checks_passed}/{len(r.checks)}")
            if r.error:
                lines.append(f"       Error: {r.error}")
            for c in r.checks:
                if c.status == CheckStatus.failed:
                    lines.append(f"       FAIL {c.category}/{c.name}: {c.detail}")
                elif c.status == CheckStatus.skipped:
                    lines.append(f"       SKIP {c.category}/{c.name}: {c.detail}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
