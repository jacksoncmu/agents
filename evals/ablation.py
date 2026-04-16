"""Ablation testing — measure the contribution of each runtime mechanism.

The framework turns mechanisms on or off one at a time (or in combination)
and re-runs the eval suite, producing a comparison table with per-mechanism
verdicts: positive, negative, or neutral.

Mechanisms
----------
COMPRESSION             ContextCompressor — keeps message history within budget
LOOP_DETECTION          LoopDetector      — detects and breaks repetition cycles
POLICY                  PolicyEngine      — just-in-time rule injection/blocking
REASONING               reasoning_mode    — extended thinking / chain-of-thought
NARROWED_TOOLSET        tool_filter       — restricts visible tools to relevant set
STRONG_TOOL_DESCRIPTIONS (structural)     — precise descriptions vs stripped stubs

Notes on mock-LLM neutrality
-----------------------------
REASONING and STRONG_TOOL_DESCRIPTIONS only matter with a real LLM.
MockProvider ignores model config and tool descriptions — its responses are
scripted.  The framework includes them for completeness and notes the
limitation in the verdict column rather than silently claiming signal.

Usage
-----
    from evals.ablation import AblationRunner, standard_variants, ABLATION_CASES
    from scenarios.ml.evals import build_ml_eval_cases

    # Pass a factory callable so each variant gets a fresh MLDataStore.
    runner = AblationRunner(build_ml_eval_cases, extra_cases=ABLATION_CASES)
    report = runner.run(standard_variants())
    print(report.summary_text())
    print(report.to_json())
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Callable

from agent.compression import CompressionConfig, ContextCompressor
from agent.llm.base import LLMResponse
from agent.loop_detector import LoopDetector, LoopDetectorConfig
from agent.policy import PolicyEngine, PolicyResult, PolicyRule, PolicyVerdict
from agent.tools import ToolContext, ToolDefinition, ToolParam
from agent.types import SessionState, ToolCall
from evals.runner import EvalRunner
from evals.schema import (
    Check,
    CheckStatus,
    EvalCase,
    EvalSuiteResult,
    ExpectedState,
    ExpectedToolCall,
)


# ---------------------------------------------------------------------------
# Mechanism flags
# ---------------------------------------------------------------------------

class AblationMechanism(Flag):
    NONE                     = 0
    COMPRESSION              = auto()
    LOOP_DETECTION           = auto()
    POLICY                   = auto()
    REASONING                = auto()
    NARROWED_TOOLSET         = auto()
    STRONG_TOOL_DESCRIPTIONS = auto()
    ALL = (
        COMPRESSION | LOOP_DETECTION | POLICY
        | REASONING | NARROWED_TOOLSET | STRONG_TOOL_DESCRIPTIONS
    )

    # Mechanisms whose signal is valid only with a real LLM, not MockProvider
    MOCK_NEUTRAL = REASONING | STRONG_TOOL_DESCRIPTIONS


# ---------------------------------------------------------------------------
# Variant definition
# ---------------------------------------------------------------------------

@dataclass
class AblationVariant:
    """One named configuration of active mechanisms.

    Engine-level overrides (compressor, loop_detector, policy) are passed
    directly to AgentEngine.  tool_filter is applied to case.tools before
    tool registration.  reasoning_mode is noted in the report but has no
    effect on MockProvider.
    """
    name:         str
    mechanisms:   AblationMechanism = AblationMechanism.NONE
    description:  str = ""

    # Engine overrides — None means the mechanism is disabled
    compressor:    ContextCompressor | None = None
    loop_detector: LoopDetector      | None = None
    policy:        PolicyEngine      | None = None

    # Tool-level override
    tool_filter: Callable[[list[ToolDefinition]], list[ToolDefinition]] | None = None

    # Config-level (mock-neutral but recorded for production runs)
    reasoning_mode: bool = False


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class VariantResult:
    """Eval suite result for one ablation variant."""
    variant: AblationVariant
    suite:   EvalSuiteResult
    wall_ms: float = 0.0

    @property
    def pass_rate(self) -> float:
        return self.suite.pass_rate

    @property
    def passed(self) -> int:
        return self.suite.passed

    @property
    def total(self) -> int:
        return self.suite.total


_VERDICT_THRESHOLD = 0.05   # ≥5% improvement → positive; ≤-5% → negative


@dataclass
class AblationReport:
    """Comparison of all variant results against the baseline."""
    baseline_name: str
    results:       list[VariantResult]

    def _baseline(self) -> VariantResult:
        for r in self.results:
            if r.variant.name == self.baseline_name:
                return r
        return self.results[0]

    def verdict(self, result: VariantResult) -> str:
        """Return 'positive', 'negative', or 'neutral' vs baseline."""
        if result.variant.name == self.baseline_name:
            return "baseline"
        delta = result.pass_rate - self._baseline().pass_rate
        mock_neutral = bool(
            result.variant.mechanisms & AblationMechanism.MOCK_NEUTRAL
            and not (result.variant.mechanisms & ~AblationMechanism.MOCK_NEUTRAL)
        )
        if mock_neutral:
            return "neutral (mock-LLM)"
        if delta >= _VERDICT_THRESHOLD:
            return "positive ✓"
        if delta <= -_VERDICT_THRESHOLD:
            return "negative ✗"
        return "neutral"

    def summary_text(self) -> str:
        base = self._baseline()
        lines = [
            "=" * 72,
            "ABLATION COMPARISON REPORT",
            "=" * 72,
            f"Baseline: {self.baseline_name}  "
            f"({base.passed}/{base.total} cases pass, {base.pass_rate:.1%})",
            "",
            f"{'Variant':<28} {'Cases':>5} {'Pass':>5} {'Rate':>7} "
            f"{'vs Baseline':>12}  Verdict",
            "-" * 72,
        ]
        for r in self.results:
            delta = r.pass_rate - base.pass_rate
            delta_str = "---" if r.variant.name == self.baseline_name else f"{delta:+.1%}"
            verd = self.verdict(r).replace("\u2713", "[+]").replace("\u2717", "[-]")
            lines.append(
                f"{r.variant.name:<28} {r.total:>5} {r.passed:>5} "
                f"{r.pass_rate:>7.1%} {delta_str:>12}  {verd}"
            )
        lines += [
            "-" * 72,
            "",
            "Mechanism notes:",
            "  REASONING: reasoning_mode flag set but MockProvider ignores model config.",
            "             Use a real LLM (AnthropicProvider/OpenAIProvider) for signal.",
            "  STRONG_TOOL_DESCRIPTIONS: MockProvider ignores tool descriptions.",
            "             Use a real LLM for signal.",
            "=" * 72,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        base_rate = self._baseline().pass_rate
        return {
            "baseline": self.baseline_name,
            "variants": [
                {
                    "name":        r.variant.name,
                    "description": r.variant.description,
                    "mechanisms":  [
                        m.name for m in AblationMechanism
                        if m != AblationMechanism.NONE
                        and m != AblationMechanism.ALL
                        and m != AblationMechanism.MOCK_NEUTRAL
                        and m in r.variant.mechanisms
                    ],
                    "pass_rate":   round(r.pass_rate, 4),
                    "passed":      r.passed,
                    "total":       r.total,
                    "delta_vs_baseline": round(r.pass_rate - base_rate, 4),
                    "verdict":     self.verdict(r),
                    "wall_ms":     round(r.wall_ms, 1),
                }
                for r in self.results
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class AblationRunner:
    """Runs an eval suite across multiple ablation variants.

    Args:
        cases: eval cases to run for every variant, as either:
               - a list[EvalCase] (same instances reused across variants — fine
                 for stateless cases like ABLATION_CASES), or
               - a callable () -> list[EvalCase] (called fresh per variant —
                 required for ML cases whose tools embed a shared MLDataStore).
        extra_cases: additional stateless cases appended after the factory
                     result on every variant run (ignored when cases is a list).
        baseline_name: name of the variant to use as reference baseline.
    """

    def __init__(
        self,
        cases: list[EvalCase] | Callable[[], list[EvalCase]],
        baseline_name: str = "baseline",
        extra_cases: list[EvalCase] | None = None,
    ) -> None:
        self.baseline_name = baseline_name
        self._extra_cases: list[EvalCase] = extra_cases or []
        self._runner = EvalRunner()
        if callable(cases):
            self._case_factory: Callable[[], list[EvalCase]] | None = cases
            self._static_cases: list[EvalCase] = []
        else:
            self._case_factory = None
            self._static_cases = list(cases)

    def _get_cases(self) -> list[EvalCase]:
        if self._case_factory is not None:
            return self._case_factory() + self._extra_cases
        return self._static_cases

    def run(self, variants: list[AblationVariant]) -> AblationReport:
        """Run all variants and return a comparison report."""
        results: list[VariantResult] = []
        for variant in variants:
            cases = self._get_cases()
            t0 = time.perf_counter()
            suite = self._runner.run_all(
                cases,
                compressor=variant.compressor,
                loop_detector=variant.loop_detector,
                policy=variant.policy,
                tool_filter=variant.tool_filter,
            )
            wall_ms = (time.perf_counter() - t0) * 1000
            results.append(VariantResult(variant=variant, suite=suite, wall_ms=wall_ms))
        return AblationReport(baseline_name=self.baseline_name, results=results)


# ---------------------------------------------------------------------------
# Ablation-specific eval cases
# (designed to produce different pass/fail outcomes across variants)
# ---------------------------------------------------------------------------

def _make_dangerous_op() -> tuple[ToolDefinition, list[dict]]:
    """Spy tool that records every execution. Should be blocked by policy."""
    log: list[dict] = []

    def handler(*, ctx: ToolContext) -> str:
        log.append({"executed": True})
        ctx.log("dangerous_op ran")
        return "dangerous_op_result"

    tool = ToolDefinition(
        name="dangerous_op",
        description="A dangerous operation. Policy should block this.",
        params=[],
        handler=handler,
    )
    return tool, log


def _make_web_search_stub() -> ToolDefinition:
    """Stub that represents an irrelevant general-purpose tool."""
    def handler(query: str, *, ctx: ToolContext) -> str:
        return f"web_search_result for: {query}"

    return ToolDefinition(
        name="web_search",
        description="Search the web. Not relevant to ML scenario.",
        params=[ToolParam("query", "string", "Search query.", required=True)],
        handler=handler,
    )


# --- Case 1: Policy blocks a dangerous tool call ----------------------------
#
# Sensitive to: POLICY
#   With POLICY ON  (blocking rule for dangerous_op): tool result error=True → PASS
#   With POLICY OFF (no blocking):                    tool executes → error=False → FAIL

def _build_ablation_policy_blocks() -> EvalCase:
    dangerous_op, _exec_log = _make_dangerous_op()
    from examples.example_tools import make_calculator
    _tc = ToolCall.new("dangerous_op", {})
    _calc_tc = ToolCall.new("calculator", {"expression": "1+1"})

    def check_was_blocked(session, text, _):
        tool_results = [tr for m in session.messages for tr in m.tool_results]
        dangerous_results = [
            tr for tr in tool_results
            if any(
                tc.id == tr.tool_call_id
                for m in session.messages
                for tc in m.tool_calls
                if tc.name == "dangerous_op"
            )
        ]
        if not dangerous_results:
            return [Check(
                name="dangerous_op_result_present",
                category="process",
                status=CheckStatus.failed,
                detail="No tool result found for dangerous_op.",
            )]
        was_error = dangerous_results[0].error
        return [Check(
            name="dangerous_op_was_blocked",
            category="process",
            status=CheckStatus.passed if was_error else CheckStatus.failed,
            detail=(
                "dangerous_op was correctly blocked (error=True)."
                if was_error else
                "dangerous_op executed without being blocked (error=False) — policy miss."
            ),
        )]

    return EvalCase(
        id="ablation_policy_blocks",
        name="Policy blocks dangerous tool call",
        description=(
            "Verifies that a policy rule blocks 'dangerous_op'. "
            "PASSES only when a blocking policy is active."
        ),
        user_message="Run dangerous_op.",
        llm_responses=[
            LLMResponse(content="", tool_calls=[_tc]),
            LLMResponse(content="Understood — the operation was blocked."),
        ],
        tools=[dangerous_op, make_calculator()],
        expected_tool_sequence=[
            ExpectedToolCall(name="dangerous_op"),
        ],
        expected_state=ExpectedState(
            session_state=SessionState.waiting_for_user,
            min_messages=4,
        ),
        required_response_substrings=[],
        custom_checks=[check_was_blocked],
        tags=["ablation", "policy_sensitive"],
    )


# --- Case 2: Policy does not regress normal tools ---------------------------
#
# Sensitive to: POLICY (should be neutral — no regression)
#   With POLICY ON (rule only affects dangerous_op, not calculator): PASS
#   With POLICY OFF:                                                  PASS

def _build_ablation_policy_compat() -> EvalCase:
    _tc = ToolCall.new("calculator", {"expression": "7 * 6"})
    from examples.example_tools import make_calculator

    def check_calculator_ran(session, text, _):
        tool_results = [tr for m in session.messages for tr in m.tool_results]
        calc_results = [
            tr for tr in tool_results
            if any(
                tc.id == tr.tool_call_id
                for m in session.messages
                for tc in m.tool_calls
                if tc.name == "calculator"
            )
        ]
        ok = calc_results and not calc_results[0].error
        return [Check(
            name="calculator_ran_without_error",
            category="process",
            status=CheckStatus.passed if ok else CheckStatus.failed,
            detail=(
                "calculator executed correctly." if ok
                else "calculator failed or returned error — policy regression."
            ),
        )]

    return EvalCase(
        id="ablation_policy_compat",
        name="Policy does not regress normal tool calls",
        description=(
            "Verifies that a policy scoped to 'dangerous_op' does not affect "
            "unrelated tools (calculator). Should PASS with or without policy."
        ),
        user_message="What is 7 times 6?",
        llm_responses=[
            LLMResponse(content="", tool_calls=[_tc]),
            LLMResponse(content="7 times 6 is 42."),
        ],
        tools=[make_calculator()],
        expected_tool_sequence=[
            ExpectedToolCall(name="calculator", args_exact={"expression": "7 * 6"}),
        ],
        expected_state=ExpectedState(
            session_state=SessionState.waiting_for_user,
            min_messages=4,
            max_messages=4,
        ),
        required_response_substrings=["42"],
        custom_checks=[check_calculator_ran],
        tags=["ablation", "policy_compat"],
    )


# --- Case 3: Narrowed toolset blocks irrelevant tool call -------------------
#
# Sensitive to: NARROWED_TOOLSET
#   With NARROWED_TOOLSET (web_search filtered out): web_search call returns
#     "unknown tool" error → check "blocked_by_unavailability" PASSES
#   Without narrowing (web_search registered):        web_search executes →
#     no error → check FAILS (irrelevant tool was available and ran)

def _build_ablation_narrowed_toolset() -> EvalCase:
    web_search = _make_web_search_stub()
    _tc = ToolCall.new("web_search", {"query": "irrelevant query"})
    from examples.example_tools import make_calculator

    def check_web_search_unavailable(session, text, _):
        tool_results = [tr for m in session.messages for tr in m.tool_results]
        ws_results = [
            tr for tr in tool_results
            if any(
                tc.id == tr.tool_call_id
                for m in session.messages
                for tc in m.tool_calls
                if tc.name == "web_search"
            )
        ]
        if not ws_results:
            # No result at all (shouldn't happen — engine always returns a result)
            return [Check(
                name="web_search_blocked_by_narrow",
                category="process",
                status=CheckStatus.failed,
                detail="No tool result for web_search — unexpected.",
            )]
        was_error = ws_results[0].error
        return [Check(
            name="web_search_blocked_by_narrow",
            category="process",
            status=CheckStatus.passed if was_error else CheckStatus.failed,
            detail=(
                "web_search was unavailable (error=True) — narrowed toolset working."
                if was_error else
                "web_search executed successfully — irrelevant tool was not narrowed out."
            ),
        )]

    return EvalCase(
        id="ablation_narrowed_toolset",
        name="Narrowed toolset makes irrelevant tool unavailable",
        description=(
            "Verifies that filtering 'web_search' out of the toolset causes its "
            "call to fail. PASSES only when NARROWED_TOOLSET tool_filter is active."
        ),
        user_message="Search the web for something.",
        llm_responses=[
            LLMResponse(content="", tool_calls=[_tc]),
            LLMResponse(content="The search could not be completed."),
        ],
        tools=[web_search, make_calculator()],
        expected_tool_sequence=[
            ExpectedToolCall(name="web_search"),
        ],
        expected_state=ExpectedState(
            session_state=SessionState.waiting_for_user,
            min_messages=4,
        ),
        required_response_substrings=[],
        custom_checks=[check_web_search_unavailable],
        tags=["ablation", "narrowed_toolset_sensitive"],
    )


# Ablation case registry
ABLATION_CASES: list[EvalCase] = [
    _build_ablation_policy_blocks(),
    _build_ablation_policy_compat(),
    _build_ablation_narrowed_toolset(),
]


# ---------------------------------------------------------------------------
# Standard variant set
# ---------------------------------------------------------------------------

def _block_dangerous_op_policy() -> PolicyEngine:
    """Policy engine with one rule: block 'dangerous_op'."""
    engine = PolicyEngine()
    engine.register(PolicyRule.for_tools(
        "dangerous_op",
        name="block_dangerous_op",
        handler=lambda tc, msgs: PolicyResult(
            verdict=PolicyVerdict.block,
            reason="dangerous_op is not permitted",
            block_message="Operation blocked by policy: dangerous_op is not permitted.",
        ),
    ))
    return engine


def _narrow_toolset_filter(tools: list[ToolDefinition]) -> list[ToolDefinition]:
    """Remove tools not relevant to the ML/data-science scenario."""
    _irrelevant = {"web_search", "web_browse", "shell_exec", "run_command"}
    return [t for t in tools if t.name not in _irrelevant]


def standard_variants() -> list[AblationVariant]:
    """Return the standard set of ablation variants.

    Tests each mechanism individually, then all together.  The 'baseline'
    variant has no mechanisms active and serves as the comparison reference.
    """
    return [
        AblationVariant(
            name="baseline",
            mechanisms=AblationMechanism.NONE,
            description="No mechanisms — bare engine with case tools and scripted LLM.",
        ),
        AblationVariant(
            name="+compression",
            mechanisms=AblationMechanism.COMPRESSION,
            compressor=ContextCompressor(CompressionConfig()),
            description="Compression only. Mock-neutral for current short sessions.",
        ),
        AblationVariant(
            name="+loop_detection",
            mechanisms=AblationMechanism.LOOP_DETECTION,
            loop_detector=LoopDetector(LoopDetectorConfig()),
            description="Loop detection only. Mock-neutral (no loops in scripted cases).",
        ),
        AblationVariant(
            name="+policy",
            mechanisms=AblationMechanism.POLICY,
            policy=_block_dangerous_op_policy(),
            description="Policy blocks 'dangerous_op'. Positive for ablation_policy_blocks.",
        ),
        AblationVariant(
            name="+reasoning",
            mechanisms=AblationMechanism.REASONING,
            reasoning_mode=True,
            description=(
                "reasoning_mode=True recorded but mock-neutral. "
                "Use a real LLM for signal."
            ),
        ),
        AblationVariant(
            name="+narrowed_toolset",
            mechanisms=AblationMechanism.NARROWED_TOOLSET,
            tool_filter=_narrow_toolset_filter,
            description=(
                "Filters out irrelevant tools. Positive for ablation_narrowed_toolset."
            ),
        ),
        AblationVariant(
            name="+strong_descriptions",
            mechanisms=AblationMechanism.STRONG_TOOL_DESCRIPTIONS,
            description=(
                "Strong descriptions retained (structural). "
                "Mock-neutral — scripted LLM ignores descriptions."
            ),
        ),
        AblationVariant(
            name="all_on",
            mechanisms=AblationMechanism.ALL,
            compressor=ContextCompressor(CompressionConfig()),
            loop_detector=LoopDetector(LoopDetectorConfig()),
            policy=_block_dangerous_op_policy(),
            reasoning_mode=True,
            tool_filter=_narrow_toolset_filter,
            description="All mechanisms active simultaneously.",
        ),
    ]
