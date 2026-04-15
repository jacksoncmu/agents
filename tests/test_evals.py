"""Tests for the eval harness itself.

Validates:
  - Schema types serialize/deserialize correctly
  - Runner produces correct checks for each built-in case
  - Machine-readable output is well-formed
  - Suite aggregation and summary are accurate
  - Custom checks execute
  - Failing cases are detected
"""
from __future__ import annotations

import json

import pytest

from agent.llm.base import LLMResponse
from agent.tools import ToolDefinition, ToolParam
from agent.types import SessionState, ToolCall

from evals.cases import (
    ALL_CASES,
    CASE_CALC_SINGLE_TOOL,
    CASE_CONFIRMATION_PAUSE,
    CASE_ERROR_RECOVERY,
    CASE_MULTI_TOOL_CHAIN,
    CASE_SEARCH_NO_RESULTS,
)
from evals.runner import EvalRunner
from evals.schema import (
    Check,
    CheckStatus,
    EvalCase,
    EvalResult,
    EvalSuiteResult,
    ExpectedState,
    ExpectedToolCall,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def runner():
    return EvalRunner()


# ===========================================================================
# Schema types
# ===========================================================================

class TestSchemaTypes:
    def test_check_to_dict(self):
        c = Check("test", "category", CheckStatus.passed, "ok")
        d = c.to_dict()
        assert d["name"] == "test"
        assert d["status"] == "passed"
        assert d["detail"] == "ok"

    def test_check_no_detail_omitted(self):
        c = Check("test", "cat", CheckStatus.failed)
        d = c.to_dict()
        assert "detail" not in d

    def test_eval_result_to_dict(self):
        r = EvalResult(
            case_id="x",
            case_name="X",
            passed=True,
            checks=[Check("a", "b", CheckStatus.passed)],
            final_text="ok",
        )
        d = r.to_dict()
        assert d["case_id"] == "x"
        assert d["passed"] is True
        assert d["checks_passed"] == 1
        assert d["checks_failed"] == 0

    def test_eval_result_to_json_is_valid(self):
        r = EvalResult(
            case_id="y", case_name="Y", passed=False,
            checks=[Check("a", "b", CheckStatus.failed, "bad")],
        )
        parsed = json.loads(r.to_json())
        assert parsed["passed"] is False

    def test_suite_result_aggregation(self):
        results = [
            EvalResult("a", "A", True, [Check("c1", "x", CheckStatus.passed)]),
            EvalResult("b", "B", False, [
                Check("c2", "x", CheckStatus.passed),
                Check("c3", "x", CheckStatus.failed),
            ]),
        ]
        suite = EvalSuiteResult(results=results)
        assert suite.total == 2
        assert suite.passed == 1
        assert suite.failed == 1
        assert suite.total_checks == 3
        assert suite.total_checks_passed == 2
        assert suite.pass_rate == 0.5

    def test_suite_to_json_is_valid(self):
        suite = EvalSuiteResult(results=[
            EvalResult("a", "A", True, []),
        ])
        parsed = json.loads(suite.to_json())
        assert parsed["cases_total"] == 1
        assert "results" in parsed

    def test_suite_summary_text(self):
        suite = EvalSuiteResult(results=[
            EvalResult("a", "A", True, [Check("c", "x", CheckStatus.passed)]),
            EvalResult("b", "B", False, [Check("c", "x", CheckStatus.failed, "bad")]),
        ])
        text = suite.summary_text()
        assert "PASS" in text
        assert "FAIL" in text
        assert "bad" in text

    def test_empty_suite(self):
        suite = EvalSuiteResult(results=[])
        assert suite.total == 0
        assert suite.pass_rate == 0.0


# ===========================================================================
# Built-in cases — all should pass
# ===========================================================================

class TestBuiltInCases:
    def test_calc_single_tool(self, runner):
        result = runner.run_one(CASE_CALC_SINGLE_TOOL)
        assert result.passed, _failure_detail(result)
        assert result.final_text == "6 times 7 is 42."
        assert result.checks_failed == 0

    def test_multi_tool_chain(self, runner):
        result = runner.run_one(CASE_MULTI_TOOL_CHAIN)
        assert result.passed, _failure_detail(result)
        assert "roadmap" in result.final_text
        assert result.checks_failed == 0

    def test_search_no_results(self, runner):
        result = runner.run_one(CASE_SEARCH_NO_RESULTS)
        assert result.passed, _failure_detail(result)
        assert result.checks_failed == 0

    def test_error_recovery(self, runner):
        result = runner.run_one(CASE_ERROR_RECOVERY)
        assert result.passed, _failure_detail(result)
        assert "zero" in result.final_text.lower()

    def test_confirmation_pause(self, runner):
        result = runner.run_one(CASE_CONFIRMATION_PAUSE)
        assert result.passed, _failure_detail(result)
        assert "[confirm_required]" in result.final_text or result.final_text == ""

    def test_all_cases_pass(self, runner):
        suite = runner.run_all(ALL_CASES)
        assert suite.failed == 0, (
            f"{suite.failed}/{suite.total} cases failed:\n"
            + suite.summary_text()
        )


# ===========================================================================
# Runner check mechanics
# ===========================================================================

class TestRunnerChecks:
    def test_tool_name_check_passes(self, runner):
        result = runner.run_one(CASE_CALC_SINGLE_TOOL)
        name_checks = [c for c in result.checks if "tool_0_name" in c.name]
        assert name_checks
        assert all(c.status == CheckStatus.passed for c in name_checks)

    def test_tool_args_exact_check(self, runner):
        result = runner.run_one(CASE_CALC_SINGLE_TOOL)
        args_checks = [c for c in result.checks if "args_exact" in c.name]
        assert args_checks
        assert all(c.status == CheckStatus.passed for c in args_checks)

    def test_session_state_check(self, runner):
        result = runner.run_one(CASE_CALC_SINGLE_TOOL)
        state_checks = [c for c in result.checks if c.name == "session_state"]
        assert state_checks
        assert state_checks[0].status == CheckStatus.passed

    def test_message_count_bounds(self, runner):
        result = runner.run_one(CASE_CALC_SINGLE_TOOL)
        min_checks = [c for c in result.checks if c.name == "min_messages"]
        max_checks = [c for c in result.checks if c.name == "max_messages"]
        assert min_checks and min_checks[0].status == CheckStatus.passed
        assert max_checks and max_checks[0].status == CheckStatus.passed

    def test_content_check_passes(self, runner):
        result = runner.run_one(CASE_CALC_SINGLE_TOOL)
        content_checks = [c for c in result.checks if c.category == "content"]
        assert content_checks
        assert all(c.status == CheckStatus.passed for c in content_checks)

    def test_custom_check_executes(self, runner):
        result = runner.run_one(CASE_ERROR_RECOVERY)
        custom_checks = [c for c in result.checks if "tool_result_is_error" in c.name]
        assert custom_checks
        assert custom_checks[0].status == CheckStatus.passed

    def test_process_completion_check(self, runner):
        result = runner.run_one(CASE_CALC_SINGLE_TOOL)
        process_checks = [c for c in result.checks if c.name == "process_completed"]
        assert process_checks
        assert process_checks[0].status == CheckStatus.passed


# ===========================================================================
# Deliberate failure detection
# ===========================================================================

class TestFailureDetection:
    def test_wrong_tool_name_detected(self, runner):
        case = EvalCase(
            id="wrong_tool",
            name="Wrong tool name",
            description="Expects calculator but gets something else",
            user_message="hi",
            llm_responses=[
                LLMResponse(content="", tool_calls=[ToolCall.new("calculator", {"expression": "1+1"})]),
                LLMResponse(content="Done."),
            ],
            tools=[
                ToolDefinition("calculator", "calc", [ToolParam("expression", "string", "expr")],
                              lambda expression: str(eval(expression, {"__builtins__": {}}))),
            ],
            expected_tool_sequence=[
                ExpectedToolCall(name="wrong_name"),
            ],
        )
        result = runner.run_one(case)
        name_checks = [c for c in result.checks if "tool_0_name" in c.name]
        assert name_checks[0].status == CheckStatus.failed

    def test_wrong_args_detected(self, runner):
        case = EvalCase(
            id="wrong_args",
            name="Wrong args",
            description="Args don't match",
            user_message="hi",
            llm_responses=[
                LLMResponse(content="", tool_calls=[ToolCall.new("calculator", {"expression": "1+1"})]),
                LLMResponse(content="Done."),
            ],
            tools=[
                ToolDefinition("calculator", "calc", [ToolParam("expression", "string", "expr")],
                              lambda expression: str(eval(expression, {"__builtins__": {}}))),
            ],
            expected_tool_sequence=[
                ExpectedToolCall(name="calculator", args_exact={"expression": "2+2"}),
            ],
        )
        result = runner.run_one(case)
        args_checks = [c for c in result.checks if "args_exact" in c.name]
        assert args_checks[0].status == CheckStatus.failed

    def test_wrong_state_detected(self, runner):
        case = EvalCase(
            id="wrong_state",
            name="Wrong final state",
            description="Expects finished but gets waiting_for_user",
            user_message="hi",
            llm_responses=[LLMResponse(content="Hello.")],
            tools=[],
            expected_state=ExpectedState(session_state=SessionState.finished),
        )
        result = runner.run_one(case)
        assert not result.passed
        state_checks = [c for c in result.checks if c.name == "session_state"]
        assert state_checks[0].status == CheckStatus.failed

    def test_missing_content_detected(self, runner):
        case = EvalCase(
            id="missing_content",
            name="Missing required content",
            description="Response missing required substring",
            user_message="hi",
            llm_responses=[LLMResponse(content="Hello.")],
            tools=[],
            required_response_substrings=["MISSING_STRING"],
        )
        result = runner.run_one(case)
        assert not result.passed

    def test_forbidden_content_detected(self, runner):
        case = EvalCase(
            id="forbidden",
            name="Contains forbidden content",
            description="Response contains forbidden substring",
            user_message="hi",
            llm_responses=[LLMResponse(content="Hello world.")],
            tools=[],
            forbidden_response_substrings=["world"],
        )
        result = runner.run_one(case)
        assert not result.passed

    def test_engine_exception_captured(self, runner):
        """If the engine raises, the result should capture the error."""
        case = EvalCase(
            id="engine_crash",
            name="Engine exception",
            description="Empty LLM script causes RuntimeError",
            user_message="hi",
            llm_responses=[],  # empty → MockProvider raises
            tools=[],
        )
        result = runner.run_one(case)
        assert not result.passed
        assert result.error is not None
        assert "MockProvider exhausted" in result.error


# ===========================================================================
# Machine-readable output
# ===========================================================================

class TestMachineReadableOutput:
    def test_result_json_roundtrip(self, runner):
        result = runner.run_one(CASE_CALC_SINGLE_TOOL)
        parsed = json.loads(result.to_json())
        assert parsed["case_id"] == "calc_single_tool"
        assert isinstance(parsed["checks"], list)
        assert all("name" in c and "status" in c for c in parsed["checks"])

    def test_suite_json_roundtrip(self, runner):
        suite = runner.run_all(ALL_CASES)
        parsed = json.loads(suite.to_json())
        assert parsed["cases_total"] == len(ALL_CASES)
        assert isinstance(parsed["results"], list)
        assert "pass_rate" in parsed

    def test_json_contains_duration(self, runner):
        result = runner.run_one(CASE_CALC_SINGLE_TOOL)
        parsed = json.loads(result.to_json())
        assert "duration_ms" in parsed
        assert parsed["duration_ms"] >= 0

    def test_json_contains_timestamp(self, runner):
        result = runner.run_one(CASE_CALC_SINGLE_TOOL)
        parsed = json.loads(result.to_json())
        assert "timestamp" in parsed


# ===========================================================================
# Tag filtering
# ===========================================================================

class TestTagFiltering:
    def test_filter_by_tag(self, runner):
        suite = runner.run_all(ALL_CASES, tags=["error_handling"])
        assert suite.total == 1
        assert suite.results[0].case_id == "error_recovery"

    def test_filter_by_multiple_tags(self, runner):
        suite = runner.run_all(ALL_CASES, tags=["basic"])
        assert suite.total == 2  # calc_single_tool and multi_tool_chain

    def test_no_matching_tags(self, runner):
        suite = runner.run_all(ALL_CASES, tags=["nonexistent_tag"])
        assert suite.total == 0


# ===========================================================================
# Helpers
# ===========================================================================

def _failure_detail(result: EvalResult) -> str:
    """Build a diagnostic string for assertion messages."""
    failures = [c for c in result.checks if c.status == CheckStatus.failed]
    lines = [f"Case {result.case_id} failed:"]
    for c in failures:
        lines.append(f"  [{c.category}] {c.name}: {c.detail}")
    if result.error:
        lines.append(f"  Error: {result.error}")
    return "\n".join(lines)
