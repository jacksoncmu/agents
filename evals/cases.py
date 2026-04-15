"""Built-in eval cases — representative agent tasks.

Each case exercises a distinct agent capability:

    calc_single_tool     single tool call → final answer (happy path)
    multi_tool_chain     two different tools called in sequence
    search_no_results    tool returns "no match" → agent handles gracefully
    error_recovery       tool raises → agent receives error → gives useful response
    confirmation_pause   tool requires confirmation → session pauses correctly
"""
from __future__ import annotations

from agent.llm.base import LLMResponse
from agent.tools import ToolDefinition, ToolParam
from agent.types import MessageRole, SessionState, ToolCall

from evals.schema import (
    Check,
    CheckStatus,
    EvalCase,
    ExpectedState,
    ExpectedToolCall,
)
from examples.example_tools import (
    make_calculator,
    make_get_current_time,
    make_request_confirmation,
    make_search_notes,
)


# ---------------------------------------------------------------------------
# Case 1: Single tool call → final answer
# ---------------------------------------------------------------------------

_calc_tc = ToolCall.new("calculator", {"expression": "6 * 7"})

CASE_CALC_SINGLE_TOOL = EvalCase(
    id="calc_single_tool",
    name="Single calculator tool call",
    description=(
        "User asks a math question. Agent calls calculator once with "
        "the correct expression, then delivers the final answer."
    ),
    user_message="What is 6 times 7?",
    llm_responses=[
        LLMResponse(content="", tool_calls=[_calc_tc]),
        LLMResponse(content="6 times 7 is 42."),
    ],
    tools=[make_calculator()],
    expected_tool_sequence=[
        ExpectedToolCall(
            name="calculator",
            args_exact={"expression": "6 * 7"},
        ),
    ],
    expected_state=ExpectedState(
        session_state=SessionState.waiting_for_user,
        min_messages=4,   # user, assistant(tc), tool_result, assistant(answer)
        max_messages=4,
    ),
    required_response_substrings=["42"],
    tags=["basic", "tool_call"],
)


# ---------------------------------------------------------------------------
# Case 2: Multi-tool chain
# ---------------------------------------------------------------------------

_search_tc = ToolCall.new("search_notes", {"query": "roadmap"})
_time_tc = ToolCall.new("get_current_time", {})

CASE_MULTI_TOOL_CHAIN = EvalCase(
    id="multi_tool_chain",
    name="Two-tool sequential chain",
    description=(
        "User asks to find meeting notes and check the time. Agent calls "
        "search_notes then get_current_time, then synthesizes both results."
    ),
    user_message="Find my meeting notes about the roadmap, and tell me the current time.",
    llm_responses=[
        LLMResponse(content="", tool_calls=[_search_tc]),
        LLMResponse(content="", tool_calls=[_time_tc]),
        LLMResponse(content="Your meeting notes mention the Q3 roadmap discussion. The current time is shown above."),
    ],
    tools=[make_search_notes(), make_get_current_time()],
    expected_tool_sequence=[
        ExpectedToolCall(name="search_notes", args_subset={"query": "roadmap"}),
        ExpectedToolCall(name="get_current_time"),
    ],
    expected_state=ExpectedState(
        session_state=SessionState.waiting_for_user,
        min_messages=6,   # user, asst(search), tr, asst(time), tr, asst(answer)
        max_messages=6,
    ),
    required_response_substrings=["roadmap"],
    tags=["basic", "multi_tool"],
)


# ---------------------------------------------------------------------------
# Case 3: Search returns no results — agent handles gracefully
# ---------------------------------------------------------------------------

_search_empty_tc = ToolCall.new("search_notes", {"query": "zzz_nonexistent_topic"})

CASE_SEARCH_NO_RESULTS = EvalCase(
    id="search_no_results",
    name="Search with no matching results",
    description=(
        "User searches for a topic that doesn't exist. The tool returns "
        "'No notes found'. Agent should relay this gracefully without "
        "fabricating content."
    ),
    user_message="Search my notes for zzz_nonexistent_topic.",
    llm_responses=[
        LLMResponse(content="", tool_calls=[_search_empty_tc]),
        LLMResponse(content="I searched your notes but found no matches for that topic."),
    ],
    tools=[make_search_notes()],
    expected_tool_sequence=[
        ExpectedToolCall(
            name="search_notes",
            args_subset={"query": "zzz_nonexistent_topic"},
        ),
    ],
    expected_state=ExpectedState(
        session_state=SessionState.waiting_for_user,
        min_messages=4,
    ),
    required_response_substrings=["no match"],
    forbidden_response_substrings=["Shopping list", "Pragmatic Programmer"],
    custom_checks=[
        lambda session, text, _: [
            Check(
                name="tool_result_indicates_no_match",
                category="tool_params",
                status=(
                    CheckStatus.passed
                    if any(
                        "No notes found" in tr.content
                        for m in session.messages
                        for tr in m.tool_results
                    )
                    else CheckStatus.failed
                ),
                detail="Tool result should contain 'No notes found'",
            )
        ],
    ],
    tags=["edge_case", "no_results"],
)


# ---------------------------------------------------------------------------
# Case 4: Tool raises an error — agent recovers
# ---------------------------------------------------------------------------

def _make_failing_calculator() -> ToolDefinition:
    """Calculator that will fail on division by zero."""
    return make_calculator()

_error_tc = ToolCall.new("calculator", {"expression": "1/0"})

CASE_ERROR_RECOVERY = EvalCase(
    id="error_recovery",
    name="Tool error recovery",
    description=(
        "User asks to divide by zero. The calculator tool raises. Agent "
        "receives the error result and provides a useful response instead "
        "of crashing."
    ),
    user_message="Calculate 1 divided by 0.",
    llm_responses=[
        LLMResponse(content="", tool_calls=[_error_tc]),
        LLMResponse(content="Division by zero is undefined. I can't compute that."),
    ],
    tools=[_make_failing_calculator()],
    expected_tool_sequence=[
        ExpectedToolCall(name="calculator", args_exact={"expression": "1/0"}),
    ],
    expected_state=ExpectedState(
        session_state=SessionState.waiting_for_user,
        min_messages=4,
    ),
    required_response_substrings=["zero"],
    custom_checks=[
        lambda session, text, _: [
            Check(
                name="tool_result_is_error",
                category="tool_params",
                status=(
                    CheckStatus.passed
                    if any(
                        tr.error
                        for m in session.messages
                        for tr in m.tool_results
                    )
                    else CheckStatus.failed
                ),
                detail="Tool result should be marked as error",
            )
        ],
    ],
    tags=["error_handling"],
)


# ---------------------------------------------------------------------------
# Case 5: Confirmation pause — session waits correctly
# ---------------------------------------------------------------------------

_delete_tc = ToolCall.new("delete_file", {"filename": "important.txt"})

CASE_CONFIRMATION_PAUSE = EvalCase(
    id="confirmation_pause",
    name="Confirmation-required tool pauses session",
    description=(
        "User requests a destructive action (delete_file). The tool "
        "requires confirmation, so the engine pauses. Session state "
        "must be waiting_for_confirmation with the pending tool call."
    ),
    user_message="Delete the file important.txt",
    llm_responses=[
        LLMResponse(content="", tool_calls=[_delete_tc]),
        # Second response only used if resumed (not in this eval)
        LLMResponse(content="File deleted."),
    ],
    tools=[make_request_confirmation()],
    expected_tool_sequence=[
        ExpectedToolCall(name="delete_file", args_exact={"filename": "important.txt"}),
    ],
    expected_state=ExpectedState(
        session_state=SessionState.waiting_for_confirmation,
    ),
    required_response_substrings=[],  # empty string returned on pause
    custom_checks=[
        lambda session, text, _: [
            Check(
                name="pending_confirmation_populated",
                category="state",
                status=(
                    CheckStatus.passed
                    if (
                        session.pending_confirmation
                        and session.pending_confirmation[0].name == "delete_file"
                    )
                    else CheckStatus.failed
                ),
                detail="Session should have pending_confirmation for delete_file",
            ),
            Check(
                name="response_is_confirm_event",
                category="task_success",
                status=(
                    CheckStatus.passed
                    if "[confirm_required]" in text or text == ""
                    else CheckStatus.failed
                ),
                detail=f"Expected confirm event or empty, got {text!r}",
            ),
        ],
    ],
    tags=["confirmation", "state_management"],
)


# ---------------------------------------------------------------------------
# Registry of all cases
# ---------------------------------------------------------------------------

ALL_CASES: list[EvalCase] = [
    CASE_CALC_SINGLE_TOOL,
    CASE_MULTI_TOOL_CHAIN,
    CASE_SEARCH_NO_RESULTS,
    CASE_ERROR_RECOVERY,
    CASE_CONFIRMATION_PAUSE,
]
