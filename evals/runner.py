"""Eval runner — executes eval cases and collects results.

Usage::

    from evals.runner import EvalRunner
    from evals.cases import ALL_CASES

    runner = EvalRunner()
    suite_result = runner.run_all(ALL_CASES)
    print(suite_result.summary_text())
    print(suite_result.to_json())
"""
from __future__ import annotations

import time
from typing import Any

from typing import Callable

from agent.compression import ContextCompressor
from agent.engine import AgentEngine
from agent.llm.mock import MockProvider
from agent.loop_detector import LoopDetector
from agent.policy import PolicyEngine
from agent.storage import InMemoryStore
from agent.telemetry import InMemoryBackend, NullBackend, set_backend
from agent.tools import ToolDefinition, ToolRegistry
from agent.types import MessageRole, Session, SessionState

from evals.schema import (
    Check,
    CheckStatus,
    EvalCase,
    EvalResult,
    EvalSuiteResult,
    ExpectedState,
    ExpectedToolCall,
)


class EvalRunner:
    """Runs eval cases against the agent engine with a mock LLM."""

    def run_all(
        self,
        cases: list[EvalCase],
        *,
        tags: list[str] | None = None,
        compressor: ContextCompressor | None = None,
        loop_detector: LoopDetector | None = None,
        policy: PolicyEngine | None = None,
        tool_filter: Callable[[list[ToolDefinition]], list[ToolDefinition]] | None = None,
    ) -> EvalSuiteResult:
        """Run all cases (or only those matching *tags*) and return aggregated results."""
        if tags:
            cases = [c for c in cases if any(t in c.tags for t in tags)]
        results = [
            self.run_one(
                case,
                compressor=compressor, loop_detector=loop_detector,
                policy=policy, tool_filter=tool_filter,
            )
            for case in cases
        ]
        return EvalSuiteResult(results=results)

    def run_one(
        self,
        case: EvalCase,
        *,
        compressor: ContextCompressor | None = None,
        loop_detector: LoopDetector | None = None,
        policy: PolicyEngine | None = None,
        tool_filter: Callable[[list[ToolDefinition]], list[ToolDefinition]] | None = None,
    ) -> EvalResult:
        """Execute a single eval case and return a detailed result.

        Optional keyword arguments override engine construction for ablation runs:
          compressor    — ContextCompressor instance (or None to disable)
          loop_detector — LoopDetector instance (or None to disable)
          policy        — PolicyEngine instance (or None to disable)
          tool_filter   — callable that receives case.tools and returns a
                          (possibly reduced) list; use to ablate narrowed toolset
        """
        # Set up telemetry capture
        tel_backend = InMemoryBackend()
        set_backend(tel_backend)

        checks: list[Check] = []
        final_text = ""
        error: str | None = None
        t0 = time.perf_counter()

        try:
            # Build engine
            registry = ToolRegistry()
            tools_to_register = tool_filter(case.tools) if tool_filter else case.tools
            for tool in tools_to_register:
                registry.register(tool)

            store = InMemoryStore()
            llm = MockProvider(list(case.llm_responses))
            engine = AgentEngine(
                store=store, llm=llm, tools=registry,
                compressor=compressor, loop_detector=loop_detector, policy=policy,
            )

            # Run
            sid = engine.create_session().id
            final_text = engine.run(sid, case.user_message)
            session = store.get(sid)

            # --- Collect checks ---

            # 1. Process completion
            checks.append(Check(
                name="process_completed",
                category="process",
                status=CheckStatus.passed,
                detail="Engine run completed without exception",
            ))

            # 2. Task success (final text not empty)
            if final_text:
                checks.append(Check(
                    name="produced_response",
                    category="task_success",
                    status=CheckStatus.passed,
                    detail=f"Response length: {len(final_text)}",
                ))
            else:
                # Empty final text might be intentional (confirmation pause)
                if case.expected_state.session_state == SessionState.waiting_for_confirmation:
                    checks.append(Check(
                        name="produced_response",
                        category="task_success",
                        status=CheckStatus.passed,
                        detail="Empty response expected (confirmation pause)",
                    ))
                else:
                    checks.append(Check(
                        name="produced_response",
                        category="task_success",
                        status=CheckStatus.failed,
                        detail="Final text is empty",
                    ))

            # 3. Tool call sequence
            checks.extend(self._check_tool_sequence(session, case.expected_tool_sequence))

            # 4. Session state
            checks.extend(self._check_state(session, case.expected_state))

            # 5. Response content
            checks.extend(self._check_response_content(
                final_text,
                case.required_response_substrings,
                case.forbidden_response_substrings,
            ))

            # 6. Custom checks
            for custom_fn in case.custom_checks:
                try:
                    custom_results = custom_fn(session, final_text, checks)
                    checks.extend(custom_results)
                except Exception as exc:
                    checks.append(Check(
                        name="custom_check_error",
                        category="custom",
                        status=CheckStatus.failed,
                        detail=f"Custom check raised: {exc}",
                    ))

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            checks.append(Check(
                name="process_completed",
                category="process",
                status=CheckStatus.failed,
                detail=error,
            ))

        finally:
            set_backend(NullBackend())

        duration_ms = (time.perf_counter() - t0) * 1000
        all_passed = all(c.status != CheckStatus.failed for c in checks)

        return EvalResult(
            case_id=case.id,
            case_name=case.name,
            passed=all_passed,
            checks=checks,
            final_text=final_text,
            error=error,
            duration_ms=duration_ms,
        )

    # ------------------------------------------------------------------
    # Check helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_tool_sequence(
        session: Session,
        expected: list[ExpectedToolCall],
    ) -> list[Check]:
        """Verify tool call sequence and parameters."""
        checks: list[Check] = []

        if not expected:
            return checks

        # Extract actual tool calls from assistant messages
        actual_calls: list[dict[str, Any]] = []
        for msg in session.messages:
            if msg.role == MessageRole.assistant:
                for tc in msg.tool_calls:
                    actual_calls.append({
                        "name": tc.name,
                        "arguments": tc.arguments,
                    })

        # Check sequence length
        if len(actual_calls) >= len(expected):
            checks.append(Check(
                name="tool_sequence_length",
                category="tool_params",
                status=CheckStatus.passed,
                detail=f"Expected {len(expected)} calls, got {len(actual_calls)}",
            ))
        else:
            checks.append(Check(
                name="tool_sequence_length",
                category="tool_params",
                status=CheckStatus.failed,
                detail=f"Expected at least {len(expected)} calls, got {len(actual_calls)}",
            ))
            # Still check what we can
            expected = expected[:len(actual_calls)]

        # Check each expected call
        for i, exp in enumerate(expected):
            if i >= len(actual_calls):
                break
            actual = actual_calls[i]

            # Tool name
            if actual["name"] == exp.name:
                checks.append(Check(
                    name=f"tool_{i}_name",
                    category="tool_params",
                    status=CheckStatus.passed,
                    detail=f"Call {i}: {exp.name}",
                ))
            else:
                checks.append(Check(
                    name=f"tool_{i}_name",
                    category="tool_params",
                    status=CheckStatus.failed,
                    detail=f"Call {i}: expected {exp.name!r}, got {actual['name']!r}",
                ))

            # Args subset match
            if exp.args_subset is not None:
                mismatches: list[str] = []
                for k, v in exp.args_subset.items():
                    if k not in actual["arguments"]:
                        mismatches.append(f"missing key {k!r}")
                    elif actual["arguments"][k] != v:
                        mismatches.append(
                            f"{k}: expected {v!r}, got {actual['arguments'][k]!r}"
                        )
                if mismatches:
                    checks.append(Check(
                        name=f"tool_{i}_args",
                        category="tool_params",
                        status=CheckStatus.failed,
                        detail=f"Call {i} args: {'; '.join(mismatches)}",
                    ))
                else:
                    checks.append(Check(
                        name=f"tool_{i}_args",
                        category="tool_params",
                        status=CheckStatus.passed,
                        detail=f"Call {i} args match subset",
                    ))

            # Args exact match
            if exp.args_exact is not None:
                if actual["arguments"] == exp.args_exact:
                    checks.append(Check(
                        name=f"tool_{i}_args_exact",
                        category="tool_params",
                        status=CheckStatus.passed,
                        detail=f"Call {i} args exact match",
                    ))
                else:
                    checks.append(Check(
                        name=f"tool_{i}_args_exact",
                        category="tool_params",
                        status=CheckStatus.failed,
                        detail=(
                            f"Call {i} args: expected {exp.args_exact!r}, "
                            f"got {actual['arguments']!r}"
                        ),
                    ))

        return checks

    @staticmethod
    def _check_state(session: Session, expected: ExpectedState) -> list[Check]:
        """Verify final session state."""
        checks: list[Check] = []

        # Session state
        if session.state == expected.session_state:
            checks.append(Check(
                name="session_state",
                category="state",
                status=CheckStatus.passed,
                detail=f"State: {session.state.value}",
            ))
        else:
            checks.append(Check(
                name="session_state",
                category="state",
                status=CheckStatus.failed,
                detail=f"Expected {expected.session_state.value}, got {session.state.value}",
            ))

        # Message count bounds
        msg_count = len(session.messages)
        if expected.min_messages is not None:
            if msg_count >= expected.min_messages:
                checks.append(Check(
                    name="min_messages",
                    category="state",
                    status=CheckStatus.passed,
                    detail=f"{msg_count} >= {expected.min_messages}",
                ))
            else:
                checks.append(Check(
                    name="min_messages",
                    category="state",
                    status=CheckStatus.failed,
                    detail=f"{msg_count} < {expected.min_messages}",
                ))

        if expected.max_messages is not None:
            if msg_count <= expected.max_messages:
                checks.append(Check(
                    name="max_messages",
                    category="state",
                    status=CheckStatus.passed,
                    detail=f"{msg_count} <= {expected.max_messages}",
                ))
            else:
                checks.append(Check(
                    name="max_messages",
                    category="state",
                    status=CheckStatus.failed,
                    detail=f"{msg_count} > {expected.max_messages}",
                ))

        # Error state
        if expected.no_error:
            if not session.error_message:
                checks.append(Check(
                    name="no_error",
                    category="state",
                    status=CheckStatus.passed,
                ))
            else:
                checks.append(Check(
                    name="no_error",
                    category="state",
                    status=CheckStatus.failed,
                    detail=f"Unexpected error: {session.error_message}",
                ))

        if expected.error_message_contains:
            if expected.error_message_contains in session.error_message:
                checks.append(Check(
                    name="error_message_content",
                    category="state",
                    status=CheckStatus.passed,
                    detail=f"Error contains {expected.error_message_contains!r}",
                ))
            else:
                checks.append(Check(
                    name="error_message_content",
                    category="state",
                    status=CheckStatus.failed,
                    detail=(
                        f"Expected error containing {expected.error_message_contains!r}, "
                        f"got {session.error_message!r}"
                    ),
                ))

        return checks

    @staticmethod
    def _check_response_content(
        text: str,
        required: list[str],
        forbidden: list[str],
    ) -> list[Check]:
        """Verify response content requirements."""
        checks: list[Check] = []

        for s in required:
            if s in text:
                checks.append(Check(
                    name=f"contains_{s[:30]}",
                    category="content",
                    status=CheckStatus.passed,
                    detail=f"Response contains {s!r}",
                ))
            else:
                checks.append(Check(
                    name=f"contains_{s[:30]}",
                    category="content",
                    status=CheckStatus.failed,
                    detail=f"Response missing {s!r}",
                ))

        for s in forbidden:
            if s not in text:
                checks.append(Check(
                    name=f"excludes_{s[:30]}",
                    category="content",
                    status=CheckStatus.passed,
                    detail=f"Response correctly excludes {s!r}",
                ))
            else:
                checks.append(Check(
                    name=f"excludes_{s[:30]}",
                    category="content",
                    status=CheckStatus.failed,
                    detail=f"Response contains forbidden {s!r}",
                ))

        return checks
