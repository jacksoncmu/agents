"""Agent engine — runs the ReAct loop."""
from __future__ import annotations

import logging
from typing import Iterator

from agent.compression import ContextCompressor
from agent.llm.base import LLMProvider
from agent.loop_detector import InterventionQueue, LoopDetector
from agent.policy import PolicyEngine, PolicyResult, PolicyVerdict
from agent.storage import LABEL_POST_LLM, LABEL_POST_TOOLS, Checkpoint, SessionStore
from agent.telemetry import Outcome, emit as _emit
from agent.tools import ExecutionResult, ToolRegistry
from agent.types import (
    Message,
    MessageRole,
    Session,
    SessionState,
    ToolCall,
    ToolResult,
)

log = logging.getLogger(__name__)

MAX_ITERATIONS = 50  # safety ceiling to prevent infinite loops


class ConcurrentRunError(RuntimeError):
    """Raised when a run is requested but another run already owns the session."""


class AgentEngine:
    def __init__(
        self,
        store: SessionStore,
        llm: LLMProvider,
        tools: ToolRegistry,
        compressor: ContextCompressor | None = None,
        loop_detector: LoopDetector | None = None,
        policy: PolicyEngine | None = None,
    ) -> None:
        self.store = store
        self.llm = llm
        self.tools = tools
        self._compressor = compressor
        self._loop_detector = loop_detector
        self._policy = policy
        self._intervention_queue = InterventionQueue()

    # ------------------------------------------------------------------
    # Ownership helpers (emit telemetry + delegate to store)
    # ------------------------------------------------------------------

    def _acquire_run(self, session_id: str) -> None:
        """Claim run ownership.  Emits run.acquire (executed|blocked).

        Raises ConcurrentRunError when another run already holds ownership.
        """
        if not self.store.try_acquire_run(session_id):
            _emit("run.acquire", Outcome.blocked, session_id=session_id)
            raise ConcurrentRunError(
                f"Session {session_id} already has an active run."
            )
        _emit("run.acquire", Outcome.executed, session_id=session_id)

    def _release_run(self, session_id: str) -> None:
        """Release run ownership and emit run.release."""
        self.store.release_run(session_id)
        _emit("run.release", Outcome.executed, session_id=session_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_session(self) -> Session:
        session = self.store.create()
        log.debug("Created session %s", session.id)
        return session

    def cancel(self, session_id: str) -> None:
        """Signal a running session to stop at the next iteration."""
        self.store.set_cancellation(session_id, True)

    def run(self, session_id: str, user_message: str) -> str:
        """
        Accept a user message, run the ReAct loop to completion, and return
        the final assistant text.

        If a tool requires confirmation the loop pauses: the session moves to
        ``waiting_for_confirmation`` and this method returns an empty string.
        Call ``resume()`` to continue.
        """
        self._acquire_run(session_id)
        try:
            session = self._require_session(session_id)
            self._assert_can_run(session)

            session.messages.append(Message.user(user_message))
            self._transition(session, SessionState.running)
            self.store.save(session)
        except Exception:
            self._release_run(session_id)
            raise

        return self._run_loop(session)

    def continue_run(self, session_id: str) -> str:
        """
        Resume an interrupted session without a new user message.

        Intended for two scenarios:
        1. Process restart recovery — the session was found in ``running``
           state (crash mid-loop).  Restores to the latest ``post_tools``
           checkpoint, then continues the loop.
        2. In-turn continuation — any other caller that wants to re-enter
           the loop from the current message history.

        If no ``post_tools`` checkpoint exists, the session is reset to
        ``waiting_for_user`` and an empty string is returned (caller must
        send a new user message).
        """
        self._acquire_run(session_id)

        try:
            session = self._require_session(session_id)

            if session.state == SessionState.running:
                # Recovery path: session was interrupted mid-loop.  Find the
                # latest post_tools checkpoint (safest resume point) and truncate
                # messages back to that state.  Skip _assert_can_run because the
                # running state is expected here.
                all_cps = self.store.list_checkpoints(session_id)
                post_tools_cps = [cp for cp in all_cps if cp.label == LABEL_POST_TOOLS]
                if post_tools_cps:
                    cp = post_tools_cps[-1]
                    log.info(
                        "Recovering session %s from checkpoint %s (msg_count=%d)",
                        session_id, cp.id, cp.message_count,
                    )
                    session.messages = session.messages[: cp.message_count]
                    self.store.save(session)
                else:
                    log.warning(
                        "Session %s in running state with no post_tools checkpoint; "
                        "resetting to waiting_for_user",
                        session_id,
                    )
                    self._transition(session, SessionState.waiting_for_user)
                    session.error_message = ""
                    self.store.save(session)
                    self._release_run(session_id)
                    return ""
            else:
                # Normal continuation from a paused-but-healthy state.
                self._assert_can_run(session)
                self._transition(session, SessionState.running)
                self.store.save(session)
        except Exception:
            self._release_run(session_id)
            raise

        return self._run_loop(session)

    def stream_run(self, session_id: str, user_message: str) -> Iterator[str]:
        """
        Generator variant of run().  Yields progress events:
          "[tool] name(args)"              — tool is about to execute
          "[result] output"                — tool returned successfully
          "[result:error] output"          — tool returned an error
          "[log] name: message"            — structured log from the tool handler
          "[confirm_required] name(args)"  — tool paused for confirmation
          final assistant text             — last event
        """
        self._acquire_run(session_id)
        try:
            session = self._require_session(session_id)
            self._assert_can_run(session)

            session.messages.append(Message.user(user_message))
            self._transition(session, SessionState.running)
            self.store.save(session)
        except Exception:
            self._release_run(session_id)
            raise

        try:
            for event in self._react_loop_events(session):
                yield event
        except Exception as exc:
            session = self.store.get(session.id)
            self._transition(session, SessionState.error)
            session.error_message = str(exc)
            self.store.save(session)
            raise
        finally:
            self._release_run(session_id)

        session = self.store.get(session.id)
        if session.state == SessionState.running:
            self._transition(session, SessionState.waiting_for_user)
            self.store.save(session)

    def resume(self, session_id: str, *, approved: bool) -> str:
        """
        Resume after ``waiting_for_confirmation``.

        approved=True  — execute the pending tools and continue the loop.
        approved=False — inject rejection results and continue the loop.
        """
        self._acquire_run(session_id)
        try:
            session = self._require_session(session_id)
            if session.state != SessionState.waiting_for_confirmation:
                raise RuntimeError(
                    f"Session {session_id} is not waiting for confirmation "
                    f"(state={session.state})"
                )

            pending = list(session.pending_confirmation)
            session.pending_confirmation = []
            self._transition(session, SessionState.running)
            self.store.save(session)

            results: list[ToolResult] = []
            for tc in pending:
                if approved:
                    exec_result = self.tools.execute_confirmed(tc.name, tc.arguments)
                    results.append(ToolResult(
                        tool_call_id=tc.id,
                        content=exec_result.output,
                        error=exec_result.error,
                    ))
                else:
                    results.append(ToolResult(
                        tool_call_id=tc.id,
                        content="Action was not approved.",
                        error=True,
                    ))

            session.messages.append(Message.tool_result_msg(results))
            self.store.save(session)
        except Exception:
            self._release_run(session_id)
            raise

        return self._run_loop(session)

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _run_loop(self, session: Session) -> str:
        """Enter the blocking ReAct loop; handle final state transitions.

        Always releases run ownership on exit — success, confirmation pause,
        cancellation, or exception.
        """
        try:
            final_text = self._react_loop(session)
        except Exception as exc:
            session = self.store.get(session.id)
            self._transition(session, SessionState.error)
            session.error_message = str(exc)
            self.store.save(session)
            raise
        finally:
            self._release_run(session.id)

        session = self.store.get(session.id)
        if session.state == SessionState.running:
            self._transition(session, SessionState.waiting_for_user)
            self.store.save(session)
        return final_text

    def _react_loop(self, session: Session) -> str:
        """Blocking ReAct loop; returns the final assistant text (may be empty
        if the loop paused for confirmation)."""
        final_text = ""
        for event in self._react_loop_events(session):
            final_text = event
        return final_text

    def _react_loop_events(self, session: Session) -> Iterator[str]:
        """
        Core loop generator.

        Tool schemas are fetched fresh at the top of every iteration so that
        tools registered between turns are immediately visible.

        Checkpoints are written at:
          LABEL_POST_LLM    — after the assistant message is appended
          LABEL_POST_TOOLS  — after all tool results are appended (safest
                              resume point; engine reads this on recovery)
        """
        for iteration in range(MAX_ITERATIONS):
            # --- cancellation check (required at top of every iteration) ---
            session = self.store.get(session.id)
            if session.cancelled:
                log.info("Session %s cancelled at iteration %d", session.id, iteration)
                _emit("loop.cancelled", Outcome.blocked,
                      session_id=session.id, iteration=iteration)
                self._transition(session, SessionState.finished)
                self.store.save(session)
                return

            # Refresh tool schemas each iteration — dynamic visibility
            tool_schemas = self.tools.all_schemas()

            log.debug("Session %s iteration %d (%d tools)", session.id, iteration, len(tool_schemas))
            _emit("loop.iteration.start", Outcome.executed,
                  session_id=session.id, iteration=iteration,
                  tool_count=len(tool_schemas))

            # Optional context compression — keep messages within the window
            if self._compressor is not None:
                compressed = self._compressor.compress(session.messages, session.id)
                if compressed is not session.messages:
                    session.messages = compressed
                    self.store.save(session)

            # Flush any queued intervention messages (from previous iteration's
            # loop detection).  Safe here: we are past the tool_result from the
            # previous iteration and before the next LLM call.
            flushed = self._intervention_queue.flush()
            if flushed:
                for msg in flushed:
                    session.messages.append(msg)
                self.store.save(session)
                _emit(
                    "loop.intervention_flushed", Outcome.executed,
                    session_id=session.id, iteration=iteration,
                    count=len(flushed),
                )

            # 1. Call the model
            llm_response = self.llm.complete(session.messages, tool_schemas)

            # 2. Append assistant message
            assistant_msg = Message.assistant(
                content=llm_response.content,
                tool_calls=llm_response.tool_calls,
            )
            session.messages.append(assistant_msg)
            self.store.save(session)

            # Checkpoint: we know what the model said
            self._save_checkpoint(session, iteration, LABEL_POST_LLM)

            # 3. No tool calls → final answer; loop exits cleanly
            if not llm_response.has_tool_calls:
                _emit("loop.final_answer", Outcome.condition_unsatisfied,
                      session_id=session.id, iteration=iteration,
                      content_length=len(llm_response.content))
                yield llm_response.content
                return

            # 4. Execute tool calls
            results: list[ToolResult] = []
            needs_confirmation: list[ToolCall] = []

            for tc in llm_response.tool_calls:
                # Per-tool cancellation check — allows fast exit between tools
                session = self.store.get(session.id)
                if session.cancelled:
                    log.info(
                        "Session %s cancelled between tools at iteration %d",
                        session.id, iteration,
                    )
                    _emit("loop.cancelled_between_tools", Outcome.blocked,
                          session_id=session.id, iteration=iteration,
                          pending_tool=tc.name)
                    self._transition(session, SessionState.finished)
                    self.store.save(session)
                    return

                # --- Policy evaluation (JIT, before execution) ---
                _pr: PolicyResult = PolicyResult(verdict=PolicyVerdict.allow)
                if self._policy is not None:
                    _pr = self._policy.evaluate(tc, session.messages)
                    _emit(
                        "policy.evaluated", Outcome.executed,
                        session_id=session.id, iteration=iteration,
                        tool_name=tc.name, verdict=_pr.verdict.value,
                    )

                if _pr.verdict == PolicyVerdict.block:
                    _block_content = _pr.block_message or _pr.reason or (
                        f"Tool {tc.name!r} was blocked by policy."
                    )
                    results.append(ToolResult(
                        tool_call_id=tc.id,
                        content=_block_content,
                        error=True,
                    ))
                    yield f"[policy:blocked] {tc.name}: {_pr.reason}"
                    _emit("policy.blocked", Outcome.blocked,
                          session_id=session.id, iteration=iteration,
                          tool_name=tc.name, reason=_pr.reason)
                    continue

                if _pr.verdict == PolicyVerdict.require_confirmation:
                    needs_confirmation.append(tc)
                    yield f"[confirm_required] {tc.name}({tc.arguments})"
                    _emit("policy.confirmation_required", Outcome.blocked,
                          session_id=session.id, iteration=iteration,
                          tool_name=tc.name, reason=_pr.reason)
                    continue
                # -------------------------------------------------

                yield f"[tool] {tc.name}({tc.arguments})"
                exec_result: ExecutionResult = self.tools.execute(tc.name, tc.arguments)

                # Emit any structured logs from the handler
                for line in exec_result.logs:
                    yield f"[log] {tc.name}: {line}"

                if exec_result.requires_confirmation:
                    log.info("Tool %r requires confirmation in session %s", tc.name, session.id)
                    yield f"[confirm_required] {tc.name}({tc.arguments})"
                    needs_confirmation.append(tc)
                else:
                    output = exec_result.output
                    if _pr.verdict == PolicyVerdict.inject_reminder and _pr.reminder:
                        output = f"[Policy reminder: {_pr.reminder}]\n\n{output}"
                        yield f"[policy:reminder] {tc.name}: {_pr.reminder}"
                        _emit("policy.reminder_injected", Outcome.executed,
                              session_id=session.id, iteration=iteration,
                              tool_name=tc.name)
                    tag = "[result:error]" if exec_result.error else "[result]"
                    yield f"{tag} {output}"
                    results.append(ToolResult(
                        tool_call_id=tc.id,
                        content=output,
                        error=exec_result.error,
                    ))

            # 5. If any tools need confirmation, pause the loop
            if needs_confirmation:
                _emit(
                    "loop.confirmation_pause", Outcome.blocked,
                    session_id=session.id, iteration=iteration,
                    pending_count=len(needs_confirmation),
                    pending_tools=[tc.name for tc in needs_confirmation],
                )
                if results:
                    session.messages.append(Message.tool_result_msg(results))
                session.pending_confirmation = needs_confirmation
                self._transition(session, SessionState.waiting_for_confirmation)
                self.store.save(session)
                return  # caller must call resume()

            # 6. Append all tool results immediately after the assistant message
            session.messages.append(Message.tool_result_msg(results))
            self.store.save(session)

            # Checkpoint: tool results persisted — safest resume point
            self._save_checkpoint(session, iteration, LABEL_POST_TOOLS)

            # 7. Loop detection — check after tool_result is safely appended
            if self._loop_detector is not None:
                detection = self._loop_detector.check(
                    session.messages,
                    session_id=session.id,
                    iteration=iteration,
                )
                if detection.detected:
                    intervention = LoopDetector.make_intervention_message(detection)
                    self._intervention_queue.enqueue(intervention)
                    yield f"[loop_detected] {detection.reason}"
            # continue loop

        _emit("loop.max_iterations_exceeded", Outcome.error,
              session_id=session.id, max_iterations=MAX_ITERATIONS)
        raise RuntimeError(
            f"Session {session.id} exceeded MAX_ITERATIONS ({MAX_ITERATIONS})"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self, session: Session, iteration: int, label: str
    ) -> None:
        cp = Checkpoint.new(
            session_id=session.id,
            iteration=iteration,
            message_count=len(session.messages),
            state=session.state,
            label=label,
        )
        self.store.save_checkpoint(cp)
        log.debug("Checkpoint %s saved for session %s (label=%s)", cp.id, session.id, label)

    def _require_session(self, session_id: str) -> Session:
        session = self.store.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id!r}")
        return session

    @staticmethod
    def _assert_can_run(session: Session) -> None:
        if session.state == SessionState.running:
            raise RuntimeError(f"Session {session.id} is already running.")
        if session.state == SessionState.error:
            raise RuntimeError(
                f"Session {session.id} is in error state: {session.error_message}"
            )
        if session.state == SessionState.waiting_for_confirmation:
            raise RuntimeError(
                f"Session {session.id} is waiting for confirmation; call resume() instead."
            )

    @staticmethod
    def _transition(session: Session, new_state: SessionState) -> None:
        log.debug("Session %s: %s → %s", session.id, session.state, new_state)
        _emit(
            "session.transition", Outcome.executed,
            session_id=session.id,
            from_state=session.state.value,
            to_state=new_state.value,
        )
        session.state = new_state
