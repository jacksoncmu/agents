"""Agent engine — runs the ReAct loop."""
from __future__ import annotations

import logging
from typing import Iterator

from agent.llm.base import LLMProvider
from agent.storage import LABEL_POST_LLM, LABEL_POST_TOOLS, Checkpoint, SessionStore
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


class AgentEngine:
    def __init__(
        self,
        store: SessionStore,
        llm: LLMProvider,
        tools: ToolRegistry,
    ) -> None:
        self.store = store
        self.llm = llm
        self.tools = tools

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
        session = self._require_session(session_id)
        self._assert_can_run(session)

        session.messages.append(Message.user(user_message))
        self._transition(session, SessionState.running)
        self.store.save(session)

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
                return ""
        else:
            # Normal continuation from a paused-but-healthy state.
            self._assert_can_run(session)
            self._transition(session, SessionState.running)
            self.store.save(session)

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
        session = self._require_session(session_id)
        self._assert_can_run(session)

        session.messages.append(Message.user(user_message))
        self._transition(session, SessionState.running)
        self.store.save(session)

        try:
            for event in self._react_loop_events(session):
                yield event
        except Exception as exc:
            session = self.store.get(session.id)
            self._transition(session, SessionState.error)
            session.error_message = str(exc)
            self.store.save(session)
            raise

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

        return self._run_loop(session)

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _run_loop(self, session: Session) -> str:
        """Enter the blocking ReAct loop; handle final state transitions."""
        try:
            final_text = self._react_loop(session)
        except Exception as exc:
            session = self.store.get(session.id)
            self._transition(session, SessionState.error)
            session.error_message = str(exc)
            self.store.save(session)
            raise

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
                self._transition(session, SessionState.finished)
                self.store.save(session)
                return

            # Refresh tool schemas each iteration — dynamic visibility
            tool_schemas = self.tools.all_schemas()

            log.debug("Session %s iteration %d (%d tools)", session.id, iteration, len(tool_schemas))

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

            # 3. No tool calls → done
            if not llm_response.has_tool_calls:
                yield llm_response.content
                return

            # 4. Execute tool calls
            results: list[ToolResult] = []
            needs_confirmation: list[ToolCall] = []

            for tc in llm_response.tool_calls:
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
                    tag = "[result:error]" if exec_result.error else "[result]"
                    yield f"{tag} {exec_result.output}"
                    results.append(ToolResult(
                        tool_call_id=tc.id,
                        content=exec_result.output,
                        error=exec_result.error,
                    ))

            # 5. If any tools need confirmation, pause the loop
            if needs_confirmation:
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
            # continue loop

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
        session.state = new_state
