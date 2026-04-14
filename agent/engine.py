"""Agent engine — runs the ReAct loop."""
from __future__ import annotations

import logging
from typing import Iterator

from agent.llm.base import LLMProvider
from agent.storage import SessionStore
from agent.tools import ToolRegistry
from agent.types import (
    Message,
    MessageRole,
    Session,
    SessionState,
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
        session = self._require_session(session_id)
        session.cancelled = True
        self.store.save(session)

    def run(self, session_id: str, user_message: str) -> str:
        """
        Accept a user message, run the ReAct loop to completion, and return
        the final assistant text.

        Raises RuntimeError if the session is in a terminal or incompatible
        state.
        """
        session = self._require_session(session_id)
        self._assert_can_run(session)

        # Append user turn
        session.messages.append(Message.user(user_message))
        self._transition(session, SessionState.running)
        self.store.save(session)

        try:
            final_text = self._react_loop(session)
        except Exception as exc:
            session = self.store.get(session.id)
            self._transition(session, SessionState.error)
            session.error_message = str(exc)
            self.store.save(session)
            raise

        # Re-read: the loop may have set a terminal state (e.g. cancellation).
        session = self.store.get(session.id)
        if session.state == SessionState.running:
            self._transition(session, SessionState.waiting_for_user)
            self.store.save(session)
        return final_text

    def stream_run(self, session_id: str, user_message: str) -> Iterator[str]:
        """
        Thin generator wrapper around run() that yields progress lines.
        Useful for the CLI without pulling in async machinery.
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

        # Re-read: the loop may have set the state to finished (e.g. cancellation).
        session = self.store.get(session.id)
        if session.state == SessionState.running:
            self._transition(session, SessionState.waiting_for_user)
            self.store.save(session)

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _react_loop(self, session: Session) -> str:
        """Blocking ReAct loop; returns the final assistant text."""
        final_text = ""
        for event in self._react_loop_events(session):
            final_text = event  # last event is the final answer
        return final_text

    def _react_loop_events(self, session: Session) -> Iterator[str]:
        """
        Core loop generator.  Yields string events:
          - "[tool] <name>(<args>)"  when a tool is called
          - "[result] <content>"     after a tool returns
          - the final assistant text when done
        """
        tool_schemas = self.tools.all_schemas()

        for iteration in range(MAX_ITERATIONS):
            # --- cancellation check (required at top of every iteration) ---
            session = self.store.get(session.id)  # re-read in case cancelled
            if session.cancelled:
                log.info("Session %s cancelled at iteration %d", session.id, iteration)
                self._transition(session, SessionState.finished)
                self.store.save(session)
                return

            log.debug("Session %s iteration %d", session.id, iteration)

            # 1. Call the model
            llm_response = self.llm.complete(session.messages, tool_schemas)

            # 2. Append assistant message (content + any tool calls)
            assistant_msg = Message.assistant(
                content=llm_response.content,
                tool_calls=llm_response.tool_calls,
            )
            session.messages.append(assistant_msg)
            self.store.save(session)

            # 3. No tool calls → we are done; leave state transition to the caller
            if not llm_response.has_tool_calls:
                self.store.save(session)
                yield llm_response.content
                return

            # 4. Execute tool calls and collect results
            results: list[ToolResult] = []
            for tc in llm_response.tool_calls:
                event_label = f"[tool] {tc.name}({tc.arguments})"
                log.debug(event_label)
                yield event_label

                try:
                    output = self.tools.execute(tc.name, tc.arguments)
                    results.append(ToolResult(tool_call_id=tc.id, content=output))
                    yield f"[result] {output}"
                except Exception as exc:
                    error_text = f"Error: {exc}"
                    results.append(
                        ToolResult(tool_call_id=tc.id, content=error_text, error=True)
                    )
                    yield f"[result:error] {error_text}"

            # 5. Append tool_result message *immediately* after assistant message
            session.messages.append(Message.tool_result_msg(results))
            self.store.save(session)
            # continue loop

        # Exceeded MAX_ITERATIONS
        raise RuntimeError(
            f"Session {session.id} exceeded MAX_ITERATIONS ({MAX_ITERATIONS})"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_session(self, session_id: str) -> Session:
        session = self.store.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id!r}")
        return session

    @staticmethod
    def _assert_can_run(session: Session) -> None:
        if session.state in (SessionState.running,):
            raise RuntimeError(
                f"Session {session.id} is already running."
            )
        if session.state == SessionState.error:
            raise RuntimeError(
                f"Session {session.id} is in error state: {session.error_message}"
            )

    @staticmethod
    def _transition(session: Session, new_state: SessionState) -> None:
        log.debug("Session %s: %s → %s", session.id, session.state, new_state)
        session.state = new_state
