#!/usr/bin/env python3
"""CLI entrypoint for the agent runtime.

Usage:
    python main.py                  # interactive REPL with mock LLM
    python main.py --real           # same but expects ANTHROPIC_API_KEY (future)

During the interactive session type:
    /quit  — end the session
    /state — print current session state
    /msgs  — dump the message log
"""
from __future__ import annotations

import logging
import sys

from agent.engine import AgentEngine
from agent.llm.mock import MockProvider
from agent.llm.base import LLMResponse
from agent.storage import InMemoryStore
from agent.tools import ToolRegistry
from agent.types import ToolCall
from examples.example_tools import make_calculator, make_get_current_time

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)


def build_engine_with_mock() -> tuple[AgentEngine, str]:
    """Build an engine wired to a simple mock LLM that uses the calculator."""

    # Script: first call asks for a tool; second call gives a final answer.
    mock_llm = MockProvider(
        responses=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall.new("calculator", {"expression": "6 * 7"})
                ],
            ),
            LLMResponse(content="The answer is 42."),
        ]
    )

    registry = ToolRegistry()
    registry.register(make_calculator())
    registry.register(make_get_current_time())

    store = InMemoryStore()
    engine = AgentEngine(store=store, llm=mock_llm, tools=registry)
    session = engine.create_session()
    return engine, session.id


def interactive_repl(engine: AgentEngine, session_id: str) -> None:
    print("Agent runtime — interactive session")
    print("Commands: /quit  /state  /msgs")
    print("-" * 40)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input == "/quit":
            print("Goodbye.")
            break
        if user_input == "/state":
            session = engine.store.get(session_id)
            print(f"State: {session.state}")
            continue
        if user_input == "/msgs":
            session = engine.store.get(session_id)
            for i, m in enumerate(session.messages):
                print(f"  [{i}] {m.role}: {m.content!r} tc={m.tool_calls} tr={m.tool_results}")
            continue

        print("Agent: ", end="", flush=True)
        try:
            for event in engine.stream_run(session_id, user_input):
                print(event)
        except RuntimeError as exc:
            print(f"[engine error] {exc}")


if __name__ == "__main__":
    engine, session_id = build_engine_with_mock()
    interactive_repl(engine, session_id)
