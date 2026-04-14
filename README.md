# Agent Runtime

A minimal, modular ReAct-style agent harness in Python.

## Structure

```
agents/
├── agent/
│   ├── engine.py      # ReAct loop and state machine
│   ├── storage.py     # Session store (in-memory)
│   ├── tools.py       # Tool registry and execution
│   ├── types.py       # Shared types (Message, Session, ToolCall, …)
│   └── llm/
│       ├── base.py    # Abstract LLMProvider
│       └── mock.py    # Scripted mock for testing
├── examples/
│   └── example_tools.py   # calculator + get_current_time
├── tests/
│   └── test_e2e.py        # End-to-end tests
├── main.py                # Interactive CLI / demo
└── requirements.txt
```

## Quick start

```bash
# Install dependencies (only pytest)
pip install -r requirements.txt

# Run the end-to-end tests
pytest tests/ -v

# Run the interactive CLI (uses the mock LLM with a scripted 2-turn exchange)
python main.py
```

## How the loop works

```
user message
     │
     ▼
[engine] append user message to session
     │
     ▼  ┌──────────────────────────────┐
[loop]  │  check cancellation flag     │
     │  │  call LLM                    │
     │  │  append assistant message    │
     │  │  if tool_calls:              │
     │  │    execute each tool         │
     │  │    append tool_result msg    │
     │  │    continue ──────────────►──┘
     │  │  else:
     │  │    yield final answer
     │  └──── stop
     ▼
session state → waiting_for_user
```

## Session states

| State                | Meaning                                    |
|---------------------|--------------------------------------------|
| `waiting_for_user`  | Ready to accept a new user message         |
| `running`           | Loop is executing                          |
| `waiting_for_confirmation` | Reserved for human-in-the-loop use  |
| `finished`          | Loop completed (or cancelled)              |
| `error`             | Unhandled exception; session is dead       |

## Adding a real LLM provider

Implement `agent.llm.base.LLMProvider`:

```python
from agent.llm.base import LLMProvider, LLMResponse
from agent.types import Message, ToolCall

class AnthropicProvider(LLMProvider):
    def complete(self, messages, tools) -> LLMResponse:
        # translate messages → Anthropic API format, call API, translate back
        ...
```

## Adding tools

```python
from agent.tools import ToolDefinition, ToolParam

def my_tool() -> ToolDefinition:
    def handler(query: str) -> str:
        return f"result for {query}"

    return ToolDefinition(
        name="my_tool",
        description="Does something useful.",
        params=[ToolParam("query", "string", "The query to run.")],
        handler=handler,
    )

registry.register(my_tool())
```
