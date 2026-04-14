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
│       ├── base.py        # LLMProvider ABC + LLMResponse
│       ├── config.py      # ModelConfig — provider/model selection
│       ├── adapter.py     # BaseAdapter ABC
│       ├── anthropic.py   # AnthropicAdapter + AnthropicProvider
│       ├── registry.py    # provider_from_config() factory
│       └── mock.py        # Scripted mock for testing
├── examples/
│   └── example_tools.py       # calculator + get_current_time
├── tests/
│   ├── test_e2e.py                  # Engine loop tests (mock LLM)
│   └── test_anthropic_adapter.py    # Adapter conversion tests
├── main.py                # Interactive CLI / demo
└── requirements.txt
```

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests (no API key needed)
pytest tests/ -v

# Run the interactive CLI (uses mock LLM with a scripted 2-turn exchange)
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

| State                      | Meaning                                   |
|----------------------------|-------------------------------------------|
| `waiting_for_user`         | Ready to accept a new user message        |
| `running`                  | Loop is executing                         |
| `waiting_for_confirmation` | Reserved for human-in-the-loop use        |
| `finished`                 | Loop completed (or cancelled)             |
| `error`                    | Unhandled exception; session is dead      |

---

## Configuring a provider

### Anthropic (built-in)

```python
from agent.llm import ModelConfig, provider_from_config
from agent.engine import AgentEngine
from agent.storage import InMemoryStore
from agent.tools import ToolRegistry

config = ModelConfig.anthropic(
    model="claude-sonnet-4-6",
    # api_key="sk-ant-..."  # or set ANTHROPIC_API_KEY env var
    max_tokens=4096,
)
llm = provider_from_config(config)

engine = AgentEngine(
    store=InMemoryStore(),
    llm=llm,
    tools=ToolRegistry(),
)
session = engine.create_session()
result = engine.run(session.id, "What is 6 times 7?")
```

The API key is read from `ANTHROPIC_API_KEY` if not passed directly.

### Portable vs provider-specific fields

**Portable** — accepted by every provider via `ModelConfig`:

| Field         | Default | Notes                            |
|---------------|---------|----------------------------------|
| `provider`    | —       | `"anthropic"`, `"openai"`, …     |
| `model`       | —       | Model ID as the provider names it|
| `api_key`     | `""`    | Falls back to env var            |
| `base_url`    | `""`    | Override endpoint                |
| `max_tokens`  | `4096`  | Max response tokens              |
| `temperature` | `None`  | Provider default when `None`     |

**Provider-specific** — pass through `extra={}` or as keyword arguments to
the convenience constructors.  Each adapter picks out what it understands and
ignores the rest.

#### Anthropic extras

```python
# Extended thinking
config = ModelConfig.anthropic(
    model="claude-sonnet-4-6",
    thinking={"type": "enabled", "budget_tokens": 5000},
)

# Custom system prompt
config = ModelConfig.anthropic(
    model="claude-sonnet-4-6",
    system="You are a concise assistant.",
)
```

`thinking` and `system` are Anthropic-specific.  When `thinking` is set,
temperature is automatically forced to `1` (API requirement).

---

## Adding a new provider

1. Implement `BaseAdapter` for message translation:

```python
from agent.llm.adapter import BaseAdapter
from agent.llm.base import LLMResponse
from agent.types import Message

class OpenAIAdapter(BaseAdapter):
    def to_provider_messages(self, messages):  ...
    def to_provider_tools(self, tools):        ...
    def from_provider_response(self, raw):     ...
```

2. Implement `LLMProvider`:

```python
from agent.llm.base import LLMProvider, LLMResponse
from agent.llm.config import ModelConfig

class OpenAIProvider(LLMProvider):
    def __init__(self, config: ModelConfig): ...
    def complete(self, messages, tools) -> LLMResponse: ...
```

3. Register it once at startup:

```python
from agent.llm.registry import register_provider
register_provider("openai", OpenAIProvider)
```

4. Use it via config:

```python
config = ModelConfig(provider="openai", model="gpt-4o", max_tokens=2048)
llm = provider_from_config(config)
```

---

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
