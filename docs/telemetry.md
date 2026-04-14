# Telemetry Reference

The runtime emits structured events at every decision point.  Each event
carries a name, an outcome, optional session/iteration context, and a
payload of key/value data.  The default backend discards all events; swap
it in to observe the system.

---

## Outcome vocabulary

| Value | Meaning |
|---|---|
| `executed` | The action was taken successfully. |
| `condition_unsatisfied` | The decision point was reached; the guard condition was not met; the action was skipped as expected. Maps to *not_executed_condition_unsatisfied* in external tracing conventions. |
| `blocked` | A policy gate, lock, or cancellation check intercepted the action before it ran. |
| `error` | An error was encountered; details are in the event payload. |

**Reading a missing event**: if a code path was never reached, its event
will simply be absent from the backend.  For example, `loop.cancelled` only
appears if cancellation was set before the loop started.  Absence is
information — it means the path was never entered.

---

## Event catalogue

### `tool.validate`

Emitted by `ToolRegistry.execute()` and `ToolRegistry.execute_confirmed()`
immediately after input validation runs.

| Outcome | Condition |
|---|---|
| `executed` | All parameters passed validation. |
| `error` | One or more parameters failed. |

Payload:

| Key | Type | Present when |
|---|---|---|
| `tool_name` | str | always |
| `errors` | list[{param, message}] | outcome = error |
| `confirmed` | bool | `execute_confirmed()` path |

---

### `tool.confirmation_gate`

Emitted by `ToolRegistry.execute()` when a tool has
`requires_confirmation=True` and is intercepted before running.  Validation
has already passed at this point.

| Outcome | Condition |
|---|---|
| `blocked` | Tool requires human approval; handler not called. |

Payload:

| Key | Type | Present when |
|---|---|---|
| `tool_name` | str | always |

**Note**: `tool.execute` is *not* emitted after `tool.confirmation_gate`
because the handler never runs.  When the human approves via `resume()`,
`execute_confirmed()` is called, which emits `tool.validate` and
`tool.execute` with `confirmed=True`.

---

### `tool.execute`

Emitted by `ToolRegistry._invoke()` after the handler returns or raises,
and by `ToolRegistry.execute()` for the unknown-tool fast path.

| Outcome | Condition |
|---|---|
| `executed` | Handler returned without raising. |
| `error` | Handler raised an exception, or the tool name was not found in the registry. |

Payload:

| Key | Type | Present when |
|---|---|---|
| `tool_name` | str | always |
| `confirmed` | bool | always (False for normal path, True for confirmed path) |
| `error` | str | outcome = error (handler exception message) |
| `reason` | str | outcome = error (unknown tool: `"unknown_tool"`) |

---

### `loop.iteration.start`

Emitted once at the top of each engine loop iteration, after the
cancellation check passes.

| Outcome | Condition |
|---|---|
| `executed` | Iteration proceeding normally. |

Payload:

| Key | Type |
|---|---|
| `tool_count` | int — number of tools visible to the LLM this iteration |

Context: `session_id`, `iteration`

---

### `loop.cancelled`

Emitted when the per-iteration cancellation flag check fires at the
*start* of an iteration (before the LLM is called).

| Outcome | Condition |
|---|---|
| `blocked` | `session.cancelled` was True at iteration start. |

Context: `session_id`, `iteration`

**Invariant**: `loop.iteration.start` is *not* emitted for the same
iteration — the check runs before that point.

---

### `loop.cancelled_between_tools`

Emitted when the per-tool cancellation check fires *between* tool
executions within a single iteration.

| Outcome | Condition |
|---|---|
| `blocked` | `session.cancelled` was True between two consecutive tool calls. |

Payload:

| Key | Type |
|---|---|
| `pending_tool` | str — name of the tool that was about to run |

Context: `session_id`, `iteration`

---

### `loop.final_answer`

Emitted when the LLM returns a response with no tool calls — the loop
exits cleanly with a final answer.

| Outcome | Condition |
|---|---|
| `condition_unsatisfied` | LLM produced text only; no further tool calls needed. |

The outcome is `condition_unsatisfied` because the "should we keep looping?"
condition was evaluated and found to be false — a completely normal,
expected exit.

Payload:

| Key | Type |
|---|---|
| `content_length` | int — character length of the assistant's final answer |

Context: `session_id`, `iteration`

---

### `loop.confirmation_pause`

Emitted when the loop pauses because one or more tools requested human
confirmation.

| Outcome | Condition |
|---|---|
| `blocked` | At least one tool returned `requires_confirmation=True`. |

Payload:

| Key | Type |
|---|---|
| `pending_count` | int — number of tools awaiting confirmation |
| `pending_tools` | list[str] — tool names in order |

Context: `session_id`, `iteration`

---

### `loop.max_iterations_exceeded`

Emitted immediately before raising `RuntimeError` when the loop hits the
`MAX_ITERATIONS` safety ceiling.

| Outcome | Condition |
|---|---|
| `error` | `MAX_ITERATIONS` iterations completed without a final answer. |

Payload:

| Key | Type |
|---|---|
| `max_iterations` | int — the ceiling that was hit |

Context: `session_id`

---

### `run.acquire`

Emitted by `AgentEngine._acquire_run()` — called at the entry point of
`run()`, `continue_run()`, `stream_run()`, and `resume()`.

| Outcome | Condition |
|---|---|
| `executed` | Ownership acquired; run may proceed. |
| `blocked` | Another run already holds ownership; `ConcurrentRunError` is raised. |

Context: `session_id`

---

### `run.release`

Emitted by `AgentEngine._release_run()` — called from `_run_loop()`'s
`finally` block and from all pre-loop exception handlers.

| Outcome | Condition |
|---|---|
| `executed` | Ownership released (always). |

Context: `session_id`

**Invariant**: every `run.acquire` with `executed` will eventually produce
a matching `run.release`, regardless of whether the run completed
successfully, errored, or was cancelled.

---

### `session.transition`

Emitted by `AgentEngine._transition()` on every session state change.

| Outcome | Condition |
|---|---|
| `executed` | State transition applied. |

Payload:

| Key | Type |
|---|---|
| `from_state` | str — previous `SessionState` value |
| `to_state` | str — new `SessionState` value |

Context: `session_id`

Common transition sequences:

```
Normal run:
  waiting_for_user → running → waiting_for_user

Cancellation:
  waiting_for_user → running → finished

LLM error:
  waiting_for_user → running → error

Confirmation required:
  waiting_for_user → running → waiting_for_confirmation → running → waiting_for_user
```

---

## Using `InMemoryBackend` in tests

```python
from agent.telemetry import InMemoryBackend, set_backend, NullBackend, Outcome

@pytest.fixture(autouse=True)
def telemetry():
    backend = InMemoryBackend()
    set_backend(backend)
    yield backend
    set_backend(NullBackend())

def test_tool_executes(telemetry):
    registry.execute("calculator", {"expression": "1+1"})
    events = telemetry.by_name("tool.execute")
    assert events[0].outcome == Outcome.executed
```

Useful `InMemoryBackend` methods:

| Method | Returns |
|---|---|
| `.events` | `list[Event]` — all events in order |
| `.by_name(name)` | `list[Event]` — filtered by event name |
| `.by_outcome(outcome)` | `list[Event]` — filtered by outcome |
| `.first(name)` | `Event \| None` — first matching event |
| `.names()` | `list[str]` — distinct names seen, in first-appearance order |
| `.clear()` | resets the event list |

---

## Swapping to a real backend

```python
from agent.telemetry import set_backend

class MyOTelBackend:
    def emit(self, event):
        span = tracer.start_span(event.name)
        span.set_attribute("outcome", event.outcome.value)
        for k, v in event.data.items():
            span.set_attribute(k, str(v))
        span.end()

set_backend(MyOTelBackend())
```

The swap is thread-safe and takes effect immediately for all subsequent
`emit()` calls.

---

## What is NOT instrumented

- Individual LLM API calls (latency, tokens) — belongs at the provider layer.
- Storage I/O (read/write latency) — belongs at the store layer.
- Message content — avoid logging PII into telemetry payloads.
