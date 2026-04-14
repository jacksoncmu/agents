"""Tests for the tool registry, execution, validation, and dynamic visibility."""
from __future__ import annotations

import pytest

from agent.engine import AgentEngine
from agent.llm.base import LLMResponse
from agent.llm.mock import MockProvider
from agent.storage import InMemoryStore
from agent.tools import (
    ExecutionResult,
    ToolContext,
    ToolDefinition,
    ToolParam,
    ToolRegistry,
    ValidationError,
)
from agent.types import SessionState, ToolCall
from examples.example_tools import (
    make_calculator,
    make_get_current_time,
    make_read_file,
    make_request_confirmation,
    make_search_notes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_tool(name: str = "echo", *, requires_confirmation: bool = False) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="Returns its input.",
        params=[ToolParam("message", "string", "The message to echo.")],
        handler=lambda message: f"echo: {message}",
        requires_confirmation=requires_confirmation,
    )


def _registry(*tools: ToolDefinition) -> ToolRegistry:
    r = ToolRegistry()
    for t in tools:
        r.register(t)
    return r


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_and_get(self):
        r = ToolRegistry()
        t = _simple_tool()
        r.register(t)
        assert r.get("echo") is t

    def test_get_unknown_returns_none(self):
        assert ToolRegistry().get("nope") is None

    def test_names_lists_registered(self):
        r = _registry(_simple_tool("a"), _simple_tool("b"))
        assert set(r.names()) == {"a", "b"}

    def test_unregister_removes_tool(self):
        r = _registry(_simple_tool())
        r.unregister("echo")
        assert r.get("echo") is None
        assert r.names() == []

    def test_unregister_nonexistent_is_silent(self):
        r = ToolRegistry()
        r.unregister("ghost")  # should not raise

    def test_re_register_replaces(self):
        r = ToolRegistry()
        t1 = _simple_tool()
        t2 = ToolDefinition("echo", "v2", [], lambda: "v2")
        r.register(t1)
        r.register(t2)
        assert r.get("echo") is t2

    def test_all_schemas_returns_all(self):
        r = _registry(make_calculator(), make_get_current_time())
        schemas = r.all_schemas()
        names = {s["name"] for s in schemas}
        assert names == {"calculator", "get_current_time"}

    def test_all_schemas_structure(self):
        r = _registry(make_calculator())
        schema = r.all_schemas()[0]
        assert schema["name"] == "calculator"
        assert "description" in schema
        assert schema["parameters"]["type"] == "object"
        assert "expression" in schema["parameters"]["properties"]


# ---------------------------------------------------------------------------
# Execution — success paths
# ---------------------------------------------------------------------------

class TestExecution:
    def test_execute_returns_execution_result(self):
        r = _registry(make_calculator())
        result = r.execute("calculator", {"expression": "2 + 2"})
        assert isinstance(result, ExecutionResult)
        assert result.output == "4"
        assert result.error is False

    def test_execute_unknown_tool_returns_error_result(self):
        r = ToolRegistry()
        result = r.execute("nope", {})
        assert result.error is True
        assert "Unknown tool" in result.output

    def test_execute_does_not_raise_on_handler_exception(self):
        def bad_handler(x: str) -> str:
            raise RuntimeError("boom")

        r = _registry(ToolDefinition("bad", "desc", [ToolParam("x", "string", "x")], bad_handler))
        result = r.execute("bad", {"x": "hi"})
        assert result.error is True
        assert "boom" in result.output

    def test_execute_no_params_tool(self):
        r = _registry(make_get_current_time())
        result = r.execute("get_current_time", {})
        assert result.error is False
        assert "UTC" in result.output

    def test_execute_calculator_division_by_zero(self):
        r = _registry(make_calculator())
        result = r.execute("calculator", {"expression": "1/0"})
        assert result.error is True

    def test_execute_calculator_unsafe_expression(self):
        r = _registry(make_calculator())
        result = r.execute("calculator", {"expression": "__import__('os')"})
        assert result.error is True


# ---------------------------------------------------------------------------
# ToolContext — logs accessible after execution
# ---------------------------------------------------------------------------

class TestToolContextLogs:
    def test_logs_from_handler_returned_in_result(self):
        def handler(path: str, ctx: ToolContext) -> str:
            ctx.log("step 1")
            ctx.log("step 2")
            return "done"

        r = _registry(ToolDefinition(
            "logged_tool", "desc",
            [ToolParam("path", "string", "p")],
            handler,
        ))
        result = r.execute("logged_tool", {"path": "x"})
        assert result.logs == ["step 1", "step 2"]

    def test_logs_empty_when_handler_does_not_use_ctx(self):
        r = _registry(_simple_tool())
        result = r.execute("echo", {"message": "hi"})
        assert result.logs == []

    def test_logs_preserved_on_handler_exception(self):
        def handler(x: str, ctx: ToolContext) -> str:
            ctx.log("before crash")
            raise ValueError("crash")

        r = _registry(ToolDefinition("crash_tool", "d", [ToolParam("x", "string", "x")], handler))
        result = r.execute("crash_tool", {"x": "y"})
        assert result.error is True
        assert result.logs == ["before crash"]

    def test_search_notes_emits_logs(self):
        r = _registry(make_search_notes())
        result = r.execute("search_notes", {"query": "agent"})
        assert len(result.logs) >= 1  # at least one log line
        assert not result.error


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_required_param(self):
        r = _registry(make_calculator())
        result = r.execute("calculator", {})
        assert result.error is True
        assert "expression" in result.output
        assert "required" in result.output.lower()

    def test_wrong_type_rejected(self):
        r = _registry(make_calculator())
        result = r.execute("calculator", {"expression": 42})   # int instead of str
        assert result.error is True
        assert "expression" in result.output

    def test_valid_params_accepted(self):
        r = _registry(make_calculator())
        result = r.execute("calculator", {"expression": "1+1"})
        assert not result.error
        assert result.output == "2"

    def test_enum_valid_value(self):
        r = _registry(ToolDefinition(
            "color", "pick a color",
            [ToolParam("choice", "string", "color", enum=["red", "green", "blue"])],
            lambda choice: f"chose {choice}",
        ))
        result = r.execute("color", {"choice": "red"})
        assert not result.error

    def test_enum_invalid_value(self):
        r = _registry(ToolDefinition(
            "color", "pick a color",
            [ToolParam("choice", "string", "color", enum=["red", "green", "blue"])],
            lambda choice: f"chose {choice}",
        ))
        result = r.execute("color", {"choice": "purple"})
        assert result.error is True
        assert "purple" in result.output

    def test_optional_param_absent_is_ok(self):
        r = _registry(ToolDefinition(
            "greet", "greet",
            [ToolParam("name", "string", "name", required=False)],
            lambda name="world": f"hello {name}",
        ))
        result = r.execute("greet", {})
        assert not result.error

    def test_boolean_type_check(self):
        r = _registry(ToolDefinition(
            "flag", "flag",
            [ToolParam("enabled", "boolean", "flag")],
            lambda enabled: str(enabled),
        ))
        assert not r.execute("flag", {"enabled": True}).error
        assert r.execute("flag", {"enabled": "yes"}).error  # string not bool

    def test_integer_type_check_rejects_float(self):
        r = _registry(ToolDefinition(
            "counter", "counter",
            [ToolParam("n", "integer", "count")],
            lambda n: str(n),
        ))
        assert not r.execute("counter", {"n": 5}).error
        assert r.execute("counter", {"n": 5.5}).error

    def test_integer_type_check_rejects_bool(self):
        """bool is a subclass of int in Python; we reject it for 'integer' type."""
        r = _registry(ToolDefinition(
            "counter", "counter",
            [ToolParam("n", "integer", "count")],
            lambda n: str(n),
        ))
        assert r.execute("counter", {"n": True}).error

    def test_tool_definition_validate_returns_errors(self):
        """ToolDefinition.validate() is usable independently."""
        td = make_calculator()
        errors = td.validate({})
        assert any(e.param == "expression" for e in errors)
        assert isinstance(errors[0], ValidationError)


# ---------------------------------------------------------------------------
# Dynamic visibility — tools registered mid-session are seen next turn
# ---------------------------------------------------------------------------

class TestDynamicVisibility:
    def test_tool_registered_after_first_turn_is_visible_second_turn(self):
        """
        Turn 1: registry has only 'calculator'.
        Between turns: 'search_notes' is registered.
        Turn 2: the schema list passed to the LLM should include 'search_notes'.
        """
        registry = ToolRegistry()
        registry.register(make_calculator())

        captured_schemas: list[list] = []

        class CapturingLLM(MockProvider):
            def complete(self, messages, tools):
                captured_schemas.append(list(tools))
                return super().complete(messages, tools)

        llm = CapturingLLM([
            LLMResponse(content="Turn 1 done."),
            LLMResponse(content="Turn 2 done."),
        ])
        engine = AgentEngine(store=InMemoryStore(), llm=llm, tools=registry)
        sid = engine.create_session().id

        engine.run(sid, "First message.")

        # Register a new tool between turns
        registry.register(make_search_notes())

        engine.run(sid, "Second message.")

        turn1_names = {s["name"] for s in captured_schemas[0]}
        turn2_names = {s["name"] for s in captured_schemas[1]}

        assert "calculator" in turn1_names
        assert "search_notes" not in turn1_names
        assert "search_notes" in turn2_names

    def test_unregistered_tool_disappears_next_turn(self):
        registry = ToolRegistry()
        registry.register(make_calculator())
        registry.register(make_get_current_time())

        captured_schemas: list[list] = []

        class CapturingLLM(MockProvider):
            def complete(self, messages, tools):
                captured_schemas.append(list(tools))
                return super().complete(messages, tools)

        llm = CapturingLLM([
            LLMResponse(content="Turn 1."),
            LLMResponse(content="Turn 2."),
        ])
        engine = AgentEngine(store=InMemoryStore(), llm=llm, tools=registry)
        sid = engine.create_session().id

        engine.run(sid, "First.")
        registry.unregister("get_current_time")
        engine.run(sid, "Second.")

        assert "get_current_time" in {s["name"] for s in captured_schemas[0]}
        assert "get_current_time" not in {s["name"] for s in captured_schemas[1]}


# ---------------------------------------------------------------------------
# Confirmation-requiring tools
# ---------------------------------------------------------------------------

class TestConfirmationRequired:
    def _build_engine_with_confirm_tool(
        self, responses: list[LLMResponse]
    ) -> tuple[AgentEngine, str]:
        registry = ToolRegistry()
        registry.register(make_request_confirmation())
        engine = AgentEngine(
            store=InMemoryStore(),
            llm=MockProvider(responses),
            tools=registry,
        )
        return engine, engine.create_session().id

    def test_confirmation_tool_pauses_session(self):
        tc = ToolCall.new("delete_file", {"filename": "report.txt"})
        engine, sid = self._build_engine_with_confirm_tool([
            LLMResponse(content="", tool_calls=[tc]),
            LLMResponse(content="File deleted."),
        ])

        engine.run(sid, "Delete report.txt")

        session = engine.store.get(sid)
        assert session.state == SessionState.waiting_for_confirmation
        assert len(session.pending_confirmation) == 1
        assert session.pending_confirmation[0].name == "delete_file"

    def test_resume_approved_executes_tool(self):
        tc = ToolCall.new("delete_file", {"filename": "report.txt"})
        engine, sid = self._build_engine_with_confirm_tool([
            LLMResponse(content="", tool_calls=[tc]),
            LLMResponse(content="File deleted."),
        ])

        engine.run(sid, "Delete report.txt")
        result = engine.resume(sid, approved=True)

        assert result == "File deleted."
        session = engine.store.get(sid)
        assert session.state == SessionState.waiting_for_user
        assert session.pending_confirmation == []

    def test_resume_rejected_sends_error_result(self):
        tc = ToolCall.new("delete_file", {"filename": "report.txt"})
        engine, sid = self._build_engine_with_confirm_tool([
            LLMResponse(content="", tool_calls=[tc]),
            LLMResponse(content="Understood, not deleting."),
        ])

        engine.run(sid, "Delete report.txt")
        result = engine.resume(sid, approved=False)

        assert result == "Understood, not deleting."
        session = engine.store.get(sid)
        assert session.state == SessionState.waiting_for_user
        # The tool_result message must contain an error result
        msgs = session.messages
        tr_msg = next(m for m in msgs if m.tool_results)
        assert tr_msg.tool_results[0].error is True
        assert "not approved" in tr_msg.tool_results[0].content.lower()

    def test_resume_on_non_confirmation_state_raises(self):
        registry = ToolRegistry()
        engine = AgentEngine(
            store=InMemoryStore(),
            llm=MockProvider([LLMResponse(content="hi")]),
            tools=registry,
        )
        sid = engine.create_session().id
        engine.run(sid, "hello")

        with pytest.raises(RuntimeError, match="not waiting for confirmation"):
            engine.resume(sid, approved=True)

    def test_execute_confirmed_bypasses_gate(self):
        """execute_confirmed() runs the handler even when requires_confirmation=True."""
        r = _registry(make_request_confirmation())
        result = r.execute_confirmed("delete_file", {"filename": "x.txt"})
        assert not result.error
        assert "deleted" in result.output.lower()

    def test_execute_without_approval_returns_confirmation_result(self):
        r = _registry(make_request_confirmation())
        result = r.execute("delete_file", {"filename": "x.txt"})
        assert result.requires_confirmation is True
        assert not result.error


# ---------------------------------------------------------------------------
# Example tools: read_file
# ---------------------------------------------------------------------------

class TestReadFile:
    def test_read_existing_file(self, tmp_path):
        (tmp_path / "hello.txt").write_text("hello world")
        r = _registry(make_read_file(tmp_path))
        result = r.execute("read_file", {"path": "hello.txt"})
        assert not result.error
        assert result.output == "hello world"

    def test_logs_resolved_path(self, tmp_path):
        (tmp_path / "f.txt").write_text("data")
        r = _registry(make_read_file(tmp_path))
        result = r.execute("read_file", {"path": "f.txt"})
        assert any("resolved path" in l for l in result.logs)

    def test_missing_file_returns_error(self, tmp_path):
        r = _registry(make_read_file(tmp_path))
        result = r.execute("read_file", {"path": "missing.txt"})
        assert result.error

    def test_path_traversal_blocked(self, tmp_path):
        r = _registry(make_read_file(tmp_path))
        result = r.execute("read_file", {"path": "../../etc/passwd"})
        assert result.error
        assert "escapes" in result.output

    def test_missing_path_param_fails_validation(self, tmp_path):
        r = _registry(make_read_file(tmp_path))
        result = r.execute("read_file", {})
        assert result.error
        assert "Validation" in result.output


# ---------------------------------------------------------------------------
# Example tools: search_notes
# ---------------------------------------------------------------------------

class TestSearchNotes:
    def test_finds_matching_note(self):
        notes = [
            {"title": "A", "text": "Python is great"},
            {"title": "B", "text": "Java is verbose"},
        ]
        r = _registry(make_search_notes(notes))
        result = r.execute("search_notes", {"query": "Python"})
        assert not result.error
        assert "Python is great" in result.output
        assert "Java" not in result.output

    def test_no_match_returns_message(self):
        r = _registry(make_search_notes([{"title": "X", "text": "nothing here"}]))
        result = r.execute("search_notes", {"query": "zzz_nomatch"})
        assert not result.error
        assert "No notes found" in result.output

    def test_case_insensitive(self):
        notes = [{"title": "T", "text": "HELLO WORLD"}]
        r = _registry(make_search_notes(notes))
        result = r.execute("search_notes", {"query": "hello"})
        assert not result.error
        assert "HELLO WORLD" in result.output
