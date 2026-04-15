"""Tests for hardened execution and output handling.

Tests are grouped by:
  - SafeExecValidation      command validation / rejection
  - SafeExecExecution       actual subprocess runs
  - BlobStore               blob storage CRUD
  - OutputLimitConfig       cap configuration
  - OutputLimiter           truncation and blob redirection
  - ToolRegistryIntegration output limiter wired into registry
  - EngineIntegration       engine handling of referenced outputs
  - Telemetry               output.capped events
"""
from __future__ import annotations

import sys

import pytest

from agent.blob_store import BLOB_REF_PREFIX, BlobRecord, InMemoryBlobStore
from agent.engine import AgentEngine
from agent.llm.base import LLMResponse
from agent.llm.mock import MockProvider
from agent.output_limiter import DEFAULT_CAPS, OutputLimitConfig, OutputLimiter
from agent.safe_exec import (
    ExecutionTimeoutError,
    ProcessResult,
    UnsafeCommandError,
    check_shell_metachars,
    safe_exec,
    safe_exec_output,
    validate_command,
)
from agent.storage import InMemoryStore
from agent.telemetry import InMemoryBackend, NullBackend, Outcome, set_backend
from agent.tools import ToolDefinition, ToolParam, ToolRegistry
from agent.types import Message, MessageRole, SessionState, ToolCall, ToolResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def telemetry_backend():
    backend = InMemoryBackend()
    set_backend(backend)
    yield backend
    set_backend(NullBackend())


# ===========================================================================
# Safe execution — validation
# ===========================================================================

class TestSafeExecValidation:
    def test_string_command_rejected(self):
        with pytest.raises(UnsafeCommandError, match="String commands"):
            validate_command("ls -la /tmp")

    def test_string_with_pipes_rejected(self):
        with pytest.raises(UnsafeCommandError, match="String commands"):
            validate_command("cat /etc/passwd | grep root")

    def test_string_with_semicolons_rejected(self):
        with pytest.raises(UnsafeCommandError, match="String commands"):
            validate_command("echo hello; rm -rf /")

    def test_empty_list_rejected(self):
        with pytest.raises(UnsafeCommandError, match="empty"):
            validate_command([])

    def test_non_string_element_rejected(self):
        with pytest.raises(UnsafeCommandError, match="index 1"):
            validate_command(["ls", 42])

    def test_non_list_non_string_rejected(self):
        with pytest.raises(UnsafeCommandError, match="must be a list"):
            validate_command(123)

    def test_valid_list_accepted(self):
        result = validate_command(["git", "log", "--oneline", "-5"])
        assert result == ["git", "log", "--oneline", "-5"]

    def test_tuple_accepted(self):
        result = validate_command(("echo", "hello"))
        assert result == ["echo", "hello"]

    def test_single_element_list_accepted(self):
        result = validate_command(["whoami"])
        assert result == ["whoami"]


class TestShellMetacharWarnings:
    def test_clean_args_no_warnings(self):
        warnings = check_shell_metachars(["git", "log", "--oneline"])
        assert warnings == []

    def test_semicolons_warn(self):
        warnings = check_shell_metachars(["echo", "hello; rm -rf /"])
        assert len(warnings) == 1
        assert "shell metacharacters" in warnings[0]

    def test_pipe_warns(self):
        warnings = check_shell_metachars(["cat", "file | grep foo"])
        assert len(warnings) == 1

    def test_backtick_warns(self):
        warnings = check_shell_metachars(["echo", "`whoami`"])
        assert len(warnings) == 1

    def test_dollar_warns(self):
        warnings = check_shell_metachars(["echo", "$HOME"])
        assert len(warnings) == 1


# ===========================================================================
# Safe execution — actual subprocess
# ===========================================================================

class TestSafeExecExecution:
    def test_simple_command(self):
        result = safe_exec([sys.executable, "-c", "print('hello')"])
        assert result.success
        assert "hello" in result.stdout

    def test_failing_command_returns_nonzero(self):
        result = safe_exec([sys.executable, "-c", "import sys; sys.exit(1)"])
        assert not result.success
        assert result.returncode == 1

    def test_stderr_captured(self):
        result = safe_exec([sys.executable, "-c",
                           "import sys; sys.stderr.write('err\\n')"])
        assert "err" in result.stderr

    def test_command_not_found(self):
        result = safe_exec(["nonexistent_command_12345"])
        assert not result.success
        assert "not found" in result.stderr.lower() or result.returncode == -1

    def test_string_command_raises_before_exec(self):
        """String commands must never reach subprocess."""
        with pytest.raises(UnsafeCommandError):
            safe_exec("echo hello")  # type: ignore[arg-type]

    def test_timeout_handling(self):
        result = safe_exec(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            timeout=0.5,
        )
        assert result.timed_out
        assert not result.success

    def test_safe_exec_output_returns_stdout(self):
        stdout = safe_exec_output([sys.executable, "-c", "print('world')"])
        assert "world" in stdout

    def test_safe_exec_output_raises_on_failure(self):
        with pytest.raises(RuntimeError, match="Command failed"):
            safe_exec_output([sys.executable, "-c", "import sys; sys.exit(1)"])

    def test_safe_exec_output_raises_on_timeout(self):
        with pytest.raises(ExecutionTimeoutError):
            safe_exec_output(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                timeout=0.5,
            )


# ===========================================================================
# Blob store
# ===========================================================================

class TestBlobStore:
    def test_put_and_get(self):
        store = InMemoryBlobStore()
        ref = store.put("hello world")
        assert ref.startswith(BLOB_REF_PREFIX)
        assert store.get(ref) == "hello world"

    def test_get_nonexistent(self):
        store = InMemoryBlobStore()
        assert store.get("blob:nonexistent") is None

    def test_get_record(self):
        store = InMemoryBlobStore()
        ref = store.put("data", metadata={"tool": "shell"})
        record = store.get_record(ref)
        assert isinstance(record, BlobRecord)
        assert record.content == "data"
        assert record.metadata["tool"] == "shell"

    def test_delete(self):
        store = InMemoryBlobStore()
        ref = store.put("temp")
        assert store.delete(ref) is True
        assert store.get(ref) is None
        assert store.delete(ref) is False

    def test_list_refs(self):
        store = InMemoryBlobStore()
        r1 = store.put("a")
        r2 = store.put("b")
        refs = store.list_refs()
        assert r1 in refs
        assert r2 in refs

    def test_count(self):
        store = InMemoryBlobStore()
        assert store.count == 0
        store.put("x")
        assert store.count == 1
        store.put("y")
        assert store.count == 2

    def test_metadata_optional(self):
        store = InMemoryBlobStore()
        ref = store.put("no meta")
        record = store.get_record(ref)
        assert record.metadata == {}

    def test_unique_refs(self):
        store = InMemoryBlobStore()
        refs = {store.put(f"data-{i}") for i in range(50)}
        assert len(refs) == 50  # all unique


# ===========================================================================
# OutputLimitConfig
# ===========================================================================

class TestOutputLimitConfig:
    def test_default_caps(self):
        cfg = OutputLimitConfig()
        effective = cfg.effective_caps()
        assert effective["file_read"] == 20_000
        assert effective["search"] == 10_000
        assert effective["shell"] == 15_000
        assert effective["default"] == 15_000

    def test_custom_cap_overrides_default(self):
        cfg = OutputLimitConfig(caps={"file_read": 5_000})
        assert cfg.effective_caps()["file_read"] == 5_000
        # Others unchanged
        assert cfg.effective_caps()["shell"] == 15_000

    def test_tool_type_mapping(self):
        cfg = OutputLimitConfig(tool_types={"read_file": "file_read"})
        assert cfg.cap_for_tool("read_file") == 20_000

    def test_unknown_tool_gets_default(self):
        cfg = OutputLimitConfig()
        assert cfg.cap_for_tool("random_tool") == 15_000

    def test_custom_type_with_custom_cap(self):
        cfg = OutputLimitConfig(
            caps={"custom_type": 5_000},
            tool_types={"my_tool": "custom_type"},
        )
        assert cfg.cap_for_tool("my_tool") == 5_000


# ===========================================================================
# OutputLimiter — truncation and blob redirection
# ===========================================================================

class TestOutputLimiter:
    def test_short_output_unchanged(self):
        limiter = OutputLimiter()
        output = "hello"
        result = limiter.limit("some_tool", output)
        assert result == "hello"

    def test_oversized_output_truncated(self):
        limiter = OutputLimiter(
            config=OutputLimitConfig(caps={"default": 100}),
        )
        big = "x" * 500
        result = limiter.limit("some_tool", big)
        assert "[OUTPUT TRUNCATED" in result
        assert "500 chars" in result
        assert "100 cap" in result

    def test_truncated_output_contains_blob_ref(self):
        limiter = OutputLimiter(
            config=OutputLimitConfig(caps={"default": 50}),
        )
        result = limiter.limit("t", "y" * 200)
        assert BLOB_REF_PREFIX in result
        assert "blob_store.get" in result

    def test_full_content_stored_in_blob(self):
        blob_store = InMemoryBlobStore()
        limiter = OutputLimiter(
            config=OutputLimitConfig(caps={"default": 50}),
            blob_store=blob_store,
        )
        original = "data" * 100  # 400 chars
        result = limiter.limit("t", original)

        # Extract ref from the stub
        ref = None
        for line in result.splitlines():
            if line.startswith("Full output stored as:"):
                ref = line.split(": ", 1)[1].strip()
                break
        assert ref is not None
        assert blob_store.get(ref) == original

    def test_blob_metadata_populated(self):
        blob_store = InMemoryBlobStore()
        limiter = OutputLimiter(
            config=OutputLimitConfig(caps={"default": 10}),
            blob_store=blob_store,
        )
        limiter.limit("my_tool", "x" * 50, session_id="s1")

        ref = blob_store.list_refs()[0]
        record = blob_store.get_record(ref)
        assert record.metadata["tool_name"] == "my_tool"
        assert record.metadata["original_length"] == 50
        assert record.metadata["session_id"] == "s1"

    def test_exact_cap_not_truncated(self):
        limiter = OutputLimiter(
            config=OutputLimitConfig(caps={"default": 100}),
        )
        result = limiter.limit("t", "a" * 100)
        assert result == "a" * 100

    def test_one_over_cap_truncated(self):
        limiter = OutputLimiter(
            config=OutputLimitConfig(caps={"default": 100}),
        )
        result = limiter.limit("t", "a" * 101)
        assert "[OUTPUT TRUNCATED" in result

    def test_per_type_caps_applied(self):
        limiter = OutputLimiter(
            config=OutputLimitConfig(
                caps={"file_read": 50, "shell": 200},
                tool_types={"read_file": "file_read", "exec_cmd": "shell"},
            ),
        )
        # read_file cap=50, should be truncated at 51+
        result_file = limiter.limit("read_file", "z" * 60)
        assert "[OUTPUT TRUNCATED" in result_file

        # exec_cmd cap=200, should pass at 60
        result_shell = limiter.limit("exec_cmd", "z" * 60)
        assert result_shell == "z" * 60

    def test_multiple_truncations_produce_unique_refs(self):
        blob_store = InMemoryBlobStore()
        limiter = OutputLimiter(
            config=OutputLimitConfig(caps={"default": 10}),
            blob_store=blob_store,
        )
        limiter.limit("t", "a" * 50)
        limiter.limit("t", "b" * 50)
        assert blob_store.count == 2
        refs = blob_store.list_refs()
        assert refs[0] != refs[1]


# ===========================================================================
# ToolRegistry integration — output limiter wired in
# ===========================================================================

class TestToolRegistryIntegration:
    def test_registry_without_limiter_no_truncation(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            "big", "returns big output", [],
            lambda: "x" * 100_000,
        ))
        result = registry.execute("big", {})
        assert len(result.output) == 100_000

    def test_registry_with_limiter_truncates(self):
        blob_store = InMemoryBlobStore()
        limiter = OutputLimiter(
            config=OutputLimitConfig(caps={"default": 100}),
            blob_store=blob_store,
        )
        registry = ToolRegistry(output_limiter=limiter)
        registry.register(ToolDefinition(
            "big", "returns big output", [],
            lambda: "x" * 500,
        ))
        result = registry.execute("big", {})
        assert "[OUTPUT TRUNCATED" in result.output
        assert not result.error
        assert blob_store.count == 1

    def test_error_results_not_limited(self):
        """Error outputs from failed handlers should NOT be capped
        (they're already short error messages)."""
        limiter = OutputLimiter(
            config=OutputLimitConfig(caps={"default": 10}),
        )
        registry = ToolRegistry(output_limiter=limiter)
        registry.register(ToolDefinition(
            "fail", "fails", [],
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        ))
        result = registry.execute("fail", {})
        assert result.error
        # Error message should be the raw error, not truncated
        assert "boom" in result.output

    def test_limiter_applies_to_confirmed_execution(self):
        blob_store = InMemoryBlobStore()
        limiter = OutputLimiter(
            config=OutputLimitConfig(caps={"default": 50}),
            blob_store=blob_store,
        )
        registry = ToolRegistry(output_limiter=limiter)
        registry.register(ToolDefinition(
            "big_confirm", "big confirmed tool", [],
            lambda: "y" * 200,
            requires_confirmation=True,
        ))
        result = registry.execute_confirmed("big_confirm", {})
        assert "[OUTPUT TRUNCATED" in result.output
        assert blob_store.count == 1

    def test_limiter_with_tool_type_mapping(self):
        blob_store = InMemoryBlobStore()
        limiter = OutputLimiter(
            config=OutputLimitConfig(
                caps={"file_read": 30},
                tool_types={"read_file": "file_read"},
            ),
            blob_store=blob_store,
        )
        registry = ToolRegistry(output_limiter=limiter)
        registry.register(ToolDefinition(
            "read_file", "reads a file", [],
            lambda: "content " * 20,  # 160 chars > 30 cap
        ))
        result = registry.execute("read_file", {})
        assert "[OUTPUT TRUNCATED" in result.output
        assert "30 cap" in result.output


# ===========================================================================
# Engine integration — handling referenced outputs
# ===========================================================================

class TestEngineIntegration:
    def test_engine_with_output_limiting(self, telemetry_backend):
        """Full engine run where a tool produces oversized output."""
        blob_store = InMemoryBlobStore()
        limiter = OutputLimiter(
            config=OutputLimitConfig(caps={"default": 50}),
            blob_store=blob_store,
        )
        registry = ToolRegistry(output_limiter=limiter)
        registry.register(ToolDefinition(
            "big_tool", "returns big output",
            [ToolParam("query", "string", "q")],
            lambda query: f"result: {'x' * 200}",
        ))

        tc = ToolCall.new("big_tool", {"query": "test"})
        responses = [
            LLMResponse(content="", tool_calls=[tc]),
            LLMResponse(content="Understood, output was large."),
        ]

        store = InMemoryStore()
        engine = AgentEngine(
            store=store,
            llm=MockProvider(responses),
            tools=registry,
        )
        sid = engine.create_session().id
        result = engine.run(sid, "Run big tool")

        assert result == "Understood, output was large."
        assert blob_store.count == 1

        # The tool_result in messages should contain the truncation stub
        session = store.get(sid)
        tr_msg = next(m for m in session.messages if m.tool_results)
        assert "[OUTPUT TRUNCATED" in tr_msg.tool_results[0].content
        assert BLOB_REF_PREFIX in tr_msg.tool_results[0].content

    def test_blob_ref_retrievable_after_engine_run(self):
        """After a run, the blob store should have the full output."""
        blob_store = InMemoryBlobStore()
        limiter = OutputLimiter(
            config=OutputLimitConfig(caps={"default": 30}),
            blob_store=blob_store,
        )
        registry = ToolRegistry(output_limiter=limiter)

        original_output = "important_data " * 20  # 300 chars
        registry.register(ToolDefinition(
            "data_tool", "returns data", [],
            lambda: original_output,
        ))

        tc = ToolCall.new("data_tool", {})
        responses = [
            LLMResponse(content="", tool_calls=[tc]),
            LLMResponse(content="Got it."),
        ]

        store = InMemoryStore()
        engine = AgentEngine(
            store=store,
            llm=MockProvider(responses),
            tools=registry,
        )
        sid = engine.create_session().id
        engine.run(sid, "fetch data")

        # Retrieve full content via blob ref
        ref = blob_store.list_refs()[0]
        assert blob_store.get(ref) == original_output

    def test_engine_session_state_correct(self):
        """Session should end normally even when output is capped."""
        limiter = OutputLimiter(
            config=OutputLimitConfig(caps={"default": 10}),
        )
        registry = ToolRegistry(output_limiter=limiter)
        registry.register(ToolDefinition(
            "t", "test", [], lambda: "x" * 100,
        ))

        tc = ToolCall.new("t", {})
        responses = [
            LLMResponse(content="", tool_calls=[tc]),
            LLMResponse(content="Done."),
        ]

        store = InMemoryStore()
        engine = AgentEngine(
            store=store,
            llm=MockProvider(responses),
            tools=registry,
        )
        sid = engine.create_session().id
        result = engine.run(sid, "go")
        assert result == "Done."

        session = store.get(sid)
        assert session.state == SessionState.waiting_for_user

    def test_llm_receives_truncation_stub_not_full_output(self):
        """The LLM should see the truncation stub, not the full output."""
        blob_store = InMemoryBlobStore()
        limiter = OutputLimiter(
            config=OutputLimitConfig(caps={"default": 30}),
            blob_store=blob_store,
        )
        registry = ToolRegistry(output_limiter=limiter)
        registry.register(ToolDefinition(
            "t", "test", [], lambda: "z" * 500,
        ))

        received_messages: list[list[Message]] = []

        class TrackingLLM(MockProvider):
            def complete(self, messages, tools):
                received_messages.append(list(messages))
                return super().complete(messages, tools)

        tc = ToolCall.new("t", {})
        responses = [
            LLMResponse(content="", tool_calls=[tc]),
            LLMResponse(content="Ok."),
        ]

        store = InMemoryStore()
        engine = AgentEngine(
            store=store,
            llm=TrackingLLM(responses),
            tools=registry,
        )
        sid = engine.create_session().id
        engine.run(sid, "go")

        # Second LLM call receives the tool result
        second_call = received_messages[1]
        tool_result_msgs = [m for m in second_call if m.tool_results]
        assert tool_result_msgs
        content = tool_result_msgs[0].tool_results[0].content
        assert "[OUTPUT TRUNCATED" in content
        assert "z" * 500 not in content


# ===========================================================================
# Telemetry
# ===========================================================================

class TestOutputCappedTelemetry:
    def test_output_capped_event_emitted(self, telemetry_backend):
        limiter = OutputLimiter(
            config=OutputLimitConfig(caps={"default": 50}),
        )
        limiter.limit("my_tool", "x" * 200, session_id="s1")

        events = telemetry_backend.by_name("output.capped")
        assert len(events) == 1
        e = events[0]
        assert e.outcome == Outcome.executed
        assert e.session_id == "s1"
        assert e.data["tool_name"] == "my_tool"
        assert e.data["original_length"] == 200
        assert e.data["cap"] == 50
        assert e.data["blob_ref"].startswith(BLOB_REF_PREFIX)

    def test_no_event_when_within_cap(self, telemetry_backend):
        limiter = OutputLimiter(
            config=OutputLimitConfig(caps={"default": 100}),
        )
        limiter.limit("t", "short")
        assert not telemetry_backend.by_name("output.capped")

    def test_event_in_engine_context(self, telemetry_backend):
        limiter = OutputLimiter(
            config=OutputLimitConfig(caps={"default": 20}),
        )
        registry = ToolRegistry(output_limiter=limiter)
        registry.register(ToolDefinition(
            "t", "test", [], lambda: "x" * 100,
        ))

        tc = ToolCall.new("t", {})
        store = InMemoryStore()
        engine = AgentEngine(
            store=store,
            llm=MockProvider([
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="Done."),
            ]),
            tools=registry,
        )
        sid = engine.create_session().id
        engine.run(sid, "go")

        events = telemetry_backend.by_name("output.capped")
        assert events
        assert events[0].data["tool_name"] == "t"
