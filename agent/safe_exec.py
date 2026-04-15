"""Safe shell/process execution wrapper.

All subprocess execution MUST go through ``safe_exec`` or ``safe_exec_output``.
These functions require an argument array (list of strings) and reject bare
string commands, preventing shell injection.

Design decisions
----------------
- ``shell=False`` is always enforced — the subprocess module receives an
  argv list, never a string parsed by /bin/sh or cmd.exe.
- A string command is rejected with ``UnsafeCommandError`` *before* any
  process is spawned.
- Timeouts default to 30 seconds.
- Output is decoded as UTF-8 with replacement for non-UTF-8 bytes.

Usage::

    from agent.safe_exec import safe_exec_output

    stdout = safe_exec_output(["git", "log", "--oneline", "-5"])
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class UnsafeCommandError(ValueError):
    """Raised when a caller passes a bare string instead of an arg list."""


class ExecutionTimeoutError(TimeoutError):
    """Raised when a subprocess exceeds its timeout."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProcessResult:
    """Structured result of a subprocess invocation."""
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.returncode == 0 and not self.timed_out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Characters that indicate shell metacharacter abuse when they appear
# in a *single* argument.  These are suspicious inside individual args
# because they would only have effect if passed through a shell.
_SHELL_METACHARS = set(";|&$`\\\"'(){}!<>\n")


def validate_command(argv: Any) -> list[str]:
    """
    Validate that *argv* is a safe argument list.

    Raises ``UnsafeCommandError`` if:
    - *argv* is a bare string (shell=True style)
    - *argv* is empty
    - Any element is not a string

    Returns the validated list for convenience.
    """
    if isinstance(argv, str):
        raise UnsafeCommandError(
            f"String commands are not allowed. Use an argument list instead. "
            f"Received: {argv!r}"
        )
    if not isinstance(argv, (list, tuple)):
        raise UnsafeCommandError(
            f"Command must be a list of strings, got {type(argv).__name__}"
        )
    if len(argv) == 0:
        raise UnsafeCommandError("Command argument list is empty")
    for i, arg in enumerate(argv):
        if not isinstance(arg, str):
            raise UnsafeCommandError(
                f"Argument at index {i} is {type(arg).__name__!r}, expected str"
            )
    return list(argv)


def check_shell_metachars(argv: list[str]) -> list[str]:
    """
    Warn (log) if any argument contains shell metacharacters.

    This is informational — it does not reject the command because some
    legitimate arguments contain these characters (e.g., grep patterns).
    Returns a list of warnings (empty if clean).
    """
    warnings: list[str] = []
    for i, arg in enumerate(argv):
        found = _SHELL_METACHARS.intersection(arg)
        if found:
            w = (
                f"Argument {i} ({arg!r}) contains shell metacharacters "
                f"{found!r} — verify this is intentional"
            )
            warnings.append(w)
            log.warning(w)
    return warnings


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def safe_exec(
    argv: list[str],
    *,
    timeout: float = 30.0,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> ProcessResult:
    """
    Run a subprocess safely with an argument array.

    Parameters
    ----------
    argv
        Command + arguments as a list of strings.  Bare strings raise
        ``UnsafeCommandError``.
    timeout
        Maximum wall-clock seconds.  Default 30.
    cwd
        Working directory for the subprocess.
    env
        Environment variables (replaces the inherited env if set).

    Returns
    -------
    ProcessResult
        Structured result including stdout, stderr, returncode.
        On timeout, ``timed_out`` is True and returncode is -1.
    """
    argv = validate_command(argv)
    check_shell_metachars(argv)

    log.debug("safe_exec: %s (timeout=%s, cwd=%s)", argv, timeout, cwd)

    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
            shell=False,  # NEVER shell=True
        )
        return ProcessResult(
            returncode=proc.returncode,
            stdout=proc.stdout.decode("utf-8", errors="replace"),
            stderr=proc.stderr.decode("utf-8", errors="replace"),
        )
    except subprocess.TimeoutExpired as exc:
        log.warning("Command timed out after %ss: %s", timeout, argv)
        return ProcessResult(
            returncode=-1,
            stdout=exc.stdout.decode("utf-8", errors="replace") if exc.stdout else "",
            stderr=exc.stderr.decode("utf-8", errors="replace") if exc.stderr else "",
            timed_out=True,
        )
    except FileNotFoundError:
        return ProcessResult(
            returncode=-1,
            stdout="",
            stderr=f"Command not found: {argv[0]!r}",
        )
    except OSError as exc:
        return ProcessResult(
            returncode=-1,
            stdout="",
            stderr=f"OS error: {exc}",
        )


def safe_exec_output(
    argv: list[str],
    *,
    timeout: float = 30.0,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> str:
    """
    Convenience wrapper: run a command and return stdout.

    Raises ``RuntimeError`` if the command fails (non-zero exit or timeout).
    """
    result = safe_exec(argv, timeout=timeout, cwd=cwd, env=env)
    if not result.success:
        error_detail = result.stderr or result.stdout or "(no output)"
        if result.timed_out:
            raise ExecutionTimeoutError(
                f"Command timed out after {timeout}s: {argv}"
            )
        raise RuntimeError(
            f"Command failed (rc={result.returncode}): {argv}\n{error_detail}"
        )
    return result.stdout
