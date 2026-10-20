"""Fresh, tool-disabled F1 Codex calls with immediate provider-limit stopping.

No call runs on import. The frozen E2 command/event/budget helpers are reused
without changing their source. Each invocation owns exactly one CLI process;
there are no Python retries or per-call deadlines. Hidden reasoning is never
returned or written. The caller owns the six permitted experiment files.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from typing import Any

import e2_transport as _e2


TRANSPORT_VERSION = "F1-streaming-terminal-stop-1.0"
_RUNTIME_LOCK = threading.Lock()
_RUNTIME: dict[str, Any] | None = None
_TERMINAL = re.compile(
    r"usage[\s_-]*limit|rate[\s_-]*limit|too\s+many\s+requests|"
    r"quota|usage_limit_reached|(?<![\d.])429(?![\d.])|"
    r"额度|配额|速率限制|请求过多|频率限制|达到.{0,8}(?:使用|调用).{0,4}上限",
    re.IGNORECASE,
)


def resolve_runtime() -> dict[str, Any]:
    """Resolve and fingerprint the installed CLI, requiring E2's exact version."""
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is not None:
            return dict(_RUNTIME)
        candidate = os.environ.get("F1_CODEX_EXE") or shutil.which("codex.exe") or shutil.which("codex")
        if not candidate:
            raise RuntimeError("Installed local Codex CLI was not found on PATH")
        executable = _e2.checked_path(candidate)
        if executable.suffix.casefold() != ".exe":
            raise RuntimeError("Resolve an actual codex.exe, not a shell wrapper")
        environment = _environment()
        version = subprocess.run(
            [str(executable), "--version"], check=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace", env=environment,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        actual_version = version.stdout.strip()
        if version.returncode or actual_version != _e2.CLI_VERSION:
            raise RuntimeError(
                f"Installed CLI differs from E2: {actual_version!r}; "
                f"expected {_e2.CLI_VERSION!r}"
            )
        digest = hashlib.sha256()
        with executable.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        # E2's former installation hash directory was removed by an app update.
        # The installed binary reports the identical frozen CLI version. Only
        # this in-memory variable changes; the old source remains read-only.
        _e2.CODEX_EXE = executable
        _RUNTIME = {
            "executable": str(executable), "sha256": digest.hexdigest(),
            "cli_version": actual_version, "model": _e2.MODEL,
            "reasoning_effort": _e2.REASONING_EFFORT,
        }
        return dict(_RUNTIME)


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    for key in tuple(environment):
        if key.upper() in {"OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL"}:
            environment.pop(key, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _terminal_error_event(line: str) -> str | None:
    """Inspect provider diagnostics only, never normal model answer/reasoning."""
    try:
        event = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(event, dict):
        return None
    item = event.get("item")
    item = item if isinstance(item, dict) else {}
    if event.get("type") in {"error", "turn.failed"}:
        diagnostic = json.dumps(event, ensure_ascii=False)
    elif item.get("type") == "error":
        diagnostic = json.dumps(item, ensure_ascii=False)
    else:
        return None
    return diagnostic if _TERMINAL.search(diagnostic) else None


def call_formula(
    prompt: str, *, schema_path: str | Path, instructions_path: str | Path,
    catalog_path: str | Path, stop_event: threading.Event,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one fresh F1 call; a shared terminal stop cancels all in-flight calls.

    The caller must fsync its call reservation before invoking this function.
    Any provider limit is detected while the CLI streams, including before its
    own transport could retry. stdout remains in memory only until E2's parser
    extracts final messages, usage, and tool/error metadata.
    """
    result: dict[str, Any] = dict(metadata or {})
    result.update({
        "prompt": prompt,
        "request_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "requested_model": _e2.MODEL, "model": _e2.MODEL,
        "reasoning_effort": _e2.REASONING_EFFORT,
        "cli_version": _e2.CLI_VERSION, "transport_version": TRANSPORT_VERSION,
        "input_tokens": None, "output_tokens": None,
        "cached_input_tokens": None, "elapsed_seconds": 0.0,
        "status": "preflight_rejected", "response_text": "",
        "tool_activity": [], "errors": [], "warnings": [],
    })
    started = time.monotonic()
    process: subprocess.Popen[str] | None = None
    readers: list[threading.Thread] = []
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    reader_errors: list[str] = []
    terminal_here = threading.Event()
    diagnostic_lock = threading.Lock()
    terminal_diagnostics: list[str] = []
    cancelled = False

    def terminal(diagnostic: str, source: str) -> None:
        with diagnostic_lock:
            terminal_diagnostics.append(f"{source}: {diagnostic}")
        terminal_here.set()
        stop_event.set()
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass

    def consume_stdout() -> None:
        assert process is not None and process.stdout is not None
        try:
            for line in process.stdout:
                stdout_lines.append(line)
                diagnostic = _terminal_error_event(line)
                if diagnostic is not None:
                    terminal(diagnostic, "stdout_provider_error")
        except Exception as exc:
            reader_errors.append(f"stdout {type(exc).__name__}: {exc}")

    def consume_stderr() -> None:
        assert process is not None and process.stderr is not None
        try:
            for line in process.stderr:
                stderr_lines.append(line)
                # stderr is CLI operational diagnostics, not model prose.
                if _TERMINAL.search(line):
                    terminal(line.rstrip(), "stderr")
        except Exception as exc:
            reader_errors.append(f"stderr {type(exc).__name__}: {exc}")

    try:
        if stop_event.is_set():
            result["status"] = "interrupted_due_terminal_stop"
            result["error"] = "Shared provider/integrity stop was set before dispatch"
            return result
        runtime = resolve_runtime()
        result["runtime"] = runtime
        budget = _e2.input_budget(
            prompt, schema_path=schema_path, instructions_path=instructions_path,
        )
        result["input_budget"] = budget
        if not budget["within_budget"]:
            result["error"] = "Conservative complete-input upper bound exceeds 15000"
            stop_event.set()
            return result
        command = _e2.build_command(
            cwd=_e2.ROOT, schema_path=schema_path,
            instructions_path=instructions_path, catalog_path=catalog_path,
        )
        if stop_event.is_set():
            result["status"] = "interrupted_due_terminal_stop"
            result["error"] = "Shared stop was set during preflight"
            return result
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            cwd=_e2.checked_path(_e2.ROOT), env=_environment(), bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        result["pid"] = process.pid
        for target, name in ((consume_stdout, "stdout"), (consume_stderr, "stderr")):
            reader = threading.Thread(target=target, name=f"F1-{process.pid}-{name}")
            readers.append(reader)
            reader.start()
        assert process.stdin is not None
        try:
            process.stdin.write(prompt)
            process.stdin.close()
        except (BrokenPipeError, OSError):
            # Early provider failure may close stdin before the prompt is read.
            pass
        while process.poll() is None:
            if stop_event.is_set():
                cancelled = not terminal_here.is_set()
                try:
                    process.kill()
                except OSError:
                    pass
                break
            time.sleep(0.05)
        result["returncode"] = process.wait()
        for reader in readers:
            reader.join()
        result.update(_e2.parse_events("".join(stdout_lines)))
        result["stderr_tail"] = "".join(stderr_lines)[-8000:]
        result["stream_reader_errors"] = list(reader_errors)
        result["terminal_diagnostics"] = list(terminal_diagnostics)
        if terminal_here.is_set():
            result["status"] = "terminal_provider_limit"
            result["error"] = "Provider quota/rate-limit diagnostic detected; no retry"
        elif cancelled:
            result["status"] = "interrupted_due_terminal_stop"
            result["error"] = "Cancelled because another call set the shared stop"
        elif result["tool_activity"]:
            result["status"] = "tool_activity_rejected"
            stop_event.set()
        elif any(model != _e2.MODEL for model in result["reported_models"]):
            result["status"] = "model_mismatch_rejected"
            stop_event.set()
        elif result["backend_protocol_errors"] or reader_errors:
            result["status"] = "backend_protocol_rejected"
            stop_event.set()
        elif result["returncode"]:
            result["status"] = "cli_error"
        elif result["failed_turn"]:
            result["status"] = "backend_error"
        elif not result["completed"] or not result["response_text"]:
            result["status"] = "incomplete_response"
        elif result["input_tokens"] is not None and result["input_tokens"] > _e2.INPUT_LIMIT:
            result["status"] = "actual_input_budget_violation"
            stop_event.set()
        else:
            result["status"] = "ok"
            if result["errors"] or result["warnings"]:
                result["recovered_cli_diagnostics"] = True
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            stop_event.set()
            result["status"] = "interrupted_due_terminal_stop"
        elif process is None:
            # Missing/changed runtime, invalid frozen tool catalog, or invalid
            # input files are batch integrity failures, not retryable examples.
            result["status"] = "preflight_rejected"
            stop_event.set()
        else:
            result["status"] = (
                "interrupted_due_terminal_stop" if stop_event.is_set() else "transport_error"
            )
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None:
            if process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass
            process.wait()
            for reader in readers:
                reader.join()
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None:
                    try:
                        pipe.close()
                    except OSError:
                        pass
            if "stderr_tail" not in result:
                result["stderr_tail"] = "".join(stderr_lines)[-8000:]
        elapsed = time.monotonic() - started
        result["elapsed_seconds"] = elapsed
        result["latency_seconds"] = elapsed
    return result
