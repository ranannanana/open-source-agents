"""Claude through your own Claude Code login (no API key): headless `claude -p`.

Each call runs the Claude Code CLI with every tool disabled, user and project settings,
hooks and MCP servers skipped, no saved session, and an empty working directory, so the
model only writes text and nothing it says is executed. The CLI reports real usage and
an API-equivalent cost, which the budget and the dataset use as is.

A reply longer than the output cap is continued by the CLI over several messages, and its
`result` holds only the last one, so the call streams events and joins every message,
repairing the seams (a reopened code fence, a repeated tail) that continuations leave.

Find the binary with RCE_SENTINEL_CLAUDE_BIN, else `claude` on PATH, else the one bundled
with the VS Code extension.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from functools import lru_cache
from typing import Any

import httpx

from rce_sentinel.orchestrator.llm import LLMResult, ModelSpec

BIN_ENV = "RCE_SENTINEL_CLAUDE_BIN"
_VSCODE = "~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude"
# Env that would send the call somewhere else, or bill a key instead of the seat.
_DROP_ENV = ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
             "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX")


class SeatError(httpx.HTTPError):
    """The CLI ran but the call failed; status is the API status when one was given."""

    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


@lru_cache(maxsize=1)
def claude_bin() -> str | None:
    if found := os.environ.get(BIN_ENV):
        return found if os.access(found, os.X_OK) else None
    if found := shutil.which("claude"):
        return found
    bundled = sorted(glob.glob(os.path.expanduser(_VSCODE)))
    return bundled[-1] if bundled else None


def wire_id(spec: ModelSpec) -> str:
    """'seat/claude-haiku-5-5' -> 'claude-haiku-5-5'."""
    return spec.id.split("/", 1)[1] if spec.id.startswith("seat/") else spec.id


def command(spec: ModelSpec, system: str, max_budget_usd: float | None = None) -> list[str]:
    exe = claude_bin()
    if exe is None:
        raise SeatError("claude CLI not found; set " + BIN_ENV)
    cmd = [exe, "-p", "--model", wire_id(spec), "--tools", "", "--system-prompt", system,
           "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence",
           "--output-format", "stream-json", "--verbose"]
    if max_budget_usd:
        cmd += ["--max-budget-usd", f"{max_budget_usd:.4f}"]
    return cmd


def _events(raw: str) -> list[Any]:
    """One JSON result object, or a stream of one JSON event per line."""
    try:
        return [json.loads(raw)]
    except ValueError:
        pass
    out = []
    for line in raw.splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


MIN_OVERLAP = 24        # shortest repeat at a seam that is trimmed
MAX_OVERLAP = 400
_REOPEN = re.compile(r"[ \t]*\n?[ \t]*```[A-Za-z][^\n]*\n")


def _trim_seam(done: str, nxt: str) -> str:
    """Undo two habits of a continued message: it may reopen the code block it is
    already inside, or repeat the last characters the previous message wrote."""
    if done.count("```") % 2 == 1 and (m := _REOPEN.match(nxt)):
        nxt = nxt[m.end():]   # a closing fence has no language, so it is never cut
    for k in range(min(len(done), len(nxt), MAX_OVERLAP), MIN_OVERLAP - 1, -1):
        if done.endswith(nxt[:k]) and re.search(r"[A-Za-z]{3}", nxt[:k]):
            return nxt[k:]
    return nxt


def _joined_text(events: list[Any]) -> str:
    """Text of every assistant message in order; continuations resume mid-word."""
    out = ""
    for e in events:
        if isinstance(e, dict) and e.get("type") == "assistant":
            for block in (e.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "text":
                    part = str(block.get("text") or "")
                    out += _trim_seam(out, part) if out else part
    return out


def parse(raw: str, spec: ModelSpec, latency_ms: float) -> LLMResult:
    """LLMResult from the CLI's result event; usage summed over every model it called."""
    events = _events(raw)
    data = next((e for e in reversed(events)
                 if isinstance(e, dict) and e.get("type", "result") == "result"), None)
    if data is None:
        raise SeatError(f"claude CLI gave no JSON result: {raw[:200]!r}")
    if data.get("is_error") or data.get("subtype") != "success":
        status = int(data.get("api_error_status") or 0)
        raise SeatError(f"claude CLI: {str(data.get('result') or data.get('subtype'))[:200]}",
                        status)
    usage: dict[str, Any] = data.get("modelUsage") or {}
    fresh = sum(int(u.get("inputTokens") or 0) for u in usage.values())
    read = sum(int(u.get("cacheReadInputTokens") or 0) for u in usage.values())
    write = sum(int(u.get("cacheCreationInputTokens") or 0) for u in usage.values())
    out = sum(int(u.get("outputTokens") or 0) for u in usage.values())
    served = next(iter(usage), "")
    cost = data.get("total_cost_usd")
    text = _joined_text(events) or str(data.get("result") or "")
    return LLMResult(text=text, tokens_in=fresh + read + write,
                     tokens_out=out, cached_tokens=read,
                     latency_ms=float(data.get("duration_api_ms") or latency_ms),
                     model=spec.id, cost=float(cost) if isinstance(cost, int | float) else None,
                     served=served)


class ClaudeSeatProvider:
    """Provider for models whose provider has `kind: claude-cli`."""

    def __init__(self, max_budget_usd: float | None = 2.0):
        self.max_budget_usd = max_budget_usd

    def complete(self, spec: ModelSpec, system: str, user: str, *, max_tokens: int,
                 timeout: float, prefix_tokens: int = 0, prefix_warm: bool = False,
                 schema: dict[str, Any] | None = None, text: bool = False) -> LLMResult:
        try:
            return self._run(spec, system, user, max(max_tokens, 1024), timeout)
        except SeatError as exc:
            # One message past the cap fails the whole call; retry once with room.
            if "output token maximum" not in str(exc):
                raise
            return self._run(spec, system, user, max(max_tokens, 1024) * 2, timeout)

    def _run(self, spec: ModelSpec, system: str, user: str, cap: int,
             timeout: float) -> LLMResult:
        env = {k: v for k, v in os.environ.items() if k not in _DROP_ENV}
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(cap)
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="rce-seat-") as cwd:
            try:
                proc = subprocess.run(command(spec, system, self.max_budget_usd), input=user,
                                      capture_output=True, text=True, timeout=timeout,
                                      cwd=cwd, env=env, check=False)
            except subprocess.TimeoutExpired as exc:
                raise SeatError(f"claude CLI timed out after {timeout:.0f}s") from exc
        if proc.returncode != 0 and not proc.stdout.strip():
            raise SeatError(f"claude CLI exit {proc.returncode}: {proc.stderr.strip()[:200]}")
        return parse(proc.stdout, spec, (time.monotonic() - t0) * 1000)
