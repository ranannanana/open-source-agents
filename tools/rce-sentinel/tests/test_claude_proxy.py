"""Claude Code pass-through: routing rules, cache-aware stickiness, usage capture, seat CLI."""

from __future__ import annotations

import json
import os
import stat
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest

from rce_sentinel.claude_proxy import ClaudeRouter, UsageMeter, conversation_key, typed_text
from rce_sentinel.config import Settings
from rce_sentinel.handoff.session import find_model
from rce_sentinel.memory.store import Store
from rce_sentinel.orchestrator import seat
from rce_sentinel.orchestrator.llm import MockProvider, ModelRegistry
from rce_sentinel.proxy import RouterProxy, make_server
from rce_sentinel.skills import MARKER

HAIKU, SONNET, OPUS = "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"
HARD = ("Design and build a multi-tenant billing service with idempotent webhooks, "
        "a Postgres schema, migrations, retries and a full test suite across three services.")
EASY = "fix the typo in README"
TOOLS = [{"name": "Read", "input_schema": {"type": "object"}}]
USAGE = {"input_tokens": 50, "cache_creation_input_tokens": 1000,
         "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 1000},
         "cache_read_input_tokens": 20_000, "output_tokens": 1}


class Anthropic(BaseHTTPRequestHandler):
    """Fake api.anthropic.com: echoes usage; refuses models listed in `refuse`."""
    seen: ClassVar[list[dict[str, Any]]] = []
    refuse: ClassVar[set[str]] = set()

    def log_message(self, *a: Any) -> None:
        return

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Anthropic.seen.append({"path": self.path, "body": body,
                               "auth": self.headers.get("Authorization"),
                               "beta": self.headers.get("anthropic-beta"),
                               "token": self.headers.get("x-rce-token")})
        if body.get("model") in Anthropic.refuse:
            self._send(400, {"type": "error", "error": {"message": "model not allowed"}})
            return
        if self.path.startswith("/v1/messages/count_tokens"):
            self._send(200, {"input_tokens": 42})
            return
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            start = {"type": "message_start", "message": {"model": body["model"],
                                                          "usage": USAGE}}
            delta = {"type": "message_delta", "usage": {"output_tokens": 300}}
            for ev in (start, {"type": "content_block_delta",
                               "delta": {"type": "text_delta", "text": "hi"}}, delta):
                self.wfile.write(f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n".encode())
            self.wfile.write(b'event: message_stop\ndata: {"type":"message_stop"}\n\n')
            return
        self._send(200, {"model": body["model"], "content": [{"type": "text", "text": "ok"}],
                         "usage": {**USAGE, "output_tokens": 300}})


@pytest.fixture
def claude(settings: Settings) -> Iterator[tuple[str, ClaudeRouter]]:
    Anthropic.seen, Anthropic.refuse = [], set()
    up = ThreadingHTTPServer(("127.0.0.1", 0), Anthropic)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    store = Store(settings.db_path)
    router = ClaudeRouter(settings, store, upstream=f"http://127.0.0.1:{up.server_address[1]}")
    proxy = RouterProxy(settings, ModelRegistry(settings.models, "mock"), MockProvider(), store,
                        use_jev=False, claude=router)
    server = make_server(proxy, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", router
    server.shutdown()
    up.shutdown()


def send(base: str, messages: list[dict[str, Any]], model: str = OPUS, *, tools: bool = True,
         stream: bool = True, path: str = "/v1/messages?beta=true") -> httpx.Response:
    body: dict[str, Any] = {"model": model, "max_tokens": 1000, "messages": messages,
                            "stream": stream}
    if tools:
        body["tools"] = TOOLS
    return httpx.post(base + path, json=body, timeout=30,
                      headers={"anthropic-version": "2023-06-01",
                               "Authorization": "Bearer seat-token",
                               "anthropic-beta": "oauth-2025-04-20"})


def user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


TOOL_RESULT = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                            "content": "file body"}]}
ASSISTANT = {"role": "assistant", "content": [{"type": "tool_use", "id": "t1",
                                               "name": "Read", "input": {}}]}


def events(router: ClaudeRouter) -> list[dict[str, Any]]:
    return [dict(r) for r in router.store.execute("SELECT * FROM usage_events ORDER BY id")]


# --- helpers -----------------------------------------------------------------------------
def test_typed_text_ignores_reminders_and_tool_results() -> None:
    msg = {"role": "user", "content": [
        {"type": "text", "text": "<system-reminder>ctx</system-reminder>"},
        {"type": "text", "text": "fix the bug"}]}
    assert typed_text(msg) == "fix the bug"
    assert typed_text(TOOL_RESULT) == ""
    assert typed_text({"role": "user", "content": "plain"}) == "plain"


def test_conversation_key_prefers_the_session_header() -> None:
    body = {"messages": [user("a"), ASSISTANT, TOOL_RESULT]}
    assert conversation_key(body, {}) == conversation_key({"messages": [user("a")]}, {})
    assert conversation_key(body, {"x-claude-code-session-id": "s1"}) == "cs1"


def test_meter_reads_streamed_and_whole_usage() -> None:
    m = UsageMeter()
    start = {"type": "message_start", "message": {"usage": USAGE}}
    raw = f"data: {json.dumps(start)}\n\ndata: ".encode()
    m.feed(raw[:30])     # events can split across chunks
    m.feed(raw[30:] + b'{"type":"message_delta","usage":{"output_tokens":7}}\n')
    assert m.usage == {"input": 50, "write_5m": 0, "write_1h": 1000, "read": 20_000,
                       "output": 7}
    assert m.ttl == 3600.0


def test_cache_aware_cost_and_haiku_long_prompt_price(settings: Settings) -> None:
    reg = ModelRegistry(settings.models, "mock")
    opus = find_model(reg, "seat/" + OPUS)
    haiku = find_model(reg, "seat/" + HAIKU)
    assert opus and haiku
    # 50 fresh + 1000 x 2 (1 h write) at $4, 20k reads at $0.20, 300 out at $20
    assert opus.usage_cost(50, 0, 1000, 20_000, 300) == pytest.approx(
        (2050 * 4 + 20_000 * 0.2 + 300 * 20) / 1e6)
    assert haiku.usage_cost(1000, 0, 0, 0, 100) == pytest.approx((1000 * 0.1 + 50) / 1e6)
    assert haiku.usage_cost(150_000, 0, 0, 0, 100) == pytest.approx(   # 5x above 100K
        5 * (150_000 * 0.1 + 100 * 0.5) / 1e6)


# --- routing -----------------------------------------------------------------------------
def test_new_conversation_is_routed_and_auth_rides_through(
        claude: tuple[str, ClaudeRouter]) -> None:
    base, router = claude
    r = send(base, [user(EASY)])
    assert r.status_code == 200 and "message_stop" in r.text
    sent = Anthropic.seen[-1]
    assert sent["auth"] == "Bearer seat-token" and sent["beta"] == "oauth-2025-04-20"
    assert sent["path"] == "/v1/messages?beta=true"
    assert sent["body"]["model"] != OPUS          # a typo fix does not need Opus
    ev = events(router)[-1]
    assert ev["rule"] == "new-conversation" and ev["rewritten"] == 1
    assert (ev["input_tokens"], ev["cache_write_1h"], ev["cache_read"], ev["tokens_out"]) == (
        50, 1000, 20_000, 300)
    assert ev["cost"] < ev["cost_requested"]
    assert ev["tag_id"] and ev["task"]


def test_tool_loop_never_switches(claude: tuple[str, ClaudeRouter]) -> None:
    base, router = claude
    first = [user(EASY)]
    send(base, first)
    chosen = Anthropic.seen[-1]["body"]["model"]
    send(base, [*first, ASSISTANT, TOOL_RESULT])
    assert Anthropic.seen[-1]["body"]["model"] == chosen
    assert events(router)[-1]["rule"] == "sticky-tool-loop"


def test_hard_follow_up_upgrades_but_warm_cache_blocks_a_downgrade(
        claude: tuple[str, ClaudeRouter]) -> None:
    base, router = claude
    convo = [user(EASY)]
    send(base, convo)
    cheap = Anthropic.seen[-1]["body"]["model"]
    convo += [{"role": "assistant", "content": [{"type": "text", "text": "done"}]}, user(HARD)]
    send(base, convo)
    strong = Anthropic.seen[-1]["body"]["model"]
    assert router.rank(strong) > router.rank(cheap) and events(router)[-1]["rule"] == "upgrade"
    assert events(router)[-1]["switched_from"] == cheap
    convo += [{"role": "assistant", "content": [{"type": "text", "text": "ok"}]}, user(EASY)]
    send(base, convo)
    assert Anthropic.seen[-1]["body"]["model"] == strong      # cache still warm: stay
    assert events(router)[-1]["rule"] == "sticky-cache-warm"
    router.convs[events(router)[-1]["session"]].last -= 4000  # past the 1 h cache
    convo += [{"role": "assistant", "content": [{"type": "text", "text": "ok"}]}, user(EASY)]
    send(base, convo)
    assert Anthropic.seen[-1]["body"]["model"] == cheap
    assert events(router)[-1]["rule"] == "cold-cache-switch"
    n = router.store.execute("SELECT COUNT(*) AS n FROM handoffs")[0]["n"]
    assert n == 2


def test_refused_model_falls_back_to_the_asked_for_one(
        claude: tuple[str, ClaudeRouter]) -> None:
    base, router = claude
    Anthropic.refuse = {HAIKU, SONNET}
    r = send(base, [user(EASY)], stream=False)
    assert r.status_code == 200 and r.json()["model"] == OPUS
    ev = events(router)[-1]
    assert ev["fallback"] == 1 and ev["model"] == OPUS and ev["tokens_out"] == 300


def test_log_only_records_the_choice_but_sends_the_request_unchanged(
        settings: Settings, claude: tuple[str, ClaudeRouter]) -> None:
    base, router = claude
    router.log_only = True
    send(base, [user(EASY)])
    assert Anthropic.seen[-1]["body"]["model"] == OPUS
    ev = events(router)[-1]
    assert ev["rewritten"] == 0 and ev["policy_choice"] != OPUS and ev["log_only"] == 1


def test_small_internal_calls_and_token_counts_pass_through(
        claude: tuple[str, ClaudeRouter]) -> None:
    base, router = claude
    send(base, [user("title this chat")], model=HAIKU, tools=False)
    assert Anthropic.seen[-1]["body"]["model"] == HAIKU
    assert events(router)[-1]["rule"] == "aux"
    r = send(base, [user("x")], path="/v1/messages/count_tokens", stream=False)
    assert r.json() == {"input_tokens": 42} and len(events(router)) == 1
    send(base, [user(EASY)], model="claude-fable-5-1")
    assert Anthropic.seen[-1]["body"]["model"] == "claude-fable-5-1"
    assert events(router)[-1]["rule"] == "not-in-pool"


def test_unreachable_upstream_is_logged_as_502(claude: tuple[str, ClaudeRouter]) -> None:
    base, router = claude
    router.upstream = "http://127.0.0.1:9"
    assert send(base, [user(EASY)]).status_code == 502
    assert events(router)[-1]["status"] == 502


def test_openai_side_still_works_next_to_claude(claude: tuple[str, ClaudeRouter]) -> None:
    base, _ = claude
    r = httpx.get(base + "/v1/models")
    assert r.json()["data"][0]["id"] == "auto"


# --- seat CLI ----------------------------------------------------------------------------
FAKE = """#!/bin/sh
printf '%s\\n' "$@" > "$ARGS_OUT"
env > "$ENV_OUT"
cat > /dev/null
cat <<'EOF'
{"type":"result","subtype":"success","is_error":false,"result":"ANSWER","duration_api_ms":900,
 "total_cost_usd":0.0123,"modelUsage":{"claude-haiku-5-5":{"inputTokens":200,"outputTokens":40,
 "cacheReadInputTokens":1000,"cacheCreationInputTokens":500}}}
EOF
"""


@pytest.fixture
def fake_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    exe = tmp_path / "claude"
    exe.write_text(FAKE)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv(seat.BIN_ENV, str(exe))
    monkeypatch.setenv("ARGS_OUT", str(tmp_path / "args"))
    monkeypatch.setenv("ENV_OUT", str(tmp_path / "env"))
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8787")
    seat.claude_bin.cache_clear()
    yield tmp_path
    seat.claude_bin.cache_clear()


def test_seat_call_is_tool_less_and_skips_the_proxy(settings: Settings,
                                                    fake_claude: Path) -> None:
    reg = ModelRegistry(settings.models, "auto")
    spec = find_model(reg, "seat/" + HAIKU)
    assert spec is not None
    res = seat.ClaudeSeatProvider().complete(spec, "SYS", "hello", max_tokens=500, timeout=30)
    assert (res.text, res.tokens_in, res.tokens_out, res.cached_tokens, res.cost) == (
        "ANSWER", 1700, 40, 1000, 0.0123)
    args = (fake_claude / "args").read_text().splitlines()
    assert args[args.index("--model") + 1] == HAIKU
    assert args[args.index("--tools") + 1] == ""
    assert args[args.index("--setting-sources") + 1] == ""
    assert "--no-session-persistence" in args and "--strict-mcp-config" in args
    env = (fake_claude / "env").read_text()
    assert "ANTHROPIC_BASE_URL" not in env and "CLAUDE_CODE_MAX_OUTPUT_TOKENS=1024" in env


def test_seat_errors_are_network_errors(fake_claude: Path, settings: Settings) -> None:
    bad = fake_claude / "claude"
    bad.write_text('#!/bin/sh\ncat >/dev/null\necho \'{"is_error":true,"subtype":"error",'
                   '"result":"rate limited","api_error_status":429}\'\n')
    spec = find_model(ModelRegistry(settings.models, "auto"), "seat/" + HAIKU)
    with pytest.raises(httpx.HTTPError) as exc:
        seat.ClaudeSeatProvider().complete(spec, "S", "u", max_tokens=10, timeout=30)  # type: ignore[arg-type]
    assert isinstance(exc.value, seat.SeatError) and exc.value.status == 429


def test_seat_models_are_available_without_a_key(settings: Settings, fake_claude: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RCE_SENTINEL_OFFLINE", "1")
    reg = ModelRegistry(settings.models, "auto")
    spec = find_model(reg, "seat/" + OPUS)
    assert spec is not None and reg.is_available(spec)
    monkeypatch.setenv(seat.BIN_ENV, "/nope/claude")
    seat.claude_bin.cache_clear()
    assert not reg.is_available(spec)
    assert os.environ[seat.BIN_ENV] == "/nope/claude"


def test_usage_report_sums_tokens_cost_and_savings(claude: tuple[str, ClaudeRouter]) -> None:
    from rce_sentinel.cli import usage_summary

    base, router = claude
    send(base, [user(EASY)])
    send(base, [user(EASY), ASSISTANT, TOOL_RESULT])
    rep = usage_summary(router.store, 0)
    assert rep["requests"] == 2 and rep["saved"] > 0
    assert rep["rules"] == {"new-conversation": 1, "sticky-tool-loop": 1}
    assert 0.9 < rep["cache_hit"] < 1.0


def test_seat_joins_a_reply_continued_past_the_output_cap(settings: Settings) -> None:
    reg = ModelRegistry(settings.models, "auto")
    spec = find_model(reg, "seat/" + HAIKU)
    assert spec is not None

    def say(text: str) -> str:
        return json.dumps({"type": "assistant",
                           "message": {"content": [{"type": "text", "text": text}]}})
    raw = "\n".join([
        json.dumps({"type": "system", "subtype": "init"}), say("## 1. Py"), say(""),
        json.dumps({"type": "user"}), say("thon\n## 2. Go"),
        json.dumps({"type": "result", "subtype": "success", "is_error": False,
                    "result": "thon\n## 2. Go", "total_cost_usd": 0.01,
                    "modelUsage": {HAIKU: {"outputTokens": 9}}})])
    res = seat.parse(raw, spec, 5.0)
    # The CLI's `result` is only the last message; every message is kept.
    assert res.text == "## 1. Python\n## 2. Go" and res.tokens_out == 9


def test_seat_join_repairs_continuation_seams() -> None:
    # Seen live: a continuation reopened the fence it was inside, mid-expression.
    a = "```dart\nreturn Message(\n  text:"
    assert seat._trim_seam(a, "```dart\n  text ?? this.text,\n") == "  text ?? this.text,\n"
    # A closing fence (no language) and text outside a fence are left alone.
    assert seat._trim_seam(a, "```\nDone.") == "```\nDone."
    assert seat._trim_seam("prose\n", "```dart\nx\n") == "```dart\nx\n"
    # A repeated tail is cut once; short or symbol-only overlaps are kept.
    tail = "publisher.publishPending(batchSize);"
    assert seat._trim_seam("x\n    " + tail, tail + "\n}") == "\n}"
    assert seat._trim_seam("abc", "abcdef") == "abcdef"
    assert seat._trim_seam("    }\n  }\n}\n" * 3, "    }\n  }\n}\n" * 3) == "    }\n  }\n}\n" * 3


# --- skills and token --------------------------------------------------------------------
def test_main_loop_gets_skills_after_claude_codes_system_blocks(
        claude: tuple[str, ClaudeRouter]) -> None:
    base, router = claude
    cached = {"type": "text", "text": "You are Claude Code.",
              "cache_control": {"type": "ephemeral"}}
    body = {"model": OPUS, "max_tokens": 1000, "messages": [user(EASY)], "tools": TOOLS,
            "system": [cached], "stream": False}
    r = httpx.post(base + "/v1/messages", json=body, timeout=30,
                   headers={"anthropic-version": "2023-06-01", "x-rce-token": "t"})
    assert r.status_code == 200
    sent = Anthropic.seen[-1]
    assert sent["body"]["system"][0] == cached
    assert sent["body"]["system"][1]["text"].startswith(MARKER)
    assert "cache_control" not in sent["body"]["system"][1]
    assert sent["token"] is None                     # the proxy's token stays local
    ev = events(router)[-1]
    assert ev["skills"] == "caveman"
    row = router.store.execute("SELECT caveman FROM llm_calls ORDER BY id DESC")[0]
    assert row["caveman"] == 1


def test_aux_calls_and_log_only_get_no_skills(claude: tuple[str, ClaudeRouter]) -> None:
    base, router = claude
    send(base, [user("title this chat")], model=HAIKU, tools=False)
    assert "system" not in Anthropic.seen[-1]["body"]
    router.log_only = True
    send(base, [user(EASY)])
    assert "system" not in Anthropic.seen[-1]["body"]
    assert events(router)[-1]["skills"] is None


def test_fallback_retry_keeps_the_skills(claude: tuple[str, ClaudeRouter]) -> None:
    base, _ = claude
    Anthropic.refuse = {HAIKU, SONNET}
    r = send(base, [user(EASY)], stream=False)
    assert r.status_code == 200 and Anthropic.seen[-1]["body"]["model"] == OPUS
    assert Anthropic.seen[-1]["body"]["system"].startswith(MARKER)


def test_proxy_token_is_checked_and_never_forwarded(settings: Settings) -> None:
    Anthropic.seen, Anthropic.refuse = [], set()
    up = ThreadingHTTPServer(("127.0.0.1", 0), Anthropic)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    store = Store(settings.db_path)
    router = ClaudeRouter(settings, store, upstream=f"http://127.0.0.1:{up.server_address[1]}")
    proxy = RouterProxy(settings, ModelRegistry(settings.models, "mock"), MockProvider(), store,
                        use_jev=False, claude=router, token="tok-0123456789abcdef")
    server = make_server(proxy, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        hdr = {"anthropic-version": "2023-06-01", "Authorization": "Bearer seat-token"}
        body = {"model": OPUS, "max_tokens": 10, "messages": [user(EASY)]}
        # A Bearer header is Claude Code's own login here, never the proxy token.
        denied = httpx.post(base + "/v1/messages", json=body, headers=hdr, timeout=30)
        assert denied.status_code == 401 and not Anthropic.seen
        ok = httpx.post(base + "/v1/messages", json=body, timeout=30,
                        headers={**hdr, "x-rce-token": "tok-0123456789abcdef"})
        assert ok.status_code == 200
        assert Anthropic.seen[-1]["auth"] == "Bearer seat-token"
        assert Anthropic.seen[-1]["token"] is None
    finally:
        server.shutdown()
        up.shutdown()
