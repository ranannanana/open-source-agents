"""Claude Code pass-through: route each conversation between Claude models, keep the cache.

    ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude

Claude Code signs in as you; its auth headers ride through untouched and nothing is
stored. Only POST /v1/messages from the main agent loop (requests that declare tools) is
routed; every other call (titles, summaries, token counts) passes through as is.

Prompt caches belong to one model, so switching mid-conversation re-sends the whole
context uncached. The router therefore decides once per conversation, then keeps it:

    new conversation          pre-eval picks a model from the pool
    new typed user message    move up when the router wants a stronger model;
                              move down only when the cache has gone cold anyway
    tool-loop continuation    never switch

Main-loop requests also get the active skills (skills.py) as a last system block, after
Claude Code's own cache breakpoints, so its prompt cache keeps hitting. Log-only mode
changes nothing on the wire, skills included.

When the upstream refuses the chosen model (400 naming the model, or 404) the request is
retried once with the model Claude Code asked for. Every request lands in usage_events
with exact token, cache and cost figures, plus what the asked-for model would have cost.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
from typing import Any

import httpx

from rce_sentinel import skills as skillmod
from rce_sentinel.config import Settings
from rce_sentinel.handoff.session import find_model
from rce_sentinel.memory.store import Store
from rce_sentinel.orchestrator.llm import ModelRegistry, ModelSpec
from rce_sentinel.orchestrator.seat import wire_id

UPSTREAM = "https://api.anthropic.com"
POOL = ("seat/claude-haiku-5-5", "seat/claude-sonnet-5-5", "seat/claude-opus-5-5")
HOP = {"host", "content-length", "connection", "keep-alive", "transfer-encoding",
       "accept-encoding", "proxy-connection", "upgrade", "te", "trailer",
       "x-rce-token"}   # the proxy's own token never goes upstream
RESP_DROP = {"content-length", "transfer-encoding", "connection", "keep-alive",
             "content-encoding"}
TTL_5M, TTL_1H = 300.0, 3600.0
MAX_SESSIONS = 2000
_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)


def typed_text(message: dict[str, Any]) -> str:
    """What the person typed in a user message: text blocks minus system reminders."""
    content = message.get("content")
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
    if not isinstance(blocks, list):
        return ""
    parts = [_REMINDER.sub("", b.get("text", "")).strip() for b in blocks
             if isinstance(b, dict) and b.get("type") == "text"]
    return "\n".join(p for p in parts if p)


def conversation_key(body: dict[str, Any], headers: dict[str, str]) -> str:
    if sid := headers.get("x-claude-code-session-id"):
        return "c" + sid[:64]
    msgs = body.get("messages") or []
    first = json.dumps(msgs[0], sort_keys=True)[:4000] if msgs else ""
    return "h" + hashlib.sha256(first.encode()).hexdigest()[:16]


@dataclass
class Conversation:
    model: str               # pool spec id the conversation is on (virtual in log-only)
    last: float              # time of the last request
    ttl: float = TTL_5M      # cache lifetime seen in the last reply
    turns: int = 0


@dataclass
class Decision:
    requested: str
    send: str                # wire model id sent upstream
    policy: str              # wire model id the policy wanted
    rule: str
    key: str = ""
    kind: str = "aux"
    task: str = ""
    tag_id: int = 0
    switched_from: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    skills: str = ""         # comma list of skills injected into this request

    @property
    def rewritten(self) -> bool:
        return self.send != self.requested


class ClaudeRouter:
    def __init__(self, settings: Settings, store: Store, *, upstream: str = UPSTREAM,
                 log_only: bool = False, pool: tuple[str, ...] = POOL,
                 mode: str = "claude-code"):
        self.settings = settings
        self.store = store
        self.upstream = upstream.rstrip("/")
        self.log_only = log_only
        self.mode = mode
        self.skills = skillmod.load(settings.policy)
        # Routing only needs prices and priors; the seat CLI is not called from here.
        self.reg = ModelRegistry(settings.models, "mock")
        self.reg.only = set(pool)
        self.pool = {wire_id(s): s for p in pool if (s := find_model(self.reg, p))}
        self.convs: OrderedDict[str, Conversation] = OrderedDict()
        self._lock = threading.Lock()

    # --- decision --------------------------------------------------------------------
    def spec(self, wire: str) -> ModelSpec | None:
        return self.pool.get(wire)

    def rank(self, wire: str) -> float:
        s = self.spec(wire)
        return s.price_out if s else -1.0

    def pick(self, text: str, key: str) -> tuple[str, str, int]:
        """(wire model, task, tag id) from pre-eval over the pool."""
        from rce_sentinel.preeval.pipeline import pre_eval

        res = pre_eval(text or "(empty)", self.settings, registry=self.reg, store=self.store,
                       use_jev=False, session_id=f"claude:{key}", mode=self.mode)
        model = res.selection.model
        return (wire_id(model) if model else "", res.tags.top_task, res.row_id)

    def decide(self, body: dict[str, Any], headers: dict[str, str]) -> Decision:
        requested = str(body.get("model") or "")
        d = Decision(requested=requested, send=requested, policy=requested, rule="aux")
        if not body.get("tools"):
            return d           # Claude Code's own small calls: leave them alone
        d.kind = "main"
        if requested not in self.pool:
            d.rule = "not-in-pool"
            return d
        msgs = body.get("messages") or []
        d.key = key = conversation_key(body, headers)
        last = msgs[-1] if msgs else {}
        text = typed_text(last) if last.get("role") == "user" else ""
        now = time.time()
        with self._lock:
            conv = self.convs.get(key)
        if conv is None:
            want, d.task, d.tag_id = self.pick(text, key)
            d.policy, d.rule = want or requested, "new-conversation"
        elif not text:
            d.policy, d.rule = conv.model, "sticky-tool-loop"
        else:
            want, d.task, d.tag_id = self.pick(text, key)
            cur = conv.model
            cold = now - conv.last > conv.ttl
            if want and self.rank(want) > self.rank(cur):
                d.policy, d.rule = want, "upgrade"
            elif want and want != cur and cold:
                d.policy, d.rule = want, "cold-cache-switch"
            else:
                d.policy = cur
                d.rule = "sticky" if want == cur or not want else "sticky-cache-warm"
            if d.policy != cur:
                d.switched_from = cur
        if d.policy not in self.pool:
            d.policy = requested
        d.send = requested if self.log_only else d.policy
        return d

    def remember(self, d: Decision, ttl: float | None) -> None:
        if d.kind != "main" or d.rule == "not-in-pool":
            return
        with self._lock:
            conv = self.convs.pop(d.key, None) or Conversation(model=d.policy, last=0.0)
            conv.model = d.policy
            conv.last = time.time()
            conv.turns += 1
            if ttl:
                conv.ttl = ttl
            self.convs[d.key] = conv
            while len(self.convs) > MAX_SESSIONS:
                self.convs.popitem(last=False)

    # --- usage and logging -----------------------------------------------------------
    def log(self, d: Decision, path: str, body: dict[str, Any], usage: dict[str, int],
            status: int, seconds: float, ttft: float | None, fallback: bool,
            sent: str) -> None:
        spec = self.spec(sent)
        asked = self.spec(d.requested)
        args = (usage.get("input", 0), usage.get("write_5m", 0), usage.get("write_1h", 0),
                usage.get("read", 0), usage.get("output", 0))
        cost = spec.usage_cost(*args) if spec else None
        cost_asked = asked.usage_cost(*args) if asked else None
        tokens_in = sum(args[:4])
        self.store.log_usage({
            "source": "claude-code", "session": d.key, "path": path, "kind": d.kind,
            "requested": d.requested, "model": sent, "policy_choice": d.policy,
            "rule": d.rule, "rewritten": int(sent != d.requested), "fallback": int(fallback),
            "log_only": int(self.log_only), "status": status, "task": d.task,
            "tag_id": d.tag_id or None, "tokens_in": tokens_in,
            "input_tokens": args[0], "cache_write_5m": args[1], "cache_write_1h": args[2],
            "cache_read": args[3], "tokens_out": args[4], "cost": cost,
            "cost_requested": cost_asked, "latency_ms": seconds * 1000,
            "ttft_ms": ttft * 1000 if ttft is not None else None,
            "n_messages": len(body.get("messages") or []), "n_tools": len(body.get("tools") or []),
            "stream": int(bool(body.get("stream"))), "switched_from": d.switched_from or None,
            "skills": d.skills or None})
        if spec and d.kind == "main" and status == 200:
            self.store.log_call({
                "run_id": f"claude:{d.key}", "task_type": f"dev:{d.task or 'session'}",
                "tier": spec.tier, "model": spec.id, "family": spec.family,
                "outcome": "proxied", "tokens_in": tokens_in, "tokens_out": args[4],
                "cached_tokens": args[3], "cost": cost, "latency_ms": seconds * 1000,
                "caveman": int("caveman" in d.skills.split(",")),
                "shadow": int(self.log_only), "simulated": 0})
            if d.switched_from and not self.log_only:
                self.store.log_handoff({
                    "run_id": f"claude:{d.key}", "mode": "transcript", "turn": 0,
                    "model": spec.id, "switched_from": d.switched_from, "reason": d.rule,
                    "context_tokens": tokens_in, "tokens_in": tokens_in,
                    "tokens_out": args[4], "cost": cost, "ok": 1, "files": "[]",
                    "rewrites": "[]"})


class UsageMeter:
    """Pull Anthropic usage out of a JSON reply or a server-sent event stream."""

    def __init__(self) -> None:
        self.usage = {"input": 0, "write_5m": 0, "write_1h": 0, "read": 0, "output": 0}
        self.ttl: float | None = None
        self._buf = b""

    def take(self, u: dict[str, Any]) -> None:
        if "input_tokens" in u:
            self.usage["input"] = int(u.get("input_tokens") or 0)
        if "cache_read_input_tokens" in u:
            self.usage["read"] = int(u.get("cache_read_input_tokens") or 0)
        if "output_tokens" in u:
            self.usage["output"] = int(u.get("output_tokens") or 0)
        if "cache_creation_input_tokens" in u:
            split = u.get("cache_creation") or {}
            h1 = int(split.get("ephemeral_1h_input_tokens") or 0)
            total = int(u.get("cache_creation_input_tokens") or 0)
            self.usage["write_1h"], self.usage["write_5m"] = h1, max(total - h1, 0)
            if total:
                self.ttl = TTL_1H if h1 else TTL_5M

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            if not line.startswith(b"data: {"):
                continue
            try:
                ev = json.loads(line[6:])
            except ValueError:
                continue
            if ev.get("type") == "message_start":
                self.take((ev.get("message") or {}).get("usage") or {})
            elif ev.get("type") == "message_delta":
                self.take(ev.get("usage") or {})

    def whole(self, raw: bytes) -> None:
        try:
            data = json.loads(raw)
        except ValueError:
            return
        if isinstance(data, dict):
            self.take(data.get("usage") or {})


def is_anthropic(handler: BaseHTTPRequestHandler) -> bool:
    return bool(handler.headers.get("anthropic-version")) or handler.path.startswith(
        "/v1/messages")


def handle(router: ClaudeRouter, h: BaseHTTPRequestHandler) -> None:
    """Forward one Anthropic API request, routing /v1/messages when it is ours to route."""
    n = int(h.headers.get("Content-Length") or 0)
    raw = h.rfile.read(n) if n > 0 else b""
    headers = {k.lower(): v for k, v in h.headers.items()}
    path_only = h.path.split("?", 1)[0].rstrip("/")
    body: dict[str, Any] = {}
    if raw:
        try:
            got = json.loads(raw)
            body = got if isinstance(got, dict) else {}
        except ValueError:
            body = {}
    route = h.command == "POST" and path_only == "/v1/messages" and "model" in body
    d = router.decide(body, headers) if route else None
    out, original = raw, raw
    send = body
    if d and d.kind == "main" and not router.log_only and router.skills.on("anthropic"):
        send, injected = skillmod.inject_anthropic(body, router.skills)
        if injected:
            d.skills = ",".join(router.skills.names)
            original = _dump(send)
    if d and (d.rewritten or d.skills):
        out = _dump({**send, "model": d.send})
    fwd = [(k, v) for k, v in h.headers.items() if k.lower() not in HOP]
    t0 = time.monotonic()
    try:
        status, meter, ttft, fallback, sent = _relay(router, h, fwd, out, original, d)
    except httpx.HTTPError as exc:
        msg = json.dumps({"type": "error", "error": {
            "type": "api_error", "message": f"rce-sentinel proxy: {type(exc).__name__}"}})
        h.send_response(502)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(msg)))
        h.end_headers()
        h.wfile.write(msg.encode())
        if d:
            router.log(d, h.path, body, {}, 502, time.monotonic() - t0, None, False, d.send)
        return
    if d:
        router.log(d, h.path, body, meter.usage, status, time.monotonic() - t0, ttft,
                   fallback, sent)
        if status == 200:
            router.remember(d, meter.ttl)


def _dump(body: dict[str, Any]) -> bytes:
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()


def _relay(router: ClaudeRouter, h: BaseHTTPRequestHandler, fwd: list[tuple[str, str]],
           out: bytes, original: bytes, d: Decision | None
           ) -> tuple[int, UsageMeter, float | None, bool, str]:
    url = router.upstream + h.path
    sent = d.send if d else ""
    t0 = time.monotonic()
    with httpx.Client(timeout=httpx.Timeout(600.0, connect=15.0)) as client:
        req = client.build_request(h.command, url, headers=fwd, content=out or None)
        resp = client.send(req, stream=True)
        fallback = False
        if d and d.rewritten and resp.status_code in (400, 404):
            detail = resp.read()
            if resp.status_code == 404 or b"model" in detail.lower():
                resp.close()
                req = client.build_request(h.command, url, headers=fwd, content=original)
                resp = client.send(req, stream=True)
                fallback, sent = True, d.requested
            else:
                resp.close()
                return (*_emit(h, resp, detail, t0), fallback, sent)
        try:
            return (*_emit(h, resp, None, t0), fallback, sent)
        finally:
            resp.close()


def _emit(h: BaseHTTPRequestHandler, resp: httpx.Response, preloaded: bytes | None,
          t0: float) -> tuple[int, UsageMeter, float | None]:
    meter = UsageMeter()
    sse = "text/event-stream" in resp.headers.get("content-type", "")
    h.send_response(resp.status_code)
    for k, v in resp.headers.items():
        if k.lower() not in RESP_DROP:
            h.send_header(k, v)
    h.send_header("Connection", "close")
    h.end_headers()
    ttft: float | None = None
    body = b""
    chunks = [preloaded] if preloaded is not None else resp.iter_bytes()
    for chunk in chunks:
        if not chunk:
            continue
        if ttft is None:
            ttft = time.monotonic() - t0
        h.wfile.write(chunk)
        h.wfile.flush()
        if sse:
            meter.feed(chunk)
        else:
            body += chunk
    if not sse:
        meter.whole(body)
    h.close_connection = True
    return resp.status_code, meter, ttft
