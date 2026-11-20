"""OpenAI-compatible local proxy: point any coding agent at it and ask for model "auto".

    POST /v1/chat/completions   model "auto": pre-eval tags the last user message and picks
                                the model; any configured model id is passed through as is
    GET  /v1/models             "auto" plus every dev model reachable over the chat API
    POST /v1/rce/outcome        {"id": <x-rce-tag-id>, "outcome": "accepted|edited|rejected",
                                 "review_min": 2.5} grades a routed reply
    GET  /health
    *    any request carrying anthropic-version, and /v1/messages: Claude Code pass-through
         (see claude_proxy: routes between Claude models, keeps the prompt cache)

Before routing, a repeated request is answered from the answer cache (memory/answers.py)
with no model call; x-rce-cache says hit, miss or bypass. The active skills (skills.py)
are added to the system message of every forwarded request; x-rce-skills names them.

The request body is forwarded unchanged apart from `model` (and OpenRouter's provider
preferences), so tools, images and streaming work as the upstream supports them. A session
(header x-rce-session, else a hash of the first messages) keeps its warm model unless the
router finds a clearly better one; the client owns the transcript, so a switch hands off
the full history. Every reply carries x-rce-model, x-rce-tier and x-rce-tag-id headers.

Binds to 127.0.0.1 by default and holds no credentials of its own: upstream keys come from
the environment, exactly as for every other command. With RCE_SENTINEL_PROXY_TOKEN set,
every route but /health needs that token (header x-rce-token, or a Bearer token on the
OpenAI routes); binding to a non-loopback address without a token is refused.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import ipaddress
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx

from rce_sentinel import skills as skillmod
from rce_sentinel.config import Settings
from rce_sentinel.handoff.session import find_model
from rce_sentinel.memory.answers import AnswerCache, Hit, bypassed, reply_body, stream_chunks
from rce_sentinel.memory.store import OUTCOMES, Store
from rce_sentinel.orchestrator.llm import TIERS, ModelRegistry, ModelSpec, Provider

AUTO = {"auto", "rce/auto", "rce-sentinel/auto"}
MAX_BODY = 32 * 1024 * 1024
UPSTREAM_TIMEOUT = 600.0


def message_text(content: Any) -> str:
    """Text of an OpenAI message content: a string or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content
                         if isinstance(p, dict) and p.get("type") == "text")
    return ""


def last_user(messages: list[dict[str, Any]]) -> str:
    return next((message_text(m.get("content")) for m in reversed(messages)
                 if m.get("role") == "user"), "")


def session_key(messages: list[dict[str, Any]], header: str | None) -> str:
    if header:
        return header[:64]
    head = [message_text(m.get("content")) for m in messages[:2]]
    return "h" + hashlib.sha256("\x00".join(head).encode()).hexdigest()[:12]


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class RouterProxy:
    def __init__(self, settings: Settings, registry: ModelRegistry, provider: Provider,
                 store: Store, *, mode: str = "scan", use_jev: bool = True,
                 claude: Any = None, token: str | None = None,
                 answers: AnswerCache | None = None):
        self.settings = settings
        self.token = token if token is not None else os.environ.get(
            "RCE_SENTINEL_PROXY_TOKEN", "")
        self.skills = skillmod.load(settings.policy)
        self.answers = answers or AnswerCache.from_policy(store, settings.policy)
        self.claude = claude       # ClaudeRouter, or None when Claude routing is off
        self.reg = registry
        self.reg.apis = {"chat"}   # bodies are forwarded as chat-completions
        self.provider = provider
        self.store = store
        self.mode = mode
        self.use_jev = use_jev
        self.sessions: dict[str, str] = {}
        self._lock = threading.Lock()

    # --- routing -----------------------------------------------------------------------
    def route(self, body: dict[str, Any], headers: dict[str, str]
              ) -> tuple[ModelSpec | None, dict[str, Any]]:
        """Model for this request, and facts for the reply headers and the log."""
        messages = body.get("messages") or []
        key = session_key(messages, headers.get("x-rce-session"))
        want = str(body.get("model") or "auto")
        info: dict[str, Any] = {"session": key, "task": "chat", "tag_id": 0}
        with self._lock:
            current = self.sessions.get(key)
        if want not in AUTO:
            spec = find_model(self.reg, want)
            info["reason"] = "requested by client"
            if spec is not None and not self.reg.is_available(spec):
                info["unavailable"] = True
            return spec, info
        from rce_sentinel.preeval.pipeline import pre_eval

        prompt = last_user(messages) or "(empty)"
        res = pre_eval(prompt, self.settings, registry=self.reg, provider=self.provider,
                       store=self.store, use_jev=self.use_jev, current=current,
                       session_id=key, mode=headers.get("x-rce-profile") or self.mode)
        info.update(task=res.tags.top_task, tag_id=res.row_id, current=current,
                    reason=f"pre-eval: {res.tags.summary()}")
        return res.selection.model, info

    def remember(self, key: str, model: str) -> None:
        with self._lock:
            self.sessions[key] = model

    def models(self) -> dict[str, Any]:
        data = [{"id": "auto", "object": "model", "owned_by": "rce-sentinel"}]
        data += [{"id": m.id, "object": "model", "owned_by": m.provider}
                 for t in TIERS for m in self.reg.available(t, "dev")]
        return {"object": "list", "data": data}

    # --- skills and cache -------------------------------------------------------------
    @property
    def skills_version(self) -> str:
        return self.skills.version if self.skills.on("openai") else ""

    def with_skills(self, body: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        if not self.skills.on("openai"):
            return body, False
        return skillmod.inject_openai(body, self.skills)

    def cache_key(self, body: dict[str, Any], headers: dict[str, str]) -> str | None:
        if bypassed(headers):
            return None
        want = str(body.get("model") or "auto")
        return self.answers.key(body, None if want in AUTO else want, self.skills_version)

    def authorized(self, headers: dict[str, str], anthropic: bool) -> bool:
        """Constant-time token check; open when no token is configured."""
        if not self.token:
            return True
        want = self.token.encode()
        got = headers.get("x-rce-token", "")
        if got and hmac.compare_digest(got.encode(), want):
            return True
        bearer = headers.get("authorization", "")
        return (not anthropic and bearer[:7].lower() == "bearer "
                and hmac.compare_digest(bearer[7:].strip().encode(), want))

    # --- logging -----------------------------------------------------------------------
    @staticmethod
    def cost_of(spec: ModelSpec, usage: dict[str, Any]) -> float:
        tin = int(usage.get("prompt_tokens") or 0)
        tout = int(usage.get("completion_tokens") or 0)
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        reported = usage.get("cost")
        return float(reported) if isinstance(reported, int | float) else spec.cost(
            tin, tout, cached)

    def log(self, spec: ModelSpec, info: dict[str, Any], usage: dict[str, Any],
            seconds: float, status: int) -> None:
        tin = int(usage.get("prompt_tokens") or 0)
        tout = int(usage.get("completion_tokens") or 0)
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        cost = self.cost_of(spec, usage)
        self.store.log_call({
            "run_id": f"proxy:{info['session']}", "task_type": f"dev:{info['task']}",
            "tier": spec.tier, "model": spec.id, "family": spec.family,
            "outcome": "proxied" if status == 200 else f"upstream_{status}",
            "tokens_in": tin, "tokens_out": tout, "cached_tokens": cached, "cost": cost,
            "latency_ms": seconds * 1000, "caveman": int(bool(info.get("caveman"))),
            "shadow": 0, "simulated": int(self.reg.simulated(spec))})
        current = info.get("current")
        if status == 200 and current and current != spec.id:
            self.store.log_handoff({
                "run_id": f"proxy:{info['session']}", "mode": "transcript", "turn": 0,
                "model": spec.id, "switched_from": current, "reason": info.get("reason", ""),
                "context_tokens": tin, "tokens_in": tin, "tokens_out": tout, "cost": cost,
                "ok": 1, "files": "[]", "rewrites": "[]"})

    # --- upstream ----------------------------------------------------------------------
    def upstream(self, spec: ModelSpec, body: dict[str, Any]
                 ) -> tuple[str, dict[str, str], dict[str, Any]]:
        p = self.reg.providers[spec.provider]
        headers = {"Content-Type": "application/json"}
        if key_env := p.get("api_key_env"):
            headers["Authorization"] = f"Bearer {os.environ.get(key_env, '')}"
        out = {**body, "model": spec.id}
        if p.get("kind") == "openrouter":
            from rce_sentinel.orchestrator.llm import OpenAICompatProvider

            for k, v in OpenAICompatProvider(self.reg).openrouter_routing(spec, p).items():
                out.setdefault(k, v)
        if out.get("stream"):
            out.setdefault("stream_options", {"include_usage": True})
        return self.reg.base_url(spec.provider) + "/chat/completions", headers, out

    def simulate(self, spec: ModelSpec, body: dict[str, Any]) -> dict[str, Any]:
        """Offline reply through the configured provider (mock mode)."""
        messages = body.get("messages") or []
        system = "\n".join(message_text(m.get("content")) for m in messages
                           if m.get("role") == "system")
        user = "\n\n".join(f"{m.get('role', 'user').upper()}:\n{message_text(m.get('content'))}"
                           for m in messages if m.get("role") != "system")
        res = self.provider.complete(spec, system, f"TASK: proxy\nREQUEST:\n{user}",
                                     max_tokens=int(body.get("max_tokens") or 4000),
                                     timeout=UPSTREAM_TIMEOUT, text=True)
        return {"id": f"rce-{int(time.time() * 1000)}", "object": "chat.completion",
                "created": int(time.time()), "model": spec.id,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": res.text}}],
                "usage": {"prompt_tokens": res.tokens_in, "completion_tokens": res.tokens_out,
                          "total_tokens": res.tokens_in + res.tokens_out}}


class Handler(BaseHTTPRequestHandler):
    proxy: RouterProxy
    server_version = "rce-sentinel-proxy"

    def log_message(self, fmt: str, *args: Any) -> None:  # keep stderr quiet
        return

    def _json(self, status: int, payload: dict[str, Any],
              extra: dict[str, str] | None = None) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": {"message": message, "type": "rce_sentinel"}})

    def _body(self) -> dict[str, Any] | None:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BODY:
            self._error(413 if n > MAX_BODY else 400, "missing or oversized body")
            return None
        try:
            got = json.loads(self.rfile.read(n))
        except ValueError:
            self._error(400, "body is not JSON")
            return None
        if not isinstance(got, dict):
            self._error(400, "body must be a JSON object")
            return None
        return got

    def _authorized(self) -> bool:
        """False (after replying 401) when a token is set and this request lacks it."""
        from rce_sentinel.claude_proxy import is_anthropic

        if self.path.split("?", 1)[0].rstrip("/") == "/health":
            return True
        headers = {k.lower(): v for k, v in self.headers.items()}
        if self.proxy.authorized(headers, is_anthropic(self)):
            return True
        self._error(401, "missing or wrong proxy token (header x-rce-token)")
        return False

    def _claude(self) -> bool:
        """Hand Anthropic API traffic to the Claude router; True when handled."""
        from rce_sentinel.claude_proxy import handle, is_anthropic

        if not is_anthropic(self):
            return False
        if self.proxy.claude is None:
            self._error(404, "Claude routing is off; start `serve` without --no-claude")
        else:
            handle(self.proxy.claude, self)
        return True

    def do_GET(self) -> None:
        if not self._authorized() or self._claude():
            return
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._json(200, self.proxy.models())
        elif self.path.rstrip("/") == "/health":
            self._json(200, {"ok": True})
        else:
            self._error(404, f"no route {self.path}")

    def do_POST(self) -> None:
        if not self._authorized() or self._claude():
            return
        path = self.path.rstrip("/")
        if path == "/v1/rce/outcome":
            self._outcome()
        elif path in ("/v1/chat/completions", "/chat/completions"):
            self._chat()
        else:
            self._error(404, f"no route {self.path}")

    def _outcome(self) -> None:
        body = self._body()
        if body is None:
            return
        outcome = str(body.get("outcome", ""))
        if outcome not in OUTCOMES:
            self._error(400, f"outcome must be one of {sorted(OUTCOMES)}")
            return
        try:
            review = body.get("review_min")
            rid = self.proxy.store.outcome_for_tag(
                int(body.get("id") or 0), outcome,
                float(review) * 60 if review is not None else None)
        except (KeyError, ValueError, TypeError) as exc:
            self._error(404 if isinstance(exc, KeyError) else 400, str(exc))
            return
        self._json(200, {"ok": True, "outcome_id": rid})

    def _chat(self) -> None:
        body = self._body()
        if body is None:
            return
        headers = {k.lower(): v for k, v in self.headers.items()}
        ckey = self.proxy.cache_key(body, headers)
        if ckey and (hit := self.proxy.answers.get(ckey)):
            self._cached(hit, body)
            return
        spec, info = self.proxy.route(body, headers)
        if spec is None:
            self._error(503 if body.get("model", "auto") in AUTO else 404,
                        "no model available" if body.get("model", "auto") in AUTO
                        else f"model {body.get('model')} is not configured")
            return
        if info.get("unavailable"):
            self._error(503, f"{spec.id} is not available (no key, or provider unreachable)")
            return
        if spec.api != "chat":
            self._error(400, f"{spec.id} uses the {spec.api} API; the proxy forwards chat only")
            return
        body, injected = self.proxy.with_skills(body)
        info["caveman"] = injected and "caveman" in self.proxy.skills.names
        extra = {"x-rce-model": spec.id, "x-rce-tier": spec.tier,
                 "x-rce-tag-id": str(info.get("tag_id", 0)),
                 "x-rce-cache": "miss" if ckey else "bypass",
                 "x-rce-skills": ",".join(self.proxy.skills.names) if injected else "none"}
        t0 = time.monotonic()
        if self.proxy.reg.simulated(spec):
            self._simulated(spec, body, info, extra, t0)   # never cached: not a real answer
            return
        url, up_headers, out = self.proxy.upstream(spec, body)
        try:
            if out.get("stream"):
                # The connection closes after this handler returns, so logging after
                # the stream still lands before the client sees the end.
                status, usage, got = self._stream(url, up_headers, out, extra)
                self._done(spec, info, usage, t0, status)
                if status == 200 and ckey and got is not None:
                    self.proxy.answers.put(ckey, spec.id, got[0], got[1], usage,
                                           self.proxy.cost_of(spec, usage))
                return
            resp = httpx.post(url, json=out, headers=up_headers, timeout=UPSTREAM_TIMEOUT)
        except httpx.HTTPError as exc:
            from rce_sentinel.preeval.pipeline import net_error

            self.proxy.log(spec, info, {}, time.monotonic() - t0, 502)
            self._error(502, f"upstream {spec.id}: {net_error(exc)}")
            return
        try:
            data = resp.json()
        except ValueError:
            data = {"error": {"message": resp.text[:500]}}
        if not isinstance(data, dict):
            data = {"data": data}
        usage = data.get("usage") or {}
        self._done(spec, info, usage, t0, resp.status_code)
        choices = data.get("choices")
        if (resp.status_code == 200 and ckey and isinstance(choices, list)
                and len(choices) == 1 and isinstance(choices[0], dict)):
            self.proxy.answers.put(ckey, spec.id, choices[0].get("message") or {},
                                   str(choices[0].get("finish_reason") or ""), usage,
                                   self.proxy.cost_of(spec, usage))
        self._json(resp.status_code, data, extra)

    def _cached(self, hit: Hit, body: dict[str, Any]) -> None:
        """Serve a cache hit in the shape the client asked for: JSON or a stream."""
        extra = {"x-rce-model": hit.model, "x-rce-tier": "cache", "x-rce-tag-id": "0",
                 "x-rce-cache": "hit", "x-rce-cache-age-s": f"{hit.age_s:.0f}",
                 "x-rce-cache-saved-usd": f"{hit.cost:.6f}"}
        if not body.get("stream"):
            self._json(200, reply_body(hit), extra)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        for k, v in extra.items():
            self.send_header(k, v)
        self.send_header("Connection", "close")
        self.end_headers()
        for chunk in stream_chunks(hit):
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    def _done(self, spec: ModelSpec, info: dict[str, Any], usage: dict[str, Any],
              t0: float, status: int) -> None:
        """Log and remember before the reply completes, so clients never race it."""
        self.proxy.log(spec, info, usage, time.monotonic() - t0, status)
        if status == 200:
            self.proxy.remember(info["session"], spec.id)

    def _simulated(self, spec: ModelSpec, body: dict[str, Any], info: dict[str, Any],
                   extra: dict[str, str], t0: float) -> None:
        reply = self.proxy.simulate(spec, body)
        self._done(spec, info, reply["usage"], t0, 200)
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            for k, v in extra.items():
                self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            chunk = {"id": reply["id"], "object": "chat.completion.chunk",
                     "created": reply["created"], "model": spec.id,
                     "choices": [{"index": 0, "finish_reason": "stop",
                                  "delta": reply["choices"][0]["message"]}],
                     "usage": reply["usage"]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode())
            self.wfile.flush()
            self.close_connection = True
        else:
            self._json(200, reply, extra)

    def _stream(self, url: str, headers: dict[str, str], body: dict[str, Any],
                extra: dict[str, str]
                ) -> tuple[int, dict[str, Any], tuple[dict[str, Any], str] | None]:
        """Relay server-sent events as they arrive; pick usage out of the last chunk.

        Also rebuilds the full reply, so a complete stream can be cached."""
        from rce_sentinel.memory.answers import StreamCollector

        usage: dict[str, Any] = {}
        collect = StreamCollector()
        with httpx.stream("POST", url, json=body, headers=headers,
                          timeout=UPSTREAM_TIMEOUT) as resp:
            self.send_response(resp.status_code)
            self.send_header("Content-Type",
                             resp.headers.get("content-type", "text/event-stream"))
            for k, v in extra.items():
                self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            for line in resp.iter_lines():
                self.wfile.write(line.encode() + b"\n")
                self.wfile.flush()
                if line.startswith("data: {"):
                    with contextlib.suppress(ValueError):
                        chunk = json.loads(line[6:])
                        if isinstance(chunk, dict):
                            usage = chunk.get("usage") or usage
                            collect.feed(chunk)
            self.close_connection = True
            return resp.status_code, usage, collect.complete()


def make_server(proxy: RouterProxy, host: str = "127.0.0.1", port: int = 8787
                ) -> ThreadingHTTPServer:
    """Fail closed: a proxy reachable from other machines must have a token."""
    if not is_loopback(host) and not proxy.token:
        raise ValueError(f"refusing to listen on {host} without RCE_SENTINEL_PROXY_TOKEN; "
                         "anyone on the network could spend your model keys")
    handler = type("BoundHandler", (Handler,), {"proxy": proxy})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server
