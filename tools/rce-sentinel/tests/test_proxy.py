"""OpenAI-compatible proxy: routing, pass-through, streaming, sessions, outcomes, errors."""

from __future__ import annotations

import copy
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

import httpx
import pytest

from rce_sentinel.config import Settings
from rce_sentinel.memory.store import Store
from rce_sentinel.orchestrator.llm import MockProvider, ModelRegistry, make_provider
from rce_sentinel.proxy import RouterProxy, last_user, make_server, session_key
from rce_sentinel.skills import MARKER

HARD = ("Build a scientific calculator app with the best possible and cleanest UI for iOS "
        "and Android. It needs to be in React Native, Swift, and Flutter.")


def serve(server: ThreadingHTTPServer) -> str:
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    return f"http://{host}:{port}"


@pytest.fixture
def mock_proxy(settings: Settings) -> Iterator[tuple[str, RouterProxy]]:
    proxy = RouterProxy(settings, ModelRegistry(settings.models, "mock"), MockProvider(),
                        Store(settings.db_path), use_jev=False)
    server = make_server(proxy, "127.0.0.1", 0)
    yield serve(server), proxy
    server.shutdown()


class Upstream(BaseHTTPRequestHandler):
    seen: ClassVar[list[dict[str, Any]]] = []

    def log_message(self, *a: Any) -> None:
        return

    def do_GET(self) -> None:
        raw = json.dumps({"data": []}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Upstream.seen.append({"body": body, "auth": self.headers.get("Authorization")})
        usage = {"prompt_tokens": 120, "completion_tokens": 30, "cost": 0.0042}
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for part in ("Hel", "lo"):
                chunk = {"choices": [{"index": 0, "delta": {"content": part},
                                      "finish_reason": "stop" if part == "lo" else None}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n"
                             "data: [DONE]\n\n".encode())
            return
        raw = json.dumps({"choices": [{"index": 0, "finish_reason": "stop",
                                       "message": {"role": "assistant", "content": "Hello"}}],
                          "usage": usage}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def real_proxy(settings: Settings, monkeypatch: pytest.MonkeyPatch
               ) -> Iterator[tuple[str, RouterProxy]]:
    Upstream.seen = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    base = serve(upstream)
    monkeypatch.setenv("FAKE_KEY", "test-key")
    cfg = copy.deepcopy(settings.models)
    cfg["providers"]["fake"] = {"base_url": base + "/v1", "api_key_env": "FAKE_KEY"}
    cfg["tiers"] = {
        "tier1": [{"id": "fake-small", "provider": "fake", "family": "a", "price_in": 0.1,
                   "price_out": 0.4, "uses": ["dev"], "quality": 0.5}],
        "tier2": [{"id": "fake-mid", "provider": "fake", "family": "b", "price_in": 0.5,
                   "price_out": 2.0, "uses": ["dev"], "quality": 0.7},
                  {"id": "fake-native", "provider": "fake", "family": "c", "price_in": 0.5,
                   "price_out": 2.0, "uses": ["dev"], "api": "messages"}],
        "tier3": [{"id": "fake-big", "provider": "fake", "family": "d", "price_in": 2.0,
                   "price_out": 10.0, "uses": ["dev"], "quality": 0.92}]}
    reg = ModelRegistry(cfg, "auto")
    proxy = RouterProxy(settings, reg, make_provider(reg), Store(settings.db_path),
                        use_jev=False)
    server = make_server(proxy, "127.0.0.1", 0)
    yield serve(server), proxy
    server.shutdown()
    upstream.shutdown()


def chat(base: str, content: str, **kw: Any) -> httpx.Response:
    headers = kw.pop("headers", {})
    body = {"model": "auto", "messages": [{"role": "user", "content": content}], **kw}
    return httpx.post(base + "/v1/chat/completions", json=body, headers=headers, timeout=30)


# --- helpers -----------------------------------------------------------------------------
def test_last_user_reads_text_parts_and_session_is_stable() -> None:
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": [{"type": "text", "text": "a"},
                                         {"type": "image_url", "image_url": {}}]},
            {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]
    assert last_user(msgs) == "c" and last_user(msgs[:2]) == "a"
    assert session_key(msgs, None) == session_key(msgs[:2], None)
    assert session_key(msgs, "abc") == "abc"


# --- mock mode ---------------------------------------------------------------------------
def test_models_and_health(mock_proxy: tuple[str, RouterProxy]) -> None:
    base, _ = mock_proxy
    ids = [m["id"] for m in httpx.get(base + "/v1/models").json()["data"]]
    assert ids[0] == "auto" and len(ids) > 5
    assert httpx.get(base + "/health").json() == {"ok": True}


def test_auto_routes_and_reports_the_choice(mock_proxy: tuple[str, RouterProxy]) -> None:
    base, proxy = mock_proxy
    r = chat(base, HARD)
    assert r.status_code == 200
    assert r.headers["x-rce-tier"] == "tier3" and int(r.headers["x-rce-tag-id"]) > 0
    assert r.json()["choices"][0]["message"]["content"]
    row = proxy.store.execute("SELECT task_type, model FROM llm_calls")[0]
    assert row["task_type"] == "dev:build" and row["model"] == r.headers["x-rce-model"]


def test_mock_stream_is_server_sent_events(mock_proxy: tuple[str, RouterProxy]) -> None:
    base, _ = mock_proxy
    r = chat(base, "fix typo in README", stream=True)
    lines = [ln for ln in r.text.splitlines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    assert json.loads(lines[0][6:])["choices"][0]["delta"]["content"]


def test_outcome_endpoint_grades_a_routed_reply(mock_proxy: tuple[str, RouterProxy]) -> None:
    base, proxy = mock_proxy
    tag = int(chat(base, HARD).headers["x-rce-tag-id"])
    ok = httpx.post(base + "/v1/rce/outcome", json={"id": tag, "outcome": "edited",
                                                     "review_min": 4})
    assert ok.status_code == 200
    row = proxy.store.execute("SELECT outcome, review_s FROM outcomes")[0]
    assert row["outcome"] == "edited" and row["review_s"] == 240
    bad = httpx.post(base + "/v1/rce/outcome", json={"id": tag, "outcome": "meh"})
    missing = httpx.post(base + "/v1/rce/outcome", json={"id": 9999, "outcome": "accepted"})
    assert bad.status_code == 400 and missing.status_code == 404


def test_bad_requests_get_clear_errors(mock_proxy: tuple[str, RouterProxy]) -> None:
    base, _ = mock_proxy
    url = base + "/v1/chat/completions"
    assert httpx.post(url, content=b"{not json",
                      headers={"Content-Type": "application/json"}).status_code == 400
    assert httpx.post(url, json=[1, 2]).status_code == 400
    r = httpx.post(url, json={"model": "no-such", "messages": []})
    assert r.status_code == 404 and "not configured" in r.json()["error"]["message"]
    assert httpx.get(base + "/nope").status_code == 404


# --- pass-through to a real upstream -----------------------------------------------------
def test_forwards_body_with_chosen_model_and_key(real_proxy: tuple[str, RouterProxy]) -> None:
    base, proxy = real_proxy
    r = chat(base, HARD, temperature=0.3, tools=[{"type": "function",
                                                 "function": {"name": "f"}}])
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "Hello"
    sent = Upstream.seen[-1]
    assert sent["body"]["model"] == r.headers["x-rce-model"] == "fake-big"
    assert sent["body"]["temperature"] == 0.3 and sent["body"]["tools"]
    assert sent["auth"] == "Bearer test-key"
    row = proxy.store.execute("SELECT * FROM llm_calls")[0]
    assert (row["tokens_in"], row["tokens_out"], row["cost"]) == (120, 30, 0.0042)
    assert row["simulated"] == 0


def test_streams_through_and_captures_usage(real_proxy: tuple[str, RouterProxy]) -> None:
    base, proxy = real_proxy
    r = chat(base, HARD, stream=True)
    text = "".join(json.loads(ln[6:])["choices"][0]["delta"]["content"]
                   for ln in r.text.splitlines()
                   if ln.startswith("data: {") and '"delta"' in ln)
    assert text == "Hello" and r.text.rstrip().endswith("data: [DONE]")
    assert Upstream.seen[-1]["body"]["stream_options"] == {"include_usage": True}
    assert proxy.store.execute("SELECT tokens_out FROM llm_calls")[0]["tokens_out"] == 30


def test_session_switch_is_logged_as_a_handoff(real_proxy: tuple[str, RouterProxy]) -> None:
    base, proxy = real_proxy
    hdr = {"x-rce-session": "s1", "x-rce-cache": "off"}   # repeats must reach a model
    first = chat(base, HARD, headers=hdr).headers["x-rce-model"]
    chat(base, HARD, headers=hdr)
    assert proxy.store.execute("SELECT COUNT(*) AS n FROM handoffs")[0]["n"] == 0
    proxy.remember("s1", "fake-small")    # pretend the session was on another model
    chat(base, HARD, headers=hdr)
    row = proxy.store.execute("SELECT * FROM handoffs")[0]
    assert row["switched_from"] == "fake-small" and row["model"] == first


def test_non_chat_models_are_never_routed_or_forwarded(
        real_proxy: tuple[str, RouterProxy]) -> None:
    base, _ = real_proxy
    ids = [m["id"] for m in httpx.get(base + "/v1/models").json()["data"]]
    assert "fake-native" not in ids
    r = httpx.post(base + "/v1/chat/completions",
                   json={"model": "fake-native", "messages": [{"role": "user",
                                                               "content": "hi"}]})
    assert r.status_code == 400 and "messages API" in r.json()["error"]["message"]


def test_unreachable_upstream_is_a_502(real_proxy: tuple[str, RouterProxy]) -> None:
    base, proxy = real_proxy
    proxy.reg.providers["fake"]["base_url"] = "http://127.0.0.1:9/v1"
    r = chat(base, HARD)
    assert r.status_code == 502 and "upstream" in r.json()["error"]["message"]


def test_explicit_model_without_key_is_a_503(real_proxy: tuple[str, RouterProxy],
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    base, _ = real_proxy
    monkeypatch.delenv("FAKE_KEY")
    r = httpx.post(base + "/v1/chat/completions",
                   json={"model": "fake-big", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503 and "not available" in r.json()["error"]["message"]


# --- answer cache ------------------------------------------------------------------------
def test_repeat_request_is_served_from_cache_without_a_model_call(
        real_proxy: tuple[str, RouterProxy]) -> None:
    base, proxy = real_proxy
    first = chat(base, HARD)
    assert first.headers["x-rce-cache"] == "miss" and len(Upstream.seen) == 1
    again = chat(base, HARD, temperature=0.7)      # sampling settings are not in the key
    assert again.status_code == 200 and len(Upstream.seen) == 1
    assert again.headers["x-rce-cache"] == "hit"
    assert again.headers["x-rce-model"] == first.headers["x-rce-model"]
    assert again.json()["choices"][0]["message"]["content"] == "Hello"
    assert float(again.headers["x-rce-cache-saved-usd"]) == 0.0042
    assert proxy.store.execute("SELECT COUNT(*) AS n FROM llm_calls")[0]["n"] == 1
    streamed = chat(base, HARD, stream=True)        # same answer, as server-sent events
    lines = [ln for ln in streamed.text.splitlines() if ln.startswith("data: ")]
    assert streamed.headers["x-rce-cache"] == "hit" and lines[-1] == "data: [DONE]"
    assert json.loads(lines[0][6:])["choices"][0]["delta"]["content"] == "Hello"
    assert len(Upstream.seen) == 1


def test_streamed_reply_is_cached_once_complete(real_proxy: tuple[str, RouterProxy]) -> None:
    base, _ = real_proxy
    chat(base, HARD, stream=True)
    hit = chat(base, HARD)
    assert hit.headers["x-rce-cache"] == "hit" and len(Upstream.seen) == 1
    assert hit.json()["choices"][0]["message"]["content"] == "Hello"


def test_cache_bypass_and_changed_requests_reach_the_model(
        real_proxy: tuple[str, RouterProxy]) -> None:
    base, _ = real_proxy
    chat(base, HARD)
    r = chat(base, HARD, headers={"x-rce-cache": "off"})
    assert r.headers["x-rce-cache"] == "bypass" and len(Upstream.seen) == 2
    chat(base, HARD, max_tokens=50)
    assert len(Upstream.seen) == 3
    named = httpx.post(base + "/v1/chat/completions", timeout=30, json={
        "model": "fake-big", "messages": [{"role": "user", "content": HARD}]})
    assert named.headers["x-rce-cache"] == "miss" and len(Upstream.seen) == 4


def test_mock_replies_are_never_cached(mock_proxy: tuple[str, RouterProxy]) -> None:
    base, proxy = mock_proxy
    chat(base, HARD)
    assert chat(base, HARD).headers["x-rce-cache"] == "miss"
    assert proxy.answers.stats()["entries"] == 0


def test_cache_off_by_env(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RCE_SENTINEL_ANSWER_CACHE", "0")
    proxy = RouterProxy(settings, ModelRegistry(settings.models, "mock"), MockProvider(),
                        Store(settings.db_path), use_jev=False)
    assert proxy.cache_key({"model": "auto", "messages": []}, {}) is None


# --- skills ------------------------------------------------------------------------------
def test_skills_are_added_to_the_forwarded_system_message(
        real_proxy: tuple[str, RouterProxy]) -> None:
    base, proxy = real_proxy
    r = chat(base, HARD)
    sent = Upstream.seen[-1]["body"]["messages"]
    assert sent[0]["role"] == "system" and sent[0]["content"].startswith(MARKER)
    assert sent[1] == {"role": "user", "content": HARD}
    assert r.headers["x-rce-skills"] == "caveman"
    assert proxy.store.execute("SELECT caveman FROM llm_calls")[0]["caveman"] == 1


def test_skills_off_forwards_messages_unchanged(settings: Settings,
                                                monkeypatch: pytest.MonkeyPatch,
                                                real_proxy: tuple[str, RouterProxy]) -> None:
    base, proxy = real_proxy
    from rce_sentinel import skills

    monkeypatch.setenv("RCE_SENTINEL_SKILLS", "")
    proxy.skills = skills.load(settings.policy)
    r = chat(base, HARD)
    assert Upstream.seen[-1]["body"]["messages"] == [{"role": "user", "content": HARD}]
    assert r.headers["x-rce-skills"] == "none"


# --- token -------------------------------------------------------------------------------
def test_token_guards_every_route_but_health(settings: Settings) -> None:
    proxy = RouterProxy(settings, ModelRegistry(settings.models, "mock"), MockProvider(),
                        Store(settings.db_path), use_jev=False, token="s3cret-token-0123")
    server = make_server(proxy, "127.0.0.1", 0)
    base = serve(server)
    try:
        assert httpx.get(base + "/health").status_code == 200
        assert httpx.get(base + "/v1/models").status_code == 401
        assert chat(base, "hi").status_code == 401
        assert chat(base, "hi", headers={"x-rce-token": "wrong"}).status_code == 401
        assert chat(base, "hi", headers={"x-rce-token": "s3cret-token-0123"}).status_code == 200
        bearer = {"Authorization": "Bearer s3cret-token-0123"}
        assert httpx.get(base + "/v1/models", headers=bearer).status_code == 200
    finally:
        server.shutdown()


def test_non_loopback_bind_needs_a_token(settings: Settings,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RCE_SENTINEL_PROXY_TOKEN", raising=False)
    proxy = RouterProxy(settings, ModelRegistry(settings.models, "mock"), MockProvider(),
                        Store(settings.db_path), use_jev=False)
    with pytest.raises(ValueError, match="RCE_SENTINEL_PROXY_TOKEN"):
        make_server(proxy, "0.0.0.0", 0)
