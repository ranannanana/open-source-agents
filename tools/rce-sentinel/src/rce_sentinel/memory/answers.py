"""Exact-match answer cache for the OpenAI-compatible proxy, shared across models.

A repeated request (same messages, tools and output limits) is answered from SQLite
without calling any model. For model "auto" the key leaves the model out, so a reply
from any model serves the same request later, whatever the router would pick then. A
request that names a model only matches replies from that model.

    key = sha256(messages, tools, tool_choice, response_format, stop, max_tokens,
                 skills version, model or "*")

Sampling settings (temperature, top_p, seed) and transport settings (stream, user,
metadata) are left out: either reply is a valid answer to the same request.

Only complete replies are stored: one choice, finish_reason "stop" or "tool_calls".
Rows are HMAC-signed like every other memory row, so a database copied in from
elsewhere, or edited by hand, is ignored. There is no near-match lookup: for code, an
answer to a similar question can be wrong.

Bypass per request with header `x-rce-cache: off` or `Cache-Control: no-cache`.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import zlib
from dataclasses import dataclass
from typing import Any

from rce_sentinel.config import env_flag
from rce_sentinel.memory.store import Store

KEY_VERSION = "a1"
KEY_FIELDS = ("messages", "tools", "tool_choice", "response_format", "stop", "max_tokens",
              "max_completion_tokens", "functions", "function_call", "modalities")
COMPLETE = {"stop", "tool_calls"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS answer_cache (
  key TEXT PRIMARY KEY, model TEXT, reply BLOB, tokens_in INTEGER, tokens_out INTEGER,
  cost REAL, created_at REAL, last_used REAL, hits INTEGER DEFAULT 0,
  saved_usd REAL DEFAULT 0, sig TEXT);
CREATE INDEX IF NOT EXISTS answer_cache_used ON answer_cache(last_used);
"""


@dataclass
class Hit:
    model: str
    message: dict[str, Any]
    finish_reason: str
    cost: float
    age_s: float


def bypassed(headers: dict[str, str]) -> bool:
    """Client opted out for this request."""
    flag = headers.get("x-rce-cache", "").strip().lower()
    cc = headers.get("cache-control", "").lower()
    return flag in ("off", "0", "no", "bypass") or "no-cache" in cc or "no-store" in cc


class AnswerCache:
    def __init__(self, store: Store, *, ttl_s: float = 86400.0, max_entries: int = 5000,
                 share_across_models: bool = True, enabled: bool = True):
        self.store = store
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self.share = share_across_models
        self.enabled = enabled
        with store._lock:
            store.conn.executescript(SCHEMA)
            store.conn.commit()

    @classmethod
    def from_policy(cls, store: Store, policy: dict[str, Any]) -> AnswerCache:
        cfg = policy.get("answer_cache") or {}
        enabled = env_flag("RCE_SENTINEL_ANSWER_CACHE", bool(cfg.get("enabled", True)))
        ttl_h = float(os.environ.get("RCE_SENTINEL_ANSWER_CACHE_TTL_H")
                      or cfg.get("ttl_hours", 24))
        return cls(store, ttl_s=ttl_h * 3600, max_entries=int(cfg.get("max_entries", 5000)),
                   share_across_models=bool(cfg.get("share_across_models", True)),
                   enabled=enabled)

    # --- key ---------------------------------------------------------------------------
    def key(self, body: dict[str, Any], model: str | None, skills_version: str) -> str | None:
        """Cache key, or None when this request must not be cached."""
        if not self.enabled or not isinstance(body.get("messages"), list):
            return None
        if int(body.get("n") or 1) != 1:
            return None
        scope = "*" if model is None and self.share else (model or "auto")
        material = {k: body[k] for k in KEY_FIELDS if k in body}
        material.update(v=KEY_VERSION, skills=skills_version, model=scope)
        raw = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    # --- read / write ------------------------------------------------------------------
    def _sig(self, key: str, model: str, blob: bytes, created: float) -> str:
        return self.store.signer.sign("answer", key, model,
                                      hashlib.sha256(blob).hexdigest(), repr(created))

    def get(self, key: str) -> Hit | None:
        rows = self.store.execute("SELECT * FROM answer_cache WHERE key=?", (key,))
        if not rows:
            return None
        r = dict(rows[0])
        now = time.time()
        if now - float(r["created_at"] or 0) > self.ttl_s:
            return None
        if not self.store.signer.verify(r["sig"], "answer", key, r["model"],
                                        hashlib.sha256(r["reply"]).hexdigest(),
                                        repr(r["created_at"])):
            self.store.rejected_rows += 1
            return None
        try:
            data = json.loads(zlib.decompress(r["reply"]))
        except (zlib.error, ValueError):
            return None
        cost = float(r["cost"] or 0.0)
        self.store.execute("UPDATE answer_cache SET hits=hits+1, last_used=?, "
                           "saved_usd=saved_usd+? WHERE key=?", (now, cost, key))
        return Hit(model=r["model"], message=data["message"],
                   finish_reason=data["finish_reason"], cost=cost,
                   age_s=now - float(r["created_at"]))

    def put(self, key: str, model: str, message: dict[str, Any], finish_reason: str,
            usage: dict[str, Any], cost: float) -> bool:
        """Store a complete reply; False when it is not complete enough to reuse."""
        if finish_reason not in COMPLETE or not isinstance(message, dict):
            return False
        if not message.get("content") and not message.get("tool_calls"):
            return False
        clean = {k: message[k] for k in ("role", "content", "tool_calls", "refusal")
                 if message.get(k) is not None}
        clean["role"] = "assistant"
        blob = zlib.compress(json.dumps({"message": clean, "finish_reason": finish_reason},
                                        ensure_ascii=False).encode(), 6)
        now = time.time()
        self.store.execute(
            "INSERT OR REPLACE INTO answer_cache(key, model, reply, tokens_in, tokens_out, "
            "cost, created_at, last_used, hits, saved_usd, sig) VALUES (?,?,?,?,?,?,?,?,0,0,?)",
            (key, model, blob, int(usage.get("prompt_tokens") or 0),
             int(usage.get("completion_tokens") or 0), cost, now, now,
             self._sig(key, model, blob, now)))
        self._evict()
        return True

    def _evict(self) -> None:
        n = self.store.execute("SELECT COUNT(*) AS n FROM answer_cache")[0]["n"]
        if n > self.max_entries:
            self.store.execute(
                "DELETE FROM answer_cache WHERE key IN (SELECT key FROM answer_cache "
                "ORDER BY last_used ASC LIMIT ?)", (n - self.max_entries,))

    # --- admin -------------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        r = dict(self.store.execute(
            "SELECT COUNT(*) AS entries, COALESCE(SUM(hits), 0) AS hits, "
            "COALESCE(SUM(saved_usd), 0) AS saved_usd, "
            "COALESCE(SUM(tokens_in * hits), 0) AS saved_in, "
            "COALESCE(SUM(tokens_out * hits), 0) AS saved_out FROM answer_cache")[0])
        r["enabled"] = self.enabled
        r["ttl_hours"] = self.ttl_s / 3600
        r["max_entries"] = self.max_entries
        return r

    def count_unsigned(self) -> int:
        """Rows whose signature fails; read-only."""
        return sum(1 for r in self.store.execute("SELECT * FROM answer_cache")
                   if not self.store.signer.verify(
                       r["sig"], "answer", r["key"], r["model"],
                       hashlib.sha256(r["reply"]).hexdigest(), repr(r["created_at"])))

    def clear(self) -> int:
        n = int(self.store.execute("SELECT COUNT(*) AS n FROM answer_cache")[0]["n"])
        self.store.execute("DELETE FROM answer_cache")
        return n


class StreamCollector:
    """Rebuild the assistant message from chat-completions stream chunks.

    Gives up (complete() returns None) on more than one choice, so a cached reply always
    matches what a single-choice client saw.
    """

    def __init__(self) -> None:
        self.content: list[str] = []
        self.tools: dict[int, dict[str, Any]] = {}
        self.finish = ""
        self.ok = True

    def feed(self, chunk: dict[str, Any]) -> None:
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict) or int(choice.get("index") or 0) != 0:
                self.ok = False
                continue
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                self.content.append(delta["content"])
            for tc in delta.get("tool_calls") or []:
                slot = self.tools.setdefault(int(tc.get("index") or 0), {
                    "id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                if tc.get("type"):
                    slot["type"] = tc["type"]
                fn = tc.get("function") or {}
                slot["function"]["name"] += fn.get("name") or ""
                slot["function"]["arguments"] += fn.get("arguments") or ""
            if choice.get("finish_reason"):
                self.finish = str(choice["finish_reason"])

    def complete(self) -> tuple[dict[str, Any], str] | None:
        if not self.ok or not self.finish:
            return None
        msg: dict[str, Any] = {"role": "assistant",
                               "content": "".join(self.content) or None}
        if self.tools:
            msg["tool_calls"] = [self.tools[i] for i in sorted(self.tools)]
        return msg, self.finish


def reply_body(hit: Hit) -> dict[str, Any]:
    """A chat.completion object for a cache hit; usage is zero, nothing was spent."""
    return {"id": f"rce-cache-{int(time.time() * 1000)}", "object": "chat.completion",
            "created": int(time.time()), "model": hit.model,
            "choices": [{"index": 0, "message": hit.message,
                         "finish_reason": hit.finish_reason}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


def stream_chunks(hit: Hit) -> list[dict[str, Any]]:
    """The same reply as stream chunks: one delta, then the finish and usage chunk."""
    base = {"id": f"rce-cache-{int(time.time() * 1000)}", "object": "chat.completion.chunk",
            "created": int(time.time()), "model": hit.model}
    delta: dict[str, Any] = {"role": "assistant"}
    if hit.message.get("content") is not None:
        delta["content"] = hit.message["content"]
    if hit.message.get("tool_calls"):
        delta["tool_calls"] = [{**tc, "index": i}
                               for i, tc in enumerate(hit.message["tool_calls"])]
    return [{**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            {**base, "choices": [{"index": 0, "delta": {},
                                  "finish_reason": hit.finish_reason}],
             "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}]
