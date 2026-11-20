"""Answer cache: keys, signatures, expiry, eviction, and stream reassembly."""

from __future__ import annotations

import time

from rce_sentinel.config import Settings
from rce_sentinel.memory.answers import (
    AnswerCache,
    StreamCollector,
    bypassed,
    reply_body,
    stream_chunks,
)
from rce_sentinel.memory.store import Store

MSG = {"role": "assistant", "content": "use pathlib"}
BODY = {"model": "auto", "messages": [{"role": "user", "content": "how to join paths"}]}


def cache(settings: Settings, **kw: object) -> AnswerCache:
    return AnswerCache(Store(settings.db_path), **kw)  # type: ignore[arg-type]


def test_key_ignores_sampling_and_transport_but_not_content(settings: Settings) -> None:
    c = cache(settings)
    k = c.key(BODY, None, "v1")
    assert k == c.key({**BODY, "temperature": 0.9, "stream": True, "user": "u"}, None, "v1")
    assert k != c.key({**BODY, "max_tokens": 50}, None, "v1")
    assert k != c.key(BODY, None, "v2")                       # skills edited
    assert k != c.key(BODY, "gpt-x", "v1")                    # named model: own scope
    assert c.key({**BODY, "n": 2}, None, "v1") is None
    assert c.key({"prompt": "x"}, None, "v1") is None
    assert cache(settings, enabled=False).key(BODY, None, "v1") is None
    assert cache(settings, share_across_models=False).key(BODY, None, "v1") != k


def test_round_trip_counts_hits_and_savings(settings: Settings) -> None:
    c = cache(settings)
    k = c.key(BODY, None, "")
    assert k and c.get(k) is None
    assert c.put(k, "model-a", MSG, "stop", {"prompt_tokens": 10, "completion_tokens": 5},
                 0.002)
    hit = c.get(k)
    assert hit and hit.model == "model-a" and hit.message == MSG and hit.cost == 0.002
    c.get(k)
    st = c.stats()
    assert (st["entries"], st["hits"], st["saved_out"]) == (1, 2, 10)
    assert abs(st["saved_usd"] - 0.004) < 1e-9


def test_incomplete_or_empty_replies_are_not_stored(settings: Settings) -> None:
    c = cache(settings)
    k = c.key(BODY, None, "")
    assert k
    assert not c.put(k, "m", MSG, "length", {}, 0.0)
    assert not c.put(k, "m", {"role": "assistant", "content": ""}, "stop", {}, 0.0)
    assert c.get(k) is None


def test_tampered_or_expired_rows_are_ignored(settings: Settings) -> None:
    c = cache(settings)
    k = c.key(BODY, None, "")
    assert k
    c.put(k, "m", MSG, "stop", {}, 0.0)
    c.store.execute("UPDATE answer_cache SET model='evil'")
    assert c.get(k) is None and c.store.rejected_rows == 1 and c.count_unsigned() == 1
    c.put(k, "m", MSG, "stop", {}, 0.0)
    c.ttl_s = 0.0
    time.sleep(0.01)
    assert c.get(k) is None


def test_least_recently_used_entries_are_evicted(settings: Settings) -> None:
    c = cache(settings, max_entries=2)
    keys = []
    for i in range(3):
        k = c.key({**BODY, "messages": [{"role": "user", "content": str(i)}]}, None, "")
        assert k
        keys.append(k)
        c.put(k, "m", MSG, "stop", {}, 0.0)
        time.sleep(0.01)
    assert c.stats()["entries"] == 2 and c.get(keys[0]) is None and c.get(keys[2])
    assert c.clear() == 2


def test_bypass_headers() -> None:
    assert bypassed({"x-rce-cache": "off"}) and bypassed({"cache-control": "no-cache"})
    assert not bypassed({}) and not bypassed({"cache-control": "max-age=0"})


def test_stream_collector_rebuilds_text_and_tool_calls() -> None:
    col = StreamCollector()
    col.feed({"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}]})
    col.feed({"choices": [{"index": 0, "delta": {"content": "lo", "tool_calls": [
        {"index": 0, "id": "c1", "type": "function",
         "function": {"name": "read", "arguments": '{"pa'}}]}}]})
    col.feed({"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "function": {"arguments": 'th": "a"}'}}]}, "finish_reason": "tool_calls"}]})
    col.feed({"choices": [], "usage": {"completion_tokens": 3}})
    got = col.complete()
    assert got is not None
    msg, finish = got
    assert finish == "tool_calls" and msg["content"] == "Hello"
    assert msg["tool_calls"] == [{"id": "c1", "type": "function",
                                  "function": {"name": "read", "arguments": '{"path": "a"}'}}]


def test_stream_collector_refuses_partial_or_multi_choice_streams() -> None:
    col = StreamCollector()
    col.feed({"choices": [{"index": 0, "delta": {"content": "x"}}]})
    assert col.complete() is None                               # never finished
    col.feed({"choices": [{"index": 1, "delta": {"content": "y"}, "finish_reason": "stop"}]})
    assert col.complete() is None


def test_hit_renders_as_completion_and_as_chunks(settings: Settings) -> None:
    c = cache(settings)
    k = c.key(BODY, None, "")
    assert k
    tool_msg = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}
    c.put(k, "m", tool_msg, "tool_calls", {}, 0.0)
    hit = c.get(k)
    assert hit
    body = reply_body(hit)
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert body["usage"]["total_tokens"] == 0
    first, last = stream_chunks(hit)
    assert first["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
    assert "content" not in first["choices"][0]["delta"]
    assert last["choices"][0]["finish_reason"] == "tool_calls"
