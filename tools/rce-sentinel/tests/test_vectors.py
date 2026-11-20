"""Vector index: hashing-based fake embedder, so tests stay offline and deterministic."""

from __future__ import annotations

import hashlib

from rce_sentinel.config import Settings
from rce_sentinel.memory.store import Store
from rce_sentinel.memory.vectors import VectorIndex


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Bag of words hashed into 64 buckets: shared words give similar vectors."""
    out = []
    for t in texts:
        v = [0.0] * 64
        for w in t.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
        out.append(v)
    return out


def test_search_finds_the_closest_text_and_skips_unchanged(settings: Settings) -> None:
    calls: list[int] = []

    def counting(texts: list[str]) -> list[list[float]]:
        calls.append(len(texts))
        return fake_embed(texts)

    idx = VectorIndex(Store(settings.db_path), counting, "fake")
    rows = [("prompt", "a", "fix the race condition in the worker pool"),
            ("prompt", "b", "write a commit message for the rename"),
            ("output", "b|m", "refactor: rename getUser to fetchUser")]
    assert idx.add_many(rows) == 3
    assert idx.add_many(rows) == 0 and idx.count() == 3     # unchanged text is skipped
    hits = idx.search("race condition in worker threads", kind="prompt", k=2)
    assert hits[0].ref == "a" and hits[0].score > hits[1].score
    assert idx.add_many([("prompt", "a", "new text")]) == 1  # changed text re-embeds


def test_models_never_mix(settings: Settings) -> None:
    store = Store(settings.db_path)
    VectorIndex(store, fake_embed, "m1").add_many([("prompt", "a", "hello world")])
    assert VectorIndex(store, fake_embed, "m2").search("hello world") == []
