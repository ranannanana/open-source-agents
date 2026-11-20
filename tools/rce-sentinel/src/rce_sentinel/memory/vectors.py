"""Small vector index in the same SQLite file: find past prompts and replies by meaning.

Embeddings come from a local embedding model (Ollama by default, free and on-device) and
are stored as float32 blobs, unit-normalised, so cosine similarity is a dot product. A
brute-force scan is exact and fast enough for tens of thousands of rows; past that, swap
the scan for an ANN index without changing callers.

Rows are keyed by (kind, ref, model) and carry the text's hash, so re-indexing skips text
that has not changed and a different embedding model never mixes with another's vectors.
"""

from __future__ import annotations

import hashlib
import math
import time
from array import array
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import httpx

from rce_sentinel.memory.store import Store

Embedder = Callable[[list[str]], list[list[float]]]
BATCH = 32
# Task prefixes some embedding models are trained with: (document, query).
PREFIXES = {"nomic-embed-text": ("search_document: ", "search_query: ")}


def ollama_embedder(base_url: str, model: str, timeout: float = 120.0) -> Embedder:
    """OpenAI-compatible /embeddings on a local server (Ollama, apfel-style proxies)."""
    url = base_url.rstrip("/") + "/embeddings"

    def embed(texts: list[str]) -> list[list[float]]:
        resp = httpx.post(url, json={"model": model, "input": texts}, timeout=timeout)
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]

    return embed


def _unit(vec: list[float]) -> array:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return array("f", (x / norm for x in vec))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass
class Hit:
    kind: str
    ref: str
    score: float


class VectorIndex:
    def __init__(self, store: Store, embed: Embedder, model: str):
        self.store = store
        self.embed = embed
        self.model = model
        self.doc_prefix, self.query_prefix = PREFIXES.get(model.split(":")[0], ("", ""))

    def add_many(self, rows: Iterable[tuple[str, str, str]]) -> int:
        """Index (kind, ref, text) rows; returns how many were (re)embedded."""
        have = {(r["kind"], r["ref"]): r["sha"] for r in self.store.execute(
            "SELECT kind, ref, sha FROM vectors WHERE model=?", (self.model,))}
        todo = [(k, ref, t) for k, ref, t in rows
                if t and have.get((k, ref)) != _sha(self.doc_prefix + t)]
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            vecs = self.embed([self.doc_prefix + t for _, _, t in chunk])
            for (kind, ref, text), vec in zip(chunk, vecs, strict=True):
                unit = _unit(vec)
                self.store.execute(
                    "INSERT OR REPLACE INTO vectors(kind, ref, model, sha, dim, vec, ts) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (kind, ref, self.model, _sha(self.doc_prefix + text), len(unit),
                     unit.tobytes(), time.time()))
        return len(todo)

    def search(self, text: str, *, kind: str | None = None, k: int = 5) -> list[Hit]:
        q = _unit(self.embed([self.query_prefix + text])[0])
        sql = "SELECT kind, ref, dim, vec FROM vectors WHERE model=?"
        args: tuple[str, ...] = (self.model,)
        if kind:
            sql += " AND kind=?"
            args += (kind,)
        hits = []
        for r in self.store.execute(sql, args):
            if r["dim"] != len(q):
                continue
            v = array("f")
            v.frombytes(r["vec"])
            hits.append(Hit(r["kind"], r["ref"], sum(a * b for a, b in zip(q, v, strict=True))))
        return sorted(hits, key=lambda h: -h.score)[:k]

    def count(self) -> int:
        return int(self.store.execute("SELECT COUNT(*) AS n FROM vectors WHERE model=?",
                                      (self.model,))[0]["n"])
