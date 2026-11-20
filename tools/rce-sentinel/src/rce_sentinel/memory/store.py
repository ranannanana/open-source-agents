"""Model-agnostic SQLite store: result cache, facts, call log, feeds, intel.

Nothing here is keyed by model name, so swapping a tier's model keeps every cached verdict.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import zlib
from pathlib import Path
from typing import Any

from rce_sentinel.memory.screen import WITHHELD, looks_like_injection
from rce_sentinel.memory.trust import UNVERIFIED, VERIFIED, Signer

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

SCHEMA = """
CREATE TABLE IF NOT EXISTS result_cache (
  key TEXT PRIMARY KEY, context_hash TEXT, verdict_json TEXT, model_used TEXT, tier TEXT,
  rules_version TEXT, cve_db_version TEXT, prompt_version TEXT, created_at REAL,
  trust INTEGER DEFAULT 0, sig TEXT);
CREATE TABLE IF NOT EXISTS facts (
  id INTEGER PRIMARY KEY AUTOINCREMENT, context_hash TEXT, file TEXT, sink TEXT,
  rule_id TEXT, finding_json TEXT, verdict TEXT, reason TEXT, evidence_json TEXT,
  model_used TEXT, rules_version TEXT, cve_db_version TEXT, commit_sha TEXT,
  invalidated_by TEXT, created_at REAL, repo_id TEXT, trust INTEGER DEFAULT 0, sig TEXT);
CREATE INDEX IF NOT EXISTS facts_sink ON facts(file, sink);
CREATE TABLE IF NOT EXISTS a2a_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, seq INTEGER, sender TEXT,
  recipient TEXT, kind TEXT, trust INTEGER, payload_hash TEXT, prev TEXT, sig TEXT,
  ts REAL);
CREATE TABLE IF NOT EXISTS dep_verdicts (
  ecosystem TEXT, name TEXT, version TEXT, status TEXT, advisories TEXT,
  cve_db_version TEXT, updated_at REAL, PRIMARY KEY (ecosystem, name, version));
CREATE TABLE IF NOT EXISTS llm_calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, task_type TEXT, tier TEXT,
  model TEXT, family TEXT, outcome TEXT, confidence REAL, tokens_in INTEGER,
  tokens_out INTEGER, cached_tokens INTEGER, cost REAL, baseline_cost REAL,
  latency_ms REAL, caveman INTEGER, shadow INTEGER DEFAULT 0, final_agree INTEGER,
  simulated INTEGER DEFAULT 0, ts REAL);
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, mode TEXT, started REAL, duration REAL, degraded TEXT,
  findings INTEGER, blocked INTEGER, units INTEGER, cache_hits INTEGER,
  prerouted INTEGER, baseline_units INTEGER, baseline_tokens INTEGER, baseline_cost REAL,
  judged INTEGER);
CREATE TABLE IF NOT EXISTS osv_queries (
  key TEXT PRIMARY KEY, vuln_ids TEXT, fetched_at REAL);
CREATE TABLE IF NOT EXISTS vulns (id TEXT PRIMARY KEY, modified TEXT, pkgs TEXT, json TEXT);
CREATE TABLE IF NOT EXISTS kev (cve TEXT PRIMARY KEY, json TEXT, catalog_version TEXT,
  simulated INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS epss (cve TEXT PRIMARY KEY, score REAL, fetched_at REAL);
CREATE TABLE IF NOT EXISTS intel_items (
  content_hash TEXT PRIMARY KEY, source TEXT, url TEXT, cve_ids TEXT, json TEXT,
  status TEXT, first_seen REAL);
CREATE TABLE IF NOT EXISTS prompt_prefix (model TEXT, prefix_hash TEXT, last_used REAL,
  PRIMARY KEY (model, prefix_hash));
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS prompt_tags (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, prompt_hash TEXT,
  prompt_tokens INTEGER, source TEXT, task_type TEXT, level REAL, tags TEXT, model TEXT,
  tier TEXT, est_tokens_in INTEGER, est_tokens_out INTEGER, est_cost REAL, outcome TEXT,
  ts REAL);
CREATE TABLE IF NOT EXISTS handoffs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, mode TEXT, turn INTEGER, model TEXT,
  switched_from TEXT, reason TEXT, context_tokens INTEGER, tokens_in INTEGER,
  tokens_out INTEGER, cost REAL, ok INTEGER, files TEXT, rewrites TEXT, ts REAL,
  event_key TEXT);
CREATE TABLE IF NOT EXISTS outcomes (
  id INTEGER PRIMARY KEY AUTOINCREMENT, tag_id INTEGER, session_id TEXT, model TEXT,
  task_type TEXT, difficulty TEXT, success INTEGER, outcome TEXT, review_s REAL,
  tokens_out INTEGER, latency_ms REAL, source TEXT, ts REAL);
CREATE INDEX IF NOT EXISTS outcomes_model ON outcomes(model, task_type);
CREATE TABLE IF NOT EXISTS bench_results (
  suite TEXT, item TEXT, model TEXT, split TEXT, task_type TEXT, difficulty TEXT,
  success INTEGER, checks TEXT, judge TEXT, tokens_in INTEGER, tokens_out INTEGER,
  cost REAL, latency_ms REAL, error TEXT, simulated INTEGER DEFAULT 0, ts REAL,
  PRIMARY KEY (suite, item, model));
CREATE TABLE IF NOT EXISTS usage_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT, session TEXT, path TEXT, kind TEXT,
  requested TEXT, model TEXT, policy_choice TEXT, rule TEXT, rewritten INTEGER,
  fallback INTEGER, log_only INTEGER, status INTEGER, task TEXT, tag_id INTEGER,
  tokens_in INTEGER, input_tokens INTEGER, cache_write_5m INTEGER, cache_write_1h INTEGER,
  cache_read INTEGER, tokens_out INTEGER, cost REAL, cost_requested REAL, latency_ms REAL,
  ttft_ms REAL, n_messages INTEGER, n_tools INTEGER, stream INTEGER, switched_from TEXT,
  ts REAL, skills TEXT);
CREATE INDEX IF NOT EXISTS usage_events_ts ON usage_events(ts);
CREATE TABLE IF NOT EXISTS bench_outputs (
  suite TEXT, item TEXT, model TEXT, reply BLOB, judge_raw TEXT, sha TEXT, chars INTEGER,
  ts REAL, PRIMARY KEY (suite, item, model));
CREATE TABLE IF NOT EXISTS vectors (
  kind TEXT, ref TEXT, model TEXT, sha TEXT, dim INTEGER, vec BLOB, ts REAL,
  PRIMARY KEY (kind, ref, model));
"""

# Outcome labels: what the developer did with the answer.
OUTCOMES = {"accepted": 1, "edited": 1, "rejected": 0, "failed": 0}


class Store:
    def __init__(self, path: Path | str, signer: Signer | None = None):
        self.path = str(path)
        self.signer = signer or Signer.load()
        self.rejected_rows = 0  # rows ignored for a bad or missing signature
        self._lock = threading.Lock()
        if self.path != ":memory:" and not Path(self.path).exists():
            # Owner-only from the first byte; SQLite's journals copy the mode.
            os.close(os.open(self.path, os.O_WRONLY | os.O_CREAT, 0o600))
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._ensure_columns()
        self.conn.commit()

    def _ensure_columns(self) -> None:
        """Add trust/signature columns to a database created before they existed."""
        wanted = {"result_cache": {"trust": "INTEGER DEFAULT 0", "sig": "TEXT"},
                  "facts": {"repo_id": "TEXT", "trust": "INTEGER DEFAULT 0", "sig": "TEXT"},
                  "usage_events": {"skills": "TEXT"},
                  "handoffs": {"event_key": "TEXT"}}
        for table, cols in wanted.items():
            have = {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            for col, decl in cols.items():
                if col not in have:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        # Idempotent handoff log: a replayed event lands once (NULL keys never clash).
        self.conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS handoffs_event "
                          "ON handoffs(event_key)")

    def close(self) -> None:
        self.conn.close()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            cur = self.conn.execute(sql, params)
            rows = cur.fetchall()
            self.conn.commit()
            return rows

    # --- meta ----------------------------------------------------------------
    def get_meta(self, k: str, default: str = "") -> str:
        rows = self.execute("SELECT v FROM meta WHERE k=?", (k,))
        return rows[0]["v"] if rows else default

    def set_meta(self, k: str, v: str) -> None:
        self.execute("INSERT OR REPLACE INTO meta(k, v) VALUES (?, ?)", (k, v))

    # --- result cache --------------------------------------------------------
    def _cache_sig(self, key: str, verdict_json: str, trust: int, v: dict[str, str]) -> str:
        return self.signer.sign("cache", key, verdict_json, trust, v["rules"], v["cve_db"],
                                v["prompt"])

    def cache_get(self, key: str) -> dict[str, Any] | None:
        """A verdict comes back only if its signature checks out, and a "safe" verdict
        only if it was cross-family verified."""
        rows = self.execute("SELECT * FROM result_cache WHERE key=?", (key,))
        if not rows:
            return None
        row = dict(rows[0])
        versions = {"rules": row["rules_version"], "cve_db": row["cve_db_version"],
                    "prompt": row["prompt_version"]}
        trust = row.get("trust") or 0
        if not self.signer.verify(row.get("sig"), "cache", key, row["verdict_json"], trust,
                                  versions["rules"], versions["cve_db"], versions["prompt"]):
            self.rejected_rows += 1
            return None
        if '"verdict": "safe"' in row["verdict_json"] and trust < VERIFIED:
            self.rejected_rows += 1
            return None
        return row

    def cache_put(self, key: str, context_hash: str, verdict: dict[str, Any], model: str,
                  tier: str, versions: dict[str, str], trust: int = 0) -> None:
        blob = json.dumps(verdict)
        self.execute(
            "INSERT OR REPLACE INTO result_cache VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (key, context_hash, blob, model, tier, versions["rules"], versions["cve_db"],
             versions["prompt"], time.time(), trust,
             self._cache_sig(key, blob, trust, versions)),
        )

    def cache_clear(self) -> int:
        n = self.execute("SELECT COUNT(*) AS n FROM result_cache")[0]["n"]
        self.execute("DELETE FROM result_cache")
        return int(n)

    # --- facts (cross-commit memory) ----------------------------------------
    def _fact_fields(self, r: dict[str, Any]) -> tuple[object, ...]:
        return ("fact", r.get("repo_id"), r.get("file"), r.get("sink"), r.get("rule_id"),
                r.get("verdict"), r.get("reason"), r.get("trust"), r.get("model_used"),
                r.get("commit_sha"))

    def add_fact(self, row: dict[str, Any], *, repo_id: str, trust: int) -> None:
        """Store a reviewed fact; instruction-like reasons are withheld and demoted."""
        if looks_like_injection(str(row.get("reason") or "")):
            row = {**row, "reason": WITHHELD}
            trust = UNVERIFIED
        row = {**row, "repo_id": repo_id, "trust": trust}
        row["sig"] = self.signer.sign(*self._fact_fields(row))
        cols = ",".join(row)
        marks = ",".join("?" for _ in row)
        self.execute(f"INSERT INTO facts({cols}) VALUES ({marks})", tuple(row.values()))

    def prior_facts(self, file: str, sink: str, *, repo_id: str, limit: int = 3,
                    min_trust: int = VERIFIED) -> list[dict[str, Any]]:
        """Earlier reviews of this sink: same repo, validly signed, cross-family verified.
        `reason` is model text, so it is stripped and shortened before anyone sees it."""
        rows = self.execute(
            "SELECT * FROM facts WHERE file=? AND sink=? AND repo_id=? AND invalidated_by "
            "IS NULL ORDER BY id DESC LIMIT 20", (file, sink, repo_id))
        out = []
        for r in rows:
            d = dict(r)
            if not self.signer.verify(d.get("sig"), *self._fact_fields(d)):
                self.rejected_rows += 1
                continue
            if (d.get("trust") or 0) < min_trust:
                continue
            d["reason"] = " ".join(_CTRL.sub("", d.get("reason") or "").split())[:160]
            out.append({k: d[k] for k in ("verdict", "reason", "model_used", "commit_sha",
                                          "created_at")})
            if len(out) >= limit:
                break
        return out

    # --- call log ------------------------------------------------------------
    def log_call(self, row: dict[str, Any]) -> int:
        row = {**row, "ts": time.time()}
        cols = ",".join(row)
        marks = ",".join("?" for _ in row)
        with self._lock:
            cur = self.conn.execute(f"INSERT INTO llm_calls({cols}) VALUES ({marks})",
                                    tuple(row.values()))
            self.conn.commit()
            return int(cur.lastrowid or 0)

    def _insert(self, table: str, row: dict[str, Any]) -> int:
        row = {**row, "ts": time.time()}
        cols = ",".join(row)
        marks = ",".join("?" for _ in row)
        with self._lock:
            cur = self.conn.execute(f"INSERT INTO {table}({cols}) VALUES ({marks})",
                                    tuple(row.values()))
            self.conn.commit()
            return int(cur.lastrowid or 0)

    def log_tags(self, row: dict[str, Any]) -> int:
        return self._insert("prompt_tags", row)

    def save_output(self, suite: str, item: str, model: str, reply: str,
                    judge_raw: str = "") -> None:
        """Keep the full reply, zlib-compressed."""
        raw = reply.encode()
        self.execute(
            "INSERT OR REPLACE INTO bench_outputs(suite, item, model, reply, judge_raw, sha, "
            "chars, ts) VALUES (?,?,?,?,?,?,?,?)",
            (suite, item, model, zlib.compress(raw, 9), judge_raw or None,
             hashlib.sha256(raw).hexdigest()[:16], len(reply), time.time()))

    def output(self, suite: str, item: str, model: str) -> str | None:
        rows = self.execute("SELECT reply FROM bench_outputs WHERE suite=? AND item=? AND "
                            "model=?", (suite, item, model))
        return zlib.decompress(rows[0]["reply"]).decode() if rows else None

    def log_usage(self, row: dict[str, Any]) -> int:
        return self._insert("usage_events", row)

    def log_handoff(self, row: dict[str, Any]) -> int:
        """Insert a handoff; a row whose event_key is already logged is skipped (0)."""
        row = {**row, "ts": time.time()}
        cols = ",".join(row)
        marks = ",".join("?" for _ in row)
        with self._lock:
            cur = self.conn.execute(f"INSERT OR IGNORE INTO handoffs({cols}) VALUES ({marks})",
                                    tuple(row.values()))
            self.conn.commit()
            return int(cur.lastrowid or 0) if cur.rowcount else 0

    def record_outcome(self, *, model: str, task_type: str, difficulty: list[float],
                       outcome: str, source: str, tag_id: int | None = None,
                       session_id: str = "", review_s: float | None = None,
                       tokens_out: int = 0, latency_ms: float = 0.0) -> int:
        """One graded answer: the evidence calibration learns from."""
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {sorted(OUTCOMES)}")
        if tag_id is not None:
            self.execute("UPDATE prompt_tags SET outcome=? WHERE id=?", (outcome, tag_id))
        return self._insert("outcomes", {
            "tag_id": tag_id, "session_id": session_id, "model": model,
            "task_type": task_type, "difficulty": json.dumps(difficulty),
            "success": OUTCOMES[outcome], "outcome": outcome, "review_s": review_s,
            "tokens_out": tokens_out, "latency_ms": latency_ms, "source": source})

    def outcome_for_tag(self, tag_id: int, outcome: str, review_s: float | None = None
                        ) -> int:
        """Grade a routed prompt by the id `route` or the proxy printed."""
        rows = self.execute("SELECT * FROM prompt_tags WHERE id=?", (tag_id,))
        if not rows:
            raise KeyError(f"no routed prompt with id {tag_id}")
        r = rows[0]
        diff = json.loads(r["tags"] or "{}").get("difficulty") or [0.0, 1.0, 0.0, 0.0]
        if isinstance(diff, dict):  # stored keyed by level name, in level order
            diff = list(diff.values())
        return self.record_outcome(model=r["model"] or "", task_type=r["task_type"] or "",
                                   difficulty=diff, outcome=outcome, source="user",
                                   tag_id=tag_id, session_id=r["session_id"] or "",
                                   review_s=review_s)

    def mark_agreement(self, row_ids: list[int], agree: bool) -> None:
        for rid in row_ids:
            self.execute("UPDATE llm_calls SET final_agree=? WHERE id=?", (int(agree), rid))
