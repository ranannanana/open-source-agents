"""Fit each model's quality from graded outcomes, starting from the hand-set priors.

P(success | q, difficulty) is the logistic step select.py routes with, so one number per
model and task group carries across difficulty levels. The fit is a MAP estimate: the
models.yaml prior acts like a few pseudo-observations, so early evidence nudges it and
plenty of evidence overrides it. A task group's quality is fitted around the model-wide
value, so a group with little data borrows from the others.

Output speed (tokens per second) comes from real dev calls in llm_calls.
"""

from __future__ import annotations

import json
import math
import statistics
import weakref
from dataclasses import dataclass, field

from rce_sentinel.memory.store import Store
from rce_sentinel.orchestrator.llm import TIERS, ModelRegistry
from rce_sentinel.preeval.select import p_success, quality_of

# Task types pooled into groups that share a quality estimate.
GROUPS = {"build": "build", "refactor": "build", "edit": "change", "debug": "change",
          "test": "change", "explain": "think", "design": "think", "chore": "chore"}
PRIOR_SD = 0.12   # how far evidence must push before it beats models.yaml
GROUP_SD = 0.08   # how far a task group may sit from its model-wide quality
GRID = tuple(i / 400 for i in range(401))
MIN_TPS_TOKENS = 200   # shorter replies are dominated by first-token time
MIN_TPS_SAMPLES = 3

Row = tuple[list[float], int]   # (difficulty distribution, success)


def group_of(task: str) -> str:
    return GROUPS.get(task.removeprefix("dev:"), "change")


def _p(q: float, diff: list[float]) -> float:
    return sum(p * p_success(q, i) for i, p in enumerate(diff))


def fit_quality(rows: list[Row], prior: float, sd: float) -> float:
    """MAP quality on a fine grid: log-likelihood plus a Gaussian prior."""
    if not rows:
        return prior

    def score(q: float) -> float:
        ll = sum(math.log(min(max(_p(q, d) if y else 1 - _p(q, d), 1e-9), 1.0))
                 for d, y in rows)
        return ll - (q - prior) ** 2 / (2 * sd * sd)

    return max(GRID, key=score)


@dataclass
class Calibration:
    quality: dict[str, float] = field(default_factory=dict)              # model-wide
    group_quality: dict[str, dict[str, float]] = field(default_factory=dict)
    n: dict[str, dict[str, int]] = field(default_factory=dict)           # "*" = all groups
    tps: dict[str, float] = field(default_factory=dict)
    rows: int = 0

    def quality_for(self, model: str, task: str, prior: float) -> float:
        g = self.group_quality.get(model, {})
        return g.get(group_of(task), self.quality.get(model, prior))

    def samples(self, model: str, task: str) -> int:
        return self.n.get(model, {}).get(group_of(task), 0)

    def to_dict(self) -> dict[str, object]:
        return {"rows": self.rows, "quality": self.quality,
                "group_quality": self.group_quality, "n": self.n, "tps": self.tps}


def _difficulty(raw: str | None) -> list[float]:
    d = json.loads(raw or "[]")
    if isinstance(d, dict):
        d = list(d.values())
    return [float(x) for x in d] if len(d) == 4 else [0.0, 1.0, 0.0, 0.0]


Evidence = tuple[str, str, list[float], int]   # (model, task type, difficulty, success)


def fit(evidence: list[Evidence], registry: ModelRegistry,
        speeds: dict[str, list[float]] | None = None) -> Calibration:
    """Fit from graded answers; `speeds` are observed output tokens per second."""
    specs = {m.id: m for t in TIERS for m in registry.all[t]}
    by_model: dict[str, dict[str, list[Row]]] = {}
    for model, task, diff, ok in evidence:
        if model:
            by_model.setdefault(model, {}).setdefault(group_of(task), []).append((diff, ok))
    cal = Calibration(rows=len(evidence))
    for model, groups in by_model.items():
        spec = specs.get(model)
        prior = quality_of(spec) if spec else quality_of("tier2")
        everything = [row for g in groups.values() for row in g]
        q = fit_quality(everything, prior, PRIOR_SD)
        cal.quality[model] = q
        cal.group_quality[model] = {g: fit_quality(rs, q, GROUP_SD) for g, rs in groups.items()}
        cal.n[model] = {g: len(rs) for g, rs in groups.items()} | {"*": len(everything)}
    cal.tps = {m: statistics.median(v) for m, v in (speeds or {}).items()
               if len(v) >= MIN_TPS_SAMPLES}
    return cal


def calibrate(store: Store, registry: ModelRegistry) -> Calibration:
    evidence = [(r["model"], r["task_type"] or "", _difficulty(r["difficulty"]),
                 int(r["success"]))
                for r in store.execute("SELECT model, task_type, difficulty, success "
                                       "FROM outcomes")]
    speeds: dict[str, list[float]] = {}
    for r in store.execute(
            "SELECT model, tokens_out, latency_ms FROM llm_calls WHERE simulated=0 AND "
            "task_type LIKE 'dev:%' AND tokens_out >= ? AND latency_ms > 0",
            (MIN_TPS_TOKENS,)):
        speeds.setdefault(r["model"], []).append(r["tokens_out"] / (r["latency_ms"] / 1000))
    return fit(evidence, registry, speeds)


# Per store object, so two databases never share a fit.
_CACHE: weakref.WeakKeyDictionary[Store, tuple[tuple[int, int, int], Calibration]] = (
    weakref.WeakKeyDictionary())


def load_calibration(store: Store, registry: ModelRegistry) -> Calibration:
    """Calibration for this database, refitted only when new evidence arrived."""
    stamp = store.execute(
        "SELECT (SELECT COALESCE(MAX(id), 0) FROM outcomes) AS o, "
        "(SELECT COALESCE(MAX(id), 0) FROM llm_calls) AS c")[0]
    key = (int(stamp["o"]), int(stamp["c"]), id(registry))
    hit = _CACHE.get(store)
    if hit and hit[0] == key:
        return hit[1]
    cal = calibrate(store, registry)
    _CACHE[store] = (key, cal)
    return cal
