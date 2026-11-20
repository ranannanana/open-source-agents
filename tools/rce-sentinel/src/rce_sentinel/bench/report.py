"""Score routing strategies against the best possible choice, on held-out prompts.

For every graded reply the realised cost is what that choice really cost:

    price paid + value_per_second * seconds waited
    + (failed ? review(tokens_out) + rework_factor * review(task_tokens) : 0)

task_tokens is the median size of the passing replies to that prompt: a short wrong
answer is quick to read but still leaves the whole task to do by hand.

The oracle takes the cheapest realised cost per prompt, knowing every model's result. A
strategy is "right" on a prompt when its realised cost is within `tolerance` (relative) or
`tolerance_usd` (absolute) of the oracle's, so near-ties and a few cents do not count as
misses. Strategies are ranked by mean regret in dollars, the quantity routing minimises;
accuracy alone would favour a cheap model that is usually fine and occasionally costly.
Strategies:

    router (calibrated)  quality fitted on the train split, scored on the test split
    router (priors)      the same router with only the models.yaml priors
    always <model>       one model for everything (openrouter/auto is just another model)

Accuracy comes with a 95% Wilson interval so a small test split is not over-read.
"""

from __future__ import annotations

import fnmatch
import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from rce_sentinel.bench.suite import Suite
from rce_sentinel.config import Settings
from rce_sentinel.memory.store import Store
from rce_sentinel.orchestrator.llm import ModelRegistry
from rce_sentinel.orchestrator.scoring import Profile, profile_for
from rce_sentinel.preeval.calibrate import MIN_TPS_TOKENS, Calibration, fit
from rce_sentinel.preeval.select import REWORK_FACTOR, review_usd, select_model
from rce_sentinel.preeval.tags import rule_tags

TOLERANCE = 0.10
TOLERANCE_USD = 0.05   # a few cents never outweighs a minute of review


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class StrategyScore:
    name: str
    n: int = 0
    right: int = 0
    passed: int = 0
    realised: float = 0.0
    regret: float = 0.0
    spend: float = 0.0
    picks: dict[str, int] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.right / self.n if self.n else 0.0

    def to_dict(self) -> dict[str, Any]:
        lo, hi = wilson(self.right, self.n)
        return {"strategy": self.name, "prompts": self.n, "accuracy": round(self.accuracy, 3),
                "ci95": [round(lo, 3), round(hi, 3)],
                "success_rate": round(self.passed / self.n, 3) if self.n else 0.0,
                "mean_regret_usd": round(self.regret / self.n, 4) if self.n else 0.0,
                "realised_usd": round(self.realised, 4), "spend_usd": round(self.spend, 4),
                "picks": self.picks}


def task_tokens(by_model: dict[str, dict[str, Any]]) -> int:
    """Median passing reply size; the longest reply when none passed."""
    ok = [r["tokens_out"] or 0 for r in by_model.values() if r["success"]]
    return int(statistics.median(ok)) if ok else max(
        (r["tokens_out"] or 0 for r in by_model.values()), default=0)


def realised(row: dict[str, Any], profile: Profile, policy: dict[str, Any],
             task: int | None = None) -> float:
    rework = float(policy.get("preeval", {}).get("rework_factor", REWORK_FACTOR))
    wait = profile.value_per_second * (row["latency_ms"] or 0) / 1000
    own = row["tokens_out"] or 0
    fail = 0.0 if row["success"] else (review_usd(own, policy)
                                       + rework * review_usd(own if task is None else task,
                                                             policy))
    return float(row["cost"] or 0) + wait + fail


def bench_report(settings: Settings, store: Store, suite: Suite, *, mode: str = "scan",
                 tolerance: float = TOLERANCE, tolerance_usd: float = TOLERANCE_USD,
                 include_simulated: bool = False,
                 models_only: list[str] | None = None,
                 only: str | None = None) -> dict[str, Any]:
    """Score strategies on the test split; `only` (an id glob) narrows the scored items,
    never the training data."""
    profile = profile_for(settings.policy, mode)
    rows = [dict(r) for r in store.execute(
        "SELECT * FROM bench_results WHERE suite=? AND error IS NULL"
        + ("" if include_simulated else " AND simulated=0"), (suite.key,))]
    if models_only:
        rows = [r for r in rows if r["model"] in set(models_only)]
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        table.setdefault(r["item"], {})[r["model"]] = r
    models = sorted({r["model"] for r in rows})
    prompts = {i.id: i for i in suite.items}
    complete = [i for i, by in table.items() if i in prompts and set(by) == set(models)]
    train = [i for i in complete if prompts[i].split == "train"]
    test = [i for i in complete if prompts[i].split == "test"
            and (not only or fnmatch.fnmatch(i, only))]

    reg = ModelRegistry(settings.models, "mock")   # every model counts as reachable here
    reg.only = set(models)
    # Re-tag with today's rules so old rows and the router agree.
    tags = {i: rule_tags(prompts[i].prompt) for i in train}
    evidence = [(m, tags[i].top_task, tags[i].difficulty, int(table[i][m]["success"]))
                for i in train for m in models]
    speeds: dict[str, list[float]] = {}
    for i in train:
        for m, r in table[i].items():
            if (r["tokens_out"] or 0) >= MIN_TPS_TOKENS and r["latency_ms"]:
                speeds.setdefault(m, []).append(r["tokens_out"] / (r["latency_ms"] / 1000))
    cal = fit(evidence, reg, speeds)
    scratch = Store(":memory:", store.signer)   # no spend history: pacing stays neutral

    def router(c: Calibration, item_id: str) -> str | None:
        prompt = prompts[item_id].prompt
        sel = select_model(prompt, rule_tags(prompt), reg, scratch, settings.policy, profile,
                           calibration=c)
        return sel.model.id if sel.model else None

    strategies: dict[str, Any] = {
        "router (calibrated)": lambda i: router(cal, i),
        "router (priors)": lambda i: router(Calibration(), i),
        **{f"always {m}": (lambda i, m=m: m) for m in models}}
    scores = {name: StrategyScore(name) for name in strategies}
    oracle = StrategyScore("oracle (best possible)")
    per_item: list[dict[str, Any]] = []
    for i in test:
        size = task_tokens(table[i])
        costs = {m: realised(r, profile, settings.policy, size) for m, r in table[i].items()}
        best_m = min(costs, key=lambda m: costs[m])
        best = costs[best_m]
        for target, pick in [(oracle, best_m)] + [(scores[n], f(i))
                                                   for n, f in strategies.items()]:
            if pick is None or pick not in costs:
                continue
            row = table[i][pick]
            target.n += 1
            target.right += (costs[pick] <= best * (1 + tolerance)
                             or costs[pick] - best <= tolerance_usd)
            target.passed += int(row["success"])
            target.realised += costs[pick]
            target.regret += costs[pick] - best
            target.spend += float(row["cost"] or 0)
            target.picks[pick] = target.picks.get(pick, 0) + 1
        t = rule_tags(prompts[i].prompt)
        pick = strategies["router (calibrated)"](i)
        per_item.append({"item": i, "task": t.top_task, "difficulty": t.difficulty,
                         "oracle": best_m, "router": pick,
                         "router_passed": bool(pick in table[i] and table[i][pick]["success"]),
                         "passed_by": sorted(m for m, r in table[i].items() if r["success"])})
    ranked = sorted(scores.values(), key=lambda s: (s.regret / max(s.n, 1), -s.accuracy))
    return {"suite": suite.key, "profile": profile.name, "tolerance": tolerance,
            "tolerance_usd": tolerance_usd,
            "models": models, "items": len(suite.items), "graded_items": len(complete),
            "only": only or "",
            "train": len(train), "test": len(test),
            "calibration": cal.to_dict(),
            "strategies": [oracle.to_dict()] + [s.to_dict() for s in ranked],
            "items_detail": per_item}
