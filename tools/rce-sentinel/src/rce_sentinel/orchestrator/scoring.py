"""Expected-cost model selection: balance price, waiting time, and answer quality.

A model is worth calling when the total expected cost of getting a correct answer through it
is lowest. For one candidate m in tier t:

    cost(m) = price(m) * pacing
            + value_per_second * latency(m)          # someone waiting, or not
            + P(escalate | m) * cost_from(t + 1)     # its answer fails a gate, next tier pays
            + P(accepted) * P(overturned | m) * error_cost

cost_from(t) is the best candidate in tier t, and the cascade starts at the tier with the
lowest cost_from. So a slow free model that usually escalates is skipped when a developer is
waiting, and used first when nobody is (overnight intel, post-commit).

Every number comes from this repo's own call log (real calls only), never from a benchmark.
Until a model has `min_samples` real calls, a mild per-tier prior is used and the router keeps
the plain cheapest-first cascade, so behaviour only changes once it has been measured.

`pacing` keeps paid usage in balance with a monthly allowance: spending ahead of the calendar
makes paid tokens count for more; reaching the allowance removes paid models entirely.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from rce_sentinel.memory.store import Store
from rce_sentinel.orchestrator.llm import TIERS, ModelSpec

# Priors used until a model has been measured: escalation rate, overturn rate, seconds.
PRIOR_ESCALATE = {"tier1": 0.5, "tier2": 0.25, "tier3": 0.1}
# Flat on purpose: every accepted verdict passed the same gates, so until a model's own
# overturn rate is measured no tier is assumed riskier than another.
PRIOR_OVERTURN = {"tier1": 0.02, "tier2": 0.02, "tier3": 0.02}
PRIOR_SECONDS = {"tier1": 10.0, "tier2": 8.0, "tier3": 12.0}
UNRESOLVED_SHARE = 0.5  # nothing left after tier3: rules decide, half an error on average
WINDOW = 50             # recent real calls per model used for the estimates

DEFAULT_PROFILES: dict[str, dict[str, Any]] = {
    # About $50/h of developer time; the commit is blocked while the hook runs.
    "pre-commit": {"value_per_second": 0.014, "latency_cap_s": 10, "error_cost_usd": 1.0,
                   "provider_sort": "latency"},
    "interactive": {"value_per_second": 0.014, "latency_cap_s": 30, "error_cost_usd": 1.0,
                    "provider_sort": "latency"},
    "post-commit": {"value_per_second": 0.001, "error_cost_usd": 1.0, "provider_sort": "price"},
    "background": {"value_per_second": 0.0, "error_cost_usd": 1.0, "provider_sort": "price"},
}
DEFAULT_MODE_PROFILE = {"pre-commit": "pre-commit", "post-commit": "post-commit",
                        "post-merge": "post-commit", "scan": "interactive",
                        "watch": "background", "intel": "background", "trace": "background"}


@dataclass(frozen=True)
class Profile:
    name: str
    value_per_second: float = 0.0
    latency_cap_s: float | None = None
    error_cost_usd: float = 1.0
    provider_sort: str = "price"   # passed to OpenRouter's provider.sort
    local_only: bool = False       # nothing may leave the machine


def profile_for(policy: dict[str, Any], mode: str) -> Profile:
    """Pick the routing profile for a scan mode. RCE_SENTINEL_PROFILE overrides it."""
    routing = policy.get("routing", {})
    profiles = {**DEFAULT_PROFILES, **routing.get("profiles", {})}
    mapping = {**DEFAULT_MODE_PROFILE, **routing.get("mode_profile", {})}
    name = os.environ.get("RCE_SENTINEL_PROFILE") or mapping.get(mode, "interactive")
    raw = profiles.get(name, DEFAULT_PROFILES["interactive"])
    local_only = bool(routing.get("local_only", False)) or os.environ.get(
        "RCE_SENTINEL_LOCAL_ONLY", "").lower() in {"1", "true", "on", "yes"}
    cap = raw.get("latency_cap_s")
    return Profile(name=name, value_per_second=float(raw.get("value_per_second", 0.0)),
                   latency_cap_s=float(cap) if cap is not None else None,
                   error_cost_usd=float(raw.get("error_cost_usd", 1.0)),
                   provider_sort=str(raw.get("provider_sort", "price")),
                   local_only=bool(raw.get("local_only", False)) or local_only)


@dataclass(frozen=True)
class ModelStats:
    n: int             # real calls seen
    p_escalate: float
    p_overturn: float
    seconds: float     # median latency
    measured: bool     # n >= min_samples


@dataclass(frozen=True)
class Score:
    model: ModelSpec
    total: float
    price: float
    wait: float
    escalate: float
    risk: float
    stats: ModelStats

    def explain(self) -> str:
        s = self.stats
        src = f"measured n={s.n}" if s.measured else "prior"
        return (f"{self.model.id}: ${self.total:.3f} = price {self.price:.4f} + wait "
                f"{self.wait:.3f} ({s.seconds:.1f}s) + escalate {self.escalate:.3f} "
                f"({s.p_escalate:.0%}) + risk {self.risk:.3f} [{src}]")


def month_spend(store: Store, now: float | None = None) -> tuple[float, float]:
    """Real dollars spent this calendar month (UTC), and the fraction of the month elapsed."""
    now = now or time.time()
    dt = datetime.fromtimestamp(now, UTC)
    start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    nxt = start.replace(year=start.year + (start.month == 12), month=start.month % 12 + 1)
    rows = store.execute("SELECT COALESCE(SUM(cost), 0) AS s FROM llm_calls WHERE simulated=0 "
                         "AND ts >= ?", (start.timestamp(),))
    frac = (now - start.timestamp()) / (nxt.timestamp() - start.timestamp())
    return float(rows[0]["s"] or 0.0), max(frac, 1e-3)


class Scorer:
    def __init__(self, store: Store, policy: dict[str, Any], profile: Profile):
        routing = policy.get("routing", {})
        self.store = store
        self.profile = profile
        self.min_samples = int(os.environ.get("RCE_SENTINEL_MIN_SAMPLES")
                               or routing.get("min_samples", 8))
        self.monthly_usd = float(routing.get("monthly_usd", 20.0))
        self._stats: dict[tuple[str, str], ModelStats] = {}
        spent, frac = month_spend(store)
        self.spent = spent
        self.pace = spent / (self.monthly_usd * frac) if self.monthly_usd > 0 else 0.0
        # Ahead of the calendar: paid tokens weigh more (squared, capped). Behind: face value.
        self.pacing = min(max(self.pace, 1.0) ** 2, 10.0)
        self.paid_allowed = self.monthly_usd <= 0 or spent < self.monthly_usd

    def stats(self, m: ModelSpec, task: str) -> ModelStats:
        key = (m.id, task)
        if key not in self._stats:
            rows = self.store.execute(
                "SELECT outcome, final_agree, latency_ms FROM llm_calls WHERE model=? AND "
                "task_type=? AND simulated=0 AND shadow=0 ORDER BY id DESC LIMIT ?",
                (m.id, task, WINDOW))
            rows = [r for r in rows if not (r["outcome"] or "").startswith("probe")]
            n = len(rows)
            measured = n >= self.min_samples
            if measured:
                bad = sum(1 for r in rows if (r["outcome"] or "").startswith(
                    ("escalated", "schema_fail")))
                labelled = [r["final_agree"] for r in rows if r["final_agree"] is not None]
                lat = sorted(r["latency_ms"] for r in rows if r["latency_ms"])
                p_over = (labelled.count(0) / len(labelled)
                          if len(labelled) >= self.min_samples else PRIOR_OVERTURN[m.tier])
                stats = ModelStats(n, bad / n, p_over,
                                   lat[len(lat) // 2] / 1000 if lat else PRIOR_SECONDS[m.tier],
                                   True)
            else:
                stats = ModelStats(n, PRIOR_ESCALATE[m.tier], PRIOR_OVERTURN[m.tier],
                                   PRIOR_SECONDS[m.tier], False)
            self._stats[key] = stats
        return self._stats[key]

    def is_paid(self, m: ModelSpec) -> bool:
        return m.price_in > 0 or m.price_out > 0

    def score(self, m: ModelSpec, task: str, price: float, next_cost: float) -> Score:
        s = self.stats(m, task)
        p = self.profile
        price_term = price * (self.pacing if self.is_paid(m) else 1.0)
        wait = p.value_per_second * s.seconds
        esc = s.p_escalate * next_cost
        risk = (1 - s.p_escalate) * s.p_overturn * p.error_cost_usd
        return Score(m, price_term + wait + esc + risk, price_term, wait, esc, risk, s)

    def too_slow(self, m: ModelSpec, task: str) -> bool:
        cap = self.profile.latency_cap_s
        s = self.stats(m, task)
        return cap is not None and s.measured and s.seconds > cap

    def any_measured(self, models: list[ModelSpec], task: str) -> bool:
        return any(self.stats(m, task).measured for m in models)

    def tail(self) -> float:
        return UNRESOLVED_SHARE * self.profile.error_cost_usd


def plan_cascade(scorer: Scorer, task: str, by_tier: dict[str, list[tuple[ModelSpec, float]]]
                 ) -> tuple[str, dict[str, Score]]:
    """Bottom-up over tiers: best score per tier and the tier the cascade should start at.

    `by_tier` maps a tier to its eligible (model, estimated price) pairs.
    """
    best: dict[str, Score] = {}
    nxt = scorer.tail()
    cost_from: dict[str, float] = {}
    for tier in reversed(TIERS):
        scores = [scorer.score(m, task, price, nxt) for m, price in by_tier.get(tier, [])]
        if scores:
            best[tier] = min(scores, key=lambda s: s.total)
            nxt = best[tier].total
        cost_from[tier] = nxt
    order = [t for t in TIERS if t in best]
    if not order:
        return TIERS[0], best
    # Ties go to the lower tier: same expected cost, less money spent up front.
    start = min(order, key=lambda t: (round(cost_from[t], 6), TIERS.index(t)))
    return start, best
