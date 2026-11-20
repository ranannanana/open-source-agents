"""Pick a model for a tagged prompt with the same expected-cost rule the scanner uses.

    cost(m) = price(m) * pacing + value_per_second * latency(m)
            + P(fail | m, tags) * (review(tokens_out) + cost_from(next tier)) + risk

Tags change two things: the token estimate (task type and difficulty set the expected
output length) and P(fail), which comes from each model's quality: a hand-set prior until
graded outcomes exist, then the calibrated value (calibrate.py).

A failed answer is not free: someone reads it, finds it broken, and asks again. That review
time grows with the size of the answer, so a cheap model that often fails is fine for a
commit message and a bad bet for a 20k-token build. If the top tier also fails, the
developer does the work: `rework_factor` times the review cost.

Staying on the current model prices its context at the cached rate, so a warm session is
not abandoned for a marginal saving.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from rce_sentinel.memory.store import Store
from rce_sentinel.orchestrator.llm import TIERS, ModelRegistry, ModelSpec
from rce_sentinel.orchestrator.prompts import approx_tokens
from rce_sentinel.orchestrator.scoring import ModelStats, Profile, Score, Scorer, plan_cascade
from rce_sentinel.preeval.tags import DIFFICULTY, Tags

if TYPE_CHECKING:
    from rce_sentinel.preeval.calibrate import Calibration

# Capability prior. Each model has a quality q in 0-1 (models.yaml `quality`, else its
# tier's default). P(success) at difficulty level d is a logistic step at THRESHOLDS[d],
# so a strong model barely notices routine work and a weak one falls off on hard work.
TIER_QUALITY = {"tier1": 0.45, "tier2": 0.65, "tier3": 0.82}
THRESHOLDS = (0.10, 0.40, 0.65, 0.85)   # trivial, routine, hard, frontier
STEEPNESS = 10.0
# Typical output tokens per task type at routine difficulty.
OUT_TOKENS = {"build": 6000, "edit": 1500, "debug": 1500, "refactor": 2500, "test": 2000,
              "explain": 600, "design": 1500, "chore": 200}
CONTEXT_OVERHEAD = 1500     # system prompt, tool schemas, a little repo context
MULTI_FILE_CONTEXT = 6000   # extra files read when work spans several
# Unmeasured latency: first-token seconds plus output tokens per second.
FIRST_TOKEN_S = {"tier1": 2.0, "tier2": 3.0, "tier3": 5.0}
TOKENS_PER_S = {"tier1": 40.0, "tier2": 60.0, "tier3": 50.0}
# Cost of a failed answer, overridable under `preeval:` in policy.yaml.
DEV_USD_PER_HOUR = 50.0        # developer time spent reviewing a reply
REVIEW_TOKENS_PER_MIN = 1000   # how fast a reply is read and judged
REWORK_FACTOR = 3.0            # every tier failed: the developer does it by hand


def quality_of(m: ModelSpec | str) -> float:
    if isinstance(m, str):
        return TIER_QUALITY[m]
    return m.quality if m.quality is not None else TIER_QUALITY[m.tier]


def p_success(q: float, level: int) -> float:
    return 1.0 / (1.0 + math.exp(-STEEPNESS * (q - THRESHOLDS[level])))


def p_fail_at(q: float, tags: Tags) -> float:
    """Expected failure chance over the tagged difficulty distribution."""
    return 1.0 - sum(p * p_success(q, i) for i, p in enumerate(tags.difficulty))


def p_fail_prior(m: ModelSpec | str, tags: Tags) -> float:
    return p_fail_at(quality_of(m), tags)


def estimate_tokens(prompt: str, tags: Tags) -> tuple[int, int]:
    """Expected (input, output) tokens for the whole answer."""
    n_targets = max(len(tags.targets), 1) if tags.predicates["multi_target"] > 0.5 else 1
    tin = (approx_tokens(prompt) + CONTEXT_OVERHEAD
           + int(tags.predicates["multi_file"] * MULTI_FILE_CONTEXT))
    # Only mass above the smoothing floor: 5% "build" must not add 300 tokens.
    floor = min(tags.task.values())
    weights = {t: p - floor for t, p in tags.task.items()}
    total = sum(weights.values())
    if total <= 0:
        weights, total = dict(tags.task), sum(tags.task.values())
    base = sum(w * OUT_TOKENS[t] for t, w in weights.items()) / total
    tout = int(base * (0.5 + tags.level) * n_targets)
    return tin, tout


def review_usd(tokens_out: int, policy: dict[str, Any]) -> float:
    """Developer time to read a reply of this size and find it broken."""
    cfg = policy.get("preeval", {})
    per_min = float(cfg.get("review_tokens_per_min", REVIEW_TOKENS_PER_MIN))
    rate = float(cfg.get("dev_usd_per_hour", DEV_USD_PER_HOUR))
    return tokens_out / per_min / 60 * rate


class DevScorer(Scorer):
    """Scanner scorer with difficulty-aware failure odds and a size-aware failure cost."""

    def __init__(self, store: Store, policy: dict[str, Any], profile: Profile, tags: Tags,
                 tokens_out: int, calibration: Calibration | None = None):
        super().__init__(store, policy, profile)
        self.tags = tags
        self.tokens_out = tokens_out
        self.cal = calibration
        self.review_usd = review_usd(tokens_out, policy)
        self.rework_factor = float(policy.get("preeval", {}).get("rework_factor",
                                                                 REWORK_FACTOR))

    def quality(self, m: ModelSpec) -> float:
        prior = quality_of(m)
        return self.cal.quality_for(m.id, self.tags.top_task, prior) if self.cal else prior

    def stats(self, m: ModelSpec, task: str) -> ModelStats:
        key = (m.id, task)
        if key not in self._stats:
            n = self.cal.samples(m.id, self.tags.top_task) if self.cal else 0
            tps = (self.cal.tps.get(m.id) if self.cal else None) or m.tps \
                or TOKENS_PER_S[m.tier]
            seconds = FIRST_TOKEN_S[m.tier] + self.tokens_out / tps
            base = super().stats(m, task)
            self._stats[key] = ModelStats(n, p_fail_at(self.quality(m), self.tags),
                                          base.p_overturn, seconds, n >= self.min_samples)
        return self._stats[key]

    def score(self, m: ModelSpec, task: str, price: float, next_cost: float) -> Score:
        s = self.stats(m, task)
        price_term = price * (self.pacing if self.is_paid(m) else 1.0)
        wait = self.profile.value_per_second * s.seconds
        esc = s.p_escalate * (self.review_usd + next_cost)
        risk = (1 - s.p_escalate) * s.p_overturn * self.profile.error_cost_usd
        return Score(m, price_term + wait + esc + risk, price_term, wait, esc, risk, s)

    def tail(self) -> float:
        return self.rework_factor * self.review_usd


@dataclass
class Selection:
    model: ModelSpec | None
    start_tier: str
    tokens_in: int
    tokens_out: int
    scores: dict[str, Score] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    fail_usd: float = 0.0     # review cost of one failed reply

    @property
    def expected_cost(self) -> float:
        return self.scores[self.start_tier].total if self.start_tier in self.scores else 0.0

    def explain(self) -> list[str]:
        lines = [f"estimate: {self.tokens_in} tokens in, {self.tokens_out} out; "
                 f"a failed reply costs ${self.fail_usd:.2f} to review"]
        for tier in TIERS:
            if tier in self.scores:
                mark = "->" if tier == self.start_tier else "  "
                lines.append(f"{mark} {tier}: {self.scores[tier].explain()}")
        lines += [f"   skipped {s}" for s in self.skipped]
        return lines


def select_model(prompt: str, tags: Tags, registry: ModelRegistry, store: Store,
                 policy: dict[str, Any], profile: Profile, current: str | None = None,
                 *, min_tier: str | None = None,
                 calibration: Calibration | None = None) -> Selection:
    if calibration is None:
        from rce_sentinel.preeval.calibrate import load_calibration

        calibration = load_calibration(store, registry)
    tin, tout = estimate_tokens(prompt, tags)
    task = f"dev:{tags.top_task}"
    scorer = DevScorer(store, policy, profile, tags, tout, calibration)
    current_family = registry.family_of(current) if current else None
    floor = TIERS.index(min_tier) if min_tier else 0
    by_tier: dict[str, list[tuple[ModelSpec, float]]] = {}
    skipped: list[str] = []
    for tier in TIERS[floor:]:
        for m in registry.available(tier, "dev"):
            if m.context_tokens < tin + min(tout, 8000):
                skipped.append(f"{m.id}: context {m.context_tokens} < needed")
                continue
            if not scorer.paid_allowed and scorer.is_paid(m):
                skipped.append(f"{m.id}: monthly allowance used")
                continue
            cached = tin if m.id == current else 0
            by_tier.setdefault(tier, []).append((m, m.cost(tin, tout, cached)))
        # Ties go to the session's family: fewer tool-call format surprises.
        if current_family and tier in by_tier:
            by_tier[tier].sort(key=lambda mp: mp[0].family != current_family)
    if not by_tier:
        return Selection(None, TIERS[floor], tin, tout, skipped=skipped,
                         fail_usd=scorer.review_usd)
    start, best = plan_cascade(scorer, task, by_tier)
    return Selection(best[start].model, start, tin, tout, best, skipped, scorer.review_usd)


def part_tags(tags: Tags, target: str) -> Tags:
    """Tags for one target of a multi-target prompt."""
    preds = {**tags.predicates, "multi_target": 0.05}
    return replace(tags, targets=[target], predicates=preds, subtasks=[])


def select_parts(prompt: str, tags: Tags, registry: ModelRegistry, store: Store,
                 policy: dict[str, Any], profile: Profile, current: str | None = None,
                 *, min_tier: str | None = None) -> list[tuple[str, Selection]]:
    """One selection per target, so each part goes to the model best placed for it."""
    if len(tags.targets) < 2 or not tags.subtasks:
        return []
    return [(sub, select_model(prompt, part_tags(tags, target), registry, store, policy,
                               profile, current, min_tier=min_tier))
            for sub, target in zip(tags.subtasks, tags.targets, strict=False)]


__all__ = ["DIFFICULTY", "Selection", "estimate_tokens", "select_model", "select_parts"]
