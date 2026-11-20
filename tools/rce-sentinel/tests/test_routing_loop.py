"""Failure cost by size, outcome capture, calibration, tier floor, and per-target parts."""

from __future__ import annotations

from typing import Any

import pytest

from rce_sentinel.config import Settings
from rce_sentinel.memory.store import Store
from rce_sentinel.orchestrator.llm import MockProvider, ModelRegistry
from rce_sentinel.orchestrator.scoring import profile_for
from rce_sentinel.preeval.calibrate import (
    Calibration,
    calibrate,
    fit,
    fit_quality,
    group_of,
    load_calibration,
)
from rce_sentinel.preeval.pipeline import pre_eval
from rce_sentinel.preeval.select import (
    estimate_tokens,
    quality_of,
    review_usd,
    select_model,
)
from rce_sentinel.preeval.tags import rule_tags

CALC = ("Build a scientific calculator app with the best possible and cleanest UI for iOS "
        "and Android. It needs to be in React Native, Swift, and Flutter.")
COMMIT = "write a commit message for the staged change"
ROUTINE = [0.0, 1.0, 0.0, 0.0]
HARD = [0.0, 0.0, 1.0, 0.0]


def reg(settings: Settings) -> ModelRegistry:
    return ModelRegistry(settings.models, "mock")


def pick(settings: Settings, prompt: str, mode: str = "scan", **kw: Any) -> Any:
    return select_model(prompt, rule_tags(prompt), kw.pop("registry", reg(settings)),
                        Store(settings.db_path), settings.policy,
                        profile_for(settings.policy, mode), **kw)


# --- failure cost ------------------------------------------------------------------------
def test_review_cost_grows_with_output_and_follows_policy() -> None:
    assert review_usd(1000, {}) == pytest.approx(50 / 60)          # 1 minute at $50/h
    assert review_usd(30_000, {}) == pytest.approx(30 * 50 / 60)
    cheap_dev = {"preeval": {"dev_usd_per_hour": 25, "review_tokens_per_min": 2000}}
    assert review_usd(1000, cheap_dev) == pytest.approx(25 / 60 / 2)


def test_big_hard_build_goes_to_a_frontier_model(settings: Settings) -> None:
    for mode in ("scan", "post-commit", "watch"):
        sel = pick(settings, CALC, mode)
        assert sel.start_tier == "tier3", mode
        assert quality_of(sel.model) >= 0.85, (mode, sel.model.id)
        assert sel.fail_usd > 10


def test_small_work_stays_off_the_frontier_tier(settings: Settings) -> None:
    for mode in ("scan", "watch"):
        sel = pick(settings, COMMIT, mode)
        assert sel.start_tier != "tier3"
        assert sel.fail_usd < 1.0


def test_token_estimate_ignores_the_smoothing_floor() -> None:
    _, out = estimate_tokens(COMMIT, rule_tags(COMMIT))
    assert out < 1000   # a commit message is not a 1k-token reply


def test_cheaper_review_makes_the_router_take_more_risk(settings: Settings) -> None:
    lax = {**settings.policy, "preeval": {"dev_usd_per_hour": 1, "rework_factor": 0}}
    strict = pick(settings, CALC, "watch")
    loose = select_model(CALC, rule_tags(CALC), reg(settings), Store(settings.db_path), lax,
                         profile_for(lax, "watch"))
    assert quality_of(loose.model) < quality_of(strict.model)


# --- outcomes ----------------------------------------------------------------------------
def test_route_then_grade_by_id(settings: Settings) -> None:
    store = Store(settings.db_path)
    res = pre_eval(CALC, settings, registry=reg(settings), provider=MockProvider(),
                   store=store)
    store.outcome_for_tag(res.row_id, "rejected", review_s=600)
    row = store.execute("SELECT * FROM outcomes")[0]
    assert row["model"] == res.selection.model.id and row["success"] == 0
    assert row["task_type"] == res.tags.top_task and row["review_s"] == 600
    assert store.execute("SELECT outcome FROM prompt_tags WHERE id=?",
                         (res.row_id,))[0]["outcome"] == "rejected"


def test_bad_outcome_and_unknown_id_are_refused(settings: Settings) -> None:
    store = Store(settings.db_path)
    with pytest.raises(ValueError):
        store.record_outcome(model="m", task_type="build", difficulty=ROUTINE,
                             outcome="great", source="user")
    with pytest.raises(KeyError):
        store.outcome_for_tag(999, "accepted")


# --- calibration -------------------------------------------------------------------------
def test_fit_stays_near_prior_with_little_data_and_moves_with_a_lot() -> None:
    few = fit_quality([(ROUTINE, 0)], 0.8, 0.12)
    many = fit_quality([(ROUTINE, 0)] * 40 + [(ROUTINE, 1)] * 10, 0.8, 0.12)
    assert many < few < 0.8
    assert many < 0.45
    assert fit_quality([(HARD, 1)] * 40, 0.5, 0.12) > 0.75


def test_task_groups_borrow_from_the_model_wide_fit(settings: Settings) -> None:
    ev = ([("m", "build", HARD, 1)] * 30 + [("m", "chore", ROUTINE, 0)] * 30
          + [("m", "explain", ROUTINE, 1)])
    cal = fit(ev, reg(settings))
    assert cal.quality_for("m", "build", 0.5) > cal.quality_for("m", "chore", 0.5)
    assert cal.samples("m", "debug") == 0 and cal.n["m"]["*"] == 61
    assert cal.quality_for("m", "debug", 0.5) == cal.quality["m"]   # no group data
    assert cal.quality_for("unknown", "build", 0.42) == 0.42
    assert group_of("dev:refactor") == group_of("build")


def test_evidence_changes_the_route(settings: Settings) -> None:
    store = Store(settings.db_path)
    registry = reg(settings)
    first = pick(settings, CALC, "scan", registry=registry).model
    for _ in range(60):
        store.record_outcome(model=first.id, task_type="build", difficulty=HARD,
                             outcome="rejected", source="bench")
    cal = load_calibration(store, registry)
    assert cal.quality[first.id] < quality_of(first) - 0.2
    second = select_model(CALC, rule_tags(CALC), registry, store, settings.policy,
                          profile_for(settings.policy, "scan")).model
    assert second.id != first.id


def test_calibration_refits_only_when_evidence_arrives(settings: Settings) -> None:
    store = Store(settings.db_path)
    registry = reg(settings)
    a = load_calibration(store, registry)
    assert load_calibration(store, registry) is a
    store.record_outcome(model="x", task_type="edit", difficulty=ROUTINE, outcome="accepted",
                         source="user")
    assert load_calibration(store, registry) is not a


def test_measured_speed_comes_from_real_dev_calls_only(settings: Settings) -> None:
    store = Store(settings.db_path)
    for sim in (0, 0, 0, 1):
        store.log_call({"task_type": "dev:build", "model": "m", "tokens_out": 1000,
                        "latency_ms": 10_000 if not sim else 1, "simulated": sim})
    assert calibrate(store, reg(settings)).tps == {"m": pytest.approx(100.0)}


# --- floor and parts ---------------------------------------------------------------------
def test_min_tier_is_a_floor(settings: Settings) -> None:
    sel = pick(settings, COMMIT, "scan", min_tier="tier3")
    assert sel.start_tier == "tier3" and "tier1" not in sel.scores


def test_each_target_is_routed_on_its_own(settings: Settings) -> None:
    res = pre_eval(CALC, settings, registry=reg(settings), provider=MockProvider(),
                   store=Store(settings.db_path), split=True)
    assert [sub for sub, _ in res.parts] == res.tags.subtasks
    assert len(res.parts) == 3
    whole = res.selection.tokens_out
    assert all(sel.tokens_out < whole and sel.model is not None for _, sel in res.parts)
    assert pre_eval(COMMIT, settings, registry=reg(settings), provider=MockProvider(),
                    store=Store(settings.db_path), split=True).parts == []


def test_empty_calibration_is_the_prior(settings: Settings) -> None:
    a = pick(settings, CALC, "scan", calibration=Calibration())
    b = pick(settings, CALC, "scan")
    assert a.model.id == b.model.id
