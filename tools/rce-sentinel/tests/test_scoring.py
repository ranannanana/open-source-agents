"""Expected-cost routing: price vs waiting vs quality, pacing, privacy, OpenRouter, Switchyard."""

from __future__ import annotations

import copy
import json
from typing import Any

import httpx
import pytest

from rce_sentinel.config import Settings
from rce_sentinel.memory.casefile import CaseFile
from rce_sentinel.memory.store import Store
from rce_sentinel.models import Finding
from rce_sentinel.orchestrator.budget import Budget
from rce_sentinel.orchestrator.llm import MockProvider, ModelRegistry, OpenAICompatProvider
from rce_sentinel.orchestrator.router import Router
from rce_sentinel.orchestrator.scoring import Scorer, profile_for

LINES = ["import os", "os.system(x)"]
VERSIONS = {"rules": "r1", "cve_db": "c1", "prompt": "p1"}


def verdict(conf: float = 0.9) -> str:
    return json.dumps({"verdict": "vulnerable", "confidence": conf,
                       "evidence": [{"file": "a.py", "line": 2, "quote": "os.system(x)"}],
                       "exploit_path": "x", "suggested_fix": "y"})


def finding() -> Finding:
    return Finding(id="f1", layer="rule", rule_id="py-os-system", title="t", cwe="CWE-78",
                   file="a.py", line=2, sink="os.system", arg_kind="unknown",
                   context_hash="ctx-scoring")


def history(store: Store, model: str, tier: str, n: int, *, latency_ms: float,
            escalated: int = 0, cost: float = 0.0, task: str = "judge") -> None:
    """n real past calls; the first `escalated` of them failed a gate."""
    for i in range(n):
        store.log_call({"run_id": "old", "task_type": task, "tier": tier, "model": model,
                        "family": "x", "outcome": "escalated: x" if i < escalated else "accepted",
                        "tokens_in": 700, "tokens_out": 100, "cached_tokens": 0, "cost": cost,
                        "latency_ms": latency_ms, "caveman": 1, "shadow": 0, "simulated": 0})


def router(settings: Settings, store: Store, mode: str, script: dict | None = None,
           models: dict | None = None) -> tuple[Router, MockProvider]:
    provider = MockProvider(script or {"judge": [verdict()]})
    registry = ModelRegistry(models or settings.models, "mock")
    r = Router(settings, store, registry, provider,
               Budget(max_tokens=10**6, max_usd=10.0, max_llm_seconds=60), "run", VERSIONS,
               CaseFile(), "rules", mode=mode)
    return r, provider


def slow_flaky_free_tier1(store: Store) -> None:
    # Measured on this machine: the free tier-1 models are slow and usually escalate,
    # the free tier-2 cloud model is fast and usually accepted.
    history(store, "apple-foundationmodel", "tier1", 10, latency_ms=15000, escalated=9)
    history(store, "qwen2.5:3b", "tier1", 10, latency_ms=12000, escalated=8)
    history(store, "gemma4:31b-cloud", "tier2", 10, latency_ms=1500, escalated=1)


def test_developer_waiting_skips_slow_free_models(settings: Settings) -> None:
    store = Store(settings.db_path)
    slow_flaky_free_tier1(store)
    r, p = router(settings, store, "pre-commit")
    out = r.judge(finding(), LINES)
    assert p.calls[0] == ("judge", "gemma4:31b-cloud")
    assert any("skipped by route plan" in t for t in out.trail)
    assert r.plans and "start at tier2" in r.plans[0]


def test_nobody_waiting_starts_with_the_free_model(settings: Settings) -> None:
    store = Store(settings.db_path)
    history(store, "apple-foundationmodel", "tier1", 10, latency_ms=15000, escalated=3)
    history(store, "gemma4:31b-cloud", "tier2", 10, latency_ms=1500, escalated=1)
    r, p = router(settings, store, "watch")  # background profile: waiting costs nothing
    r.judge(finding(), LINES)
    assert p.calls[0][1] in {"apple-foundationmodel", "qwen2.5:3b"}


def test_unmeasured_models_keep_the_plain_cascade(settings: Settings) -> None:
    r, p = router(settings, Store(settings.db_path), "pre-commit")
    r.judge(finding(), LINES)
    assert p.calls[0] == ("judge", "apple-foundationmodel") and not r.plans


def test_paid_model_wins_when_its_time_saving_is_worth_more_than_its_price(
        settings: Settings) -> None:
    store = Store(settings.db_path)
    models = copy.deepcopy(settings.models)
    models["tiers"]["tier1"] = [
        {"id": "free-slow", "provider": "ollama", "family": "a", "price_in": 0.0,
         "price_out": 0.0},
        {"id": "paid-fast", "provider": "gemini", "family": "b", "price_in": 3.0,
         "price_out": 15.0}]
    history(store, "free-slow", "tier1", 10, latency_ms=120_000, escalated=2)  # 2 min
    history(store, "paid-fast", "tier1", 10, latency_ms=50_000, escalated=1)   # 50 s
    waiting, _ = router(settings, store, "interactive", models=models)
    assert waiting.pick("tier1").id == "paid-fast"
    nobody, _ = router(settings, store, "watch", models=models)
    assert nobody.pick("tier1").id == "free-slow"


def test_pacing_makes_paid_tokens_dearer_then_switches_them_off(settings: Settings) -> None:
    store = Store(settings.db_path)
    policy = copy.deepcopy(settings.policy)
    policy["routing"]["monthly_usd"] = 1.0
    history(store, "kimi-k2.6", "tier3", 1, latency_ms=1000, cost=0.9)
    ahead = Scorer(store, policy, profile_for(policy, "pre-commit"))
    assert ahead.pacing > 1.0 and ahead.paid_allowed
    history(store, "kimi-k2.6", "tier3", 1, latency_ms=1000, cost=0.2)
    settings.policy["routing"]["monthly_usd"] = 1.0
    r, _ = router(settings, store, "pre-commit")
    assert not r.scorer.paid_allowed
    assert r.pick("tier3") is None  # every tier-3 model is paid
    assert r.pick("tier1").id == "apple-foundationmodel"  # free models still work
    assert any("monthly allowance" in d for d in r.degraded)


def test_latency_cap_drops_a_measured_slow_model_when_a_faster_one_exists(
        settings: Settings) -> None:
    store = Store(settings.db_path)
    history(store, "apple-foundationmodel", "tier1", 10, latency_ms=40_000)
    r, _ = router(settings, store, "pre-commit")  # 10 s cap
    assert r.pick("tier1").id == "qwen2.5:3b"


def test_local_only_removes_every_model_that_sends_code_off_the_machine(
        settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RCE_SENTINEL_LOCAL_ONLY", "1")
    r, _ = router(settings, Store(settings.db_path), "pre-commit")
    reg = r.registry
    left = {m.id for t in ("tier1", "tier2", "tier3") for m in reg.available(t)}
    assert left == {"apple-foundationmodel", "qwen2.5:3b"}
    assert next(m for m in reg.all["tier2"] if m.id == "gemma4:31b-cloud").egress


def test_profile_mapping_and_override(settings: Settings,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    assert profile_for(settings.policy, "pre-commit").latency_cap_s == 10
    assert profile_for(settings.policy, "watch").value_per_second == 0
    monkeypatch.setenv("RCE_SENTINEL_PROFILE", "interactive")
    assert profile_for(settings.policy, "watch").name == "interactive"


def test_openrouter_request_carries_routing_and_reported_cost_is_used(
        settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    models = copy.deepcopy(settings.models)
    models["tiers"]["tier3"] = [{"id": "moonshotai/kimi-k2.6", "provider": "openrouter",
                                 "family": "moonshot", "price_in": 0.95, "price_out": 4.0,
                                 "fallbacks": ["moonshotai/kimi-k2.5"]}]
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    sent: dict[str, Any] = {}

    def fake_post(url: str, json: dict, headers: dict, **_: Any) -> httpx.Response:
        sent.update(json)
        body = {"model": "moonshotai/kimi-k2.5",
                "choices": [{"message": {"content": verdict()}}],
                "usage": {"prompt_tokens": 700, "completion_tokens": 90, "cost": 0.00123}}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    registry = ModelRegistry(models, "auto")
    registry.provider_sort = "latency"
    spec = registry.all["tier3"][0]
    res = OpenAICompatProvider(registry).complete(spec, "sys", "user", max_tokens=100,
                                                  timeout=5)
    assert sent["provider"]["sort"] == "latency"
    assert sent["provider"]["max_price"] == {"prompt": 1.1875, "completion": 5.0}
    assert sent["provider"]["data_collection"] == "deny"
    assert sent["models"] == ["moonshotai/kimi-k2.6", "moonshotai/kimi-k2.5"]
    assert sent["usage"] == {"include": True}
    assert res.cost == 0.00123 and res.served == "moonshotai/kimi-k2.5"


def test_switchyard_route_is_local_but_counts_as_egress(settings: Settings) -> None:
    models = copy.deepcopy(settings.models)
    models["tiers"]["tier2"].insert(0, {"id": "switchyard/gemma", "provider": "switchyard",
                                        "family": "gemma", "price_in": 0.0, "price_out": 0.0})
    reg = ModelRegistry(models, "auto")
    spec = reg.all["tier2"][0]
    assert spec.egress
    reg._local_up = {"switchyard": {"switchyard/gemma"}}
    assert reg.is_real(spec)
    reg.local_only = True
    assert not reg.is_available(spec)


def test_verifier_avoids_the_family_a_gateway_actually_used(settings: Settings) -> None:
    models = copy.deepcopy(settings.models)
    models["tiers"]["tier1"].insert(0, {"id": "switchyard-stage", "provider": "switchyard",
                                        "family": "switchyard", "price_in": 0.0,
                                        "price_out": 0.0})
    r, _ = router(settings, Store(settings.db_path), "watch", models=models)
    spec = r.registry.all["tier1"][0]
    r._served[7] = "qwen2.5:3b"  # what Switchyard reported in the reply's `model` field
    finder = r._as_served(spec, 7)
    assert finder.family == "qwen"
    assert r.pick_verifier(finder).family != "qwen"
