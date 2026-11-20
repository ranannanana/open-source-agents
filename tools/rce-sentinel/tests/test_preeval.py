"""Prompt pre-eval: rule tags, Jev decision model, local fallback, model selection."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from rce_sentinel.config import Settings
from rce_sentinel.memory.store import Store
from rce_sentinel.orchestrator.llm import MockProvider, ModelRegistry
from rce_sentinel.preeval.jev import parse_answers, questions
from rce_sentinel.preeval.pipeline import pre_eval, tag_prompt
from rce_sentinel.preeval.tags import DIFFICULTY, PREDICATES, TASK_TYPES, rule_tags

CALC = ("Build a scientific calculator app with the best possible and cleanest UI for iOS "
        "and Android. It needs to be in React Native, Swift, and Flutter.")
VAGUE = "hmm the thing on the page looks off sometimes"


def jev_reply(task: str = "debug", level: int = 1) -> dict[str, Any]:
    probs = {t: (0.72 if t == task else 0.04) for t in TASK_TYPES}
    return {"model": "typesafe/jev-1.14",
            "answers": {"task_type": {"type": "choice", "choice": task, "probabilities": probs,
                                      "confidence": 0.8},
                        "difficulty": {"type": "score", "score": float(level),
                                       "probabilities": {str(i): (0.7 if i == level else 0.1)
                                                         for i in range(4)},
                                       "confidence": 0.75},
                        **{p: {"type": "noul", "noul": 0.3} for p in PREDICATES}},
            "usage": {"input_tokens": 120, "output_tokens": 0, "cost": 0.000005}}


def test_benchmark_prompt_is_a_hard_multi_target_build() -> None:
    t = rule_tags(CALC)
    assert t.top_task == "build"
    assert set(t.targets) == {"react native", "swift", "flutter"}
    assert {"ios", "android"} <= set(t.platforms)
    assert len(t.subtasks) == 3
    assert t.predicates["multi_target"] > 0.9
    assert t.level >= 2
    assert abs(sum(t.task.values()) - 1) < 1e-9 and abs(sum(t.difficulty) - 1) < 1e-9


@pytest.mark.parametrize("prompt,task", [
    ("why does the login test fail with a KeyError?", "debug"),
    ("refactor the auth module to remove duplicated token parsing", "refactor"),
    ("write a commit message for the staged change", "chore"),
])
def test_rule_task_types(prompt: str, task: str) -> None:
    assert rule_tags(prompt).top_task == task


def test_trivial_prompt_scores_low() -> None:
    assert rule_tags("fix typo in README").level < 1


def test_vague_prompt_is_unsure() -> None:
    assert rule_tags(VAGUE).confidence < 0.6


def test_jev_questions_cover_every_label() -> None:
    q = questions()
    assert set(q["task_type"]["criteria"]) == set(TASK_TYPES)
    assert len(q["difficulty"]["criteria"]) == len(DIFFICULTY)
    assert all(q[p]["type"] == "noul" for p in PREDICATES)


def test_parse_jev_answers_keeps_rule_targets() -> None:
    base = rule_tags(CALC)
    t = parse_answers(jev_reply("build", 3), base)
    assert t.source == "jev" and t.top_task == "build"
    assert t.difficulty.index(max(t.difficulty)) == 3
    assert t.targets == base.targets and t.cost_usd == pytest.approx(0.000005)


def _online(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    monkeypatch.delenv("RCE_SENTINEL_OFFLINE", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    sent: list[dict[str, Any]] = []

    def fake_post(url: str, json: dict[str, Any], **_: Any) -> httpx.Response:
        sent.append({"url": url, "body": json})
        return httpx.Response(200, json=jev_reply(), request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    return sent


def test_unsure_rules_ask_jev(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _online(monkeypatch)
    reg = ModelRegistry(settings.models, "mock")
    tags, _ = tag_prompt(VAGUE, settings, reg, MockProvider())
    assert tags.source == "jev" and tags.top_task == "debug"
    assert sent[0]["url"].endswith("/systemone")
    assert sent[0]["body"]["state"] == VAGUE          # prompt only ever goes in state
    assert VAGUE not in json.dumps(sent[0]["body"]["questions"])


def test_confident_rules_skip_jev(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _online(monkeypatch)
    tags, _ = tag_prompt(CALC, settings, ModelRegistry(settings.models, "mock"), MockProvider())
    assert tags.source == "rules" and not sent


def test_jev_failure_falls_back_to_local(settings: Settings,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    _online(monkeypatch)

    def down(url: str, **_: Any) -> httpx.Response:
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "post", down)
    tags, notes = tag_prompt(VAGUE, settings, ModelRegistry(settings.models, "mock"),
                             MockProvider())
    assert tags.source == "local"
    assert any("failed (ConnectError)" in n for n in notes)


def test_local_only_never_calls_jev(settings: Settings,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _online(monkeypatch)
    reg = ModelRegistry(settings.models, "mock")
    reg.local_only = True
    tags, _ = tag_prompt(VAGUE, settings, reg, MockProvider())
    assert not sent and tags.source == "local"


def test_offline_never_calls_jev(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(httpx, "post", lambda *a, **k: pytest.fail("network used offline"))
    tags, _ = tag_prompt(VAGUE, settings, ModelRegistry(settings.models, "mock"),
                         MockProvider())
    assert tags.source == "local"


def run(settings: Settings, prompt: str, **kw: Any) -> Any:
    return pre_eval(prompt, settings, registry=ModelRegistry(settings.models, "mock"),
                    provider=MockProvider(), store=Store(settings.db_path), **kw)


def test_hard_work_gets_a_stronger_model_than_trivial_work(settings: Settings) -> None:
    from rce_sentinel.preeval.select import quality_of

    hard = run(settings, CALC, mode="scan").selection
    easy = run(settings, "fix typo in README", mode="scan").selection
    assert hard.model is not None and easy.model is not None
    assert quality_of(hard.model) > quality_of(easy.model)
    # Small-context local models cannot hold the answer.
    assert any("context" in s for s in hard.skipped)


def test_models_in_a_tier_differ_by_quality_not_just_price(settings: Settings) -> None:
    from rce_sentinel.preeval.select import p_fail_prior

    reg = ModelRegistry(settings.models, "mock")
    t3 = {m.id: m for m in reg.available("tier3", "dev")}
    tags = rule_tags(CALC)
    assert p_fail_prior(t3["claude-opus-5-5"], tags) < p_fail_prior(t3["deepseek-v4-pro"], tags)


def test_proxy_blocked_provider_is_not_available(settings: Settings,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RCE_SENTINEL_OFFLINE")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("OPENCODE_API_KEY", "k")

    def fake_get(url: str, **_: Any) -> httpx.Response:
        blocked = "openrouter" in url
        return httpx.Response(403 if blocked else 401,
                              headers={"server": "Zscaler/6.2" if blocked else "cloudflare"},
                              request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    reg = ModelRegistry(settings.models, "auto")
    ids = {m.id for t in ("tier1", "tier2", "tier3") for m in reg.available(t, "dev")}
    assert "openai/gpt-6.1-sol" not in ids and "gpt-6.1-sol" in ids
    assert "Zscaler" in reg.unreachable("openrouter")


def test_trivial_task_stays_cheap(settings: Settings) -> None:
    for mode in ("scan", "watch"):
        sel = run(settings, "fix typo in README", mode=mode).selection
        assert sel.start_tier != "tier3"
        assert sel.model is not None
        assert sel.scores[sel.start_tier].price < 0.01   # under a cent per call
        assert sel.fail_usd < 1.0                        # a small reply is quick to review


def test_warm_session_model_is_cheaper(settings: Settings) -> None:
    cold = run(settings, CALC, mode="scan")
    warm = run(settings, CALC, mode="scan", current=cold.selection.scores["tier3"].model.id)
    assert warm.selection.scores["tier3"].price < cold.selection.scores["tier3"].price


def test_tags_logged_without_prompt_text(settings: Settings) -> None:
    res = run(settings, CALC, session_id="s1")
    store = Store(settings.db_path)
    row = dict(store.execute("SELECT * FROM prompt_tags WHERE id=?", (res.row_id,))[0])
    assert row["session_id"] == "s1" and row["task_type"] == "build"
    assert row["model"] == res.selection.model.id
    assert "calculator" not in json.dumps(row).lower()


def test_opencode_jev_preferred_when_its_key_is_set(settings: Settings,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _online(monkeypatch)
    monkeypatch.setenv("OPENCODE_API_KEY", "oc-key")
    tags, notes = tag_prompt(VAGUE, settings, ModelRegistry(settings.models, "mock"),
                             MockProvider())
    assert sent[0]["url"] == "https://opencode.ai/inference/systemone/v1/systemone"
    assert sent[0]["body"]["model"] == "jev-1.13-free"
    assert tags.source == "jev" and "tagged by jev-1.13-free" in notes


def test_opencode_models_are_configured(settings: Settings) -> None:
    reg = ModelRegistry(settings.models, "auto")
    ids = {m.id for t in ("tier1", "tier2", "tier3") for m in reg.all[t]
           if m.provider == "opencode"}
    assert {"big-pickle", "deepseek-v4.1-flash", "kimi-k2.7-code"} <= ids
    assert reg.base_url("opencode") == "https://opencode.ai/inference/openai/v1"


def test_env_file_loads_keys_without_overriding(tmp_path: Any,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    from rce_sentinel.config import load_env_file

    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    monkeypatch.setenv("KEEP_ME", "original")
    env = tmp_path / ".env"
    env.write_text("# comment\nexport OPENCODE_API_KEY='sk-test'\nKEEP_ME=new\nbad line\n")
    assert load_env_file(env) == ["OPENCODE_API_KEY"]
    import os

    assert os.environ["OPENCODE_API_KEY"] == "sk-test" and os.environ["KEEP_ME"] == "original"
    monkeypatch.delenv("OPENCODE_API_KEY")


def _reply(api: str) -> dict[str, Any]:
    if api == "messages":
        return {"model": "claude-sonnet-5-5", "content": [{"type": "text", "text": '{"ok": true}'}],
                "usage": {"input_tokens": 10, "cache_read_input_tokens": 90,
                          "output_tokens": 5}}
    if api == "responses":
        return {"model": "gpt-6.1-sol", "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "content": [{"type": "output_text", "text": '{"ok": true}'}]}],
            "usage": {"input_tokens": 100, "output_tokens": 40,
                      "input_tokens_details": {"cached_tokens": 64}}}
    return {"modelVersion": "gemini-4-argon", "candidates": [{"content": {"parts": [
        {"text": "thinking", "thought": True}, {"text": '{"ok": true}'}]}}],
        "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 5,
                          "thoughtsTokenCount": 20, "cachedContentTokenCount": 50}}


@pytest.mark.parametrize("model,api,path,auth", [
    ("claude-sonnet-5-5", "messages", "/inference/anthropic/v1/messages", "x-api-key"),
    ("gpt-6.1-sol", "responses", "/inference/openai/v1/responses", "Authorization"),
    ("gemini-4-argon", "gemini", "/inference/google/v1beta/models/gemini-4-argon:generateContent",
     "x-goog-api-key"),
])
def test_zen_frontier_apis(settings: Settings, monkeypatch: pytest.MonkeyPatch, model: str,
                           api: str, path: str, auth: str) -> None:
    from rce_sentinel.orchestrator.llm import OpenAICompatProvider

    monkeypatch.setenv("OPENCODE_API_KEY", "oc-key")
    seen: dict[str, Any] = {}

    def fake_post(url: str, json: dict[str, Any], headers: dict[str, str],
                  **_: Any) -> httpx.Response:
        seen.update(url=url, headers=headers, body=json)
        return httpx.Response(200, json=_reply(api), request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    reg = ModelRegistry(settings.models, "auto")
    spec = next(m for t in ("tier1", "tier2", "tier3") for m in reg.all[t] if m.id == model)
    assert spec.api == api and reg.is_available(spec)
    r = OpenAICompatProvider(reg, retries=0).complete(spec, "sys", "hi", max_tokens=64,
                                                      timeout=5)
    assert seen["url"] == "https://opencode.ai" + path
    assert "oc-key" in seen["headers"][auth]
    assert r.text == '{"ok": true}' and r.cached_tokens > 0 and r.tokens_out > 0


def test_env_autoload_reads_explicit_file(tmp_path: Any, monkeypatch: pytest.MonkeyPatch
                                          ) -> None:
    from rce_sentinel import config

    env = tmp_path / "keys.env"
    env.write_text("OPENCODE_API_KEY=sk-auto\n")
    monkeypatch.delenv("RCE_SENTINEL_NO_ENV_AUTOLOAD")
    monkeypatch.setenv("RCE_SENTINEL_ENV_FILE", str(env))
    monkeypatch.setattr(config, "_autoloaded", [])
    monkeypatch.setattr(config, "TOOL_ROOT", tmp_path)  # skip the real tool .env
    monkeypatch.setattr(config.Path, "home", lambda: tmp_path)
    assert "OPENCODE_API_KEY" in config.autoload_env()
    import os

    assert os.environ["OPENCODE_API_KEY"] == "sk-auto"
    monkeypatch.delenv("OPENCODE_API_KEY")
