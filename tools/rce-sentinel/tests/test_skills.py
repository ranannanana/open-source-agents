"""Built-in response skills: policy loading and injection into both request shapes."""

from __future__ import annotations

import pytest

from rce_sentinel import skills
from rce_sentinel.skills import MARKER, inject_anthropic, inject_openai


def test_default_policy_turns_on_caveman_everywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RCE_SENTINEL_SKILLS", raising=False)
    sk = skills.load({})
    assert sk.names == ["caveman"] and sk.on("openai") and sk.on("anthropic")
    assert sk.text.startswith(MARKER) and len(sk.version) == 12


def test_env_overrides_policy_and_empty_turns_everything_off(
        monkeypatch: pytest.MonkeyPatch) -> None:
    policy = {"skills": {"enabled": ["caveman"],
                         "custom": {"cite": "Cite the file and line for every claim."}}}
    monkeypatch.setenv("RCE_SENTINEL_SKILLS", "cite,caveman,cite")
    sk = skills.load(policy)
    assert sk.names == ["cite", "caveman"] and "file and line" in sk.text
    monkeypatch.setenv("RCE_SENTINEL_SKILLS", "")
    off = skills.load(policy)
    assert off.names == [] and off.text == "" and not off.on("openai") and off.version == ""


def test_bad_names_and_surfaces_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RCE_SENTINEL_SKILLS", raising=False)
    with pytest.raises(ValueError, match="unknown skill"):
        skills.load({"skills": {"enabled": ["nope"]}})
    with pytest.raises(ValueError, match="surface"):
        skills.load({"skills": {"surfaces": ["grpc"]}})
    only_anthropic = skills.load({"skills": {"surfaces": ["anthropic"]}})
    assert not only_anthropic.on("openai") and only_anthropic.on("anthropic")


def test_editing_a_skill_changes_its_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RCE_SENTINEL_SKILLS", "x")
    a = skills.load({"skills": {"custom": {"x": "one"}}})
    b = skills.load({"skills": {"custom": {"x": "two"}}})
    assert a.version != b.version


@pytest.fixture
def sk(monkeypatch: pytest.MonkeyPatch) -> skills.SkillSet:
    monkeypatch.delenv("RCE_SENTINEL_SKILLS", raising=False)
    return skills.load({})


def test_openai_adds_a_system_message_or_extends_the_first(sk: skills.SkillSet) -> None:
    user = {"role": "user", "content": "hi"}
    out, done = inject_openai({"messages": [user]}, sk)
    assert done and out["messages"][0]["role"] == "system" and out["messages"][1] == user
    body = {"messages": [{"role": "system", "content": "be kind"}, user], "temperature": 0}
    out, done = inject_openai(body, sk)
    assert done and len(out["messages"]) == 2 and out["temperature"] == 0
    assert out["messages"][0]["content"].startswith("be kind\n\n" + MARKER)
    assert body["messages"][0]["content"] == "be kind"          # input left untouched
    parts = {"messages": [{"role": "developer", "content": [{"type": "text", "text": "s"}]},
                          user]}
    out, done = inject_openai(parts, sk)
    assert done and out["messages"][0]["content"][-1]["text"].startswith(MARKER)


def test_openai_injection_is_idempotent(sk: skills.SkillSet) -> None:
    once, _ = inject_openai({"messages": [{"role": "user", "content": "hi"}]}, sk)
    twice, done = inject_openai(once, sk)
    assert not done and twice == once
    assert inject_openai({"prompt": "legacy"}, sk) == ({"prompt": "legacy"}, False)


def test_anthropic_appends_after_cache_breakpoints(sk: skills.SkillSet) -> None:
    cached = {"type": "text", "text": "You are Claude Code.",
              "cache_control": {"type": "ephemeral"}}
    out, done = inject_anthropic({"system": [cached], "messages": []}, sk)
    assert done and out["system"][0] == cached and "cache_control" not in out["system"][1]
    assert out["system"][1]["text"].startswith(MARKER)
    again, done = inject_anthropic(out, sk)
    assert not done and again == out
    plain, done = inject_anthropic({"system": "S", "messages": []}, sk)
    assert done and plain["system"].startswith("S\n\n" + MARKER)
    bare, done = inject_anthropic({"messages": []}, sk)
    assert done and bare["system"].startswith(MARKER)
