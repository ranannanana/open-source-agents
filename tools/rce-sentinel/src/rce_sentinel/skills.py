"""Built-in response skills: short, fixed system instructions the proxy adds to requests.

A skill changes how a model writes, never what it may do. The text is constant, so after
the first request each provider's own prompt cache hits on it like any other prefix.

    caveman   terse prose, normal code: measured at about half the output tokens with
              about 90% of the key points kept (docs/ab.json, 48 paired answers)

Policy (config/policy.yaml):

    skills:
      enabled: [caveman]            # RCE_SENTINEL_SKILLS=caveman,... overrides; "" = none
      surfaces: [openai, anthropic] # which proxy front ends inject them
      custom: {name: "instruction"} # extra skills; a custom name replaces a built-in one

Requests that already carry the marker are left alone, so a client that loops the proxy's
output back in never gets the text twice.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

MARKER = "[rce-sentinel skills]"
SURFACES = ("openai", "anthropic")

BUILTIN: dict[str, str] = {
    "caveman": (
        "Response style, to save output tokens: write prose tersely. No greetings, "
        "pleasantries, filler, hedges, or restating the question. Short sentences; "
        "fragments are fine. Code, commands, file paths, identifiers, error messages and "
        "code comments stay complete and normal. Never drop technical content that the "
        "answer needs."),
}


@dataclass(frozen=True)
class Skill:
    name: str
    text: str


@dataclass(frozen=True)
class SkillSet:
    skills: tuple[Skill, ...]
    surfaces: frozenset[str]

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.skills]

    @property
    def text(self) -> str:
        """The instruction block; empty when no skill is on."""
        if not self.skills:
            return ""
        body = "\n".join(f"- {s.name}: {s.text}" for s in self.skills)
        return f"{MARKER}\n{body}"

    @property
    def version(self) -> str:
        """Short hash of the text: part of answer-cache keys, so edits invalidate."""
        return hashlib.sha256(self.text.encode()).hexdigest()[:12] if self.skills else ""

    def on(self, surface: str) -> bool:
        return bool(self.skills) and surface in self.surfaces


def load(policy: dict[str, Any]) -> SkillSet:
    """Active skills from policy, with RCE_SENTINEL_SKILLS taking precedence."""
    cfg = policy.get("skills") or {}
    custom = {str(k): " ".join(str(v).split()) for k, v in (cfg.get("custom") or {}).items()}
    known = {**BUILTIN, **custom}
    env = os.environ.get("RCE_SENTINEL_SKILLS")
    if env is not None:
        names = [n.strip() for n in env.split(",") if n.strip()]
    else:
        names = [str(n) for n in cfg.get("enabled", ["caveman"]) or []]
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"unknown skill(s) {', '.join(unknown)}; known: "
                         f"{', '.join(sorted(known))}")
    surfaces = frozenset(str(s) for s in cfg.get("surfaces", SURFACES) or [])
    if bad := sorted(surfaces - set(SURFACES)):
        raise ValueError(f"unknown skill surface(s) {', '.join(bad)}; known: "
                         f"{', '.join(SURFACES)}")
    names = list(dict.fromkeys(names))
    return SkillSet(tuple(Skill(n, known[n]) for n in names), surfaces)


def _has_marker(value: Any) -> bool:
    if isinstance(value, str):
        return MARKER in value
    if isinstance(value, list):
        return any(isinstance(p, dict) and MARKER in str(p.get("text", "")) for p in value)
    return False


def inject_openai(body: dict[str, Any], skills: SkillSet) -> tuple[dict[str, Any], bool]:
    """Chat-completions body with the skill block in the system message.

    Appends to an existing leading system message rather than adding a second one: some
    providers and local chat templates accept only one system message, and only first.
    """
    text = skills.text
    messages = body.get("messages")
    if not text or not isinstance(messages, list):
        return body, False
    if any(isinstance(m, dict) and m.get("role") in ("system", "developer")
           and _has_marker(m.get("content")) for m in messages):
        return body, False
    first = messages[0] if messages else None
    if isinstance(first, dict) and first.get("role") in ("system", "developer"):
        content = first.get("content")
        if isinstance(content, str):
            new_first = {**first, "content": f"{content}\n\n{text}" if content else text}
        elif isinstance(content, list):
            new_first = {**first, "content": [*content, {"type": "text", "text": text}]}
        else:
            return body, False
        return {**body, "messages": [new_first, *messages[1:]]}, True
    return {**body, "messages": [{"role": "system", "content": text}, *messages]}, True


def inject_anthropic(body: dict[str, Any], skills: SkillSet) -> tuple[dict[str, Any], bool]:
    """Messages-API body with the skill block as the last system block.

    It goes after every cache_control breakpoint the client set and carries none itself,
    so the client's cached prefix and its breakpoint count are unchanged.
    """
    text = skills.text
    if not text:
        return body, False
    system = body.get("system")
    if _has_marker(system):
        return body, False
    if system is None or system == "":
        return {**body, "system": text}, True
    if isinstance(system, str):
        return {**body, "system": f"{system}\n\n{text}"}, True
    if isinstance(system, list):
        return {**body, "system": [*system, {"type": "text", "text": text}]}, True
    return body, False
