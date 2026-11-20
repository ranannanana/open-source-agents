"""Pre-eval pipeline: rules -> Jev (online, when rules are unsure) -> local model -> select.

Every result is logged to the prompt_tags table with a hash of the prompt, never the
prompt itself, so outcomes can later be joined back to the tags that predicted them.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import httpx

from rce_sentinel.config import Settings
from rce_sentinel.memory.store import Store
from rce_sentinel.orchestrator.llm import ModelRegistry, Provider, make_provider
from rce_sentinel.orchestrator.prompts import approx_tokens
from rce_sentinel.orchestrator.scoring import profile_for
from rce_sentinel.preeval.jev import JevTagger, LocalTagger
from rce_sentinel.preeval.select import Selection, select_model, select_parts
from rce_sentinel.preeval.tags import Tags, rule_tags

DEFAULT_JEV_BELOW = 0.6  # rules confidence under this asks a model


@dataclass
class PreEval:
    tags: Tags
    selection: Selection
    notes: list[str] = field(default_factory=list)
    row_id: int = 0
    parts: list[tuple[str, Selection]] = field(default_factory=list)  # one per target


def _offline() -> bool:
    return os.environ.get("RCE_SENTINEL_OFFLINE", "").lower() in {"1", "true", "on"}


def net_error(exc: Exception) -> str:
    """Short reason, naming a blocking proxy when the reply came from one."""
    from rce_sentinel.orchestrator.seat import SeatError

    if isinstance(exc, SeatError):
        return str(exc)[:500]
    if isinstance(exc, httpx.HTTPStatusError):
        server = exc.response.headers.get("server", "")
        proxy = " (blocked by network proxy)" if "zscaler" in server.lower() else ""
        return f"HTTP {exc.response.status_code} from {server or 'server'}{proxy}"
    if isinstance(exc, httpx.ConnectError) and "CERTIFICATE_VERIFY_FAILED" in str(exc):
        return "TLS certificate not trusted (proxy inspecting HTTPS?)"
    return type(exc).__name__


def decision_models(settings: Settings, registry: ModelRegistry) -> list[JevTagger]:
    """Configured Jev endpoints in order; `decision` may be one entry or a list."""
    raw = settings.models.get("decision") or []
    out = []
    for cfg in raw if isinstance(raw, list) else [raw]:
        name = cfg.get("provider", "openrouter")
        prov = registry.providers.get(name, {})
        base = cfg.get("base_url") or registry.base_url(name)
        out.append(JevTagger(cfg, base, prov.get("api_key_env", "OPENROUTER_API_KEY")))
    return out


def tag_prompt(prompt: str, settings: Settings, registry: ModelRegistry, provider: Provider,
               use_jev: bool = True) -> tuple[Tags, list[str]]:
    notes: list[str] = []
    base = rule_tags(prompt)
    below = float(settings.policy.get("preeval", {}).get("jev_below_confidence",
                                                         DEFAULT_JEV_BELOW))
    if base.confidence >= below:
        return base, notes
    notes.append(f"rules unsure (conf {base.confidence:.2f} < {below})")
    if use_jev and not _offline() and not registry.local_only:
        jevs = [j for j in decision_models(settings, registry) if j.available()]
        if not jevs:
            notes.append("jev skipped: no API key")
        for jev in jevs:
            try:
                tags = jev.tag(prompt, base)
                notes.append(f"tagged by {jev.model}")
                return tags, notes
            except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
                notes.append(f"{jev.model} failed ({net_error(exc)})")
    local = next((m for m in registry.available("tier1", "dev") if not m.egress), None)
    if local:
        tags = LocalTagger(provider, local).tag(prompt, base)
        if tags.source == "local":
            notes.append(f"tagged by {registry.label(local)}")
            return tags, notes
        notes.append("local tagger reply invalid, kept rules")
    return base, notes


def pre_eval(prompt: str, settings: Settings, *, registry: ModelRegistry | None = None,
             provider: Provider | None = None, store: Store | None = None,
             use_jev: bool = True, current: str | None = None, session_id: str = "",
             mode: str = "scan", min_tier: str | None = None,
             split: bool = False) -> PreEval:
    registry = registry or ModelRegistry(settings.models, settings.provider_mode)
    profile = profile_for(settings.policy, mode)
    registry.local_only = profile.local_only
    provider = provider or make_provider(registry)
    store = store or Store(settings.db_path)
    tags, notes = tag_prompt(prompt, settings, registry, provider, use_jev)
    sel = select_model(prompt, tags, registry, store, settings.policy, profile, current,
                       min_tier=min_tier)
    parts = (select_parts(prompt, tags, registry, store, settings.policy, profile, current,
                          min_tier=min_tier) if split else [])
    row_id = store.log_tags({
        "session_id": session_id,
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
        "prompt_tokens": approx_tokens(prompt), "source": tags.source,
        "task_type": tags.top_task, "level": round(tags.level, 3),
        "tags": json.dumps(tags.to_dict()), "model": sel.model.id if sel.model else None,
        "tier": sel.start_tier, "est_tokens_in": sel.tokens_in,
        "est_tokens_out": sel.tokens_out, "est_cost": round(sel.expected_cost, 6)})
    return PreEval(tags, sel, notes, row_id, parts)
