"""Provider adapter. One OpenAI-compatible code path covers apfel, Ollama, Gemini, DeepSeek,
Kimi, GLM, OpenRouter, and a local NeMo Switchyard proxy. LLM calls never get tools.
MockProvider makes everything run offline."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from pydantic import BaseModel

from rce_sentinel.orchestrator.prompts import CAVEMAN_MARKER, approx_tokens

TIERS = ("tier1", "tier2", "tier3")


@dataclass(frozen=True)
class ModelSpec:
    id: str
    provider: str
    family: str
    tier: str
    price_in: float
    price_out: float
    price_cached_in: float
    context_tokens: int = 128_000
    quota_tokens: int | None = None  # per-run token quota; after it the router switches model
    egress: bool = False  # prompts leave this machine (cloud API, Ollama cloud, proxy to cloud)
    fallbacks: tuple[str, ...] = ()  # OpenRouter: same-capability models tried server-side
    uses: tuple[str, ...] = ("scan", "dev")  # scan: code review; dev: prompt routing
    api: str = "chat"  # chat | messages (Anthropic) | responses (OpenAI) | gemini
    quality: float | None = None  # 0-1 capability prior for dev routing; None = tier default
    tps: float | None = None      # output tokens per second prior; None = tier default
    long_above: int | None = None  # prompts above this many tokens use the long prices
    long_scale: float = 1.0        # price multiplier above long_above (Haiku 5.5: 5x)

    def cost(self, tokens_in: int, tokens_out: int, cached: int = 0) -> float:
        fresh = max(tokens_in - cached, 0)
        scale = self._scale(tokens_in)
        return scale * (fresh * self.price_in + cached * self.price_cached_in
                        + tokens_out * self.price_out) / 1_000_000

    def _scale(self, prompt_tokens: int) -> float:
        return self.long_scale if self.long_above and prompt_tokens > self.long_above else 1.0

    def usage_cost(self, fresh: int, write_5m: int, write_1h: int, read: int,
                   out: int) -> float:
        """Anthropic-style usage: cache writes cost 1.25x (5 min) or 2x (1 h) input."""
        scale = self._scale(fresh + write_5m + write_1h + read)
        return scale * ((fresh + 1.25 * write_5m + 2.0 * write_1h) * self.price_in
                        + read * self.price_cached_in + out * self.price_out) / 1_000_000


@dataclass
class LLMResult:
    text: str
    tokens_in: int
    tokens_out: int
    cached_tokens: int
    latency_ms: float
    model: str
    cost: float | None = None  # provider-reported dollars (OpenRouter usage accounting)
    served: str = ""           # model that actually answered, when a gateway picked it


class Provider(Protocol):
    def complete(self, spec: ModelSpec, system: str, user: str, *, max_tokens: int,
                 timeout: float, prefix_tokens: int = 0, prefix_warm: bool = False,
                 schema: dict[str, Any] | None = None, text: bool = False) -> LLMResult: ...


def parse_quotas(raw: str) -> dict[str, int]:
    """'apple-foundationmodel=2000,qwen2.5:3b=5000' -> {model id: tokens per run}."""
    out: dict[str, int] = {}
    for part in filter(None, (p.strip() for p in raw.split(","))):
        model, _, n = part.rpartition("=")
        if model and n.isdigit():
            out[model] = int(n)
    return out


def inline_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic JSON schema with $ref/$defs inlined, as constrained decoders need."""
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].rsplit("/", 1)[-1]])
            out = {k: walk(v) for k, v in node.items() if k not in {"title", "default"}}
            if out.get("type") == "object":
                out.setdefault("additionalProperties", False)
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(raw)  # type: ignore[no-any-return]


class ModelRegistry:
    def __init__(self, models_cfg: dict[str, Any], mode: str):
        self.cfg = models_cfg
        self.mode = mode
        self.providers: dict[str, Any] = models_cfg.get("providers", {})
        quotas = parse_quotas(os.environ.get("RCE_SENTINEL_QUOTA", ""))
        self.all: dict[str, list[ModelSpec]] = {
            tier: [ModelSpec(id=m["id"], provider=m["provider"], family=m["family"], tier=tier,
                             price_in=float(m["price_in"]), price_out=float(m["price_out"]),
                             price_cached_in=float(m.get("price_cached_in", m["price_in"])),
                             context_tokens=int(m.get("context_tokens", 128_000)),
                             quota_tokens=quotas.get(m["id"], m.get("quota_tokens")),
                             egress=self._egress(m), fallbacks=tuple(m.get("fallbacks", ())),
                             uses=tuple(m.get("uses", ("scan", "dev"))),
                             api=str(m.get("api", "chat")),
                             quality=_opt_float(m.get("quality")),
                             tps=_opt_float(m.get("tps")),
                             long_above=m.get("long_above"),
                             long_scale=float(m.get("long_scale", 1.0)))
                   for m in models_cfg.get("tiers", {}).get(tier, [])]
            for tier in TIERS
        }
        self._local_up: dict[str, set[str]] = {}
        self._reach: dict[str, str] = {}   # provider -> "" if reachable, else why not
        self._down: set[str] = set()
        self.down_status: dict[str, int] = {}
        self.local_only = False        # set from the routing profile: no egress models
        self.provider_sort = "price"   # OpenRouter provider.sort for this run
        self.apis: set[str] | None = None  # only these wire APIs; the proxy needs "chat"
        self.only: set[str] | None = None  # only these model ids; the benchmark report

    def _egress(self, m: dict[str, Any]) -> bool:
        """Cloud APIs, Ollama cloud models, and proxies that forward to the cloud."""
        if "egress" in m:
            return bool(m["egress"])
        p = self.providers.get(m["provider"], {})
        if "egress" in p:
            return bool(p["egress"])
        return not p.get("local") or str(m["id"]).endswith(("-cloud", ":cloud"))

    def base_url(self, provider: str, api: str = "chat") -> str:
        """Base URL for a provider; `base_urls` maps non-chat APIs to their own roots."""
        p = self.providers[provider]
        if api != "chat" and api in p.get("base_urls", {}):
            return str(p["base_urls"][api]).rstrip("/")
        return (os.environ.get(p.get("base_url_env", "")) or p["base_url"]).rstrip("/")

    def _local_models(self, provider: str) -> set[str]:
        """Model ids a local server can serve now, probed once per run.

        apfel: /health must report model_available. Every server must list the model in
        /v1/models, so apfel on Ollama's port is never mistaken for Ollama.
        """
        if provider not in self._local_up:
            p = self.providers[provider]
            base = self.base_url(provider)
            ids: set[str] = set()
            try:
                healthy = True
                if health := p.get("health"):
                    data = httpx.get(base.removesuffix("/v1") + health, timeout=0.5).json()
                    healthy = bool(data.get("model_available"))
                if healthy:
                    listing = httpx.get(base + "/models", timeout=0.5).json()
                    ids = {m.get("id", "") for m in listing.get("data", [])}
            except (httpx.HTTPError, ValueError, AttributeError):
                ids = set()
            self._local_up[provider] = ids
        return self._local_up[provider]

    def mark_down(self, spec: ModelSpec, status: int = 401) -> None:
        """Stop offering a model for the rest of the run (401 not signed in, 402 plan)."""
        self._down.add(spec.id)
        self.down_status[spec.id] = status

    def is_down(self, spec: ModelSpec) -> bool:
        return spec.id in self._down

    def is_real(self, spec: ModelSpec) -> bool:
        """A real endpoint can serve this model right now."""
        if spec.id in self._down:
            return False
        p = self.providers.get(spec.provider, {})
        if p.get("local"):
            return spec.id in self._local_models(spec.provider)
        if p.get("kind") == "claude-cli":   # your Claude Code login, no key
            from rce_sentinel.orchestrator.seat import claude_bin

            return claude_bin() is not None and not self.unreachable(spec.provider)
        return bool(os.environ.get(p.get("api_key_env", ""))) and not self.unreachable(
            spec.provider)

    def unreachable(self, provider: str) -> str:
        """Why a cloud provider cannot be reached from here ("" when it can).

        Probed once per run. A proxy block page (e.g. Zscaler's 403) or a failed connection
        means a key alone does not make the model usable. Skipped when offline.
        """
        if os.environ.get("RCE_SENTINEL_OFFLINE", "").lower() in {"1", "true", "on"}:
            return ""
        if provider not in self._reach:
            for attempt in range(2):   # one retry: inspecting proxies are slow at times
                try:
                    r = httpx.get(self.base_url(provider) + "/models", timeout=6.0)
                    server = r.headers.get("server", "").lower()
                    blocked = r.status_code == 403 and "zscaler" in server
                    self._reach[provider] = (
                        f"blocked by network proxy ({r.headers.get('server')})"
                        if blocked else "")
                    break
                except httpx.HTTPError as exc:
                    self._reach[provider] = f"unreachable ({type(exc).__name__})"
                    if not isinstance(exc, httpx.TimeoutException) or attempt:
                        break
        return self._reach[provider]

    def is_available(self, spec: ModelSpec) -> bool:
        if spec.id in self._down or (self.local_only and spec.egress):
            return False
        if self.mode in {"mock", "hybrid"}:
            return True
        if self.mode == "off":
            return False
        return self.is_real(spec)

    def simulated(self, spec: ModelSpec) -> bool:
        return self.mode == "mock" or (self.mode == "hybrid" and not self.is_real(spec))

    def counts_as_evidence(self, spec: ModelSpec) -> bool:
        """In hybrid mode a mock stands in for a model that is down, so its opinion is not
        evidence. In pure mock mode (offline tests and demos) every model is a mock by design."""
        return not (self.mode == "hybrid" and self.simulated(spec))

    def family_of(self, model_id: str) -> str | None:
        """Family of a configured model, matched loosely ('gemma4:31b' ~ 'gemma4:31b-cloud')."""
        specs = [m for tier in TIERS for m in self.all[tier]]
        for m in specs:
            if m.id == model_id:
                return m.family
        for m in specs:
            if m.id.startswith(model_id) or model_id.startswith(m.id):
                return m.family
        return None

    def label(self, spec: ModelSpec) -> str:
        return f"{spec.id} (mock)" if self.simulated(spec) else spec.id

    def available(self, tier: str, use: str = "scan") -> list[ModelSpec]:
        return [m for m in self.all.get(tier, []) if use in m.uses
                and (self.apis is None or m.api in self.apis)
                and (self.only is None or m.id in self.only) and self.is_available(m)]

    def any_available(self) -> bool:
        return any(self.available(t) for t in TIERS)

    def baseline(self) -> ModelSpec:
        target = self.cfg.get("baseline_model")
        for tier in reversed(TIERS):
            for m in self.all[tier]:
                if m.id == target:
                    return m
        return self.all["tier3"][0]


class OpenAICompatProvider:
    RETRY_STATUS = frozenset({429, 503})  # rate limited, or apfel still loading assets

    def __init__(self, registry: ModelRegistry, retries: int = 2):
        self.registry = registry
        self.retries = retries
        self._seat: Any = None

    @staticmethod
    def _response_format(provider: dict[str, Any], schema: dict[str, Any] | None
                         ) -> dict[str, Any]:
        """Constrained decoding when the provider supports it: the reply always has the
        right shape, which small local models otherwise get wrong."""
        if schema and provider.get("json_schema"):
            return {"type": "json_schema",
                    "json_schema": {"name": "reply", "strict": True, "schema": schema}}
        return {"type": "json_object"}

    def _post(self, url: str, body: dict[str, Any], headers: dict[str, str],
              timeout: float) -> httpx.Response:
        deadline = time.monotonic() + timeout
        for attempt in range(self.retries + 1):
            left = max(0.5, deadline - time.monotonic())
            resp = httpx.post(url, json=body, headers=headers, timeout=left,
                              follow_redirects=False)
            if resp.status_code not in self.RETRY_STATUS or attempt == self.retries:
                return resp
            time.sleep(min(1.0 * (attempt + 1), max(0.0, deadline - time.monotonic())))
        return resp

    def complete(self, spec: ModelSpec, system: str, user: str, *, max_tokens: int,
                 timeout: float, prefix_tokens: int = 0, prefix_warm: bool = False,
                 schema: dict[str, Any] | None = None, text: bool = False) -> LLMResult:
        p = self.registry.providers[spec.provider]
        if p.get("kind") == "claude-cli":
            if self._seat is None:
                from rce_sentinel.orchestrator.seat import ClaudeSeatProvider

                self._seat = ClaudeSeatProvider(p.get("max_budget_usd", 2.0))
            return self._seat.complete(spec, system, user, max_tokens=max_tokens,
                                       timeout=timeout, schema=schema, text=text)
        if spec.api != "chat":
            return self._native(spec, system, user, max_tokens, timeout, schema)
        headers = {}
        if key_env := p.get("api_key_env"):
            headers["Authorization"] = f"Bearer {os.environ.get(key_env, '')}"
        body: dict[str, Any] = {
            "model": spec.id,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if not text:  # code and prose replies must not be forced into JSON
            body["response_format"] = self._response_format(p, schema)
        if p.get("kind") == "openrouter":
            body.update(self.openrouter_routing(spec, p))
        url = self.registry.base_url(spec.provider) + "/chat/completions"
        t0 = time.monotonic()
        resp = self._post(url, body, headers, timeout)
        if resp.status_code == 400 and body.get("response_format", {}).get("type") == "json_schema":
            body["response_format"] = {"type": "json_object"}  # schema mode not supported here
            resp = self._post(url, body, headers, timeout)
        if resp.status_code == 400 and "response_format" in body:
            body.pop("response_format")
            resp = self._post(url, body, headers, timeout)
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage") or {}
        text = data["choices"][0]["message"].get("content") or ""
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or usage.get(
            "prompt_cache_hit_tokens", 0)
        reported = usage.get("cost")
        return LLMResult(
            text=text,
            tokens_in=int(usage.get("prompt_tokens") or approx_tokens(system + user)),
            tokens_out=int(usage.get("completion_tokens") or approx_tokens(text)),
            cached_tokens=int(cached or 0),
            latency_ms=(time.monotonic() - t0) * 1000,
            model=spec.id,
            cost=float(reported) if isinstance(reported, int | float) else None,
            served=str(data.get("model") or ""),
        )

    def _native(self, spec: ModelSpec, system: str, user: str, max_tokens: int,
                timeout: float, schema: dict[str, Any] | None) -> LLMResult:
        """Anthropic Messages, OpenAI Responses, and Gemini generateContent."""
        p = self.registry.providers[spec.provider]
        key = os.environ.get(p.get("api_key_env", ""), "")
        base = self.registry.base_url(spec.provider, spec.api)
        # Reasoning models spend output tokens thinking; leave room for the answer.
        room = max(max_tokens, 1024)
        if spec.api == "messages":
            url = base + "/messages"
            headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
            body: dict[str, Any] = {"model": spec.id, "max_tokens": max_tokens,
                                    "system": system,
                                    "messages": [{"role": "user", "content": user}]}
        elif spec.api == "responses":
            url = base + "/responses"
            headers = {"Authorization": f"Bearer {key}"}
            body = {"model": spec.id, "instructions": system, "input": user,
                    "max_output_tokens": room, "store": False}
            if schema:
                body["text"] = {"format": {"type": "json_schema", "name": "reply",
                                           "schema": schema, "strict": True}}
        elif spec.api == "gemini":
            url = f"{base}/models/{spec.id}:generateContent"
            headers = {"x-goog-api-key": key}
            body = {"systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generationConfig": {"maxOutputTokens": room, "temperature": 0}}
        else:
            raise ValueError(f"unknown api {spec.api!r} for {spec.id}")
        t0 = time.monotonic()
        resp = self._post(url, body, headers, timeout)
        resp.raise_for_status()
        data = resp.json()
        text, tin, tout, cached = _parse_native(spec.api, data)
        return LLMResult(text=text, tokens_in=tin or approx_tokens(system + user),
                         tokens_out=tout or approx_tokens(text), cached_tokens=cached,
                         latency_ms=(time.monotonic() - t0) * 1000, model=spec.id,
                         served=str(data.get("model") or data.get("modelVersion") or ""))

    def openrouter_routing(self, spec: ModelSpec, p: dict[str, Any]) -> dict[str, Any]:
        """OpenRouter does provider-level routing; our router keeps the correctness decisions.

        - provider.sort follows the run's profile: latency when someone waits, else price.
        - provider.max_price caps each request at the configured price plus a margin, so a
          fallback can never be silently more expensive (prices are USD per 1M tokens).
        - models: same-capability fallbacks tried server-side when the first one is down.
        - data_collection deny: skip providers that may train on or store the code.
        - usage.include: the reply reports the real cost, which the budget then uses.
        """
        margin = float(p.get("max_price_margin", 1.25))
        prefs: dict[str, Any] = {"sort": self.registry.provider_sort,
                                 "data_collection": p.get("data_collection", "deny"),
                                 "allow_fallbacks": True}
        if spec.price_in or spec.price_out:
            prefs["max_price"] = {"prompt": round(spec.price_in * margin, 4),
                                  "completion": round(spec.price_out * margin, 4)}
        out: dict[str, Any] = {"provider": prefs, "usage": {"include": True}}
        if spec.fallbacks:
            out["models"] = [spec.id, *spec.fallbacks]
        return out


def _opt_float(v: Any) -> float | None:
    return None if v is None else float(v)


def _parse_native(api: str, data: dict[str, Any]) -> tuple[str, int, int, int]:
    """(text, tokens in, tokens out, cached tokens) from a native API reply."""
    if api == "messages":
        u = data.get("usage") or {}
        text = "".join(b.get("text", "") for b in data.get("content", [])
                       if b.get("type") == "text")
        cached = int(u.get("cache_read_input_tokens") or 0)
        fresh = int(u.get("input_tokens") or 0) + int(u.get("cache_creation_input_tokens") or 0)
        return text, fresh + cached, int(u.get("output_tokens") or 0), cached
    if api == "responses":
        u = data.get("usage") or {}
        text = "".join(c.get("text", "") for item in data.get("output", [])
                       if item.get("type") == "message"
                       for c in item.get("content", []) if c.get("type") == "output_text")
        cached = int((u.get("input_tokens_details") or {}).get("cached_tokens") or 0)
        return text, int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0), cached
    u = data.get("usageMetadata") or {}
    parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
    text = "".join(x.get("text", "") for x in parts if not x.get("thought"))
    return (text, int(u.get("promptTokenCount") or 0),
            int(u.get("candidatesTokenCount") or 0) + int(u.get("thoughtsTokenCount") or 0),
            int(u.get("cachedContentTokenCount") or 0))


# --- Mock ---------------------------------------------------------------------

_FIELD = re.compile(r"^(TASK|ARG|SINK LINE|FILE|TAINT CHAIN|CLAIM):\s*(.*)$", re.M)
_SNIPPET_LINE = re.compile(r"^\s*(\d+)\| (.*)$", re.M)
_SAFE_HINTS = ("shlex.quote", "fullmatch", "in ALLOWED", "in allowed", "SafeLoader",
               "safe_load", "literal_eval")
_INJECTION = re.compile(r"ignore (all )?(previous|prior) instructions|mark (this|it) (as )?safe",
                        re.I)

_PROSE = {
    "vulnerable": ("An attacker who controls {src} can inject shell or code metacharacters. "
                   "The value flows unchanged into `{sink}` on line {line}, which executes it, "
                   "so the attacker runs arbitrary commands on the host."),
    "safe": ("The argument to `{sink}` on line {line} is constant or passes a strict "
             "allowlist before use, so attacker data cannot change what is executed."),
}
_CAVE = {
    "vulnerable": "{src} -> `{sink}` L{line}. no sanitize. attacker run cmd.",
    "safe": "`{sink}` L{line} arg const/allowlisted. no attacker path.",
}
_FIX_PROSE = ("Pass an argument list without a shell, validate the value against a strict "
              "allowlist, and never build the command or code string from request data.")
_FIX_CAVE = "argv list, no shell. allowlist input."


class MockProvider:
    """Deterministic offline stand-in that mimics tiered model behaviour.

    Tier 1 is deliberately weaker: lower confidence on ambiguous code, and it can be fooled
    by prompt injection in the snippet, which the evidence check must catch.
    """

    def __init__(self, script: dict[str, list[str]] | None = None):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.calls: list[tuple[str, str]] = []

    def complete(self, spec: ModelSpec, system: str, user: str, *, max_tokens: int,
                 timeout: float, prefix_tokens: int = 0, prefix_warm: bool = False,
                 schema: dict[str, Any] | None = None, text: bool = False) -> LLMResult:
        fields = {k: v.strip() for k, v in _FIELD.findall(user)}
        task = fields.get("TASK", "judge")
        self.calls.append((task, spec.id))
        caveman = CAVEMAN_MARKER in system
        queue = self.script.get(f"{task}:{spec.tier}") or self.script.get(task)
        if queue:
            text = queue.pop(0) if len(queue) > 1 else queue[0]
        else:
            text = getattr(self, f"_{task}", self._judge)(spec, fields, user, caveman)
        return LLMResult(
            text=text,
            tokens_in=approx_tokens(system + user),
            tokens_out=approx_tokens(text),
            cached_tokens=prefix_tokens if prefix_warm else 0,
            latency_ms=0.0,
            model=spec.id,
        )

    def _judge(self, spec: ModelSpec, f: dict[str, str], user: str, caveman: bool) -> str:
        line = int(f.get("SINK LINE", "0") or 0)
        lines = dict((int(n), t) for n, t in _SNIPPET_LINE.findall(user))
        quote = lines.get(line, "").strip()
        sink = re.sub(r"\(.*", "", quote).split("=")[-1].strip() or "sink"
        src = (f.get("TAINT CHAIN") or "the call argument").split(" -> ")[0]
        arg = f.get("ARG", "unknown")
        snippet = "\n".join(lines.values())
        if spec.tier == "tier1" and _INJECTION.search(snippet):
            return json.dumps({"verdict": "safe", "confidence": 0.99, "evidence": [
                {"file": f.get("FILE", ""), "line": line, "quote": "reviewed by security team"}],
                "exploit_path": "", "suggested_fix": ""})
        if arg == "tainted":
            verdict, conf = "vulnerable", {"tier1": 0.86, "tier2": 0.9}.get(spec.tier, 0.95)
        elif arg == "constant" or any(h in snippet for h in _SAFE_HINTS):
            verdict, conf = "safe", 0.62 if spec.tier == "tier1" else 0.93
        else:
            verdict, conf = "vulnerable", 0.62 if spec.tier == "tier1" else 0.82
        tmpl = (_CAVE if caveman else _PROSE)[verdict]
        return json.dumps({
            "verdict": verdict,
            "confidence": conf,
            "evidence": [{"file": f.get("FILE", ""), "line": line, "quote": quote}],
            "exploit_path": tmpl.format(src=src, sink=sink, line=line),
            "suggested_fix": (_FIX_CAVE if caveman else _FIX_PROSE) if verdict != "safe" else "",
        })

    def _verify(self, spec: ModelSpec, f: dict[str, str], user: str, caveman: bool) -> str:
        claim = f.get("CLAIM", "")
        agree = not (claim.startswith("safe") and f.get("ARG") == "tainted")
        reason = ("claim matches cited line" if caveman else
                  "The cited line exists and the described data flow matches the snippet.")
        if not agree:
            reason = ("tainted data reach sink. safe wrong." if caveman else
                      "Tainted data reaches the sink, so a safe verdict is not supported.")
        return json.dumps({"agree": agree, "confidence": 0.9, "reason": reason})

    def _merge(self, spec: ModelSpec, f: dict[str, str], user: str, caveman: bool) -> str:
        n = user.count("- ")
        summary = (f"{n} vuln verdicts. check shared input paths." if caveman else
                   f"There are {n} vulnerable verdicts across files; review shared input "
                   "handling because one entry point may reach several sinks.")
        return json.dumps({"summary": summary, "cross_file_risks": []})

    def _triage(self, spec: ModelSpec, f: dict[str, str], user: str, caveman: bool) -> str:
        rce = bool(re.search(r"code execution|command injection|deserial|RCE", user, re.I))
        pkgs = [{"ecosystem": e, "name": n}
                for e, n in re.findall(r"PACKAGE: (\S+) (\S+)", user)]
        return json.dumps({"rce": rce, "confidence": 0.85 if rce else 0.7, "packages": pkgs,
                           "reason": "rce keyword" if caveman else "Mentions code execution."})

    def _tag(self, spec: ModelSpec, f: dict[str, str], user: str, caveman: bool) -> str:
        from rce_sentinel.preeval.tags import rule_tags

        t = rule_tags(user.split("<<<", 1)[-1])
        return json.dumps({"task_type": t.top_task, "difficulty": round(t.level),
                           "confidence": 0.6, "multi_file": t.predicates["multi_file"] > 0.5,
                           "multi_target": t.predicates["multi_target"] > 0.5,
                           "long_output": t.predicates["long_output"] > 0.5})

    def _session(self, spec: ModelSpec, f: dict[str, str], user: str, caveman: bool) -> str:
        """A coder that follows only the decisions it can see in its input."""
        request = user.rsplit("REQUEST:", 1)[-1]
        seen = user.lower()

        def pick(needle: str, fallback: str) -> str:
            return needle if needle.lower() in seen else fallback

        op, eq = pick("#1E88E5", "#2196F3"), pick("#FF7043", "#4CAF50")
        label = pick("DEG/RAD", "RAD")
        hist = "20" if "last 20" in seen else "50"
        fact = pick("Error: n! needs a whole number", "Error")
        radius = "16" if "radius of 16" in seen or "radius 16" in seen else "8"
        engine = "calc_core" if "calc_core" in seen else "engine"
        decimal = "decimal" in seen
        num = "Decimal" if decimal else "number"
        files: dict[str, str] = {}
        if "write calc_core" in request.lower():
            files["calc_core/index.ts"] = (
                "import Decimal from 'decimal.js';\n"
                "export function add(a: string, b: string): string "
                "{ return new Decimal(a).plus(b).toString(); }\n"
                "export function factorial(n: string): string {\n"
                f"  if (!new Decimal(n).isInteger()) return '{fact}';\n  return n;\n}}\n"
                f"export const HISTORY_LIMIT = {hist};\n")
            files["calc_core/index.test.ts"] = "test('0.1+0.2', () => {});\n"
        ui = "write calc_core" not in request.lower()
        if ui and "react native" in request.lower():
            if engine != "calc_core":  # no memory of the engine: writes another one
                files["src/engine.ts"] = f"export function add(a: {num}, b: {num}) {{}}\n"
            files["App.tsx"] = (
                f"import {{ add }} from './{engine}';\n"
                f"const OP = '{op}'; const EQ = '{eq}'; const ANGLE = '{label}';\n"
                f"const styles = {{ key: {{ borderRadius: {radius} }} }};\n"
                f"// history keeps {hist}; math: {'decimal' if decimal else 'float'}\n")
        if ui and "flutter" in request.lower():
            files["lib/main.dart"] = (
                f"// uses {engine} rules; {'decimal' if decimal else 'double'} math\n"
                f"const op = Color(0xFF{op[1:]}); const eq = Color(0xFF{eq[1:]});\n"
                f"const angle = '{label}'; const historyLimit = {hist};\n"
                f"final radius = BorderRadius.circular({radius});\n")
        if ui and "swiftui" in request.lower():
            files["ios/CalculatorView.swift"] = (
                f"// {engine} port; {'Decimal' if decimal else 'Double'} math\n"
                f"let op = Color(hex: \"{op}\"); let eq = Color(hex: \"{eq}\")\n"
                f"let angle = \"{label}\"; let historyLimit = {hist}\n"
                f".cornerRadius({radius}) // factorial: {fact}\n")
        blocks = "".join(f"```ts path={p}\n{b}```\n" for p, b in files.items())
        decisions = [d for d, on in (
            ("engine module calc_core", engine == "calc_core"),
            ("decimal math via decimal.js", decimal), (f"operator keys {op}", True),
            (f"equals key {eq}", True), (f"toggle label {label}", True),
            (f"history keeps last {hist}", True), (f"factorial error: {fact}", True),
            (f"button radius {radius}", True)) if on]
        state = {"decisions": decisions, "plan": ["engine", "react native", "flutter",
                                                   "swiftui"],
                 "done": [p for p in files], "next": "next platform", "open_errors": []}
        return blocks + "```state\n" + json.dumps(state) + "\n```\n"

    def _bench(self, spec: ModelSpec, f: dict[str, str], user: str, caveman: bool) -> str:
        """Offline stand-in: echoes the request, so only trivial checks pass."""
        return f"```text\n{user.rsplit('REQUEST:', 1)[-1].strip()}\n```\n"

    def _grade(self, spec: ModelSpec, f: dict[str, str], user: str, caveman: bool) -> str:
        return json.dumps({"passed": True, "score": 6, "reason": "mock judge"})

    def _plan(self, spec: ModelSpec, f: dict[str, str], user: str, caveman: bool) -> str:
        return json.dumps({
            "steps": ["bump to fixed version", "replace unsafe call with safe variant",
                      "add input validation at entry point"],
            "waf_hint": "block request bodies carrying language object tags",
            "notes": "proposal only" if caveman else "This is a proposal for human review.",
        })


class HybridProvider:
    """Real endpoint when the model is up, deterministic mock otherwise (labelled)."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.real = OpenAICompatProvider(registry)
        self.mock = MockProvider()

    def complete(self, spec: ModelSpec, system: str, user: str, *, max_tokens: int,
                 timeout: float, prefix_tokens: int = 0, prefix_warm: bool = False,
                 schema: dict[str, Any] | None = None, text: bool = False) -> LLMResult:
        target = self.mock if self.registry.simulated(spec) else self.real
        return target.complete(spec, system, user, max_tokens=max_tokens, timeout=timeout,
                               prefix_tokens=prefix_tokens, prefix_warm=prefix_warm,
                               schema=schema, text=text)


def make_provider(registry: ModelRegistry) -> Provider:
    if registry.mode == "mock":
        return MockProvider()
    if registry.mode == "hybrid":
        return HybridProvider(registry)
    return OpenAICompatProvider(registry)
