"""rce-sentinel console: prompt the models, watch Smart Switching decide, inspect model usage.

Run:  streamlit run ui/app.py   (from tools/rce-sentinel, after `pip install -e '.[ui]'`)
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

TOOL = Path(__file__).resolve().parents[1]
STATE = TOOL / ".rce-sentinel-ui"
CFG = STATE / "config"
MODELS_FILE = CFG / "models.yaml"
PROMPTS = TOOL / "demo" / "prompts"
TIERS = ("tier1", "tier2", "tier3")
PROFILES = ("pre-commit", "interactive", "post-commit", "background")

st.set_page_config(page_title="rce-sentinel", page_icon="🛡️", layout="wide")


# --- config: the UI edits its own copy of models.yaml, never the packaged one ---------------
def models_cfg() -> dict:
    CFG.mkdir(parents=True, exist_ok=True)
    if not MODELS_FILE.exists():
        shutil.copy(TOOL / "config" / "models.yaml", MODELS_FILE)
    cfg = yaml.safe_load(MODELS_FILE.read_text())
    # Bring in providers and models added to the packaged config since the copy was
    # made, except models removed here on purpose.
    packaged = yaml.safe_load((TOOL / "config" / "models.yaml").read_text())
    removed = set(cfg.get("ui_removed", []))
    have = {m["id"] for t in TIERS for m in cfg["tiers"].get(t, [])}
    changed = False
    for name, prov in packaged.get("providers", {}).items():
        if cfg["providers"].get(name) != prov:
            cfg["providers"][name] = prov
            changed = True
    shipped = {m["id"]: m for t in TIERS for m in packaged["tiers"].get(t, [])}
    for tier in TIERS:
        # Packaged models follow the packaged entry; models added here stay as added.
        for i, m in enumerate(cfg["tiers"].get(tier, [])):
            if m["id"] in shipped and m != shipped[m["id"]]:
                cfg["tiers"][tier][i] = shipped[m["id"]]
                changed = True
        for m in packaged["tiers"].get(tier, []):
            if m["id"] not in have and m["id"] not in removed:
                cfg["tiers"].setdefault(tier, []).append(m)
                changed = True
    for key in ("decision", "bench"):
        if cfg.get(key) != packaged.get(key):
            cfg[key] = packaged.get(key, [] if key == "decision" else {})
            changed = True
    if changed:
        save_models(cfg)
    return cfg


def save_models(cfg: dict) -> None:
    MODELS_FILE.write_text(yaml.safe_dump(cfg, sort_keys=False))


def apply_env(provider_mode: str, profile: str, quota: str, local_only: bool,
              min_samples: int, urls: dict[str, str]) -> None:
    os.environ["RCE_SENTINEL_CONFIG_DIR"] = str(CFG)
    os.environ["RCE_SENTINEL_PROVIDER"] = provider_mode
    os.environ["RCE_SENTINEL_PROFILE"] = profile
    os.environ["RCE_SENTINEL_QUOTA"] = quota
    os.environ["RCE_SENTINEL_LOCAL_ONLY"] = "1" if local_only else "0"
    os.environ["RCE_SENTINEL_MIN_SAMPLES"] = str(min_samples)
    os.environ["RCE_SENTINEL_TRACE_IO"] = "1"
    for k, v in urls.items():
        if v:
            os.environ[k] = v


def scanner():
    from rce_sentinel.config import load_settings
    from rce_sentinel.engine import Scanner
    from rce_sentinel.trace import workspace

    return Scanner(load_settings(workspace(STATE)))


# --- sidebar --------------------------------------------------------------------------------
cfg = models_cfg()
sb = st.sidebar
sb.title("🛡️ rce-sentinel")
sb.caption("RCE gate + Smart Switching")

sb.subheader("Routing")
profile = sb.selectbox("Profile (how much waiting costs)", PROFILES, index=0,
                       help="pre-commit ≈ $50/h of developer time, 10 s cap; "
                            "background = nobody waits")
provider_mode = sb.selectbox("Model mode", ("auto", "hybrid", "mock", "off"), index=0,
                             help="auto: real models only · hybrid: labelled mocks for "
                                  "models that are down · off: rules only")
quota = sb.text_input("Token quotas per run", placeholder="apple-foundationmodel=1500")
local_only = sb.toggle("Local only (no code leaves the machine)")
min_samples = sb.slider("Trust a model's own numbers after N real calls", 1, 20, 8)

with sb.expander("API keys"):
    env_path = st.text_input("Env file (KEY=VALUE lines)",
                             os.environ.get("RCE_SENTINEL_ENV_FILE", ""),
                             placeholder="~/path/to/.env")
    if env_path:
        from rce_sentinel.config import load_env_file

        try:
            names = load_env_file(env_path)
            st.caption(f"loaded: {', '.join(names) or 'already set'}")
        except OSError as exc:
            st.error(f"cannot read env file: {exc.strerror}")
    for key in ("OPENCODE_API_KEY", "OPENROUTER_API_KEY"):
        st.caption(f"{key}: {'set' if os.environ.get(key) else 'not set'}")

with sb.expander("Endpoints"):
    urls = {
        "APFEL_BASE_URL": st.text_input("apfel", os.environ.get("APFEL_BASE_URL",
                                                                 "http://127.0.0.1:11434/v1")),
        "OLLAMA_BASE_URL": st.text_input("Ollama", os.environ.get("OLLAMA_BASE_URL",
                                                                   "http://127.0.0.1:11436/v1")),
        "SWITCHYARD_BASE_URL": st.text_input("Switchyard", os.environ.get(
            "SWITCHYARD_BASE_URL", "http://127.0.0.1:4000/v1")),
    }

sb.subheader("➕ Add a model")
with sb.form("add_model", clear_on_submit=True):
    mid = st.text_input("Model id", placeholder="llama3.2:3b or switchyard-stage")
    prov = st.selectbox("Provider", list(cfg.get("providers", {})))
    fam = st.text_input("Family (for cross-family second opinions)", placeholder="llama")
    tier = st.selectbox("Tier", TIERS)
    c1, c2 = st.columns(2)
    pin = c1.number_input("$ / 1M in", 0.0, 100.0, 0.0, 0.01)
    pout = c2.number_input("$ / 1M out", 0.0, 200.0, 0.0, 0.01)
    ctx = st.number_input("Context window (tokens)", 512, 2_000_000, 128_000, 512)
    egress = st.checkbox("Prompts leave the machine (cloud)")
    if st.form_submit_button("Add model") and mid and fam:
        entry = {"id": mid, "provider": prov, "family": fam, "price_in": pin,
                 "price_out": pout, "price_cached_in": pin, "context_tokens": int(ctx),
                 "egress": egress}
        cfg["tiers"].setdefault(tier, []).append(entry)
        save_models(cfg)
        st.success(f"added {mid} to {tier}")
        st.rerun()

with sb.expander("Remove a model"):
    all_ids = [f"{t}: {m['id']}" for t in TIERS for m in cfg["tiers"].get(t, [])]
    gone = st.selectbox("Model", all_ids)
    if st.button("Remove") and gone:
        t, m = gone.split(": ", 1)
        cfg["tiers"][t] = [x for x in cfg["tiers"][t] if x["id"] != m]
        cfg.setdefault("ui_removed", []).append(m)
        save_models(cfg)
        st.rerun()
    if st.button("Reset to packaged models.yaml"):
        MODELS_FILE.unlink(missing_ok=True)
        st.rerun()

apply_env(provider_mode, profile, quota, local_only, min_samples, urls)

# --- main -----------------------------------------------------------------------------------
st.title("rce-sentinel")
st.caption("Feature 1 finds RCE paths with rules and taint (zero tokens). Feature 2, Smart "
           "Switching, decides which model reviews what: by expected cost (price + waiting "
           "time + measured redo rate), token windows, quotas, answer quality, and access "
           "failures.")

tab_run, tab_switch, tab_models, tab_pre, tab_hand = st.tabs(
    ["① Prompt the models", "② Did it switch?", "③ Models: capabilities & usage",
     "④ Pre-eval: tag & route", "⑤ Switch test: context pack"])

with tab_run:
    presets = {p.name: p.read_text() for p in sorted(PROMPTS.glob("*")) if p.is_file()}
    left, right = st.columns([3, 2])
    pick = left.selectbox("Start from a demo prompt", ["(write your own)", *presets], key="preset")
    code = left.text_area("Code to review", presets.get(pick, ""), height=300,
                          placeholder="paste Python, JavaScript/TypeScript, or Go")
    lang = right.selectbox("Language", ("auto", "py", "js", "ts", "go"))
    req = right.text_input("Pinned dependencies (comma separated)",
                           placeholder="PyYAML==5.3")
    fresh = right.checkbox("Ignore cached verdicts (force model calls)", value=True)
    right.info(f"Profile **{profile}** · mode **{provider_mode}**"
               + (" · **local only**" if local_only else "")
               + (f" · quota `{quota}`" if quota else ""))
    go = right.button("Run review", type="primary", width="stretch",
                      disabled=not code.strip())

    if go:
        from rce_sentinel.engine import FileSet
        from rce_sentinel.trace import EXT, detect_lang

        lg = detect_lang(code) if lang == "auto" else lang
        files = {f"snippet{EXT.get(lg, '.py')}": code}
        deps = [d.strip() for d in req.split(",") if d.strip()]
        if deps:
            files["requirements.txt"] = "\n".join(deps) + "\n"
        sc = scanner()
        if fresh:
            sc.store.cache_clear()
        lines: list[str] = []
        with st.status("Reviewing…", expanded=True) as status:
            box = st.empty()
            t0 = time.monotonic()
            ctx = get_script_run_ctx()
            lock = threading.Lock()

            def emit(msg: str) -> None:
                # The scanner judges files on worker threads; give them this session.
                add_script_run_ctx(threading.current_thread(), ctx)
                with lock:
                    lines.append(f"[{time.monotonic() - t0:6.1f}s] {msg}")
                    box.code("\n".join(lines[-40:]), language="text")

            sc.emit = emit
            rep = sc.scan(FileSet(files), "trace")
            status.update(label=f"Done in {time.monotonic() - t0:.1f}s", state="complete")
        st.session_state["last"] = {"rep": rep, "router": sc.last_router, "log": lines,
                                    "store": sc.store}

    last = st.session_state.get("last")
    if last:
        rep = last["rep"]
        bad = [f for f in rep.findings if f.risk and f.risk.label in ("critical", "high")]
        s = rep.stats
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Verdict", "BLOCK" if bad else "allow")
        m2.metric("Findings", len(rep.findings))
        m3.metric("Model calls", s.llm_calls)
        m4.metric("Cost", f"${s.cost_usd:.5f}")
        for f in rep.findings:
            label = f.risk.label.upper() if f.risk else "?"
            with st.expander(f"{label} · {f.rule_id} · line {f.line} · {f.arg_kind}",
                             expanded=label in ("CRITICAL", "HIGH")):
                if f.taint_chain:
                    st.markdown("**Data flow:** " + " → ".join(f"`{c}`" for c in f.taint_chain))
                st.markdown(f"**How an attacker gets RCE:** {f.attacker_path}")
                st.markdown(f"**Fix:** {f.fix}")
                st.caption(f"decided by: {f.decided_by}")

with tab_switch:
    last = st.session_state.get("last")
    if not last:
        st.info("Run a review first.")
    else:
        rep, router, store = last["rep"], last["router"], last["store"]
        if router is not None:
            st.subheader("Route plan")
            prof = router.profile
            st.caption(f"{prof.name} profile · waiting worth ${prof.value_per_second * 3600:.0f}/h"
                       f" · paid pacing ×{router.scorer.pacing:.1f} · "
                       f"${router.scorer.spent:.2f} of ${router.scorer.monthly_usd:.2f} "
                       "spent this month")
            if router.plans:
                for plan in router.plans:
                    st.code(plan, language="text")
            else:
                st.write("Not enough measured calls yet, so the plain cheapest-first cascade "
                         "ran from tier 1. Lower the N slider or run a few reviews.")
        rows = store.execute(
            "SELECT task_type AS task, tier, model, simulated, outcome, tokens_in, tokens_out, "
            "latency_ms, cost FROM llm_calls WHERE run_id=? AND shadow=0 ORDER BY id",
            (rep.run_id,))
        df = pd.DataFrame([dict(r) for r in rows])
        st.subheader("Calls, in order")
        if df.empty:
            st.write("No model was called (rules decided, or a signed cache hit).")
        else:
            df.insert(0, "#", range(1, len(df) + 1))
            df["real"] = df.pop("simulated").map({0: "REAL", 1: "mock"})
            df["latency_s"] = (df.pop("latency_ms") / 1000).round(1)
            st.dataframe(df, hide_index=True, width="stretch")
            moves = df["tier"].ne(df["tier"].shift()) | df["model"].ne(df["model"].shift())
            st.caption(f"{int(moves.sum()) - 1} model change(s) in this run")
        switches = [d for d in rep.degraded if "fallback" in d]
        notes = [step for f in rep.findings if f.internal_notes
                 for step in f.internal_notes.split(" | ") if not step.startswith("model:")]
        st.subheader("Why it switched")
        if not switches and not notes:
            st.write("No switch was needed.")
        for d in switches:
            st.warning(d)
        for n in notes:
            icon = "⏭️" if "skipped" in n else "⬆️" if "escalated" in n else "✅" \
                if "accepted" in n else "•"
            st.markdown(f"{icon} {n}")
        if router is not None and router.io:
            st.subheader("Exact prompts and replies")
            for i, c in enumerate(router.io, 1):
                with st.expander(f"{i}. {c['task']} → {c['model']} "
                                 f"({c['tokens_in']} in / {c['tokens_out']} out)"):
                    body = c["user"]
                    st.code(body[body.find("TASK:"):] if "TASK:" in body else body,
                            language="text")
                    st.code(c["reply"], language="json")

with tab_models:
    from rce_sentinel.orchestrator.llm import ModelRegistry
    from rce_sentinel.orchestrator.scoring import Scorer, profile_for

    sc = scanner()
    reg = ModelRegistry(sc.settings.models, "auto")
    prof = profile_for(sc.settings.policy, "trace")
    scorer = Scorer(sc.store, sc.settings.policy, prof)
    usage = {r["model"]: dict(r) for r in sc.store.execute(
        "SELECT model, COUNT(*) AS calls, SUM(tokens_in) AS tok_in, SUM(tokens_out) AS tok_out, "
        "SUM(cost) AS usd FROM llm_calls WHERE simulated=0 AND shadow=0 GROUP BY model")}
    out = []
    for t in TIERS:
        for m in reg.all[t]:
            stt = scorer.stats(m, "judge")
            u = usage.get(m.id, {})
            down = reg.down_status.get(m.id)
            out.append({
                "tier": t, "model": m.id, "provider": m.provider, "family": m.family,
                "status": "🟢 up" if reg.is_real(m) else (f"🔴 HTTP {down}" if down
                                                          else "⚪ not reachable / no key"),
                "window": m.context_tokens, "$/1M in": m.price_in, "$/1M out": m.price_out,
                "cloud": "yes" if m.egress else "local",
                "calls": u.get("calls", 0), "tokens": (u.get("tok_in") or 0)
                + (u.get("tok_out") or 0), "spent $": round(u.get("usd") or 0.0, 5),
                "p50 s": round(stt.seconds, 1), "escalates": f"{stt.p_escalate:.0%}",
                "basis": f"measured n={stt.n}" if stt.measured else f"prior (n={stt.n})",
            })
    st.subheader("Configured models")
    st.dataframe(pd.DataFrame(out), hide_index=True, width="stretch")
    st.caption("Status is probed live. p50 and escalation rate are this repo's real judge "
               "calls; 'prior' means fewer than N calls so far.")
    st.subheader("Usage by tier (real calls)")
    tiers = pd.DataFrame([dict(r) for r in sc.store.execute(
        "SELECT tier, COUNT(*) AS calls, SUM(tokens_in + tokens_out) AS tokens, "
        "ROUND(SUM(cost), 5) AS usd FROM llm_calls WHERE simulated=0 AND shadow=0 "
        "GROUP BY tier ORDER BY tier")])
    if tiers.empty:
        st.write("No real calls yet.")
    else:
        c1, c2 = st.columns(2)
        c1.bar_chart(tiers, x="tier", y="calls")
        c2.bar_chart(tiers, x="tier", y="tokens")
    st.caption(f"Model config edited here lives in {MODELS_FILE.relative_to(TOOL)}; the "
               "packaged config/models.yaml is never changed.")

with tab_pre:
    from rce_sentinel.orchestrator.llm import ModelRegistry, make_provider
    from rce_sentinel.orchestrator.scoring import DEFAULT_PROFILES, Profile, profile_for
    from rce_sentinel.preeval.pipeline import pre_eval
    from rce_sentinel.preeval.select import p_fail_prior, quality_of, select_model
    from rce_sentinel.preeval.tags import DIFFICULTY

    EXAMPLES = {
        "Benchmark: scientific calculator (3 frameworks)": (
            "Build a scientific calculator app with the best possible and cleanest UI for "
            "iOS and Android. It needs to be in React Native, Swift, and Flutter."),
        "Debug": "why does the login test fail with a KeyError?",
        "Refactor": "refactor the auth module to remove duplicated token parsing",
        "Chore": "fix typo in README",
        "Vague (rules unsure)": "hmm the thing on the page looks off sometimes",
    }
    st.caption("Tags a developer prompt before any model works on it, then picks the model "
               "by expected cost. Free rules first; Jev (decision model) only when rules are "
               "unsure; a local model when offline. The sidebar profile sets how much "
               "waiting costs.")
    pick = st.selectbox("Example", list(EXAMPLES))
    prompt = st.text_area("Prompt", EXAMPLES[pick], height=110, key=f"pre_{pick}")
    sc = scanner()
    settings = sc.settings
    if "decision" not in settings.models:  # UI copy predates the Jev entry
        settings.models["decision"] = yaml.safe_load(
            (TOOL / "config" / "models.yaml").read_text()).get("decision", {})
    ids = ["(none)"] + [m["id"] for t in TIERS for m in settings.models["tiers"].get(t, [])]
    c1, c2, c3 = st.columns(3)
    current = c1.selectbox("Session is on (warm cache)", ids)
    use_jev = c2.toggle("Allow Jev", True,
                        help="needs OPENROUTER_API_KEY; never used offline or local-only")
    session = c3.text_input("Session id", "ui")
    if st.button("Tag & route", type="primary") and prompt.strip():
        reg = ModelRegistry(settings.models, settings.provider_mode)
        with st.spinner("tagging…"):
            res = pre_eval(prompt, settings, registry=reg, provider=make_provider(reg),
                           store=sc.store, use_jev=use_jev,
                           current=None if current == "(none)" else current,
                           session_id=session)
        st.session_state["pre"] = {"res": res, "prompt": prompt, "reg": reg,
                                   "current": None if current == "(none)" else current}

    pre = st.session_state.get("pre")
    if pre:
        res, tags, sel = pre["res"], pre["res"].tags, pre["res"].selection
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Task", tags.top_task, f"{tags.task[tags.top_task]:.0%}")
        m2.metric("Difficulty", DIFFICULTY[round(tags.level)], f"level {tags.level:.2f}")
        m3.metric("Tagged by", tags.source, f"conf {tags.confidence:.2f}")
        m4.metric("Model", sel.model.id if sel.model else "none", sel.start_tier)
        m5.metric("Expected cost", f"${sel.expected_cost:.3f}",
                  f"{sel.tokens_in} in / {sel.tokens_out} out", delta_color="off")
        for n in res.notes:
            st.info(n)
        if tags.subtasks:
            st.markdown("**Split into:** " + " · ".join(tags.subtasks))

        g1, g2, g3 = st.columns(3)
        g1.caption("P(task type)")
        g1.bar_chart(pd.DataFrame({"p": tags.task}), horizontal=True)
        g2.caption("P(difficulty)")
        g2.bar_chart(pd.DataFrame({"p": dict(zip(DIFFICULTY, tags.difficulty, strict=True))}),
                     horizontal=True)
        g3.caption("Predicates")
        g3.bar_chart(pd.DataFrame({"p": tags.predicates}), horizontal=True)

        st.subheader("Why this model")
        rows = []
        for t in TIERS:
            s = sel.scores.get(t)
            if s:
                rows.append({"": "→" if t == sel.start_tier else "", "tier": t,
                             "best model": s.model.id,
                             "quality": quality_of(s.model),
                             "tok/s": s.model.tps or "tier default",
                             "total $": round(s.total, 3),
                             "price $": round(s.price, 4), "wait $": round(s.wait, 3),
                             "redo $": round(s.escalate, 3), "risk $": round(s.risk, 3),
                             "P(fail)": f"{s.stats.p_escalate:.0%}",
                             "est. s": round(s.stats.seconds, 1),
                             "basis": f"measured n={s.stats.n}" if s.stats.measured
                             else "prior"})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("total = price + wait (value of the developer's time) + redo "
                       "(P(fail) × cost of the next tier) + risk. A long task a cheap model "
                       "would likely fail moves up, because a retry repeats the whole wait.")
        else:
            st.warning("No model available. Check the sidebar mode and endpoints.")
        for s in sel.skipped:
            st.caption(f"skipped {s}")

        st.subheader("Same tags, every profile")
        what_if = []
        base = profile_for(settings.policy, "scan")
        for name, raw in DEFAULT_PROFILES.items():
            prof = Profile(name=name, value_per_second=raw["value_per_second"],
                           latency_cap_s=raw.get("latency_cap_s"),
                           error_cost_usd=raw["error_cost_usd"], local_only=base.local_only)
            alt = select_model(pre["prompt"], tags, pre["reg"], sc.store, settings.policy,
                               prof, pre["current"])
            what_if.append({"profile": name,
                            "waiting $/h": round(raw["value_per_second"] * 3600),
                            "model": alt.model.id if alt.model else "none",
                            "tier": alt.start_tier,
                            "expected $": round(alt.expected_cost, 3)})
        st.dataframe(pd.DataFrame(what_if), hide_index=True, width="stretch")
        st.caption("quality and tok/s are hand-set priors per model (models.yaml) until real "
                   "outcomes exist. P(fail) at each tier's default quality: "
                   + ", ".join(f"{t} {p_fail_prior(t, tags):.0%}" for t in TIERS))

    st.subheader("Tag history")
    hist = sc.store.execute(
        "SELECT datetime(ts, 'unixepoch', 'localtime') AS time, session_id AS session, "
        "task_type AS task, level, source, model, tier, est_tokens_in AS tok_in, "
        "est_tokens_out AS tok_out, est_cost AS 'expected $', outcome FROM prompt_tags "
        "ORDER BY id DESC LIMIT 25")
    if hist:
        st.dataframe(pd.DataFrame([dict(r) for r in hist]), hide_index=True, width="stretch")
        st.caption("Prompts are stored as a hash only. `outcome` fills in once test/build "
                   "results are captured.")
    else:
        st.write("No prompts tagged yet.")


with tab_hand:
    from rce_sentinel.handoff.session import MODES, SCENARIOS, Scenario, SessionRunner
    from rce_sentinel.orchestrator.llm import ModelRegistry, make_provider

    st.caption("A multi-turn build where each turn runs on a different model. Turn 1 fixes "
               "the decisions; later turns never repeat them, so the incoming model knows "
               "them only through the handoff. Canaries check the files written after "
               "turn 1. Generated code is written to a sandbox and never run.")
    names = sorted(p.stem for p in SCENARIOS.glob("*.yaml"))
    sc = Scenario.load(st.selectbox("Scenario", names))
    with st.expander("Turns and canaries"):
        for i, t in enumerate(sc.turns, 1):
            st.markdown(f"**Turn {i}**")
            st.code(t, language="text")
        st.table(pd.DataFrame([{"canary": c.id, "checks": c.desc} for c in sc.canaries]))
    sc_h = scanner()
    hs = sc_h.settings
    all_ids = ["auto (pre-eval picks)"] + [m["id"] for t in TIERS
                                           for m in hs.models["tiers"].get(t, [])]
    cols = st.columns(len(sc.turns))
    chosen = []
    for i, col in enumerate(cols):
        default = sc.models[i] if i < len(sc.models) and sc.models[i] in all_ids else all_ids[0]
        chosen.append(col.selectbox(f"Turn {i + 1} model", all_ids, all_ids.index(default),
                                    key=f"hand_m{i}"))
    c1, c2, c3 = st.columns(3)
    modes = c1.multiselect("Handoff modes", MODES, list(MODES))
    keep_last = c2.number_input("Pack: also pass last N turns verbatim", 0, 3, 0)
    max_usd = c3.number_input("Stop a session past $", 0.1, 50.0, 2.0, 0.5)
    auto = any(c.startswith("auto") for c in chosen)
    models = [] if auto else chosen
    if provider_mode != "mock":
        st.warning(f"Model mode is **{provider_mode}**: this makes real paid calls, up to "
                   f"${max_usd:.2f} per mode ({len(modes)} mode(s)). Use mock to rehearse.")
    if st.button("Run switch test", type="primary") and modes:
        reg = ModelRegistry(hs.models, hs.provider_mode)
        runner = SessionRunner(hs, reg, make_provider(reg), sc_h.store, keep_last=keep_last,
                               max_usd=max_usd)
        done = []
        bar = st.progress(0.0, "starting")
        for k, mode in enumerate(modes):
            bar.progress(k / len(modes), f"running {mode}…")
            done.append(runner.run(sc, mode, models))
        bar.progress(1.0, "done")
        st.session_state["hand"] = done

    for_hand = st.session_state.get("hand")
    if for_hand:
        st.subheader("Result")
        summ = pd.DataFrame([r.summary() for r in for_hand])
        view = summ[["mode", "retention", "handoff_tokens", "tokens_in", "tokens_out",
                     "cost_usd", "switches", "rewrites", "failed_turns"]]
        st.dataframe(view, hide_index=True, width="stretch")
        g1, g2, g3 = st.columns(3)
        g1.caption("Decisions kept after switching (higher is better)")
        g1.bar_chart(view, x="mode", y="retention")
        g2.caption("Tokens the incoming models read at switches (lower is better)")
        g2.bar_chart(view, x="mode", y="handoff_tokens")
        g3.caption("Session cost $")
        g3.bar_chart(view, x="mode", y="cost_usd")
        st.subheader("Canaries")
        st.dataframe(pd.DataFrame({r.mode: {k: "✅" if v else "❌"
                                            for k, v in r.canaries.items()}
                                   for r in for_hand}), width="stretch")
        for r in for_hand:
            with st.expander(f"{r.mode}: turns, files, context pack"):
                st.dataframe(pd.DataFrame([{
                    "turn": t.turn, "model": t.model, "switched from": t.switched_from or "",
                    "why": t.reason, "handoff tok": t.context_tokens, "in": t.tokens_in,
                    "out": t.tokens_out, "cached": t.cached, "$": round(t.cost, 5),
                    "s": round(t.seconds, 1), "ok": "✅" if t.ok else f"❌ {t.error}",
                    "files": ", ".join(t.files), "rewrote": ", ".join(t.rewrites)}
                    for t in r.turns]), hide_index=True, width="stretch")
                if r.mode == "pack" and r.pack is not None:
                    st.markdown("**What the next model reads (context pack)**")
                    st.code(r.pack.render(), language="text")
                    st.caption(f"{r.pack.tokens()} tokens · signed by the memory shield · "
                               f"{r.pack.dropped} injection attempt(s) dropped · "
                               "[reported] = not yet seen in the code")
                files = sorted(p for p in r.workspace.rglob("*")
                               if p.is_file() and p.name != "context_pack.json")
                if files:
                    pick = st.selectbox("File", [str(p.relative_to(r.workspace))
                                                 for p in files], key=f"f_{r.run_id}")
                    st.code((r.workspace / pick).read_text(errors="replace"))
                st.caption(f"sandbox: {r.workspace}")
