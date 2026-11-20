"""rce-sentinel command line."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from rce_sentinel import gitio, hooks, report
from rce_sentinel.config import load_settings
from rce_sentinel.engine import Scanner, collect_staged, collect_tree


def _root(path: str | None) -> tuple[Path, bool]:
    start = Path(path or ".").resolve()
    try:
        return gitio.repo_root(start), True
    except RuntimeError:
        return start, False


def cmd_install_hooks(args: argparse.Namespace) -> int:
    root, is_git = _root(args.repo)
    if not is_git:
        print("not a git repository", file=sys.stderr)
        return 2
    for line in hooks.install(root):
        print(line)
    return 0


def _run(root: Path, is_git: bool, mode: str, as_json: bool, live: bool = False) -> int:
    settings = load_settings(root)
    scanner = Scanner(settings)
    if live:
        from rce_sentinel.trace import live_printer

        scanner.emit = live_printer(sys.stderr)
    budget = None
    if mode == "pre-commit":
        files = collect_staged(root)
        budget = float(settings.policy.get("pre_commit_budget_seconds", 10))
    else:
        files = collect_tree(root, use_git=is_git)
    commit = gitio.head_sha(root) if is_git else ""
    rep = scanner.scan(files, mode, wall_seconds=budget, commit=commit)
    jpath, mpath = report.write(rep, settings.state_dir)
    if as_json:
        print(rep.model_dump_json(indent=2))
    else:
        report.terminal(rep)
        print(f"  reports: {jpath} and {mpath}", file=sys.stderr)
    return 1 if rep.blocked else 0


def cmd_scan(args: argparse.Namespace) -> int:
    root, is_git = _root(args.repo)
    mode = "pre-commit" if args.staged else "full"
    if args.staged and not is_git:
        print("--staged needs a git repository", file=sys.stderr)
        return 2
    return _run(root, is_git, mode, args.json, args.live)


def cmd_hook(args: argparse.Namespace) -> int:
    root, is_git = _root(None)
    if args.name == "pre-commit":
        return _run(root, is_git, "pre-commit", False)
    _run(root, is_git, args.name, False)  # post-commit / post-merge never block
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    from rce_sentinel.intel.agent import IntelAgent, load_simulated

    root, _ = _root(args.repo)
    settings = load_settings(root)
    agent = IntelAgent(Scanner(settings))
    simulated = load_simulated(args.simulate) if args.simulate else None
    interval = float(settings.sources.get("poll_interval_minutes", 30)) * 60
    while True:
        res = agent.run_once(simulated)
        print(f"intel: fetched {res.fetched}, new {res.new}, rce {res.rce}, "
              f"matched {res.matched}")
        for e in res.errors:
            print(f"  ! {e}")
        for n in res.notices:
            print(f"  ! {n}")
        for r in res.realerts:
            print(f"  re-alert: {r}")
        for p in res.proposals:
            print(f"  draft mitigation (not applied, not pushed): {p}")
        for n in res.notified:
            print(f"  notified: {n}")
        if args.once or simulated is not None:
            return 0
        time.sleep(interval)


def cmd_trace(args: argparse.Namespace) -> int:
    import os

    from rce_sentinel import trace as tr

    if args.text == "-":
        text = sys.stdin.read()
    elif args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        print("give code as an argument, --file FILE, or '-' for stdin", file=sys.stderr)
        return 2
    os.environ.setdefault("RCE_SENTINEL_PROVIDER", "hybrid" if not args.real_only else "auto")
    state = Path(args.state).resolve()
    seed = Path(args.osv_seed) if args.osv_seed else None
    if args.show_io:
        os.environ["RCE_SENTINEL_TRACE_IO"] = "1"
    if args.profile:
        os.environ["RCE_SENTINEL_PROFILE"] = args.profile
    print(tr.run(text, args.lang, state, args.req or [], seed, args.show_io,
                 live=not args.no_live))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Real one-line call to every configured model, so a demo never meets a surprise."""
    import time

    import httpx

    from rce_sentinel.orchestrator.llm import ModelRegistry, OpenAICompatProvider
    from rce_sentinel.posture import posture

    root, _ = _root(args.repo)
    settings = load_settings(root)
    bad = 0
    print(f"{'posture':<16} {'status':<6} detail")
    for chk in posture(settings):
        print(f"{chk.name:<16} {chk.level:<6} {chk.detail}")
        bad += chk.level == "fail"
    if args.posture:
        return 1 if bad else 0
    print()
    reg = ModelRegistry(settings.models, "auto")
    prov = OpenAICompatProvider(reg, retries=0)
    print(f"{'tier':<6} {'model':<28} {'status':<8} detail")
    for tier, specs in reg.all.items():
        for spec in specs:
            if args.provider and spec.provider != args.provider:
                continue
            if not reg.is_available(spec):
                p = reg.providers.get(spec.provider, {})
                why = (f"not running or model not pulled ({reg.base_url(spec.provider)})"
                       if p.get("local") else
                       (reg.unreachable(spec.provider) if os.environ.get(
                           p.get("api_key_env", "")) else f"no key ({p.get('api_key_env')})"))
                print(f"{tier:<6} {spec.id:<28} {'skip':<8} {why}")
                continue
            t0 = time.monotonic()
            try:
                r = prov.complete(spec, "Reply with a JSON object only.",
                                  'Reply with {"ok": true}', max_tokens=16, timeout=args.timeout)
                ok = '"ok"' in r.text.replace(" ", "") or "true" in r.text
                print(f"{tier:<6} {spec.id:<28} {'ok' if ok else 'odd':<8} "
                      f"{(time.monotonic() - t0) * 1000:.0f} ms")
                bad += 0 if ok else 1
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                from rce_sentinel.orchestrator.router import STATUS_REASON

                print(f"{tier:<6} {spec.id:<28} {'FAIL':<8} HTTP {code}: "
                      f"{STATUS_REASON.get(code, 'error')}")
                bad += 1
            except Exception as exc:
                print(f"{tier:<6} {spec.id:<28} {'FAIL':<8} {type(exc).__name__}")
                bad += 1
    return 1 if bad else 0


def cmd_cost_report(args: argparse.Namespace) -> int:
    from rce_sentinel.costreport import gather, render
    from rce_sentinel.memory.store import Store
    from rce_sentinel.orchestrator.llm import ModelRegistry

    root, _ = _root(args.repo)
    settings = load_settings(root)
    data = gather(Store(settings.db_path), ModelRegistry(settings.models, "off"), args.last)
    print(json.dumps(data, indent=2, default=str) if args.json else render(data))
    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    from rce_sentinel.memory.answers import AnswerCache
    from rce_sentinel.memory.store import Store

    root, _ = _root(args.repo)
    settings = load_settings(root)
    store = Store(settings.db_path)
    answers = AnswerCache.from_policy(store, settings.policy)
    if args.action == "stats":
        st = answers.stats()
        verdicts = store.execute("SELECT COUNT(*) AS n FROM result_cache")[0]["n"]
        if args.json:
            print(json.dumps({"verdicts": verdicts, "answers": st}, indent=2))
            return 0
        print(f"verdict cache: {verdicts} entr{'y' if verdicts == 1 else 'ies'}")
        print(f"answer cache:  {'on' if st['enabled'] else 'off'}, {st['entries']} entries, "
              f"{st['hits']} hit(s), saved {st['saved_in']} input and {st['saved_out']} "
              f"output tokens, ${st['saved_usd']:.4f} API-equivalent; ttl "
              f"{st['ttl_hours']:g} h, max {st['max_entries']}")
        return 0
    what = args.which
    parts = []
    if what in ("verdicts", "all"):
        parts.append(f"{store.cache_clear()} cached verdict(s)")
    if what in ("answers", "all"):
        parts.append(f"{answers.clear()} cached answer(s)")
    print(f"cleared {' and '.join(parts)}; facts and call history kept")
    return 0


def cmd_osv_import(args: argparse.Namespace) -> int:
    from rce_sentinel.memory.store import Store
    from rce_sentinel.scanner.osv import OsvClient

    root, _ = _root(args.repo)
    settings = load_settings(root)
    records = json.loads(Path(args.file).read_text())
    n = OsvClient(Store(settings.db_path), None, {}).import_seed(records)
    print(f"imported {n} OSV record(s) into the local cache")
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    from rce_sentinel.preeval.pipeline import pre_eval

    root, _ = _root(args.repo)
    prompt = sys.stdin.read() if args.prompt in (None, "-") else args.prompt
    if not prompt.strip():
        print("empty prompt", file=sys.stderr)
        return 2
    res = pre_eval(prompt, load_settings(root), use_jev=not args.no_jev, current=args.current,
                   session_id=args.session or "", mode=args.mode, split=args.split)
    sel = res.selection
    parts = [{"subtask": sub, "model": ps.model.id if ps.model else None,
              "expected_cost": round(ps.expected_cost, 6)} for sub, ps in res.parts]
    if args.json:
        print(json.dumps({"id": res.row_id, "tags": res.tags.to_dict(), "notes": res.notes,
                          "model": sel.model.id if sel.model else None,
                          "tier": sel.start_tier, "tokens_in": sel.tokens_in,
                          "tokens_out": sel.tokens_out, "fail_usd": round(sel.fail_usd, 4),
                          "expected_cost": round(sel.expected_cost, 6),
                          "parts": parts}, indent=2))
        return 0 if sel.model else 1
    print(f"tags:  {res.tags.summary()}")
    if res.tags.subtasks:
        print(f"split: {', '.join(res.tags.subtasks)}")
    for n in res.notes:
        print(f"note:  {n}")
    for line in sel.explain():
        print(line)
    print(f"model: {sel.model.id if sel.model else 'none available'}")
    for part in parts:
        print(f"part:  {part['subtask']} -> {part['model']}  ${part['expected_cost']:.3f}")
    print(f"id:    {res.row_id}  (grade it: rce-sentinel outcome {res.row_id} accepted)")
    return 0 if sel.model else 1


def cmd_outcome(args: argparse.Namespace) -> int:
    from rce_sentinel.memory.store import Store

    root, _ = _root(args.repo)
    store = Store(load_settings(root).db_path)
    try:
        store.outcome_for_tag(args.id, args.outcome,
                              args.review_min * 60 if args.review_min is not None else None)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 1
    print(f"recorded {args.outcome} for routed prompt {args.id}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from rce_sentinel.memory.store import Store
    from rce_sentinel.orchestrator.llm import ModelRegistry
    from rce_sentinel.preeval.calibrate import calibrate
    from rce_sentinel.preeval.select import quality_of

    root, _ = _root(args.repo)
    settings = load_settings(root)
    reg = ModelRegistry(settings.models, settings.provider_mode)
    cal = calibrate(Store(settings.db_path), reg)
    if args.json:
        print(json.dumps(cal.to_dict(), indent=2))
        return 0
    from rce_sentinel.handoff.session import find_model

    print(f"{cal.rows} graded outcome(s)")
    print(f"{'model':<44} {'n':>4} {'prior':>6} {'fitted':>6}  groups")
    for model, q in sorted(cal.quality.items(), key=lambda kv: -kv[1]):
        spec = find_model(reg, model)
        prior = quality_of(spec) if spec else float("nan")
        groups = ", ".join(f"{g} {v:.2f} (n={cal.n[model][g]})"
                           for g, v in cal.group_quality[model].items())
        print(f"{model:<44} {cal.n[model]['*']:>4} {prior:>6.2f} {q:>6.2f}  {groups}")
    for model, tps in sorted(cal.tps.items()):
        print(f"speed  {model}: {tps:.0f} tok/s measured")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from rce_sentinel.bench.report import bench_report
    from rce_sentinel.bench.runner import BenchRunner
    from rce_sentinel.bench.suite import Suite
    from rce_sentinel.handoff.session import find_model
    from rce_sentinel.memory.store import Store
    from rce_sentinel.orchestrator.llm import ModelRegistry, make_provider

    root, _ = _root(args.repo)
    settings = load_settings(root)
    suite = Suite.load(args.suite)
    store = Store(settings.db_path)
    bench_cfg = settings.models.get("bench") or {}
    if args.set:
        if args.set not in (bench_cfg.get("sets") or {}):
            print(f"no bench set {args.set!r} in models.yaml", file=sys.stderr)
            return 2
        bench_cfg = bench_cfg["sets"][args.set]
    if args.action == "matrix":
        from rce_sentinel.bench.matrix import bench_matrix, format_matrix

        only = (args.models.split(",") if args.models
                else list(bench_cfg.get("models", [])) if args.set else None)
        mx = bench_matrix(store, suite, models_only=only, only=args.only,
                          include_simulated=args.include_simulated)
        print(json.dumps(mx, indent=2) if args.json else format_matrix(mx))
        return 0
    if args.action == "report":
        only = (args.models.split(",") if args.models
                else list(bench_cfg.get("models", [])) if args.set else None)
        rep = bench_report(settings, store, suite, mode=args.mode,
                           include_simulated=args.include_simulated, models_only=only,
                           only=args.only or None)
        if args.json:
            print(json.dumps(rep, indent=2))
            return 0
        print(f"{rep['suite']}: {rep['graded_items']}/{rep['items']} items graded on every "
              f"model; train {rep['train']}, test {rep['test']}; profile {rep['profile']}; "
              f"right = within {rep['tolerance']:.0%} or ${rep['tolerance_usd']:.2f} of the "
              "best possible cost; ranked by mean regret")
        print(f"{'strategy':<46} {'n':>3} {'right':>6} {'95% CI':>13} {'pass':>5} "
              f"{'regret $':>9} {'spend $':>8}")
        for s in rep["strategies"]:
            lo, hi = s["ci95"]
            print(f"{s['strategy']:<46} {s['prompts']:>3} {s['accuracy']:>6.0%} "
                  f"{f'{lo:.0%}-{hi:.0%}':>13} {s['success_rate']:>5.0%} "
                  f"{s['mean_regret_usd']:>9.4f} {s['spend_usd']:>8.4f}")
        if rep["test"] < 30:
            print("note: fewer than 30 test prompts; the interval is wide. Grow the suite.")
        return 0
    reg = ModelRegistry(settings.models, settings.provider_mode)
    ids = args.models.split(",") if args.models else list(bench_cfg.get("models", []))
    if not ids:
        print("no models: pass --models or set bench.models in models.yaml", file=sys.stderr)
        return 2
    models, missing = [], []
    for mid in ids:
        spec = find_model(reg, mid)
        if spec is None or not reg.is_available(spec):
            missing.append(mid)
        else:
            models.append(spec)
    for mid in missing:
        print(f"skip {mid}: not configured or not reachable", file=sys.stderr)
    if not models:
        return 1
    judges = [j for jid in bench_cfg.get("judges", []) if (j := find_model(reg, jid))]
    runner = BenchRunner(reg, make_provider(reg), store, judges, timeout=args.timeout,
                         log=None if args.quiet else print,
                         same_family_judge=bool(bench_cfg.get("same_family_judge")),
                         per_model=lambda s: args.per_model if s.egress
                         else args.local_per_model)
    items = [i for i in suite.items if fnmatch.fnmatch(i.id, args.only)] if args.only \
        else suite.items
    items = items[:args.limit] if args.limit else items
    out = runner.run(suite, models, max_usd=args.max_usd, fresh=args.fresh, items=items,
                     parallel=not args.serial)
    print(f"ran {out.runs}, skipped {out.skipped} already graded, {out.errors} error(s), "
          f"{out.passed} passed, ${out.cost:.4f}{'; ' + out.stopped if out.stopped else ''}")
    for n in out.notes:
        print(f"note: {n}")
    return 0 if not out.errors else 1


def cmd_serve(args: argparse.Namespace) -> int:
    from rce_sentinel.memory.store import Store
    from rce_sentinel.orchestrator.llm import ModelRegistry, make_provider
    from rce_sentinel.proxy import RouterProxy, make_server

    root, _ = _root(args.repo)
    settings = load_settings(root)
    reg = ModelRegistry(settings.models, settings.provider_mode)
    store = Store(settings.db_path)
    claude = None
    if not args.no_claude:
        from rce_sentinel.claude_proxy import ClaudeRouter

        claude = ClaudeRouter(settings, store, upstream=args.anthropic_upstream,
                              log_only=args.log_only)
    try:
        proxy = RouterProxy(settings, reg, make_provider(reg), store,
                            mode=args.mode, use_jev=not args.no_jev, claude=claude)
        server = make_server(proxy, args.host, args.port)
    except ValueError as exc:
        print(f"serve: {exc}", file=sys.stderr)
        return 2
    host, port = server.server_address[:2]
    print(f"rce-sentinel proxy on http://{host}:{port}/v1  (model 'auto' routes; Ctrl-C stops)",
          flush=True)
    sk = proxy.skills
    print(f"skills: {', '.join(sk.names) or 'none'}"
          + (f" on {', '.join(sorted(sk.surfaces))}" if sk.names else "")
          + f"; answer cache {'on' if proxy.answers.enabled else 'off'}"
          + f"; token {'required' if proxy.token else 'off (loopback only)'}", flush=True)
    if claude:
        state = "log-only: decides and records, sends the asked-for model" if args.log_only \
            else "routing between " + ", ".join(claude.pool)
        print(f"Claude Code: ANTHROPIC_BASE_URL=http://{host}:{port} claude   ({state})",
              flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def usage_summary(store: Any, since: float) -> dict[str, Any]:
    """Claude Code traffic through the proxy: tokens, cache, cost, and routing rules."""
    rows = [dict(r) for r in store.execute(
        "SELECT * FROM usage_events WHERE ts >= ? AND status = 200", (since,))]
    by: dict[str, dict[str, float]] = {}
    for r in rows:
        m = by.setdefault(r["model"] or "?", {k: 0.0 for k in (
            "requests", "input", "write", "read", "out", "cost", "cost_requested")})
        m["requests"] += 1
        m["input"] += r["input_tokens"] or 0
        m["write"] += (r["cache_write_5m"] or 0) + (r["cache_write_1h"] or 0)
        m["read"] += r["cache_read"] or 0
        m["out"] += r["tokens_out"] or 0
        m["cost"] += r["cost"] or 0.0
        m["cost_requested"] += r["cost_requested"] if r["cost_requested"] is not None \
            else (r["cost"] or 0.0)
    rules: dict[str, int] = {}
    for r in rows:
        rules[r["rule"]] = rules.get(r["rule"], 0) + 1
    skilled = sum(1 for r in rows if r.get("skills"))
    total = {k: sum(m[k] for m in by.values()) for k in (
        "requests", "input", "write", "read", "out", "cost", "cost_requested")}
    prompt = total["input"] + total["write"] + total["read"]
    return {"requests": int(total["requests"]), "models": by, "rules": rules,
            "cost": total["cost"], "cost_requested": total["cost_requested"],
            "saved": total["cost_requested"] - total["cost"],
            "cache_hit": total["read"] / prompt if prompt else 0.0,
            "fallbacks": sum(r["fallback"] or 0 for r in rows), "with_skills": skilled,
            "switches": sum(1 for r in rows if r["switched_from"])}


def cmd_usage(args: argparse.Namespace) -> int:
    from rce_sentinel.memory.store import Store

    root, _ = _root(args.repo)
    settings = load_settings(root)
    rep = usage_summary(Store(settings.db_path), time.time() - args.days * 86400)
    if args.json:
        print(json.dumps(rep, indent=2))
        return 0
    print(f"Claude Code via proxy, last {args.days:g} day(s): {rep['requests']} requests, "
          f"cache hit {rep['cache_hit']:.0%}, {rep['switches']} switch(es), "
          f"{rep['fallbacks']} fallback(s), skills on {rep['with_skills']}")
    print(f"{'model':<22} {'req':>5} {'input':>9} {'cache w':>9} {'cache r':>11} "
          f"{'output':>9} {'cost $':>9} {'as asked $':>10}")
    for model, m in sorted(rep["models"].items()):
        print(f"{model:<22} {m['requests']:>5.0f} {m['input']:>9.0f} {m['write']:>9.0f} "
              f"{m['read']:>11.0f} {m['out']:>9.0f} {m['cost']:>9.4f} "
              f"{m['cost_requested']:>10.4f}")
    print(f"API-equivalent cost ${rep['cost']:.4f} vs ${rep['cost_requested']:.4f} on the "
          f"asked-for models (saved ${rep['saved']:.4f}); quality is not in this number, "
          "grade replies with `rce-sentinel outcome`")
    print("rules: " + ", ".join(f"{k} {v}" for k, v in sorted(rep["rules"].items())))
    return 0


def cmd_vectors(args: argparse.Namespace) -> int:
    import httpx

    from rce_sentinel.bench.suite import Suite
    from rce_sentinel.memory.store import Store
    from rce_sentinel.memory.vectors import VectorIndex, ollama_embedder
    from rce_sentinel.orchestrator.llm import ModelRegistry

    root, _ = _root(args.repo)
    settings = load_settings(root)
    store = Store(settings.db_path)
    base = args.base_url or ModelRegistry(settings.models, "auto").base_url("ollama")
    index = VectorIndex(store, ollama_embedder(base, args.embed_model), args.embed_model)
    try:
        if args.action == "build":
            suite = Suite.load(args.suite)
            rows = [("prompt", i.id, i.prompt) for i in suite.items]
            for r in store.execute("SELECT item, model FROM bench_outputs WHERE suite=?",
                                   (suite.key,)):
                text = store.output(suite.key, r["item"], r["model"]) or ""
                rows.append(("output", f"{r['item']}|{r['model']}", text[:8000]))
            n = index.add_many(rows)
            print(f"embedded {n} new or changed text(s); {index.count()} vectors for "
                  f"{args.embed_model}")
            return 0
        for h in index.search(args.query, kind=args.kind or None, k=args.k):
            print(f"{h.score:.3f}  {h.kind:<6}  {h.ref}")
        return 0
    except httpx.HTTPError as exc:
        print(f"embedding model unavailable at {base} ({type(exc).__name__}); "
              f"run `ollama pull {args.embed_model}` and check OLLAMA_BASE_URL",
              file=sys.stderr)
        return 1


def cmd_session(args: argparse.Namespace) -> int:
    from rce_sentinel.handoff.session import MODES, Scenario, SessionRunner
    from rce_sentinel.memory.store import Store
    from rce_sentinel.orchestrator.llm import ModelRegistry, make_provider

    root, _ = _root(args.repo)
    settings = load_settings(root)
    sc = Scenario.load(args.scenario)
    reg = ModelRegistry(settings.models, settings.provider_mode)
    runner = SessionRunner(settings, reg, make_provider(reg), Store(settings.db_path),
                           keep_last=args.keep_last, max_usd=args.max_usd, split=args.split,
                           timeout=args.timeout)
    modes = MODES if args.modes == "all" else tuple(args.modes.split(","))
    models = None if args.models is None else (
        [] if args.models == "auto" else args.models.split(","))
    out = []
    for mode in modes:
        res = runner.run(sc, mode, models)
        out.append(res.summary())
        if not args.json:
            print(f"\n=== {mode}  ({res.run_id})")
            for t in res.turns:
                sw = f"switch {t.switched_from} -> " if t.switched_from else ""
                print(f"  turn {t.turn}: {sw}{t.model}  ctx {t.context_tokens} tok  "
                      f"in {t.tokens_in} out {t.tokens_out}  ${t.cost:.4f}  {t.seconds:.1f}s"
                      f"  {'ok' if t.ok else 'FAIL ' + t.error}"
                      f"{'  rewrote ' + ', '.join(t.rewrites) if t.rewrites else ''}")
            miss = [k for k, v in res.canaries.items() if not v]
            print(f"  retention {res.retention:.0%}  handoff tokens {res.handoff_tokens}  "
                  f"cost ${res.cost:.4f}{'  missed: ' + ', '.join(miss) if miss else ''}")
            print(f"  files: {res.workspace}")
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"\n{'mode':<11} {'retention':>9} {'handoff tok':>11} {'tokens in':>9} "
              f"{'cost $':>8} {'rewrites':>8}")
        for r in out:
            print(f"{r['mode']:<11} {r['retention']:>9.0%} {r['handoff_tokens']:>11} "
                  f"{r['tokens_in']:>9} {r['cost_usd']:>8.4f} {r['rewrites']:>8}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rce-sentinel",
                                description="Cost-aware RCE scanner and intel agent")
    p.add_argument("--repo", help="repository path (default: current directory)")
    p.add_argument("--env-file", help="load API keys from this KEY=VALUE file "
                                      "(also RCE_SENTINEL_ENV_FILE); set vars win")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("install-hooks", help="install pre-commit, post-commit, post-merge hooks")
    s = sub.add_parser("scan", help="scan the full tree, or the staged diff with --staged")
    s.add_argument("--staged", action="store_true")
    s.add_argument("--json", action="store_true")
    s.add_argument("--live", action="store_true",
                   help="print progress as it happens (which tool fired, which model is asked)")
    h = sub.add_parser("hook", help=argparse.SUPPRESS)
    h.add_argument("name", choices=list(hooks.HOOKS))
    w = sub.add_parser("watch", help="run the zero-day intel agent")
    w.add_argument("--once", action="store_true")
    w.add_argument("--simulate", help="JSON file of intel items to inject instead of polling")
    t = sub.add_parser("trace", help="run one prompt (code snippet) and show the pipeline")
    t.add_argument("text", nargs="?", help="code to review, or '-' to read stdin")
    t.add_argument("--file", help="read the code from a file")
    t.add_argument("--lang", choices=["py", "js", "ts", "go"], help="default: auto-detect")
    t.add_argument("--req", action="append", help="add a pinned dependency, e.g. PyYAML==5.3")
    t.add_argument("--no-live", action="store_true",
                   help="skip the live progress log, print only the final summary")
    t.add_argument("--show-io", action="store_true",
                   help="print the exact prompt sent to each model and its raw reply")
    t.add_argument("--osv-seed", help="OSV records JSON to load (offline advisories)")
    t.add_argument("--state", default=".rce-sentinel-trace",
                   help="persistent state dir (cache, memory, signed verdicts)")
    t.add_argument("--real-only", action="store_true",
                   help="never fall back to the labelled mock provider")
    t.add_argument("--profile", choices=["pre-commit", "interactive", "post-commit",
                                          "background"],
                   help="routing profile: how much waiting costs (default: background)")
    d = sub.add_parser("doctor", help="make a real call to every configured model")
    d.add_argument("--timeout", type=float, default=60.0)
    d.add_argument("--provider", help="only this provider, e.g. opencode")
    d.add_argument("--posture", action="store_true",
                   help="only the security posture checks; no model calls")
    c = sub.add_parser("cost-report", help="tokens and dollars per tier vs baseline")
    c.add_argument("--last", type=int, help="only the last N runs")
    c.add_argument("--json", action="store_true")
    cache = sub.add_parser("cache", help="verdict and answer caches: stats or clear")
    cache.add_argument("action", choices=["clear", "stats"])
    cache.add_argument("--which", choices=["verdicts", "answers", "all"], default="verdicts",
                       help="clear: which cache (default verdicts, as before)")
    cache.add_argument("--json", action="store_true")
    o = sub.add_parser("osv-import", help="load OSV records into the offline cache")
    o.add_argument("file")
    se = sub.add_parser("session", help="multi-turn switch test: hand off between models")
    se.add_argument("--scenario", default="calculator", help="name in demo/scenarios or path")
    se.add_argument("--modes", default="all", help="all, or a comma list of none,transcript,pack")
    se.add_argument("--models", help="comma list, one per turn; 'auto' = pre-eval picks")
    se.add_argument("--keep-last", type=int, default=0,
                    help="pack mode: also pass the last N turns verbatim")
    se.add_argument("--max-usd", type=float, default=2.0, help="stop a session past this")
    se.add_argument("--timeout", type=float, default=300.0, help="seconds per model call")
    se.add_argument("--split", action="store_true",
                    help="auto mode: one call per target, each routed on its own")
    se.add_argument("--json", action="store_true")
    r = sub.add_parser("route", help="tag a dev prompt and pick the model for it")
    r.add_argument("prompt", nargs="?", help="the prompt, or '-' to read stdin")
    r.add_argument("--current", help="model the session is on now (warm cache)")
    r.add_argument("--session", help="session id to log the tags under")
    r.add_argument("--mode", default="scan",
                   help="routing profile key, e.g. scan (interactive) or watch (background)")
    r.add_argument("--no-jev", action="store_true", help="never call the Jev decision model")
    r.add_argument("--split", action="store_true", help="also route each target on its own")
    r.add_argument("--json", action="store_true")
    oc = sub.add_parser("outcome", help="grade a routed prompt: what you did with the answer")
    oc.add_argument("id", type=int, help="id printed by route, or the proxy's x-rce-tag-id")
    oc.add_argument("outcome", choices=["accepted", "edited", "rejected", "failed"])
    oc.add_argument("--review-min", type=float, help="minutes spent reviewing the answer")
    ca = sub.add_parser("calibrate", help="fit model quality from graded outcomes")
    ca.add_argument("--json", action="store_true")
    b = sub.add_parser("bench", help="graded benchmark: run models on a suite, or report")
    b.add_argument("action", choices=["run", "report", "matrix"])
    b.add_argument("--suite", default="dev", help="name in demo/bench or a path")
    b.add_argument("--models", help="comma list (default: bench.models in models.yaml)")
    b.add_argument("--max-usd", type=float, default=5.0, help="stop the run past this")
    b.add_argument("--limit", type=int, help="only the first N items")
    b.add_argument("--serial", action="store_true", help="one model, one prompt at a time")
    b.add_argument("--only", default="",
                   help="item id glob, e.g. 'hard-*' (run: items to run; report: test items "
                        "to score)")
    b.add_argument("--per-model", type=int, default=4,
                   help="prompts in flight per remote model")
    b.add_argument("--local-per-model", type=int, default=1,
                   help="prompts in flight per local model")
    b.add_argument("--set", default="", help="named model set from bench.sets (e.g. seat)")
    b.add_argument("--fresh", action="store_true", help="re-run items already graded")
    b.add_argument("--timeout", type=float, default=300.0)
    b.add_argument("--quiet", action="store_true")
    b.add_argument("--mode", default="scan", help="report: routing profile key")
    b.add_argument("--include-simulated", action="store_true",
                   help="report: also count mock replies (for demos only)")
    b.add_argument("--json", action="store_true")
    vx = sub.add_parser("vectors", help="semantic index of bench prompts and replies")
    vx.add_argument("action", choices=["build", "search"])
    vx.add_argument("query", nargs="?", default="")
    vx.add_argument("--suite", default="dev")
    vx.add_argument("--kind", choices=["prompt", "output", ""], default="")
    vx.add_argument("-k", type=int, default=5)
    vx.add_argument("--embed-model", default="nomic-embed-text")
    vx.add_argument("--base-url", default="", help="OpenAI-compatible /v1 root")
    us = sub.add_parser("usage", help="Claude Code token, cache and cost log from the proxy")
    us.add_argument("--days", type=float, default=7.0)
    us.add_argument("--json", action="store_true")
    sv = sub.add_parser("serve", help="OpenAI-compatible proxy; model 'auto' routes")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8787)
    sv.add_argument("--mode", default="scan", help="routing profile key")
    sv.add_argument("--no-jev", action="store_true")
    sv.add_argument("--no-claude", action="store_true", help="turn off Claude Code routing")
    sv.add_argument("--log-only", action="store_true",
                    help="Claude Code: decide and record, but send the asked-for model")
    sv.add_argument("--anthropic-upstream", default="https://api.anthropic.com")
    return p


HANDLERS = {"install-hooks": cmd_install_hooks, "scan": cmd_scan, "hook": cmd_hook,
            "watch": cmd_watch, "trace": cmd_trace, "doctor": cmd_doctor,
            "cost-report": cmd_cost_report, "cache": cmd_cache, "osv-import": cmd_osv_import,
            "route": cmd_route, "session": cmd_session, "outcome": cmd_outcome,
            "calibrate": cmd_calibrate, "bench": cmd_bench, "serve": cmd_serve,
            "usage": cmd_usage, "vectors": cmd_vectors}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if env_file := args.env_file or os.environ.get("RCE_SENTINEL_ENV_FILE"):
        from rce_sentinel.config import load_env_file

        names = load_env_file(env_file)
        print(f"env: loaded {', '.join(names) or 'nothing new'} from {env_file}",
              file=sys.stderr)
    for sub in ("repo",):
        if getattr(args, sub, None) is None:
            setattr(args, sub, None)
    return HANDLERS[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
