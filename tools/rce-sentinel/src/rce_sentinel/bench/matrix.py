"""Model-by-group comparison of graded bench replies.

Groups are the item family from its id (debug-*, hard-build-*, ...) and the difficulty
the current rule tagger gives the prompt. Each cell is pass rate over graded replies.
Per model it also reports cost, latency, output size, errors, and "cheapest pass" wins:
on how many prompts the model was the cheapest one that passed.
"""

from __future__ import annotations

import statistics
from typing import Any

from rce_sentinel.bench.suite import Suite
from rce_sentinel.memory.store import Store
from rce_sentinel.preeval.tags import DIFFICULTY, rule_tags


def family(item_id: str) -> str:
    """'hard-debug-x' -> 'hard-debug', 'div-go-x' -> 'div-go', 'debug-x' -> 'debug'."""
    parts = item_id.split("-")
    return "-".join(parts[:2]) if parts[0] in ("hard", "div") and len(parts) > 1 else parts[0]


def _cell(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    ok = sum(int(r["success"]) for r in rows)
    return {"n": n, "passed": ok, "rate": round(ok / n, 3) if n else None}


def bench_matrix(store: Store, suite: Suite, *, models_only: list[str] | None = None,
                 only: str = "", include_simulated: bool = False) -> dict[str, Any]:
    import fnmatch

    prompts = {i.id: i for i in suite.items
               if not only or fnmatch.fnmatch(i.id, only)}
    sql = "SELECT * FROM bench_results WHERE suite=?"
    if not include_simulated:
        sql += " AND simulated=0"
    rows = [dict(r) for r in store.execute(sql, (suite.key,)) if r["item"] in prompts]
    if models_only:
        rows = [r for r in rows if r["model"] in set(models_only)]
    graded = [r for r in rows if r["error"] is None]
    models = sorted({r["model"] for r in rows})
    level = {i: DIFFICULTY[round(rule_tags(p.prompt).level)] for i, p in prompts.items()}
    fams = sorted({family(i) for i in prompts})

    by_item: dict[str, dict[str, dict[str, Any]]] = {}
    for r in graded:
        by_item.setdefault(r["item"], {})[r["model"]] = r
    common = [i for i, by in by_item.items() if set(by) == set(models)]
    wins = dict.fromkeys(models, 0)
    for i in common:
        passing = [r for r in by_item[i].values() if r["success"]]
        if passing:
            wins[min(passing, key=lambda r: r["cost"] or 0)["model"]] += 1

    out: dict[str, Any] = {"suite": suite.key, "models": models, "families": fams,
                           "levels": [d for d in DIFFICULTY if d in level.values()],
                           "common_items": len(common), "rows": []}
    for m in models:
        mine = [r for r in graded if r["model"] == m]
        errs = [r for r in rows if r["model"] == m and r["error"] is not None]
        lat = [r["latency_ms"] / 1000 for r in mine if r["latency_ms"]]
        out["rows"].append({
            "model": m, "all": _cell(mine),
            "common": _cell([r for r in mine if r["item"] in common]),
            "families": {f: _cell([r for r in mine if family(r["item"]) == f]) for f in fams},
            "levels": {d: _cell([r for r in mine if level[r["item"]] == d])
                       for d in out["levels"]},
            "errors": len(errs), "cost_usd": round(sum(r["cost"] or 0 for r in mine), 4),
            "usd_per_prompt": round(sum(r["cost"] or 0 for r in mine) / len(mine), 5)
            if mine else None,
            "median_s": round(statistics.median(lat), 1) if lat else None,
            "mean_tokens_out": round(statistics.mean(r["tokens_out"] or 0 for r in mine))
            if mine else None,
            "cheapest_pass_wins": wins[m]})
    return out


def format_matrix(mx: dict[str, Any]) -> str:
    def pct(c: dict[str, Any]) -> str:
        return "-" if c["rate"] is None else f"{c['rate']:.0%}/{c['n']}"

    cols = mx["families"] + mx["levels"]
    w = max([len(m) for m in mx["models"]] + [5])
    lines = [f"{mx['suite']}: pass rate/graded per group; {mx['common_items']} items graded "
             "on every model",
             f"{'model':<{w}} {'all':>8} {'common':>8} " + " ".join(f"{c:>12}" for c in cols)
             + f" {'$/prompt':>9} {'med s':>6} {'tok out':>7} {'wins':>5} {'err':>4}"]
    for r in mx["rows"]:
        cells = [pct(r["families"][c]) for c in mx["families"]]
        cells += [pct(r["levels"][c]) for c in mx["levels"]]
        lines.append(
            f"{r['model']:<{w}} {pct(r['all']):>8} {pct(r['common']):>8} "
            + " ".join(f"{c:>12}" for c in cells)
            + f" {r['usd_per_prompt'] if r['usd_per_prompt'] is not None else '-':>9} "
            f"{r['median_s'] if r['median_s'] is not None else '-':>6} "
            f"{r['mean_tokens_out'] if r['mean_tokens_out'] is not None else '-':>7} "
            f"{r['cheapest_pass_wins']:>5} {r['errors']:>4}")
    return "\n".join(lines)
