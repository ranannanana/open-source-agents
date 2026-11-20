"""Benchmark: suite loading and splits, grading, judge, retries, evidence, and the report."""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from rce_sentinel.bench.report import bench_report, realised, task_tokens, wilson
from rce_sentinel.bench.runner import BenchRunner
from rce_sentinel.bench.suite import Suite
from rce_sentinel.config import Settings
from rce_sentinel.handoff.session import find_model
from rce_sentinel.memory.store import Store
from rce_sentinel.orchestrator.llm import LLMResult, MockProvider, ModelRegistry, ModelSpec
from rce_sentinel.preeval.select import quality_of

WEAK, MID, STRONG = ("deepseek/deepseek-v4.1-flash", "moonshotai/kimi-k2.7-code",
                     "anthropic/claude-opus-5.5")
JUDGE = "openai/gpt-6.1-sol"


class Graded(MockProvider):
    """Strong passes everything, mid passes easy items, weak passes nothing."""

    def __init__(self, down: set[str] | None = None, judge_pass: bool = True):
        super().__init__()
        self.down = down or set()
        self.judge_pass = judge_pass

    def complete(self, spec: ModelSpec, system: str, user: str, **kw: Any) -> LLMResult:
        if spec.id in self.down:
            raise httpx.ConnectError("no route")
        if user.startswith("TASK: grade"):
            text = json.dumps({"passed": self.judge_pass, "score": 7, "reason": "ok"})
            return LLMResult(text, 300, 20, 0, 500.0, spec.id)
        hard = "hard" in user
        ok = spec.id == STRONG or (spec.id == MID and not hard)
        body = ("ANSWER-OK " if ok else "nope ") * 1500
        return LLMResult(body, 200, 2500, 0, 20_000.0, spec.id)


def write_suite(tmp_path: Path, n: int = 40, rubric: str = "") -> Suite:
    items = [{"id": f"item-{i}", "prompt": f"{'hard' if i % 2 else 'easy'} task {i}: "
              "build a small parser", "must": ["ANSWER-OK"], "must_not": ["nope"],
              **({"rubric": rubric} if rubric else {})} for i in range(n)]
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump({"name": "t", "version": 1, "items": items}))
    return Suite.load(str(path))


def setup(settings: Settings, provider: Any, real: bool = False
          ) -> tuple[BenchRunner, list[ModelSpec], Store]:
    reg = ModelRegistry(settings.models, "mock")
    if real:   # behave like reachable cloud models, so outcomes count as evidence
        reg.simulated = lambda spec: False  # type: ignore[method-assign]
    store = Store(settings.db_path)
    models = [find_model(reg, m) for m in (WEAK, MID, STRONG)]
    judges = [find_model(reg, JUDGE)]
    return BenchRunner(reg, provider, store, judges), models, store  # type: ignore[list-item]


def test_shipped_suite_loads_with_both_splits() -> None:
    suite = Suite.load("dev")
    assert len(suite.items) >= 100
    splits = {i.split for i in suite.items}
    assert splits == {"train", "test"}
    assert all(i.must for i in suite.items)


def test_split_is_stable_per_id(tmp_path: Path) -> None:
    a = write_suite(tmp_path, 40)
    b = write_suite(tmp_path, 60)
    assert [i.split for i in a.items] == [i.split for i in b.items[:40]]


def test_duplicate_ids_and_bad_regex_are_refused(tmp_path: Path) -> None:
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump({"name": "x", "items": [{"id": "a", "prompt": "p"},
                                                         {"id": "a", "prompt": "q"}]}))
    with pytest.raises(ValueError):
        Suite.load(str(p))
    p.write_text(yaml.safe_dump({"name": "x", "items": [{"id": "a", "prompt": "p",
                                                         "must": ["("]}]}))
    with pytest.raises(re.error):
        Suite.load(str(p))


def test_run_grades_resumes_and_records_evidence(settings: Settings, tmp_path: Path) -> None:
    suite = write_suite(tmp_path, 10)
    runner, models, store = setup(settings, Graded(), real=True)
    out = runner.run(suite, models)
    assert out.runs == 30 and out.errors == 0
    assert out.passed == 10 + 5   # strong all, mid the easy half
    again = runner.run(suite, models)
    assert again.runs == 0 and again.skipped == 30
    n = store.execute("SELECT COUNT(*) AS n FROM outcomes WHERE source='bench'")[0]["n"]
    assert n == 30


def test_mock_replies_are_not_evidence(settings: Settings, tmp_path: Path) -> None:
    runner, models, store = setup(settings, Graded())
    runner.run(write_suite(tmp_path, 4), models)
    assert store.execute("SELECT COUNT(*) AS n FROM outcomes")[0]["n"] == 0


def test_network_errors_are_retried_not_graded(settings: Settings, tmp_path: Path) -> None:
    suite = write_suite(tmp_path, 4)
    runner, models, store = setup(settings, Graded(down={MID}), real=True)
    out = runner.run(suite, models)
    assert out.errors == 4 and out.runs == 8
    runner.provider = Graded()
    assert runner.run(suite, models).runs == 4
    assert store.execute("SELECT COUNT(*) AS n FROM outcomes WHERE model=?",
                         (MID,))[0]["n"] == 4


def test_judge_can_fail_a_reply_that_passes_checks(settings: Settings,
                                                   tmp_path: Path) -> None:
    suite = write_suite(tmp_path, 2, rubric="complete parser")
    runner, models, _ = setup(settings, Graded(judge_pass=False), real=True)
    assert runner.run(suite, [models[2]]).passed == 0
    runner2, models2, _ = setup(settings, Graded(judge_pass=True), real=True)
    runner2.store.execute("DELETE FROM bench_results")
    assert runner2.run(suite, [models2[2]]).passed == 2


def test_judge_is_never_the_same_family(settings: Settings) -> None:
    runner, _, _ = setup(settings, Graded())
    sol = find_model(runner.reg, JUDGE)
    assert runner.judge_for(sol) is None    # only judge configured is its own family
    opus = find_model(runner.reg, STRONG)
    assert runner.judge_for(opus) == sol


def test_cost_cap_stops_the_run(settings: Settings, tmp_path: Path) -> None:
    runner, models, _ = setup(settings, Graded(), real=True)
    out = runner.run(write_suite(tmp_path, 10), models, max_usd=0.0)
    assert out.runs == 0 and "cap" in out.stopped


def test_report_scores_strategies_on_held_out_prompts(settings: Settings,
                                                      tmp_path: Path) -> None:
    suite = write_suite(tmp_path, 60)
    runner, models, store = setup(settings, Graded(), real=True)
    runner.run(suite, models)
    rep = bench_report(settings, store, suite, mode="watch")
    assert rep["graded_items"] == 60 and rep["test"] + rep["train"] == 60 and rep["test"] > 5
    by = {s["strategy"]: s for s in rep["strategies"]}
    assert by["oracle (best possible)"]["accuracy"] == 1.0
    assert by[f"always {WEAK}"]["accuracy"] == 0.0
    assert by[f"always {STRONG}"]["success_rate"] == 1.0
    cal, prior = by["router (calibrated)"], by["router (priors)"]
    assert cal["mean_regret_usd"] <= prior["mean_regret_usd"]
    assert cal["accuracy"] >= prior["accuracy"]
    assert rep["strategies"][1]["strategy"] != f"always {WEAK}"   # ranked by regret
    assert by["router (calibrated)"]["prompts"] == rep["test"]
    weak_spec = find_model(runner.reg, WEAK)
    assert rep["calibration"]["quality"][WEAK] < quality_of(weak_spec)


def test_report_excludes_mock_rows_by_default(settings: Settings, tmp_path: Path) -> None:
    suite = write_suite(tmp_path, 8)
    runner, models, store = setup(settings, Graded())
    runner.run(suite, models)
    assert bench_report(settings, store, suite)["graded_items"] == 0
    assert bench_report(settings, store, suite, include_simulated=True)["graded_items"] == 8


def test_wilson_interval() -> None:
    lo, hi = wilson(85, 100)
    assert 0.76 < lo < 0.80 and 0.89 < hi < 0.92
    assert wilson(0, 0) == (0.0, 1.0)


def test_row_cost_excludes_the_judge_but_the_cap_counts_it(settings: Settings,
                                                            tmp_path: Path) -> None:
    suite = write_suite(tmp_path, 2, rubric="complete parser")
    runner, models, store = setup(settings, Graded(), real=True)
    out = runner.run(suite, [models[2]])
    row_total = store.execute("SELECT SUM(cost) AS c FROM bench_results")[0]["c"]
    assert out.cost > row_total > 0


def test_small_context_models_get_a_smaller_output_cap(settings: Settings,
                                                       tmp_path: Path) -> None:
    from rce_sentinel.bench.runner import out_cap

    item = write_suite(tmp_path, 1).items[0]
    item.max_tokens = 8000
    reg = ModelRegistry(settings.models, "mock")
    assert out_cap(find_model(reg, "qwen2.5:3b"), item) == 2048  # type: ignore[arg-type]
    assert out_cap(find_model(reg, STRONG), item) == 8000  # type: ignore[arg-type]


def test_parallel_and_serial_runs_grade_the_same(settings: Settings, tmp_path: Path) -> None:
    suite = write_suite(tmp_path, 10)
    a, models, _ = setup(settings, Graded(), real=True)
    par = a.run(suite, models)
    a.store.execute("DELETE FROM bench_results")
    ser = a.run(suite, models, parallel=False)
    assert (par.runs, par.passed, par.errors) == (ser.runs, ser.passed, ser.errors) == (30, 15, 0)


class JudgeDown(Graded):
    def complete(self, spec: ModelSpec, system: str, user: str, **kw: Any) -> LLMResult:
        if user.startswith("TASK: grade"):
            raise httpx.ConnectError("offline")
        return super().complete(spec, system, user, **kw)


def test_failed_judge_is_an_error_to_retry_not_a_grade(settings: Settings,
                                                       tmp_path: Path) -> None:
    suite = write_suite(tmp_path, 2, rubric="complete parser")
    runner, models, store = setup(settings, JudgeDown(), real=True)
    out = runner.run(suite, [models[2]])
    assert out.errors == 2 and out.runs == 0
    rows = store.execute("SELECT error FROM bench_results")
    assert all("judge" in r["error"] and "ConnectError" in r["error"] for r in rows)
    assert store.execute("SELECT COUNT(*) AS n FROM outcomes")[0]["n"] == 0
    runner.provider = Graded()
    assert runner.run(suite, [models[2]]).passed == 2      # retried and graded


def test_replies_are_kept_compressed(settings: Settings, tmp_path: Path) -> None:
    suite = write_suite(tmp_path, 1)
    runner, models, store = setup(settings, Graded(), real=True)
    runner.run(suite, [models[2]])
    text = store.output(suite.key, "item-0", models[2].id)
    assert text and text.startswith("ANSWER-OK")
    row = store.execute("SELECT LENGTH(reply) AS z, chars FROM bench_outputs")[0]
    assert row["z"] < row["chars"] / 10      # repetitive text compresses well


def test_seat_error_text_is_saved(settings: Settings, tmp_path: Path) -> None:
    from rce_sentinel.orchestrator.seat import SeatError

    class SeatDown(Graded):
        def complete(self, spec: ModelSpec, system: str, user: str, **kw: Any) -> LLMResult:
            raise SeatError("claude CLI: Output token limit exceeded", 400)

    runner, models, store = setup(settings, SeatDown(), real=True)
    runner.run(write_suite(tmp_path, 1), [models[0]])
    assert "Output token limit" in store.execute("SELECT error FROM bench_results")[0]["error"]


class Overlap(Graded):
    """Records the most calls in flight at once."""

    def __init__(self) -> None:
        super().__init__()
        self.lock = threading.Lock()
        self.now = self.peak = 0

    def complete(self, spec: ModelSpec, system: str, user: str, **kw: Any) -> LLMResult:
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
        time.sleep(0.02)
        try:
            return super().complete(spec, system, user, **kw)
        finally:
            with self.lock:
                self.now -= 1


def test_prompts_run_in_parallel_within_a_model(settings: Settings, tmp_path: Path) -> None:
    suite = write_suite(tmp_path, 8)
    fake = Overlap()
    runner, models, _ = setup(settings, fake, real=True)
    runner.per_model = lambda spec: 4
    out = runner.run(suite, [models[0]])
    assert out.runs == 8 and fake.peak > 1
    fake.peak = 0
    runner.per_model = lambda spec: 1
    runner.run(suite, [models[0]], fresh=True)
    assert fake.peak == 1


def test_matrix_groups_by_family_and_counts_cheapest_wins(settings: Settings,
                                                          tmp_path: Path) -> None:
    from rce_sentinel.bench.matrix import bench_matrix, family, format_matrix

    assert family("hard-debug-x") == "hard-debug" and family("debug-x") == "debug"
    assert family("div-go-explain-defer") == "div-go"
    suite = write_suite(tmp_path, 4)
    runner, models, store = setup(settings, Graded(), real=True)
    runner.run(suite, models)
    mx = bench_matrix(store, suite)
    assert mx["common_items"] == 4 and len(mx["rows"]) == 3
    assert sum(r["cheapest_pass_wins"] for r in mx["rows"]) == 4
    assert "common" in format_matrix(mx)


def test_long_or_fenced_verdict_is_read() -> None:
    from rce_sentinel.bench.runner import read_verdict

    v = read_verdict('```json\n{"passed": true, "score": 9, "reason": "' + "x" * 900 + '"}\n```')
    assert v.passed and len(v.reason) == 300
    with pytest.raises(ValueError):
        read_verdict("no verdict here")


class Counting(Graded):
    def __init__(self) -> None:
        super().__init__()
        self.model_calls = 0

    def complete(self, spec: ModelSpec, system: str, user: str, **kw: Any) -> LLMResult:
        if not user.startswith("TASK: grade"):
            self.model_calls += 1
        return super().complete(spec, system, user, **kw)


def test_judge_retry_reuses_the_saved_reply(settings: Settings, tmp_path: Path) -> None:
    suite = write_suite(tmp_path, 2, rubric="complete parser")
    runner, models, store = setup(settings, JudgeDown(), real=True)
    runner.run(suite, [models[2]])
    cost = [r["cost"] for r in store.execute("SELECT cost FROM bench_results")]
    runner.provider = again = Counting()
    out = runner.run(suite, [models[2]])
    assert out.passed == 2 and again.model_calls == 0
    assert [r["cost"] for r in store.execute("SELECT cost FROM bench_results")] == cost


def test_short_wrong_reply_still_pays_for_the_whole_task(settings: Settings) -> None:
    from rce_sentinel.orchestrator.scoring import profile_for

    prof = profile_for(settings.policy, "background")
    by = {"big": {"success": 1, "tokens_out": 2000, "cost": 0.0, "latency_ms": 0},
          "tiny": {"success": 0, "tokens_out": 20, "cost": 0.0, "latency_ms": 0}}
    size = task_tokens(by)
    assert size == 2000
    # The rework a failure leaves is sized by the task, not the wrong reply.
    assert realised(by["tiny"], prof, settings.policy, size) > realised(
        by["tiny"], prof, settings.policy)
