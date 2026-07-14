"""Plan 35 Unit 3 — BYO outcome-check seam.

Custom checks run in the same _post_call_success pipeline as the built-in
inspectors: None is a pass, a verdict folds into the worst-severity rollup,
custom reasons are namespaced under user:, and a broken/slow check is a
contained pass (an observer never fails the run). Covers the deterministic
kind, the inline sampled LLM-judge, and the @agent registration threading.
"""

from __future__ import annotations

import time

import pytest

from papayya import CheckVerdict, llm_judge, agent
from papayya.checks import degraded, failed, run_checks
from papayya.outcomes import OK, OutcomeVerdict
from papayya.durable.run import Item
from papayya.durable.sqlite_store import SQLiteStore
from papayya.durable.types import DurableRunConfig


# --- helpers ------------------------------------------------------------ #

def short_answer(result):
    """A deterministic customer check: strings under 20 chars are degraded."""
    if isinstance(result, str) and len(result) < 20:
        return CheckVerdict("degraded", "too_short")
    return None


# --- 1. pure verdict helpers + runner ----------------------------------- #

def test_degraded_failed_namespace():
    assert degraded("too_short").reason == "user:too_short"
    assert failed("bad").reason == "user:bad"
    # idempotent — an already-namespaced reason is left alone.
    assert degraded("user:x").reason == "user:x"
    assert degraded("too_short").status == "degraded"
    assert failed("bad").status == "failed"


def test_run_checks_worst_severity_wins():
    # built-in ok + custom degraded → degraded
    v = run_checks([lambda r: degraded("x")], "res", OK, "run-1")
    assert v.status == "degraded"
    # built-in degraded + custom failed → failed
    base = OutcomeVerdict("degraded", "empty_none")
    v = run_checks([lambda r: failed("bad")], "res", base, "run-1")
    assert v.status == "failed"
    # built-in degraded + custom ok(None) → base unchanged
    v = run_checks([lambda r: None], "res", base, "run-1")
    assert v.status == "degraded" and v.reason == "empty_none"


def test_run_checks_contains_a_raising_check():
    def boom(result):
        raise RuntimeError("check bug")

    # A broken check must NOT propagate — it's an observer. Base is returned.
    v = run_checks([boom], "res", OK, "run-1")
    assert v.status == "ok"


def test_run_checks_namespaces_custom_reason():
    # A verdict whose reason forgot the user: prefix is namespaced by the runner.
    v = run_checks([lambda r: OutcomeVerdict("degraded", "raw")], "res", OK, "run-1")
    assert v.reason == "user:raw"


# --- 2. integration: check flips the store aggregate -------------------- #

def test_custom_check_flips_worst_outcome(tmp_path):
    store = SQLiteStore(str(tmp_path / "chk.db"))
    try:
        run = Item(DurableRunConfig(agent="a", store=store, checks=[short_answer]))
        run.step("gen", lambda: "hi")()  # built-in OK (non-empty), check → degraded
        loaded = store.load(run.run_id)
        assert loaded is not None
        assert loaded.worst_outcome_status == "degraded"
        assert loaded.tasks[0].outcome_status == "degraded"
        assert loaded.tasks[0].outcome_reason == "user:too_short"  # prefix reaches aggregation
    finally:
        store.close()


def test_none_is_a_pass(tmp_path):
    store = SQLiteStore(str(tmp_path / "chk2.db"))
    try:
        run = Item(DurableRunConfig(agent="a", store=store, checks=[short_answer]))
        run.step("gen", lambda: "a sufficiently long answer that passes")()
        loaded = store.load(run.run_id)
        assert loaded.worst_outcome_status == "ok"
        assert loaded.tasks[0].outcome_status == "ok"
    finally:
        store.close()


def test_broken_check_does_not_fail_the_run(tmp_path):
    store = SQLiteStore(str(tmp_path / "chk3.db"))
    try:
        def boom(result):
            raise ValueError("bug in customer check")

        run = Item(DurableRunConfig(agent="a", store=store, checks=[boom]))
        out = run.step("gen", lambda: "hi")()  # must return normally
        assert out == "hi"
        loaded = store.load(run.run_id)
        assert loaded.worst_outcome_status == "ok"  # broken check contained
    finally:
        store.close()


def test_agent_registration_checks_thread_through(tmp_path):
    @agent(name="checked-agent-thread", checks=[short_answer])
    def _a():
        return None

    store = SQLiteStore(str(tmp_path / "chk4.db"))
    try:
        # No explicit config.checks — init() pulls them off the @agent registration.
        run = Item(DurableRunConfig(agent="checked-agent-thread", store=store))
        run.step("gen", lambda: "hi")()
        loaded = store.load(run.run_id)
        assert loaded.worst_outcome_status == "degraded"
        assert loaded.tasks[0].outcome_reason == "user:too_short"
    finally:
        store.close()


# --- 3. LLM-judge scaffold ---------------------------------------------- #

def test_llm_judge_fail_flags_degraded():
    judge = llm_judge(
        name="tone", model=lambda prompt: "FAIL — too terse", rubric="Is it polite?",
        sample_rate=1.0,
    )
    v = judge("some output")
    assert v is not None
    assert v.status == "degraded"
    assert v.reason == "user:judge:tone"


def test_llm_judge_pass_returns_none():
    judge = llm_judge(name="tone", model=lambda p: "PASS", rubric="ok?", sample_rate=1.0)
    assert judge("out") is None


def test_llm_judge_unparseable_is_contained_pass():
    judge = llm_judge(name="tone", model=lambda p: "¯\\_(ツ)_/¯", rubric="ok?", sample_rate=1.0)
    assert judge("out") is None  # can't parse → pass, never a failure


def test_llm_judge_error_is_contained_pass():
    def broken_model(prompt):
        raise RuntimeError("provider down")

    judge = llm_judge(name="tone", model=broken_model, rubric="ok?", sample_rate=1.0)
    assert judge("out") is None


def test_llm_judge_timeout_is_contained_pass():
    def slow_model(prompt):
        time.sleep(2.0)
        return "FAIL"

    judge = llm_judge(name="tone", model=slow_model, rubric="ok?", sample_rate=1.0, timeout=0.2)
    start = time.monotonic()
    result = judge("out")
    elapsed = time.monotonic() - start
    assert result is None            # timed out → contained pass
    assert elapsed < 1.5             # returned near the timeout, not after 2s


def test_llm_judge_sampling_is_per_run():
    # sample_rate=0 → the runner skips the judge entirely (base unchanged),
    # regardless of what the model would say.
    judge = llm_judge(name="tone", model=lambda p: "FAIL", rubric="ok?", sample_rate=0.0)
    v = run_checks([judge], "out", OK, "run-xyz")
    assert v.status == "ok"
    # sample_rate=1 → it runs.
    judge_on = llm_judge(name="tone", model=lambda p: "FAIL", rubric="ok?", sample_rate=1.0)
    v = run_checks([judge_on], "out", OK, "run-xyz")
    assert v.status == "degraded"
