"""Integration tests for structural outcome detection through PapayyaRun.

Plan 02 wires papayya.outcomes inspectors into ``_post_call_success``.
These tests drive a real run through the wrapper and assert that
TaskEntry rows land with the right outcome_status/outcome_reason, and
that Plan 01's incremental aggregation in SQLiteStore reflects the
verdict on the parent run.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from papayya.durable.run import PapayyaRun
from papayya.durable.sqlite_store import SQLiteStore
from papayya.durable.types import DurableRunConfig


def _make_run(store=None) -> PapayyaRun:
    return PapayyaRun(DurableRunConfig(agent="test-agent", store=store))


# --- structural detection through @step -------------------------------- #

def test_step_returning_none_is_degraded():
    run = _make_run()

    def returns_none():
        return None

    run.step("noop", returns_none)()

    entry = run._cache["noop"]
    assert entry.outcome_status == "degraded"
    assert entry.outcome_reason == "empty_none"


def test_step_returning_value_is_ok():
    run = _make_run()

    def returns_value():
        return {"hits": ["a", "b"]}

    run.step("search", returns_value)()

    entry = run._cache["search"]
    assert entry.outcome_status == "ok"
    assert entry.outcome_reason is None


def test_step_returning_zero_embedding_is_degraded():
    run = _make_run()

    def embed():
        return {"embedding": [0.0] * 1536}

    run.step("embed", embed)()

    entry = run._cache["embed"]
    assert entry.outcome_status == "degraded"
    assert entry.outcome_reason == "degenerate_embedding"


# --- aggregation under SQLiteStore ------------------------------------- #

def test_mixed_ok_and_degraded_run_aggregates_to_degraded(tmp_path):
    store = SQLiteStore(str(tmp_path / "agg.db"))
    try:
        run = _make_run(store=store)

        run.step("ok-step", lambda: {"x": 1})()
        run.step("degraded-step", lambda: [])()

        loaded = store.load(run.run_id)
        assert loaded is not None
        assert loaded.worst_outcome_status == "degraded"
        assert loaded.degraded_count == 1

        # Verify each row landed with the expected per-step status.
        by_label = {t.label: t for t in loaded.tasks}
        assert by_label["ok-step"].outcome_status == "ok"
        assert by_label["degraded-step"].outcome_status == "degraded"
        assert by_label["degraded-step"].outcome_reason == "empty_sequence"
    finally:
        store.close()


# --- LLM stop-reason path --------------------------------------------- #

def test_llm_step_with_length_stop_reason_is_degraded():
    run = _make_run()

    def call_llm():
        return SimpleNamespace(
            model="gpt-4o-mini",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4096, total_tokens=4106),
            choices=[SimpleNamespace(finish_reason="length")],
        )

    run.llm_step("call-llm", call_llm)()

    entry = run._cache["call-llm"]
    assert entry.outcome_status == "degraded"
    assert entry.outcome_reason == "llm_stop_reason:length"
    # The pre-existing LLM extraction must still populate normally.
    assert entry.llm_stop_reason == "length"
    assert entry.llm_total_tokens == 4106


def test_llm_step_with_normal_stop_is_ok():
    run = _make_run()

    def call_llm():
        return SimpleNamespace(
            model="gpt-4o-mini",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            choices=[SimpleNamespace(finish_reason="stop")],
        )

    run.llm_step("call-llm", call_llm)()

    entry = run._cache["call-llm"]
    assert entry.outcome_status == "ok"
    assert entry.outcome_reason is None


# --- negative regression: exception path produces no task row ---------- #

def test_step_that_raises_writes_no_task_row(tmp_path):
    """An exception in a non-LLM step must propagate and NOT produce a
    task row; the parent run's degraded_count stays at 0.

    This guards against accidentally introducing an exception-path
    save_task call — Plan 02 explicitly does not write failed rows.
    """
    store = SQLiteStore(str(tmp_path / "neg.db"))
    try:
        run = _make_run(store=store)

        def boom():
            raise RuntimeError("boom")

        wrapped = run.step("boom", boom)
        with pytest.raises(RuntimeError, match="boom"):
            wrapped()

        loaded = store.load(run.run_id)
        assert loaded is not None
        assert loaded.tasks == []
        assert loaded.worst_outcome_status == "ok"
        assert loaded.degraded_count == 0
    finally:
        store.close()
