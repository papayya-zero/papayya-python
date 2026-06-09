"""Tests for the leaf-level adoption surface: ``@papayya.llm`` / ``@papayya.step``.

The contract: decorate the function that calls your model (not your
orchestration code), drive items with ``papayya.iter``, and get automatic
ran-vs-worked detection + tenant attribution with no ``run`` threaded through
any business signature. Called outside an active iter run, the function runs
bare.
"""

from __future__ import annotations

import asyncio
import inspect

import papayya
from papayya.durable.sqlite_store import SQLiteStore


def _refusal(text):
    # Anthropic-dict response shape with a degenerate stop reason.
    return {
        "model": "m",
        "stop_reason": "refusal",
        "usage": {"input_tokens": 5, "output_tokens": 1},
        "content": "",
    }


def _ok(text):
    return {
        "model": "m",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 3},
        "content": text,
    }


def _drive(call, item, store):
    rid = None
    for it in papayya.iter(
        [item],
        workload="w",
        item_id=lambda i: i["id"],
        partition_key=lambda i: i["t"],
        store=store,
    ):
        call(it["id"])
        rid = papayya.active_run_id()
    return rid


def test_leaf_llm_auto_detects_degraded_inside_iter(tmp_path):
    db = SQLiteStore(str(tmp_path / "leaf.db"))
    try:
        decorated = papayya.llm(_refusal)
        rid = _drive(decorated, {"id": "a", "t": "acme"}, db)
        cp = db.load(rid)
        assert cp is not None
        assert cp.worst_outcome_status == "degraded"
        # Tenant rides onto the task row with no signature change.
        assert any(t.partition_key == "acme" for t in cp.tasks)
    finally:
        db.close()


def test_leaf_llm_ok_when_healthy(tmp_path):
    db = SQLiteStore(str(tmp_path / "leaf_ok.db"))
    try:
        rid = _drive(papayya.llm(_ok), {"id": "a", "t": "acme"}, db)
        assert db.load(rid).worst_outcome_status == "ok"
    finally:
        db.close()


def test_leaf_runs_bare_outside_iter():
    @papayya.llm
    def call_model(text):
        return {"content": "x"}

    # No active iter run — the function just calls through, no recording, no error.
    assert call_model("hi") == {"content": "x"}
    assert papayya.active_run_id() is None


def test_leaf_step_runs_empty_result_inspector(tmp_path):
    db = SQLiteStore(str(tmp_path / "leaf_step.db"))
    try:
        @papayya.step
        def retrieve(qid):
            return []  # empty result → degraded via inspect_empty

        rid = _drive(retrieve, {"id": "a", "t": "acme"}, db)
        assert db.load(rid).worst_outcome_status == "degraded"
    finally:
        db.close()


def test_leaf_async_llm_detects_degraded_and_stays_a_coroutine(tmp_path):
    db = SQLiteStore(str(tmp_path / "leaf_async.db"))
    try:
        @papayya.llm
        async def acall(text):
            return _refusal(text)

        # The decorated function must still introspect as a coroutine fn, or
        # frameworks that decide whether to await it will break.
        assert inspect.iscoroutinefunction(acall)

        rid = None
        for it in papayya.iter(
            [{"id": "a", "t": "acme"}],
            workload="w",
            item_id=lambda i: i["id"],
            partition_key=lambda i: i["t"],
            store=db,
        ):
            asyncio.run(acall(it["id"]))
            rid = papayya.active_run_id()
        assert db.load(rid).worst_outcome_status == "degraded"
    finally:
        db.close()


def test_leaf_async_runs_bare_outside_iter():
    @papayya.llm
    async def acall(text):
        return {"content": "x"}

    assert inspect.iscoroutinefunction(acall)
    assert asyncio.run(acall("hi")) == {"content": "x"}


def test_leaf_multiple_calls_per_item_are_distinct_steps(tmp_path):
    db = SQLiteStore(str(tmp_path / "leaf_multi.db"))
    try:
        decorated = papayya.llm(_ok)
        rid = None
        for it in papayya.iter(
            [{"id": "a", "t": "acme"}],
            workload="w",
            item_id=lambda i: i["id"],
            partition_key=lambda i: i["t"],
            store=db,
        ):
            decorated("one")
            decorated("two")  # same fn, second call in same run
            rid = papayya.active_run_id()
        labels = [t.label for t in db.load(rid).tasks]
        # First call keeps the clean label; the second is suffixed so it is its
        # own durable step rather than a cache collision.
        assert "_ok" in labels
        assert "_ok#1" in labels
    finally:
        db.close()
