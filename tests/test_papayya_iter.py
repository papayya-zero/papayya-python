"""Tests for ``papayya.iter`` (Plan 10).

The iterator opens a per-item :class:`PapayyaRun`, installs it on the
``_ACTIVE_RUN`` contextvar, and auto-closes on body exit. Module-level
``mark_degraded`` / ``mark_outcome`` write a synthetic ``TaskEntry``
through the run's store. ``PapayyaRun`` defaults to ``MemoryStore``;
where these tests need run-level outcome aggregation (Plan 01's
``worst_outcome_status`` / ``degraded_count`` math) they swap to
``SQLiteStore``, which is the only in-tree store with that wiring.

Test seam: ``papayya.iter`` constructs ``PapayyaRun`` directly with no
``store=`` kwarg. To inject a store we monkey-patch
``papayya.iterators.PapayyaRun`` with a tiny subclass that overrides the
config's ``store`` field. Documented inline.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

import papayya
from papayya import iterators
from papayya.durable import PapayyaRun
from papayya.durable.sqlite_store import SQLiteStore
from papayya.durable.types import DurableRunConfig


# ── Test seam: capture + optional store injection ─────────────────────────


class _CapturingRun(PapayyaRun):
    """Subclass that records construction args and (optionally) forces a
    shared store into every run the iterator opens.

    The iterator's source of truth is ``papayya.iterators.PapayyaRun``;
    swapping that binding swaps everything the iterator builds. Tests use
    this both to inspect what configs the iterator passed and to point
    every per-item run at a single ``SQLiteStore`` so they can read back
    the aggregated outcome columns.
    """

    captured_configs: list[DurableRunConfig] = []
    captured_runs: list["_CapturingRun"] = []
    shared_store: Any = None
    fail_complete: bool = False  # toggled by test 8

    @classmethod
    def reset(cls) -> None:
        cls.captured_configs = []
        cls.captured_runs = []
        cls.shared_store = None
        cls.fail_complete = False

    def __init__(self, config: DurableRunConfig) -> None:
        if _CapturingRun.shared_store is not None:
            config.store = _CapturingRun.shared_store
        _CapturingRun.captured_configs.append(config)
        super().__init__(config)
        self.fail_calls: list[Any] = []
        _CapturingRun.captured_runs.append(self)

    def fail(self, error: Any = None):  # type: ignore[override]
        self.fail_calls.append(error)
        return super().fail(error)

    def complete(self, output: Any = None):  # type: ignore[override]
        if _CapturingRun.fail_complete:
            raise RuntimeError("complete-time boom")
        return super().complete(output)


@pytest.fixture(autouse=True)
def _patch_papayya_run(monkeypatch):
    """Swap Item (né PapayyaRun) for the capturing subclass for every test."""
    _CapturingRun.reset()
    # _iter_gen constructs via the module-level ``Item`` binding (Plan 34
    # rename); patch both names so either spelling routes through the seam.
    monkeypatch.setattr(iterators, "Item", _CapturingRun)
    monkeypatch.setattr(iterators, "PapayyaRun", _CapturingRun)
    yield
    _CapturingRun.reset()


# ── 1. Happy path: items pass through unchanged ───────────────────────────

def test_iter_yields_items_by_identity():
    items = [{"id": "1", "t": "a"}, {"id": "2", "t": "b"}, {"id": "3", "t": "a"}]
    out = list(
        papayya.iter(
            items,
            workload="ingest",
            item_id=lambda i: i["id"],
            partition_key=lambda i: i["t"],
        )
    )
    assert len(out) == 3
    for original, yielded in zip(items, out):
        assert yielded is original  # identity, not equality


# ── 2. Per-item PapayyaRun construction ───────────────────────────────────

def test_iter_constructs_one_run_per_item():
    items = [{"id": "a", "t": "T1"}, {"id": "b", "t": "T2"}]
    for _ in papayya.iter(
        items,
        workload="my-workload",
        item_id=lambda i: i["id"],
        partition_key=lambda i: i["t"],
    ):
        pass

    assert len(_CapturingRun.captured_configs) == 2
    cfg1, cfg2 = _CapturingRun.captured_configs
    assert cfg1.agent == "my-workload"
    assert cfg1.item_id == "a"
    assert cfg1.partition_key == "T1"
    assert cfg2.item_id == "b"
    assert cfg2.partition_key == "T2"


# ── 3. item_id / partition_key stringified ────────────────────────────────

def test_iter_stringifies_item_id_and_partition_key():
    for _ in papayya.iter(
        [42],
        workload="w",
        item_id=lambda _: 42,
        partition_key=lambda _: 7,
    ):
        pass

    (cfg,) = _CapturingRun.captured_configs
    assert cfg.item_id == "42"
    assert cfg.partition_key == "7"
    assert isinstance(cfg.item_id, str)
    assert isinstance(cfg.partition_key, str)


# ── 4. Missing required kwargs → TypeError ────────────────────────────────

def test_iter_requires_workload():
    with pytest.raises(TypeError):
        list(papayya.iter([1], item_id=lambda i: "x", partition_key=lambda i: "y"))  # type: ignore[call-arg]


def test_iter_requires_item_id():
    with pytest.raises(TypeError):
        list(papayya.iter([1], workload="w", partition_key=lambda i: "y"))  # type: ignore[call-arg]


def test_iter_requires_partition_key():
    with pytest.raises(TypeError):
        list(papayya.iter([1], workload="w", item_id=lambda i: "x"))  # type: ignore[call-arg]


# ── 5. Empty iterable → no runs opened ────────────────────────────────────

def test_iter_empty_iterable_opens_no_runs():
    out = list(
        papayya.iter(
            [],
            workload="w",
            item_id=lambda i: "x",
            partition_key=lambda i: "y",
        )
    )
    assert out == []
    assert _CapturingRun.captured_configs == []
    assert _CapturingRun.captured_runs == []


# ── 6. Exception in body re-raises; run.fail called ───────────────────────

def test_iter_exception_in_body_calls_fail_and_reraises():
    items = [1, 2, 3]
    seen: list[int] = []
    with pytest.raises(RuntimeError, match="boom"):
        for x in papayya.iter(
            items,
            workload="w",
            item_id=lambda i: str(i),
            partition_key=lambda i: "p",
        ):
            seen.append(x)
            if x == 2:
                raise RuntimeError("boom")

    # Body saw items 1 and 2 only.
    assert seen == [1, 2]
    # Two runs were opened (one per item).
    assert len(_CapturingRun.captured_runs) == 2
    r1, r2 = _CapturingRun.captured_runs
    assert r1.fail_calls == []  # item 1 completed cleanly
    # item 2's run.fail() carries the synthetic marker, not the original
    # exception message — Python's for-loop cleanup sends GeneratorExit
    # into the generator, so the iterator cannot observe the original
    # RuntimeError. The original still propagates to the caller (asserted
    # via the pytest.raises match above); the audit trail of the failure
    # lives on the synthetic TaskEntry (test 7).
    assert r2.fail_calls == ["loop_body_exception"]


# ── 7. Failed-body produces a synthetic 'failed' TaskEntry ────────────────

def test_iter_failed_body_writes_synthetic_failed_entry(tmp_path):
    db = SQLiteStore(str(tmp_path / "iter.db"))
    try:
        _CapturingRun.shared_store = db
        with pytest.raises(ValueError):
            for x in papayya.iter(
                [{"id": "only", "t": "T"}],
                workload="w",
                item_id=lambda i: i["id"],
                partition_key=lambda i: i["t"],
            ):
                raise ValueError("schema mismatch")

        (run,) = _CapturingRun.captured_runs
        loaded = db.load(run.run_id)
        assert loaded is not None
        assert any(
            t.outcome_status == "failed" and t.outcome_reason == "loop_body_exception"
            for t in loaded.tasks
        )
        # Run status should be 'failed' from run.fail().
        assert loaded.status == "failed"
    finally:
        db.close()


# ── 7b. Item captured as the run's input_snapshot (item-replay) ───────────

def test_iter_passes_item_as_input_snapshot_to_config():
    """The iterator hands each item to DurableRunConfig.input_snapshot so the
    run row carries the payload that produced it (the data foundation for
    replay). Without a decorator above it, iter has no other way to capture
    the input."""
    items = [{"id": "a", "t": "T1"}, {"id": "b", "t": "T2"}]
    for _ in papayya.iter(
        items,
        workload="w",
        item_id=lambda i: i["id"],
        partition_key=lambda i: i["t"],
    ):
        pass

    cfg1, cfg2 = _CapturingRun.captured_configs
    assert cfg1.input_snapshot == {"id": "a", "t": "T1"}
    assert cfg2.input_snapshot == {"id": "b", "t": "T2"}
    # Identity: the exact item object is captured, not a copy.
    assert cfg1.input_snapshot is items[0]


def test_iter_persists_item_snapshot_through_store(tmp_path):
    """End-to-end: a completed iter-run's row carries the item as
    input_snapshot after a real SQLite round-trip — the column that was NULL
    before this change."""
    db = SQLiteStore(str(tmp_path / "snap.db"))
    try:
        _CapturingRun.shared_store = db
        for _ in papayya.iter(
            [{"id": "only", "t": "T", "text": "hello"}],
            workload="w",
            item_id=lambda i: i["id"],
            partition_key=lambda i: i["t"],
        ):
            pass

        (run,) = _CapturingRun.captured_runs
        loaded = db.load(run.run_id)
        assert loaded is not None
        assert loaded.input_snapshot == {"id": "only", "t": "T", "text": "hello"}
    finally:
        db.close()


def test_iter_failed_run_carries_snapshot_for_replay(tmp_path):
    """The replay-relevant case: when the loop body raises, the failed run
    still carries its input_snapshot, so the run is re-drivable from its id
    (status='failed' + non-NULL snapshot are the two preconditions the replay
    path checks)."""
    db = SQLiteStore(str(tmp_path / "fail_snap.db"))
    try:
        _CapturingRun.shared_store = db
        with pytest.raises(ValueError):
            for x in papayya.iter(
                [{"id": "boom", "t": "T"}],
                workload="w",
                item_id=lambda i: i["id"],
                partition_key=lambda i: i["t"],
            ):
                raise ValueError("schema mismatch")

        (run,) = _CapturingRun.captured_runs
        loaded = db.load(run.run_id)
        assert loaded is not None
        assert loaded.status == "failed"
        assert loaded.input_snapshot == {"id": "boom", "t": "T"}
    finally:
        db.close()


# ── 8. Contextvar always reset, even when complete() raises ───────────────

def test_iter_resets_contextvar_when_complete_raises(caplog):
    _CapturingRun.fail_complete = True
    with caplog.at_level(logging.ERROR, logger="papayya.iter"):
        for _ in papayya.iter(
            [{"id": "x", "t": "T"}],
            workload="w",
            item_id=lambda i: i["id"],
            partition_key=lambda i: i["t"],
        ):
            pass

    assert iterators._ACTIVE_RUN.get() is None
    assert any("run.complete()" in rec.message for rec in caplog.records)


# ── 9. mark_degraded inside iter writes a degraded TaskEntry ──────────────

def test_mark_degraded_inside_iter_writes_degraded_entry(tmp_path):
    db = SQLiteStore(str(tmp_path / "mark_degraded.db"))
    try:
        _CapturingRun.shared_store = db
        for _ in papayya.iter(
            [{"id": "alpha", "t": "acme"}],
            workload="w",
            item_id=lambda i: i["id"],
            partition_key=lambda i: i["t"],
        ):
            papayya.mark_degraded("schema_mismatch")

        (run,) = _CapturingRun.captured_runs
        loaded = db.load(run.run_id)
        assert loaded is not None
        mark_entries = [t for t in loaded.tasks if t.outcome_status == "degraded"]
        assert len(mark_entries) == 1
        entry = mark_entries[0]
        assert entry.outcome_reason == "schema_mismatch"
        assert entry.item_id == "alpha"
        assert entry.partition_key == "acme"
        assert entry.label.startswith("papayya.mark/")
    finally:
        db.close()


# ── 10. mark_degraded outside iter logs warning, no raise ─────────────────

def test_mark_degraded_outside_iter_logs_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="papayya.iter"):
        result = papayya.mark_degraded("nothing-to-mark")
    assert result is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("outside an active papayya.iter" in r.message for r in warnings)


# ── 11. mark_outcome('failed') writes a failed entry ──────────────────────

def test_mark_outcome_failed_writes_failed_entry(tmp_path):
    db = SQLiteStore(str(tmp_path / "mark_failed.db"))
    try:
        _CapturingRun.shared_store = db
        for _ in papayya.iter(
            [{"id": "x", "t": "T"}],
            workload="w",
            item_id=lambda i: i["id"],
            partition_key=lambda i: i["t"],
        ):
            papayya.mark_outcome("failed", "schema_violation")

        (run,) = _CapturingRun.captured_runs
        loaded = db.load(run.run_id)
        assert loaded is not None
        marks = [t for t in loaded.tasks if t.outcome_status == "failed"]
        assert len(marks) == 1
        assert marks[0].outcome_reason == "schema_violation"
    finally:
        db.close()


# ── 12. mark_outcome('ok') still writes a row ─────────────────────────────

def test_mark_outcome_ok_writes_audit_row(tmp_path):
    db = SQLiteStore(str(tmp_path / "mark_ok.db"))
    try:
        _CapturingRun.shared_store = db
        for _ in papayya.iter(
            [{"id": "x", "t": "T"}],
            workload="w",
            item_id=lambda i: i["id"],
            partition_key=lambda i: i["t"],
        ):
            papayya.mark_outcome("ok")

        (run,) = _CapturingRun.captured_runs
        loaded = db.load(run.run_id)
        assert loaded is not None
        ok_marks = [t for t in loaded.tasks if t.label.startswith("papayya.mark/")]
        assert len(ok_marks) == 1
        assert ok_marks[0].outcome_status == "ok"
        assert ok_marks[0].outcome_reason is None
    finally:
        db.close()


# ── 13. mark_outcome rejects unknown status ───────────────────────────────

def test_mark_outcome_rejects_unknown_status():
    with pytest.raises(ValueError, match="must be"):
        papayya.mark_outcome("weird")


# ── 14. Aggregation: mark_degraded escalates worst_outcome_status ─────────

def test_mark_degraded_escalates_run_worst_outcome(tmp_path):
    db = SQLiteStore(str(tmp_path / "agg.db"))
    try:
        _CapturingRun.shared_store = db
        for x in papayya.iter(
            [1, 2, 3],
            workload="w",
            item_id=lambda i: str(i),
            partition_key=lambda i: "p",
        ):
            if x == 2:
                papayya.mark_degraded("only-item-2")

        assert len(_CapturingRun.captured_runs) == 3
        r1, r2, r3 = _CapturingRun.captured_runs
        l1 = db.load(r1.run_id)
        l2 = db.load(r2.run_id)
        l3 = db.load(r3.run_id)
        assert l1 is not None and l2 is not None and l3 is not None
        assert l1.worst_outcome_status == "ok"
        assert l1.degraded_count == 0
        assert l2.worst_outcome_status == "degraded"
        assert l2.degraded_count >= 1
        assert l3.worst_outcome_status == "ok"
        assert l3.degraded_count == 0
    finally:
        db.close()


# ── 15. Nested iter: inner replaces outer in the contextvar ───────────────

def test_iter_nested_inner_overrides_outer(tmp_path):
    db = SQLiteStore(str(tmp_path / "nested.db"))
    try:
        _CapturingRun.shared_store = db
        for outer in papayya.iter(
            [{"id": "outer-1", "t": "T"}],
            workload="outer",
            item_id=lambda i: i["id"],
            partition_key=lambda i: i["t"],
        ):
            for inner in papayya.iter(
                [{"id": "inner-1", "t": "U"}],
                workload="inner",
                item_id=lambda i: i["id"],
                partition_key=lambda i: i["t"],
            ):
                papayya.mark_degraded("inner-mark")

        # Two runs captured, in construction order outer then inner.
        assert len(_CapturingRun.captured_runs) == 2
        outer_run, inner_run = _CapturingRun.captured_runs
        outer_ckpt = db.load(outer_run.run_id)
        inner_ckpt = db.load(inner_run.run_id)
        assert outer_ckpt is not None
        assert inner_ckpt is not None
        # Mark went to inner only.
        assert inner_ckpt.worst_outcome_status == "degraded"
        assert outer_ckpt.worst_outcome_status == "ok"
        assert not any(t.outcome_status == "degraded" for t in outer_ckpt.tasks)
    finally:
        db.close()


# ── 16. No contextvar leak after iteration ────────────────────────────────

def test_iter_clears_contextvar_on_clean_exit():
    for _ in papayya.iter(
        [{"id": "x", "t": "T"}],
        workload="w",
        item_id=lambda i: i["id"],
        partition_key=lambda i: i["t"],
    ):
        assert iterators._ACTIVE_RUN.get() is not None

    assert iterators._ACTIVE_RUN.get() is None


# ── 17. Public symbols are exported ───────────────────────────────────────

def test_top_level_symbols_exported():
    assert callable(papayya.iter)
    assert callable(papayya.mark_degraded)
    assert callable(papayya.mark_outcome)
    # The function is ours, not the builtin.
    assert papayya.iter is iterators.iter
