"""Outcome-field round-trip + aggregation tests for Plan 01.

Covers TaskEntry.outcome_status / outcome_reason and the denormalized
RunCheckpoint.worst_outcome_status / degraded_count aggregates. Plan 01
ships only the data shape; Plan 02 wires the writer. These tests
construct entries with explicit outcome values and verify they round-trip
through SQLiteStore and that the parent-run aggregates update incrementally
on each save_task.
"""

from __future__ import annotations

from datetime import datetime, timezone

from papayya.durable import _schema
from papayya.durable.sqlite_store import SQLiteStore
from papayya.durable.types import RunCheckpoint, TaskEntry, _outcome_severity


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_run(store: SQLiteStore, run_id: str = "r1") -> None:
    store.create(
        RunCheckpoint(
            run_id=run_id,
            agent="test_agent",
            tasks=[],
            status="running",
            created_at=_now(),
            updated_at=_now(),
        )
    )


def _make_task(label: str, *, status: str = "ok", reason: str | None = None) -> TaskEntry:
    return TaskEntry(
        label=label,
        result={"v": label},
        duration_ms=1,
        completed_at=_now(),
        outcome_status=status,
        outcome_reason=reason,
    )


# --- 1. defaults ------------------------------------------------------- #

def test_task_entry_defaults_to_ok():
    entry = TaskEntry(label="x", result=None, duration_ms=0, completed_at=_now())
    assert entry.outcome_status == "ok"
    assert entry.outcome_reason is None


def test_run_checkpoint_defaults_to_ok():
    ckpt = RunCheckpoint(run_id="r", agent="a", tasks=[], status="running")
    assert ckpt.worst_outcome_status == "ok"
    assert ckpt.degraded_count == 0


# --- 2. SQLite round-trip: TaskEntry ----------------------------------- #

def test_sqlite_task_entry_round_trips_outcome_fields(tmp_path):
    store = SQLiteStore(str(tmp_path / "p01.db"))
    try:
        _make_run(store)
        store.save_task(
            "r1",
            _make_task("step", status="degraded", reason="empty_collection"),
        )

        loaded = store.load("r1")
        assert loaded is not None
        assert len(loaded.tasks) == 1
        assert loaded.tasks[0].outcome_status == "degraded"
        assert loaded.tasks[0].outcome_reason == "empty_collection"
    finally:
        store.close()


# --- 3. SQLite round-trip: RunCheckpoint aggregate --------------------- #

def test_sqlite_run_checkpoint_round_trips_aggregates(tmp_path):
    store = SQLiteStore(str(tmp_path / "p01.db"))
    try:
        _make_run(store)
        store.save_task("r1", _make_task("a", status="degraded", reason="empty_none"))

        loaded = store.load("r1")
        assert loaded is not None
        assert loaded.worst_outcome_status == "degraded"
        assert loaded.degraded_count == 1
    finally:
        store.close()


# --- 4. aggregation ---------------------------------------------------- #

def test_aggregation_counts_only_non_ok_tasks(tmp_path):
    store = SQLiteStore(str(tmp_path / "p01.db"))
    try:
        _make_run(store)
        store.save_task("r1", _make_task("a", status="ok"))
        store.save_task("r1", _make_task("b", status="degraded", reason="empty_sequence"))
        store.save_task("r1", _make_task("c", status="ok"))

        loaded = store.load("r1")
        assert loaded is not None
        assert loaded.worst_outcome_status == "degraded"
        assert loaded.degraded_count == 1
    finally:
        store.close()


# --- 5. severity order ------------------------------------------------- #

def test_severity_helper_order():
    assert _outcome_severity("ok") < _outcome_severity("degraded") < _outcome_severity("failed")
    # Unknown statuses fall back to severity 0 (ok-equivalent) — fail-safe.
    assert _outcome_severity("bogus") == _outcome_severity("ok")


def test_severity_order_monotonic_in_aggregate(tmp_path):
    # degraded then failed → failed
    store_a = SQLiteStore(str(tmp_path / "asc.db"))
    try:
        _make_run(store_a, "r_asc")
        store_a.save_task("r_asc", _make_task("a", status="degraded", reason="empty_dict"))
        store_a.save_task("r_asc", _make_task("b", status="failed", reason="boom"))
        loaded = store_a.load("r_asc")
        assert loaded is not None
        assert loaded.worst_outcome_status == "failed"
        assert loaded.degraded_count == 2
    finally:
        store_a.close()

    # failed then degraded → stays failed
    store_b = SQLiteStore(str(tmp_path / "desc.db"))
    try:
        _make_run(store_b, "r_desc")
        store_b.save_task("r_desc", _make_task("a", status="failed", reason="boom"))
        store_b.save_task("r_desc", _make_task("b", status="degraded", reason="empty_dict"))
        loaded = store_b.load("r_desc")
        assert loaded is not None
        assert loaded.worst_outcome_status == "failed"
        assert loaded.degraded_count == 2
    finally:
        store_b.close()


# --- 6. fresh-DB schema ------------------------------------------------ #

def test_fresh_db_has_outcome_columns(tmp_path):
    store = SQLiteStore(str(tmp_path / "fresh.db"))
    try:
        conn = store._conn
        task_cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()}
        run_cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)").fetchall()}

        assert _schema.COL_TASK_OUTCOME_STATUS in task_cols
        assert _schema.COL_TASK_OUTCOME_REASON in task_cols
        assert _schema.COL_RUN_WORST_OUTCOME_STATUS in run_cols
        assert _schema.COL_RUN_DEGRADED_COUNT in run_cols

        # Schema version must have advanced to '11'.
        version = conn.execute(
            "SELECT value FROM _meta WHERE key = 'schema_version'"
        ).fetchone()[0]
        assert version == "11"
    finally:
        store.close()
