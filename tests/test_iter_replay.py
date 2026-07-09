"""Tests for handler-mode replay of ``papayya.iter`` runs (Tier 2, Layer B1).

An iter loop body is a suspended frame, not a registered ``@agent``, so
``replay(run_id)`` cannot discover a callable to re-invoke. Handler mode
(``replay(run_id, handler=fn)``) re-drives the captured item through
``papayya.iter`` once: the replay is itself a fully-recorded run and the
original is marked ``disposition='replayed'``.

These tests exercise the local-SQLite path end to end — create a failed
iter-run in a temp DB, then replay it.
"""

from __future__ import annotations

import pytest

import papayya
from papayya.durable._replay import ReplayError, replay
from papayya.durable.sqlite_store import SQLiteStore


def _make_failed_iter_run(db_path, item) -> str:
    """Drive one item through papayya.iter with a raising body, returning the
    failed run's id. The run lands in ``db_path`` with status='failed',
    disposition NULL, and the item captured as input_snapshot (Layer A)."""
    store = SQLiteStore(str(db_path))
    captured: dict[str, str] = {}
    try:
        with pytest.raises(ValueError):
            for it in papayya.iter(
                [item],
                workload="triage",
                item_id=lambda i: i["id"],
                partition_key=lambda i: i["tenant"],
                store=store,
            ):
                captured["run_id"] = papayya.active_run_id()
                raise ValueError("boom")
    finally:
        store.close()
    return captured["run_id"]


def test_handler_replay_reexecutes_item_and_resolves_original(tmp_path):
    db = tmp_path / "local.db"
    item = {"id": "tkt-1", "tenant": "acme", "text": "refund please"}
    run_id = _make_failed_iter_run(db, item)

    seen: list = []

    def handler(received):
        seen.append(received)
        return {"ok": True}

    result = replay(run_id, handler=handler, db_path=db)

    # Handler ran with the captured item, passed positionally (whole item,
    # not unpacked kwargs).
    assert seen == [item]
    assert result == {"ok": True}

    # Original run left the dead-letter queue.
    assert _disposition(db, run_id) == "replayed"


def test_handler_replay_records_a_fresh_completed_run(tmp_path):
    db = tmp_path / "local.db"
    item = {"id": "tkt-9", "tenant": "globex", "text": "where is my order"}
    run_id = _make_failed_iter_run(db, item)

    replay(run_id, handler=lambda received: "done", db_path=db)

    # A new run distinct from the original exists, recorded as completed,
    # carrying the same workload / attribution / snapshot.
    new_runs = _other_runs(db, exclude=run_id)
    assert len(new_runs) == 1
    store = SQLiteStore(str(db))
    try:
        new = store.load(new_runs[0])
    finally:
        store.close()
    assert new is not None
    assert new.run_id != run_id
    assert new.status == "completed"
    assert new.agent == "triage"
    assert new.partition_key == "globex"
    assert new.input_snapshot == item


def test_handler_replay_propagates_handler_error_but_still_resolves_original(tmp_path):
    db = tmp_path / "local.db"
    item = {"id": "tkt-err", "tenant": "initech", "text": "x"}
    run_id = _make_failed_iter_run(db, item)

    def bad_handler(_received):
        raise RuntimeError("still broken")

    with pytest.raises(RuntimeError, match="still broken"):
        replay(run_id, handler=bad_handler, db_path=db)

    # Original is marked replayed regardless of the replay's own outcome,
    # and the replay run itself recorded as a fresh failure (its own dead
    # letter), mirroring registration-mode semantics.
    assert _disposition(db, run_id) == "replayed"
    store = SQLiteStore(str(db))
    try:
        new = store.load(_other_runs(db, exclude=run_id)[0])
    finally:
        store.close()
    assert new.status == "failed"


def test_handler_and_from_step_are_mutually_exclusive(tmp_path):
    db = tmp_path / "local.db"
    run_id = _make_failed_iter_run(db, {"id": "a", "tenant": "t", "text": "x"})
    with pytest.raises(ReplayError, match="from_step= is not supported with handler="):
        replay(run_id, handler=lambda r: r, from_step=1, db_path=db)


def test_handler_and_agent_module_are_mutually_exclusive(tmp_path):
    db = tmp_path / "local.db"
    run_id = _make_failed_iter_run(db, {"id": "a", "tenant": "t", "text": "x"})
    with pytest.raises(ReplayError, match="not both"):
        replay(run_id, handler=lambda r: r, agent_module="agent.py", db_path=db)


def test_iter_run_without_handler_has_no_registration_to_discover(tmp_path):
    """Capture (Layer A) alone does not make replay(run_id) work on an
    iter-run: with no handler and no matching @agent, discovery fails. This
    pins the design boundary — handler= is required for iter replay."""
    db = tmp_path / "local.db"
    run_id = _make_failed_iter_run(db, {"id": "a", "tenant": "t", "text": "x"})
    # No agent.py with a 'triage' registration in cwd → discovery raises.
    with pytest.raises(ReplayError):
        replay(run_id, db_path=db, agent_module=tmp_path / "no_such_agent.py")


# --------------------------------------------------------------------------- #


def _other_runs(db_path, *, exclude: str) -> list[str]:
    """Run ids in the DB other than ``exclude``, oldest first."""
    import sqlite3

    from papayya.durable import _schema

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            f"SELECT {_schema.COL_ITEM_ID} AS run_id FROM {_schema.TBL_ITEMS} WHERE {_schema.COL_ITEM_ID} != ? "
            f"ORDER BY created_at",
            (exclude,),
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def _disposition(db_path, run_id: str):
    """Read a run's dlq_disposition column directly (not on RunCheckpoint)."""
    import sqlite3

    from papayya.durable import _schema

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            f"SELECT {_schema.COL_ITEM_DLQ_DISPOSITION} AS disp "
            f"FROM {_schema.TBL_ITEMS} WHERE {_schema.COL_ITEM_ID} = ?",
            (run_id,),
        ).fetchone()
    finally:
        conn.close()
    return row["disp"] if row is not None else None
