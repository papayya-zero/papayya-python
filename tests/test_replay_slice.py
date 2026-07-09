"""Plan 34 Unit 2b — slice replay.

``replay_slice(run_id)`` opens a NEW run over the items of an existing
run whose ``worst_outcome_status != 'ok'`` (optionally narrowed to one
tenant), re-drives each captured input, and links everything via
``replayed_from`` at both the run and item level. This is the acceptance
sentence's recovery verb; single-item ``replay()`` stays available.
"""

from __future__ import annotations

import sqlite3
import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

import papayya
from papayya import cli as cli_module
from papayya.durable import _schema
from papayya.durable._replay import ReplayError, replay_slice
from papayya.durable.sqlite_store import SQLiteStore


TICKETS = [
    {"id": "t1", "tenant": "acme", "ok": True},
    {"id": "t2", "tenant": "acme", "ok": False},   # degraded
    {"id": "t3", "tenant": "globex", "ok": False}, # degraded
]


def _seed_run(db_path: Path) -> str:
    """Drive one iter() invocation: t1 ok, t2/t3 marked degraded.
    Returns the run row's id."""
    store = SQLiteStore(str(db_path))
    try:
        for t in papayya.iter(
            TICKETS,
            agent="triage",
            item_id=lambda t: t["id"],
            partition_key=lambda t: t["tenant"],
            store=store,
        ):
            if not t["ok"]:
                papayya.mark_degraded("refusal")
    finally:
        store.close()
    conn = sqlite3.connect(db_path)
    try:
        (run_id,) = conn.execute("SELECT run_id FROM runs").fetchone()
    finally:
        conn.close()
    return run_id


def _rows(db_path: Path, sql: str, args: tuple = ()) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, args)]
    finally:
        conn.close()


class TestSliceReplayHandlerMode:
    def test_replays_not_ok_items_into_new_run(self, tmp_path: Path) -> None:
        db = tmp_path / "slice.db"
        old_run = _seed_run(db)

        handled: list = []
        summary = replay_slice(
            old_run, handler=lambda t: handled.append(t["id"]), db_path=db
        )

        assert summary["selected"] == 2
        assert summary["replayed_ok"] == 2
        assert summary["replay_failed"] == 0
        assert summary["skipped_no_snapshot"] == 0
        assert sorted(handled) == ["t2", "t3"]

        # New run row: linked to the source, correct totals, terminal.
        (new_run,) = _rows(
            db, "SELECT * FROM runs WHERE run_id = ?", (summary["new_run_id"],)
        )
        assert new_run["replayed_from"] == old_run
        assert new_run["total_items"] == 2
        assert new_run["status"] == "completed"

        # Each fresh item links back to the item it re-drove.
        new_items = _rows(
            db, "SELECT * FROM items WHERE run_id = ?", (summary["new_run_id"],)
        )
        assert len(new_items) == 2
        sources = {i["replayed_from"] for i in new_items}
        degraded_sources = {
            r["id"] for r in _rows(
                db,
                "SELECT id FROM items WHERE run_id = ? AND worst_outcome_status != 'ok'",
                (old_run,),
            )
        }
        assert sources == degraded_sources

    def test_tenant_filter_narrows_slice(self, tmp_path: Path) -> None:
        db = tmp_path / "slice.db"
        old_run = _seed_run(db)

        handled: list = []
        summary = replay_slice(
            old_run, tenant="acme",
            handler=lambda t: handled.append(t["id"]), db_path=db,
        )
        assert summary["selected"] == 1
        assert handled == ["t2"]
        new_items = _rows(
            db, "SELECT * FROM items WHERE run_id = ?", (summary["new_run_id"],)
        )
        assert [i["partition_key"] for i in new_items] == ["acme"]

    def test_failed_source_items_leave_the_dlq(self, tmp_path: Path) -> None:
        db = tmp_path / "slice.db"
        store = SQLiteStore(str(db))
        try:
            with pytest.raises(ValueError):
                for t in papayya.iter(
                    TICKETS,
                    agent="triage",
                    item_id=lambda t: t["id"],
                    partition_key=lambda t: t["tenant"],
                    store=store,
                ):
                    if t["id"] == "t3":
                        raise ValueError("boom")
        finally:
            store.close()
        (run_id,) = [r["run_id"] for r in _rows(db, "SELECT run_id FROM runs")]

        summary = replay_slice(run_id, handler=lambda t: None, db_path=db)
        # t3 failed (synthetic failed entry escalates worst_outcome_status).
        assert summary["selected"] == 1

        (source,) = _rows(db, "SELECT * FROM items WHERE id IN "
                              "(SELECT replayed_from FROM items WHERE run_id = ?)",
                          (summary["new_run_id"],))
        assert source["item_id"] == "t3"
        assert source["dlq_disposition"] == _schema.DLQ_REPLAYED

    def test_failing_replay_items_do_not_abort_the_slice(self, tmp_path: Path) -> None:
        db = tmp_path / "slice.db"
        old_run = _seed_run(db)

        def flaky(t):
            if t["id"] == "t2":
                raise RuntimeError("still broken")

        summary = replay_slice(old_run, handler=flaky, db_path=db)
        assert summary["selected"] == 2
        assert summary["replayed_ok"] == 1
        assert summary["replay_failed"] == 1
        (new_run,) = _rows(
            db, "SELECT * FROM runs WHERE run_id = ?", (summary["new_run_id"],)
        )
        assert new_run["status"] == "partial"

    def test_no_not_ok_items_raises(self, tmp_path: Path) -> None:
        db = tmp_path / "slice.db"
        store = SQLiteStore(str(db))
        try:
            for _ in papayya.iter(
                [{"id": "clean", "tenant": "acme"}],
                agent="triage",
                item_id=lambda t: t["id"],
                partition_key=lambda t: t["tenant"],
                store=store,
            ):
                pass
        finally:
            store.close()
        (run_id,) = [r["run_id"] for r in _rows(db, "SELECT run_id FROM runs")]
        with pytest.raises(ReplayError, match="nothing to replay"):
            replay_slice(run_id, handler=lambda t: None, db_path=db)

    def test_unknown_run_raises(self, tmp_path: Path) -> None:
        db = tmp_path / "slice.db"
        SQLiteStore(str(db)).close()
        with pytest.raises(ReplayError, match="not found"):
            replay_slice("nope", handler=lambda t: None, db_path=db)

    def test_handler_and_agent_module_are_exclusive(self, tmp_path: Path) -> None:
        db = tmp_path / "slice.db"
        SQLiteStore(str(db)).close()
        with pytest.raises(ReplayError, match="not both"):
            replay_slice("x", handler=lambda t: None, agent_module="agent.py", db_path=db)


class TestSliceReplayRegistrationMode:
    def _write_agent(self, dir: Path) -> Path:
        path = dir / "agent.py"
        path.write_text(textwrap.dedent("""\
            import papayya
            from papayya import agent

            @agent(name="triage")
            def triage(item):
                # Ambient verbs resolve against the slice-replay item.
                return {"got": item}
        """))
        return path

    def test_cli_slice_replay_on_run_id(self, tmp_path: Path) -> None:
        db = tmp_path / "slice.db"
        old_run = _seed_run(db)
        agent_file = self._write_agent(tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            cli_module.main,
            ["replay", "--run", old_run, "--db", str(db),
             "--file", str(agent_file), "--latest"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert "not-ok slice" in result.output
        assert "2 ok, 0 failed" in result.output

        new_runs = _rows(db, "SELECT * FROM runs WHERE replayed_from = ?", (old_run,))
        assert len(new_runs) == 1

    def test_cli_run_flag_falls_back_to_single_item(self, tmp_path: Path) -> None:
        """--run <pre-0.3.0 per-item id> keeps working (the dashboard's DLQ
        Replay button sends item ids through --run)."""
        db = tmp_path / "slice.db"
        agent_file = self._write_agent(tmp_path)
        # A failed direct-call item with a snapshot.
        from datetime import datetime, timezone
        from papayya.durable.types import RunCheckpoint
        store = SQLiteStore(str(db))
        now = datetime.now(timezone.utc).isoformat()
        store.create(RunCheckpoint(
            run_id="dead-item", agent="triage", tasks=[], status="running",
            created_at=now, updated_at=now, input_snapshot={"id": "x"},
        ))
        store.set_status("dead-item", "failed", output="boom")
        store.close()

        runner = CliRunner()
        result = runner.invoke(
            cli_module.main,
            ["replay", "--run", "dead-item", "--db", str(db),
             "--file", str(agent_file), "--latest"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert "Replaying item dead-item" in result.output

    def test_cli_requires_exactly_one_of_run_or_item(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli_module.main, ["replay"])
        assert result.exit_code == 1
        assert "exactly one of --run or --item" in result.output
