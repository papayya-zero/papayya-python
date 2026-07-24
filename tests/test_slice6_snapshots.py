"""Slice 6 tests — per-object state snapshots at step boundaries.

Covers:
  * Schema v3 migration (from fresh, from v2, chained from v1).
  * `run.step(..., item_id=..., snapshot=...)` persistence.
  * Run-level item_id inheritance across steps.
  * First-step item_id seeds the run-level id for later inheritance.
  * Per-step override does not rewrite the run-level id.
  * Calls without item_id remain snapshot-free (status quo behaviour).
  * Snapshot JSON encoding, size-cap truncation, and replay-load round-trip.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from papayya.durable import _schema
from papayya.durable.run import PapayyaRun
from papayya.durable.sqlite_store import (
    SQLiteStore,
    _SNAPSHOT_BYTE_CAP,
    _decode_snapshot,
    _encode_snapshot,
)
from papayya.durable.types import DurableRunConfig


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "local.db"


# --------------------------------------------------------------------------- #
#  Schema                                                                      #
# --------------------------------------------------------------------------- #


class TestSchemaV3:
    def test_fresh_db_has_v3_columns(self, tmp_db: Path) -> None:
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        step_cols = {r[1] for r in conn.execute("PRAGMA table_info(steps)")}
        item_cols = {r[1] for r in conn.execute("PRAGMA table_info(items)")}
        assert _schema.COL_STEP_CUSTOMER_ITEM_ID in step_cols
        assert _schema.COL_STEP_INPUT_SNAPSHOT in step_cols
        assert _schema.COL_STEP_OUTPUT_SNAPSHOT in step_cols
        assert _schema.COL_ITEM_ITEM_ID in item_cols

    def test_fresh_db_has_v3_indexes(self, tmp_db: Path) -> None:
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        indexes = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert _schema.IDX_STEPS_CUSTOMER_ITEM in indexes
        assert _schema.IDX_ITEMS_ITEM in indexes

    def test_fresh_db_reports_current(self, tmp_db: Path) -> None:
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        version = conn.execute(
            "SELECT value FROM _meta WHERE key='schema_version'"
        ).fetchone()[0]
        assert version == _schema.SCHEMA_VERSION


# --------------------------------------------------------------------------- #
#  Snapshot encode/decode                                                      #
# --------------------------------------------------------------------------- #


class TestSnapshotCodec:
    def test_none_encodes_to_none(self) -> None:
        assert _encode_snapshot(None) is None

    def test_round_trip_primitive(self) -> None:
        encoded = _encode_snapshot({"k": "v", "n": 42})
        assert _decode_snapshot(encoded) == {"k": "v", "n": 42}

    def test_unencodable_raises(self) -> None:
        class Opaque:
            pass

        with pytest.raises(ValueError, match="JSON-encodable"):
            _encode_snapshot({"bad": Opaque()})

    def test_oversize_truncates_with_sentinel(self) -> None:
        payload = {"blob": "x" * (_SNAPSHOT_BYTE_CAP + 1024)}
        encoded = _encode_snapshot(payload)
        assert encoded is not None
        decoded = json.loads(encoded)
        assert decoded["__truncated__"] is True
        assert decoded["bytes"] > _SNAPSHOT_BYTE_CAP
        assert "preview" in decoded


# --------------------------------------------------------------------------- #
#  Write path — run.step(item_id, snapshot)                                    #
# --------------------------------------------------------------------------- #


class TestStepWriteWithSnapshot:
    def _run(self, tmp_db: Path, **config_kwargs: object) -> PapayyaRun:
        store = SQLiteStore(str(tmp_db))
        return PapayyaRun(
            DurableRunConfig(agent="test", store=store, **config_kwargs)  # type: ignore[arg-type]
        )

    def test_persists_item_id_and_snapshots(self, tmp_db: Path) -> None:
        run = self._run(tmp_db)

        run.step(
            "enrich", lambda co: {**co, "enriched": True},
            item_id="co_42", snapshot={"id": "co_42"},
        )({"id": "co_42"})

        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM steps WHERE label='enrich'").fetchone()
        assert row[_schema.COL_STEP_CUSTOMER_ITEM_ID] == "co_42"
        assert json.loads(row[_schema.COL_STEP_INPUT_SNAPSHOT]) == {"id": "co_42"}
        assert json.loads(row[_schema.COL_STEP_OUTPUT_SNAPSHOT]) == {
            "id": "co_42",
            "enriched": True,
        }

    def test_calls_without_item_id_write_null_snapshots(self, tmp_db: Path) -> None:
        run = self._run(tmp_db)

        run.step("plain", lambda: "ok")()

        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM steps WHERE label='plain'").fetchone()
        assert row[_schema.COL_STEP_CUSTOMER_ITEM_ID] is None
        assert row[_schema.COL_STEP_INPUT_SNAPSHOT] is None
        assert row[_schema.COL_STEP_OUTPUT_SNAPSHOT] is None

    def test_runs_item_id_denormalized(self, tmp_db: Path) -> None:
        run = self._run(tmp_db)
        run.step("a", lambda: "ok", item_id="co_7")()

        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM items WHERE id=?", (run.run_id,)
        ).fetchone()
        assert row[_schema.COL_ITEM_ITEM_ID] == "co_7"


# --------------------------------------------------------------------------- #
#  item_id inheritance semantics                                               #
# --------------------------------------------------------------------------- #


class TestItemIdInheritance:
    def _run(self, tmp_db: Path, **kwargs: object) -> PapayyaRun:
        store = SQLiteStore(str(tmp_db))
        return PapayyaRun(
            DurableRunConfig(agent="test", store=store, **kwargs)  # type: ignore[arg-type]
        )

    def test_config_level_item_id_propagates_to_all_steps(self, tmp_db: Path) -> None:
        run = self._run(tmp_db, item_id="co_1")
        run.step("a", lambda: "ok")()
        run.step("b", lambda: "ok")()

        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT label, customer_item_id FROM steps ORDER BY id"
        ).fetchall()
        assert [r["customer_item_id"] for r in rows] == ["co_1", "co_1"]

    def test_first_step_item_id_seeds_run_and_later_steps_inherit(
        self, tmp_db: Path
    ) -> None:
        run = self._run(tmp_db)
        run.step("a", lambda: "ok", item_id="co_2")()
        run.step("b", lambda: "ok")()  # no kwarg → inherits

        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT label, customer_item_id FROM steps ORDER BY id"
        ).fetchall()
        assert rows[0]["customer_item_id"] == "co_2"
        assert rows[1]["customer_item_id"] == "co_2"

    def test_per_step_override_does_not_rewrite_run_level(
        self, tmp_db: Path
    ) -> None:
        run = self._run(tmp_db, item_id="co_primary")
        run.step("a", lambda: "ok", item_id="co_override")()

        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        step_row = conn.execute(
            "SELECT customer_item_id FROM steps WHERE label='a'"
        ).fetchone()
        item_row = conn.execute(
            "SELECT item_id FROM items WHERE id=?", (run.run_id,)
        ).fetchone()
        assert step_row["customer_item_id"] == "co_override"
        assert item_row["item_id"] == "co_primary"


# --------------------------------------------------------------------------- #
#  Replay load path                                                            #
# --------------------------------------------------------------------------- #


class TestReplayRestoresSnapshots:
    def test_load_restores_item_id_and_snapshots(self, tmp_db: Path) -> None:
        store = SQLiteStore(str(tmp_db))
        run = PapayyaRun(
            DurableRunConfig(agent="test", store=store, run_id="r1")
        )
        run.step("enrich", lambda: {"out": 1}, item_id="co_9", snapshot={"in": 0})()

        # Fresh store + run with same run_id → hydrates from disk.
        store2 = SQLiteStore(str(tmp_db))
        checkpoint = store2.load("r1")
        assert checkpoint is not None
        (entry,) = checkpoint.tasks
        assert entry.item_id == "co_9"
        assert entry.input_snapshot == {"in": 0}
        assert entry.output_snapshot == {"out": 1}
