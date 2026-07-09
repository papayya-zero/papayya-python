"""Tests for the SQLite store forward-migration chain (v1 → v12).

Guards the properties the ``LOCAL_DEV_EXECUTION.md`` plan pins down:

1. Never corrupt a populated old-schema DB — data survives migration intact.
2. Idempotent — re-opening a current-version DB is a no-op with no ALTERs.
3. Atomic — a mid-migration crash leaves the DB at its prior version.

Plus Plan 34's v11→v12 noun consolidation:

* fresh DBs are created at v12 DIRECTLY (no chain walk, no backup storm)
* v11 (and older) DBs migrate: batches→runs, runs→items (run_id→id,
  batch_id→run_id), tasks→steps (item_id→customer_item_id, run_id→item_id),
  dead legacy steps table dropped, indexes rebuilt under v12 names.

The v1 fixture is built from code rather than a checked-in binary because
the v1 schema is short and reproducible. If that ever stops being true,
replace ``_build_v1_db`` with a binary fixture.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from papayya.durable import _schema
from papayya.durable.sqlite_store import (
    SQLiteStore,
    _apply_v1_to_v2,
    _apply_v2_to_v3,
    _apply_v3_to_v4,
    _apply_v4_to_v5,
    _apply_v5_to_v6,
    _apply_v6_to_v7,
    _apply_v7_to_v8,
    _apply_v8_to_v9,
    _apply_v9_to_v10,
    _apply_v10_to_v11,
)


# --------------------------------------------------------------------------- #
#  v1 fixture — the schema as it shipped before Slice 1                        #
# --------------------------------------------------------------------------- #


_V1_SCHEMA = """\
CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
INSERT INTO _meta (key, value) VALUES ('schema_version', '1');

CREATE TABLE runs (
    run_id               TEXT PRIMARY KEY,
    agent                TEXT NOT NULL,
    status               TEXT NOT NULL DEFAULT 'running',
    budget_limit_usd     REAL,
    budget_consumed_usd  REAL NOT NULL DEFAULT 0.0,
    total_input_tokens   INTEGER NOT NULL DEFAULT 0,
    total_output_tokens  INTEGER NOT NULL DEFAULT 0,
    budget_input_tokens  INTEGER,
    budget_output_tokens INTEGER,
    output               TEXT,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL
);

CREATE TABLE tasks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL REFERENCES runs(run_id),
    label         TEXT NOT NULL,
    result        TEXT,
    cost_usd      REAL NOT NULL DEFAULT 0.0,
    duration_ms   INTEGER NOT NULL DEFAULT 0,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    completed_at  TEXT NOT NULL
);
CREATE INDEX idx_tasks_run_id ON tasks(run_id);

CREATE TABLE steps (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL REFERENCES runs(run_id),
    task_label     TEXT,
    step_index     INTEGER NOT NULL DEFAULT 0,
    model          TEXT NOT NULL DEFAULT 'unknown',
    input_tokens   INTEGER NOT NULL DEFAULT 0,
    output_tokens  INTEGER NOT NULL DEFAULT 0,
    cost_usd       REAL NOT NULL DEFAULT 0.0,
    duration_ms    INTEGER NOT NULL DEFAULT 0,
    tool_calls     TEXT,
    response_text  TEXT,
    created_at     TEXT NOT NULL
);
CREATE INDEX idx_steps_run_id ON steps(run_id);
"""


def _build_v1_db(db_path: Path) -> None:
    """Create a v1-schema DB with representative data."""
    conn = sqlite3.connect(db_path)
    conn.executescript(_V1_SCHEMA)
    conn.execute(
        """INSERT INTO runs (run_id, agent, status, budget_consumed_usd,
           total_input_tokens, total_output_tokens, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        ("run-1", "test-agent", "completed", 0.42, 1000, 200,
         "2026-04-01T00:00:00+00:00", "2026-04-01T00:00:05+00:00"),
    )
    conn.execute(
        """INSERT INTO tasks (run_id, label, result, cost_usd, duration_ms,
           input_tokens, output_tokens, completed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        ("run-1", "search", '"hit"', 0.21, 500, 500, 100,
         "2026-04-01T00:00:02+00:00"),
    )
    conn.execute(
        """INSERT INTO steps (run_id, task_label, step_index, model,
           input_tokens, output_tokens, cost_usd, duration_ms,
           tool_calls, response_text, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("run-1", "search", 0, "gpt-4", 500, 100, 0.21, 500,
         None, "hit", "2026-04-01T00:00:02+00:00"),
    )
    conn.commit()
    conn.close()


def _chain(conn: sqlite3.Connection, tmp_db: Path, upto: int) -> None:
    """Apply the frozen legacy migrations v1→v<upto> in order."""
    applies = [
        _apply_v1_to_v2, _apply_v2_to_v3, _apply_v3_to_v4, _apply_v4_to_v5,
        _apply_v5_to_v6, _apply_v6_to_v7, _apply_v7_to_v8, _apply_v8_to_v9,
        _apply_v9_to_v10, _apply_v10_to_v11,
    ]
    for fn in applies[: upto - 1]:
        fn(conn, tmp_db)


def _build_vn_db(tmp_db: Path, version: int) -> None:
    """Build a populated DB frozen at schema v<version> (2..11)."""
    _build_v1_db(tmp_db)
    conn = sqlite3.connect(tmp_db)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        _chain(conn, tmp_db, version)
    finally:
        conn.close()


def _version_of(tmp_db: Path) -> str:
    conn = sqlite3.connect(tmp_db)
    try:
        return conn.execute(
            "SELECT value FROM _meta WHERE key='schema_version'"
        ).fetchone()[0]
    finally:
        conn.close()


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }


# --------------------------------------------------------------------------- #
#  Tests                                                                       #
# --------------------------------------------------------------------------- #


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "local.db"


class TestFreshInstall:
    def test_fresh_db_gets_current_schema(self, tmp_db: Path) -> None:
        SQLiteStore(str(tmp_db))
        assert _version_of(tmp_db) == _schema.SCHEMA_VERSION

    def test_fresh_db_creates_v12_tables_only(self, tmp_db: Path) -> None:
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        tables = _tables(conn)
        assert {"_meta", "runs", "items", "steps"} <= tables
        # Pre-v12 nouns must not exist on a fresh DB.
        assert "batches" not in tables
        assert "tasks" not in tables

    def test_fresh_db_no_backup_storm(self, tmp_db: Path) -> None:
        """Fresh DBs are created at head directly — the pre-fix behavior
        walked the whole chain and left one backup-vN file per step."""
        SQLiteStore(str(tmp_db))
        backups = list(tmp_db.parent.glob("local.db.backup-*"))
        assert backups == [], f"fresh DB left backups: {backups}"

    def test_fresh_db_items_shape(self, tmp_db: Path) -> None:
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(items)")}
        for col in (
            _schema.COL_ITEM_ID,           # surrogate 'id' (was run_id)
            _schema.COL_ITEM_RUN_ID,       # invocation FK (was batch_id)
            _schema.COL_ITEM_ITEM_ID,      # customer identity
            _schema.COL_ITEM_ERROR_CODE,
            _schema.COL_ITEM_INPUT_SNAPSHOT,
            _schema.COL_ITEM_DLQ_DISPOSITION,
            _schema.COL_ITEM_DLQ_RESOLVED_AT,
            _schema.COL_ITEM_REPLAYED_FROM,
            _schema.COL_ITEM_AGENT_VERSION,
            _schema.COL_ITEM_METADATA,
            _schema.COL_ITEM_PARTITION_KEY,
            _schema.COL_ITEM_PARENT_RUN_ID,
            _schema.COL_ITEM_WORST_OUTCOME_STATUS,
            _schema.COL_ITEM_DEGRADED_COUNT,
        ):
            assert col in cols, f"items.{col} missing"
        # Budget/cost columns never existed on v12-fresh DBs.
        for col in ("budget_limit_usd", "budget_consumed_usd",
                    "total_input_tokens", "total_output_tokens"):
            assert col not in cols

    def test_fresh_db_runs_shape(self, tmp_db: Path) -> None:
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
        for col in (
            _schema.COL_RUN_ID, _schema.COL_RUN_AGENT, _schema.COL_RUN_STATUS,
            _schema.COL_RUN_TOTAL_ITEMS, _schema.COL_RUN_COMPLETED,
            _schema.COL_RUN_FAILED, _schema.COL_RUN_CONCURRENCY_CAP,
            _schema.COL_RUN_CREATED_AT, _schema.COL_RUN_COMPLETED_AT,
            _schema.COL_RUN_REPLAYED_FROM,
        ):
            assert col in cols, f"runs.{col} missing"
        assert "aggregate_cost_usd" not in cols
        assert "budget_limit_usd" not in cols

    def test_fresh_db_steps_shape(self, tmp_db: Path) -> None:
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(steps)")}
        for col in (
            _schema.COL_STEP_ITEM_ID,            # FK -> items.id (was run_id)
            _schema.COL_STEP_CUSTOMER_ITEM_ID,   # was tasks.item_id
            _schema.COL_STEP_LABEL,
            _schema.COL_STEP_KIND,
            _schema.COL_STEP_LLM_PROMPT_TOKENS,
            _schema.COL_STEP_LLM_COMPLETION_TOKENS,
            _schema.COL_STEP_LLM_TOTAL_TOKENS,
            _schema.COL_STEP_LLM_MODEL,
            _schema.COL_STEP_LLM_STOP_REASON,
            _schema.COL_STEP_LLM_PROVIDER_SHAPE,
            _schema.COL_STEP_ERROR_CATEGORY,
            _schema.COL_STEP_AGENT_VERSION,
            _schema.COL_STEP_DELIVERY_ATTEMPTS,
            _schema.COL_STEP_JOURNALED_AT,
            _schema.COL_STEP_METADATA,
            _schema.COL_STEP_PARTITION_KEY,
            _schema.COL_STEP_OUTCOME_STATUS,
            _schema.COL_STEP_OUTCOME_REASON,
        ):
            assert col in cols, f"steps.{col} missing"
        # The dead legacy LLM-call-log columns never existed on v12 steps.
        for col in ("tool_name", "input_hash", "step_index",
                    "response_text", "tool_calls", "model"):
            assert col not in cols

    def test_fresh_db_has_v12_indexes(self, tmp_db: Path) -> None:
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        indexes = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        for idx in (
            _schema.IDX_ITEMS_RUN, _schema.IDX_ITEMS_ITEM,
            _schema.IDX_ITEMS_DLQ, _schema.IDX_ITEMS_PARTITION,
            _schema.IDX_ITEMS_PARENT, _schema.IDX_STEPS_ITEM,
            _schema.IDX_STEPS_CUSTOMER_ITEM, _schema.IDX_STEPS_PARTITION,
        ):
            assert idx in indexes, f"index {idx} missing"
        # Old-named indexes must not exist on fresh DBs.
        for idx in ("idx_runs_batch", "idx_tasks_run_id", "idx_runs_item",
                    "idx_tasks_item", "idx_runs_dlq", "idx_runs_partition",
                    "idx_tasks_partition", "idx_runs_parent",
                    "idx_steps_tool", "idx_steps_error"):
            assert idx not in indexes, f"stale index {idx} on fresh DB"


class TestV1Migration:
    def test_v1_data_survives_migration(self, tmp_db: Path) -> None:
        _build_v1_db(tmp_db)
        SQLiteStore(str(tmp_db))

        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        items = conn.execute("SELECT * FROM items").fetchall()
        steps = conn.execute("SELECT * FROM steps").fetchall()
        assert len(items) == 1
        assert len(steps) == 1
        # The per-item id and step label are preserved through v1→v12
        # (old runs.run_id -> items.id; old tasks -> steps).
        assert items[0]["id"] == "run-1"
        assert steps[0]["label"] == "search"
        assert steps[0]["item_id"] == "run-1"

    def test_v1_migration_bumps_to_current_schema(self, tmp_db: Path) -> None:
        _build_v1_db(tmp_db)
        SQLiteStore(str(tmp_db))
        assert _version_of(tmp_db) == _schema.SCHEMA_VERSION

    def test_v1_migration_creates_backup(self, tmp_db: Path) -> None:
        _build_v1_db(tmp_db)
        SQLiteStore(str(tmp_db))
        backup = tmp_db.with_suffix(tmp_db.suffix + ".backup-v1")
        assert backup.exists(), "expected backup-v1 sibling file"

    def test_v1_migration_backup_is_recoverable(self, tmp_db: Path) -> None:
        """Backup is the recovery artifact, so openable + v1-shape + data intact."""
        _build_v1_db(tmp_db)
        SQLiteStore(str(tmp_db))
        backup = tmp_db.with_suffix(tmp_db.suffix + ".backup-v1")

        conn = sqlite3.connect(backup)
        version = conn.execute(
            "SELECT value FROM _meta WHERE key='schema_version'"
        ).fetchone()[0]
        assert version == "1"

        run_cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
        assert "batch_id" not in run_cols
        assert "error_code" not in run_cols

        runs = conn.execute("SELECT run_id FROM runs").fetchall()
        assert [r[0] for r in runs] == ["run-1"]
        conn.close()


class TestV11ToV12Migration:
    """The Plan 34 rename step, exercised from a real v11 fixture."""

    def _build_v11_db(self, tmp_db: Path) -> None:
        _build_vn_db(tmp_db, 11)
        # Enrich the v11 fixture so the rename mapping is observable:
        # a real batch row, a batch-linked run, and a customer item_id on
        # both the run and the task.
        conn = sqlite3.connect(tmp_db)
        try:
            conn.execute(
                """INSERT INTO batches (batch_id, agent, status, total_items,
                   completed, failed, created_at)
                   VALUES ('batch-9', 'test-agent', 'running', 2, 0, 0,
                           '2026-04-02T00:00:00+00:00')"""
            )
            conn.execute(
                "UPDATE runs SET batch_id='batch-9', item_id='co_007' "
                "WHERE run_id='run-1'"
            )
            conn.execute(
                "UPDATE tasks SET item_id='co_007' WHERE run_id='run-1'"
            )
            conn.commit()
        finally:
            conn.close()
        assert _version_of(tmp_db) == "11"

    def test_v11_bumps_to_v12(self, tmp_db: Path) -> None:
        self._build_v11_db(tmp_db)
        SQLiteStore(str(tmp_db))
        assert _version_of(tmp_db) == "12"

    def test_v11_tables_renamed(self, tmp_db: Path) -> None:
        self._build_v11_db(tmp_db)
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        tables = _tables(conn)
        assert {"runs", "items", "steps"} <= tables
        assert "batches" not in tables
        assert "tasks" not in tables

    def test_v11_batch_row_becomes_run_row(self, tmp_db: Path) -> None:
        self._build_v11_db(tmp_db)
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        run = conn.execute(
            "SELECT * FROM runs WHERE run_id='batch-9'"
        ).fetchone()
        assert run is not None
        assert run["agent"] == "test-agent"
        assert run["total_items"] == 2
        # v12 adds replayed_from to the invocation table.
        assert run["replayed_from"] is None

    def test_v11_run_row_becomes_item_row(self, tmp_db: Path) -> None:
        self._build_v11_db(tmp_db)
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        item = conn.execute("SELECT * FROM items WHERE id='run-1'").fetchone()
        assert item is not None
        assert item["run_id"] == "batch-9"     # was batch_id
        assert item["item_id"] == "co_007"     # customer identity unchanged
        assert item["agent"] == "test-agent"
        assert item["status"] == "completed"

    def test_v11_task_row_becomes_step_row(self, tmp_db: Path) -> None:
        self._build_v11_db(tmp_db)
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        step = conn.execute(
            "SELECT * FROM steps WHERE item_id='run-1'"
        ).fetchone()
        assert step is not None
        assert step["label"] == "search"
        assert step["customer_item_id"] == "co_007"  # was tasks.item_id
        assert step["result"] == '"hit"'

    def test_v11_legacy_steps_table_dropped(self, tmp_db: Path) -> None:
        """The pre-v12 `steps` table (raw LLM-call log, no production
        writer) is dropped, so its rows do not survive — by design."""
        self._build_v11_db(tmp_db)
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(steps)")}
        assert "tool_calls" not in cols
        assert "response_text" not in cols

    def test_v11_creates_backup(self, tmp_db: Path) -> None:
        self._build_v11_db(tmp_db)
        SQLiteStore(str(tmp_db))
        backup = tmp_db.with_suffix(tmp_db.suffix + ".backup-v11")
        assert backup.exists()

    def test_v11_indexes_rebuilt(self, tmp_db: Path) -> None:
        self._build_v11_db(tmp_db)
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        indexes = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert _schema.IDX_ITEMS_RUN in indexes
        assert _schema.IDX_STEPS_ITEM in indexes
        for idx in ("idx_runs_batch", "idx_tasks_run_id", "idx_runs_dlq",
                    "idx_steps_tool", "idx_steps_error"):
            assert idx not in indexes

    def test_migrated_db_round_trips_through_store(self, tmp_db: Path) -> None:
        """The store API works against a chain-migrated DB: load the old
        item, write a new one."""
        self._build_v11_db(tmp_db)
        store = SQLiteStore(str(tmp_db))
        loaded = store.load("run-1")
        assert loaded is not None
        assert loaded.agent == "test-agent"
        assert loaded.item_id == "co_007"
        assert loaded.invocation_id == "batch-9"
        assert [t.label for t in loaded.tasks] == ["search"]
        assert loaded.tasks[0].item_id == "co_007"


class TestV4OnwardMigration:
    """Exercise forward migration starting from a v4 DB."""

    def test_v4_db_bumps_to_current(self, tmp_db: Path) -> None:
        _build_vn_db(tmp_db, 4)
        SQLiteStore(str(tmp_db))
        assert _version_of(tmp_db) == _schema.SCHEMA_VERSION

    def test_v4_db_adds_llm_columns_as_null(self, tmp_db: Path) -> None:
        _build_vn_db(tmp_db, 4)
        SQLiteStore(str(tmp_db))

        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        step = conn.execute("SELECT * FROM steps WHERE item_id='run-1'").fetchone()
        assert step[_schema.COL_STEP_KIND] is None
        assert step[_schema.COL_STEP_LLM_PROMPT_TOKENS] is None
        assert step[_schema.COL_STEP_LLM_COMPLETION_TOKENS] is None
        assert step[_schema.COL_STEP_LLM_TOTAL_TOKENS] is None
        assert step[_schema.COL_STEP_LLM_MODEL] is None
        assert step[_schema.COL_STEP_LLM_STOP_REASON] is None
        assert step[_schema.COL_STEP_LLM_PROVIDER_SHAPE] is None
        assert step[_schema.COL_STEP_ERROR_CATEGORY] is None

    def test_v4_onward_preserves_existing_rows(self, tmp_db: Path) -> None:
        _build_vn_db(tmp_db, 4)
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        item = conn.execute("SELECT * FROM items WHERE id='run-1'").fetchone()
        step = conn.execute("SELECT * FROM steps WHERE item_id='run-1'").fetchone()
        assert item["agent"] == "test-agent"
        assert step["label"] == "search"
        # result column survived every migration through the chain.
        assert step["result"] == '"hit"'

    def test_v4_to_v5_creates_backup(self, tmp_db: Path) -> None:
        _build_vn_db(tmp_db, 4)
        SQLiteStore(str(tmp_db))
        backup = tmp_db.with_suffix(tmp_db.suffix + ".backup-v4")
        assert backup.exists()


class TestV5ToV6Migration:
    """Exercise the v5→v6 step: DLQ + input_snapshot on per-item rows."""

    def test_v5_db_bumps_to_head(self, tmp_db: Path) -> None:
        _build_vn_db(tmp_db, 5)
        SQLiteStore(str(tmp_db))
        assert _version_of(tmp_db) == _schema.SCHEMA_VERSION

    def test_v5_db_adds_dlq_columns_as_null(self, tmp_db: Path) -> None:
        _build_vn_db(tmp_db, 5)
        SQLiteStore(str(tmp_db))

        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        item = conn.execute("SELECT * FROM items WHERE id='run-1'").fetchone()
        assert item[_schema.COL_ITEM_INPUT_SNAPSHOT] is None
        assert item[_schema.COL_ITEM_DLQ_DISPOSITION] is None
        assert item[_schema.COL_ITEM_DLQ_RESOLVED_AT] is None
        assert item[_schema.COL_ITEM_REPLAYED_FROM] is None

    def test_v5_to_v6_creates_backup(self, tmp_db: Path) -> None:
        _build_vn_db(tmp_db, 5)
        SQLiteStore(str(tmp_db))
        backup = tmp_db.with_suffix(tmp_db.suffix + ".backup-v5")
        assert backup.exists()


class TestV6ToV7Migration:
    """Exercise the v6→v7 step: agent_version (ADR-0002 #7)."""

    def test_v6_db_bumps_to_head(self, tmp_db: Path) -> None:
        _build_vn_db(tmp_db, 6)
        SQLiteStore(str(tmp_db))
        assert _version_of(tmp_db) == _schema.SCHEMA_VERSION

    def test_v6_db_adds_agent_version_columns_as_null(self, tmp_db: Path) -> None:
        _build_vn_db(tmp_db, 6)
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        item = conn.execute("SELECT * FROM items WHERE id='run-1'").fetchone()
        step = conn.execute("SELECT * FROM steps WHERE item_id='run-1'").fetchone()
        assert item[_schema.COL_ITEM_AGENT_VERSION] is None
        assert step[_schema.COL_STEP_AGENT_VERSION] is None

    def test_v6_to_v7_creates_backup(self, tmp_db: Path) -> None:
        _build_vn_db(tmp_db, 6)
        SQLiteStore(str(tmp_db))
        backup = tmp_db.with_suffix(tmp_db.suffix + ".backup-v6")
        assert backup.exists()


class TestV7ToV8Migration:
    """Exercise the v7→v8 step: delivery audit columns (ADR-0002 #8)."""

    def test_v7_db_bumps_to_head(self, tmp_db: Path) -> None:
        _build_vn_db(tmp_db, 7)
        SQLiteStore(str(tmp_db))
        assert _version_of(tmp_db) == _schema.SCHEMA_VERSION

    def test_v7_db_adds_delivery_audit_columns_as_null(self, tmp_db: Path) -> None:
        _build_vn_db(tmp_db, 7)
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        step = conn.execute("SELECT * FROM steps WHERE item_id='run-1'").fetchone()
        assert step[_schema.COL_STEP_DELIVERY_ATTEMPTS] is None
        assert step[_schema.COL_STEP_JOURNALED_AT] is None

    def test_v7_to_v8_creates_backup(self, tmp_db: Path) -> None:
        _build_vn_db(tmp_db, 7)
        SQLiteStore(str(tmp_db))
        backup = tmp_db.with_suffix(tmp_db.suffix + ".backup-v7")
        assert backup.exists()


class TestV8ToV9Migration:
    """Exercise the v8→v9 step: metadata + partition_key."""

    def test_v8_db_bumps_to_head(self, tmp_db: Path) -> None:
        _build_vn_db(tmp_db, 8)
        SQLiteStore(str(tmp_db))
        assert _version_of(tmp_db) == _schema.SCHEMA_VERSION

    def test_v8_db_adds_metadata_partition_columns_as_null(self, tmp_db: Path) -> None:
        _build_vn_db(tmp_db, 8)
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        item = conn.execute("SELECT * FROM items WHERE id='run-1'").fetchone()
        step = conn.execute("SELECT * FROM steps WHERE item_id='run-1'").fetchone()
        assert item[_schema.COL_ITEM_METADATA] is None
        assert item[_schema.COL_ITEM_PARTITION_KEY] is None
        assert step[_schema.COL_STEP_METADATA] is None
        assert step[_schema.COL_STEP_PARTITION_KEY] is None

    def test_v8_to_v9_creates_partition_indexes(self, tmp_db: Path) -> None:
        _build_vn_db(tmp_db, 8)
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        indexes = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        # After chaining through v12 the partition indexes carry v12 names.
        assert _schema.IDX_ITEMS_PARTITION in indexes
        assert _schema.IDX_STEPS_PARTITION in indexes

    def test_v8_to_v9_creates_backup(self, tmp_db: Path) -> None:
        _build_vn_db(tmp_db, 8)
        SQLiteStore(str(tmp_db))
        backup = tmp_db.with_suffix(tmp_db.suffix + ".backup-v8")
        assert backup.exists()

    def test_v8_to_v9_preserves_existing_rows(self, tmp_db: Path) -> None:
        _build_vn_db(tmp_db, 8)
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        item = conn.execute("SELECT * FROM items WHERE id='run-1'").fetchone()
        step = conn.execute("SELECT * FROM steps WHERE item_id='run-1'").fetchone()
        assert item["agent"] == "test-agent"
        assert step["label"] == "search"
        assert step["result"] == '"hit"'


class TestIdempotence:
    def test_reopening_current_is_noop(self, tmp_db: Path) -> None:
        SQLiteStore(str(tmp_db))
        # Second open must not raise (ALTER/RENAME would fail if re-attempted)
        SQLiteStore(str(tmp_db))
        SQLiteStore(str(tmp_db))

    def test_v1_backup_not_overwritten_on_reopen(self, tmp_db: Path) -> None:
        _build_v1_db(tmp_db)
        SQLiteStore(str(tmp_db))
        backup = tmp_db.with_suffix(tmp_db.suffix + ".backup-v1")
        first = backup.read_bytes()
        # Open the migrated DB again — no backup should be written because we're
        # not migrating from v1 anymore.
        SQLiteStore(str(tmp_db))
        assert backup.read_bytes() == first


class TestUnknownVersion:
    def test_future_version_raises(self, tmp_db: Path) -> None:
        _build_v1_db(tmp_db)
        conn = sqlite3.connect(tmp_db)
        conn.execute(
            "UPDATE _meta SET value='99' WHERE key='schema_version'"
        )
        conn.commit()
        conn.close()
        with pytest.raises(RuntimeError, match="Unknown schema version"):
            SQLiteStore(str(tmp_db))
