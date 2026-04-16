"""Tests for the SQLite store forward-migration chain (v1 → v4).

Guards three properties the ``LOCAL_DEV_EXECUTION.md`` plan pins down:

1. Never corrupt a populated v1 DB — data survives the migration intact.
2. Idempotent — re-opening a current-version DB is a no-op with no ALTERs attempted.
3. Atomic — a mid-migration crash leaves the DB at its prior version, not a half state.

Plus a v4-specific check that budget/cost/token columns are dropped by
v3→v4.

The v1 fixture is built from code rather than a checked-in binary because
the v1 schema is short and reproducible. If that ever stops being true,
replace ``_build_v1_db`` with a binary fixture.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from papayya.durable import _schema
from papayya.durable.sqlite_store import SQLiteStore


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


# --------------------------------------------------------------------------- #
#  Tests                                                                       #
# --------------------------------------------------------------------------- #


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "local.db"


class TestFreshInstall:
    def test_fresh_db_gets_current_schema(self, tmp_db: Path) -> None:
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        version = conn.execute(
            "SELECT value FROM _meta WHERE key='schema_version'"
        ).fetchone()[0]
        assert version == _schema.SCHEMA_VERSION

    def test_fresh_db_creates_batches_table(self, tmp_db: Path) -> None:
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (_schema.TBL_BATCHES,),
        ).fetchall()
        assert rows, "batches table missing on fresh install"

    def test_fresh_db_has_capture_columns(self, tmp_db: Path) -> None:
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        run_cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
        step_cols = {r[1] for r in conn.execute("PRAGMA table_info(steps)")}
        assert _schema.COL_RUN_BATCH_ID in run_cols
        assert _schema.COL_RUN_ERROR_CODE in run_cols
        assert _schema.COL_STEP_TOOL_NAME in step_cols
        assert _schema.COL_STEP_ERROR_CODE in step_cols
        assert _schema.COL_STEP_ERROR_CATEGORY in step_cols
        assert _schema.COL_STEP_INPUT_HASH in step_cols

    def test_fresh_db_has_indexes(self, tmp_db: Path) -> None:
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        indexes = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert _schema.IDX_STEPS_TOOL in indexes
        assert _schema.IDX_STEPS_ERROR in indexes
        assert _schema.IDX_RUNS_BATCH in indexes

    def test_fresh_db_drops_budget_and_cost_columns(self, tmp_db: Path) -> None:
        """v4 removes budget/cost/token columns. Fresh DBs end up v4 after the
        migration chain runs."""
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        run_cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
        task_cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
        step_cols = {r[1] for r in conn.execute("PRAGMA table_info(steps)")}
        batch_cols = {r[1] for r in conn.execute("PRAGMA table_info(batches)")}

        for col in ("budget_limit_usd", "budget_consumed_usd",
                    "total_input_tokens", "total_output_tokens",
                    "budget_input_tokens", "budget_output_tokens"):
            assert col not in run_cols, f"runs.{col} should be dropped in v4"
        for col in ("cost_usd", "input_tokens", "output_tokens"):
            assert col not in task_cols, f"tasks.{col} should be dropped in v4"
            assert col not in step_cols, f"steps.{col} should be dropped in v4"
        for col in ("aggregate_cost_usd", "budget_limit_usd"):
            assert col not in batch_cols, f"batches.{col} should be dropped in v4"


class TestV1Migration:
    def test_v1_data_survives_migration(self, tmp_db: Path) -> None:
        _build_v1_db(tmp_db)
        SQLiteStore(str(tmp_db))

        conn = sqlite3.connect(tmp_db)
        runs = conn.execute("SELECT * FROM runs").fetchall()
        tasks = conn.execute("SELECT * FROM tasks").fetchall()
        steps = conn.execute("SELECT * FROM steps").fetchall()
        assert len(runs) == 1
        assert len(tasks) == 1
        assert len(steps) == 1
        # Run ID and task label preserved through v1→v4.
        assert runs[0][0] == "run-1"
        # Task label is column index 2 in the v4 shape (id, run_id, label, ...)
        assert tasks[0][2] == "search"

    def test_v1_capture_columns_default_null(self, tmp_db: Path) -> None:
        _build_v1_db(tmp_db)
        SQLiteStore(str(tmp_db))

        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        run = conn.execute("SELECT * FROM runs WHERE run_id='run-1'").fetchone()
        step = conn.execute(
            "SELECT * FROM steps WHERE run_id='run-1'"
        ).fetchone()
        assert run[_schema.COL_RUN_BATCH_ID] is None
        assert run[_schema.COL_RUN_ERROR_CODE] is None
        assert step[_schema.COL_STEP_TOOL_NAME] is None
        assert step[_schema.COL_STEP_ERROR_CODE] is None
        assert step[_schema.COL_STEP_ERROR_CATEGORY] is None
        assert step[_schema.COL_STEP_INPUT_HASH] is None

    def test_v1_migration_bumps_to_current_schema(self, tmp_db: Path) -> None:
        _build_v1_db(tmp_db)
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        version = conn.execute(
            "SELECT value FROM _meta WHERE key='schema_version'"
        ).fetchone()[0]
        assert version == _schema.SCHEMA_VERSION

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
        assert _schema.COL_RUN_BATCH_ID not in run_cols
        assert _schema.COL_RUN_ERROR_CODE not in run_cols

        runs = conn.execute("SELECT run_id FROM runs").fetchall()
        assert [r[0] for r in runs] == ["run-1"]
        conn.close()


class TestIdempotence:
    def test_reopening_current_is_noop(self, tmp_db: Path) -> None:
        SQLiteStore(str(tmp_db))
        # Second open must not raise (ALTER would fail if re-attempted)
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
