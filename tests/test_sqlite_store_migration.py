"""Tests for the SQLite store forward-migration chain (v1 → v5).

Guards three properties the ``LOCAL_DEV_EXECUTION.md`` plan pins down:

1. Never corrupt a populated v1 DB — data survives the migration intact.
2. Idempotent — re-opening a current-version DB is a no-op with no ALTERs attempted.
3. Atomic — a mid-migration crash leaves the DB at its prior version, not a half state.

Plus version-specific checks: v3→v4 drops the budget/cost/token columns,
and v4→v5 adds the BYOF observability columns on tasks.

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

    def test_fresh_db_has_llm_columns(self, tmp_db: Path) -> None:
        """v5 adds BYOF observability fields to the tasks table."""
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        task_cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
        assert _schema.COL_TASK_KIND in task_cols
        assert _schema.COL_TASK_LLM_PROMPT_TOKENS in task_cols
        assert _schema.COL_TASK_LLM_COMPLETION_TOKENS in task_cols
        assert _schema.COL_TASK_LLM_TOTAL_TOKENS in task_cols
        assert _schema.COL_TASK_LLM_MODEL in task_cols
        assert _schema.COL_TASK_LLM_STOP_REASON in task_cols
        assert _schema.COL_TASK_LLM_PROVIDER_SHAPE in task_cols
        assert _schema.COL_TASK_ERROR_CATEGORY in task_cols

    def test_fresh_db_has_dlq_columns(self, tmp_db: Path) -> None:
        """v6 adds DLQ + run-level input_snapshot to the runs table."""
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        run_cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
        assert _schema.COL_RUN_INPUT_SNAPSHOT in run_cols
        assert _schema.COL_RUN_DLQ_DISPOSITION in run_cols
        assert _schema.COL_RUN_DLQ_RESOLVED_AT in run_cols
        assert _schema.COL_RUN_REPLAYED_FROM in run_cols

    def test_fresh_db_has_agent_version_columns(self, tmp_db: Path) -> None:
        """v7 adds agent_version on runs (source) + tasks (denormalized)."""
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        run_cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
        task_cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
        assert _schema.COL_RUN_AGENT_VERSION in run_cols
        assert _schema.COL_TASK_AGENT_VERSION in task_cols

    def test_fresh_db_has_delivery_audit_columns(self, tmp_db: Path) -> None:
        """v8 adds delivery_attempts + journaled_at on tasks (ADR-0002 #8)."""
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        task_cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
        assert _schema.COL_TASK_DELIVERY_ATTEMPTS in task_cols
        assert _schema.COL_TASK_JOURNALED_AT in task_cols

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


class TestV4OnwardMigration:
    """Exercise forward migration starting from a v4 DB.

    A v4-shaped DB is built by running the v1→v4 migrations manually, then
    ``SQLiteStore`` is opened to trigger everything from v4 forward.
    """

    def _build_v4_db(self, tmp_db: Path) -> None:
        _build_v1_db(tmp_db)
        conn = sqlite3.connect(tmp_db)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            _apply_v1_to_v2(conn, tmp_db)
            _apply_v2_to_v3(conn, tmp_db)
            _apply_v3_to_v4(conn, tmp_db)
        finally:
            conn.close()

    def test_v4_db_bumps_to_current(self, tmp_db: Path) -> None:
        self._build_v4_db(tmp_db)
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        version = conn.execute(
            "SELECT value FROM _meta WHERE key='schema_version'"
        ).fetchone()[0]
        assert version == _schema.SCHEMA_VERSION

    def test_v4_db_adds_llm_columns_as_null(self, tmp_db: Path) -> None:
        self._build_v4_db(tmp_db)
        SQLiteStore(str(tmp_db))

        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        task = conn.execute("SELECT * FROM tasks WHERE run_id='run-1'").fetchone()
        assert task[_schema.COL_TASK_KIND] is None
        assert task[_schema.COL_TASK_LLM_PROMPT_TOKENS] is None
        assert task[_schema.COL_TASK_LLM_COMPLETION_TOKENS] is None
        assert task[_schema.COL_TASK_LLM_TOTAL_TOKENS] is None
        assert task[_schema.COL_TASK_LLM_MODEL] is None
        assert task[_schema.COL_TASK_LLM_STOP_REASON] is None
        assert task[_schema.COL_TASK_LLM_PROVIDER_SHAPE] is None
        assert task[_schema.COL_TASK_ERROR_CATEGORY] is None

    def test_v4_onward_preserves_existing_rows(self, tmp_db: Path) -> None:
        self._build_v4_db(tmp_db)
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        run = conn.execute("SELECT * FROM runs WHERE run_id='run-1'").fetchone()
        task = conn.execute("SELECT * FROM tasks WHERE run_id='run-1'").fetchone()
        assert run["agent"] == "test-agent"
        assert task["label"] == "search"
        # result column survived every migration through the chain.
        assert task["result"] == '"hit"'

    def test_v4_to_v5_creates_backup(self, tmp_db: Path) -> None:
        self._build_v4_db(tmp_db)
        SQLiteStore(str(tmp_db))
        backup = tmp_db.with_suffix(tmp_db.suffix + ".backup-v4")
        assert backup.exists()


class TestV5ToV6Migration:
    """Exercise the v5→v6 step: DLQ + input_snapshot on runs."""

    def _build_v5_db(self, tmp_db: Path) -> None:
        _build_v1_db(tmp_db)
        conn = sqlite3.connect(tmp_db)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            _apply_v1_to_v2(conn, tmp_db)
            _apply_v2_to_v3(conn, tmp_db)
            _apply_v3_to_v4(conn, tmp_db)
            _apply_v4_to_v5(conn, tmp_db)
        finally:
            conn.close()

    def test_v5_db_bumps_to_v6(self, tmp_db: Path) -> None:
        self._build_v5_db(tmp_db)
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        version = conn.execute(
            "SELECT value FROM _meta WHERE key='schema_version'"
        ).fetchone()[0]
        # Migrations chain: v5 → head. Test name reflects the v5→v6 step
        # this class exercises; the assertion tracks the current head.
        assert version == _schema.SCHEMA_VERSION

    def test_v5_db_adds_dlq_columns_as_null(self, tmp_db: Path) -> None:
        self._build_v5_db(tmp_db)
        SQLiteStore(str(tmp_db))

        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        run = conn.execute("SELECT * FROM runs WHERE run_id='run-1'").fetchone()
        assert run[_schema.COL_RUN_INPUT_SNAPSHOT] is None
        assert run[_schema.COL_RUN_DLQ_DISPOSITION] is None
        assert run[_schema.COL_RUN_DLQ_RESOLVED_AT] is None
        assert run[_schema.COL_RUN_REPLAYED_FROM] is None

    def test_v5_to_v6_creates_dlq_index(self, tmp_db: Path) -> None:
        self._build_v5_db(tmp_db)
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        indexes = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert _schema.IDX_RUNS_DLQ in indexes

    def test_v5_to_v6_creates_backup(self, tmp_db: Path) -> None:
        self._build_v5_db(tmp_db)
        SQLiteStore(str(tmp_db))
        backup = tmp_db.with_suffix(tmp_db.suffix + ".backup-v5")
        assert backup.exists()


class TestV6ToV7Migration:
    """Exercise the v6→v7 step: agent_version on runs + tasks (ADR-0002 #7)."""

    def _build_v6_db(self, tmp_db: Path) -> None:
        _build_v1_db(tmp_db)
        conn = sqlite3.connect(tmp_db)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            _apply_v1_to_v2(conn, tmp_db)
            _apply_v2_to_v3(conn, tmp_db)
            _apply_v3_to_v4(conn, tmp_db)
            _apply_v4_to_v5(conn, tmp_db)
            _apply_v5_to_v6(conn, tmp_db)
        finally:
            conn.close()

    def test_v6_db_bumps_to_head(self, tmp_db: Path) -> None:
        self._build_v6_db(tmp_db)
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        version = conn.execute(
            "SELECT value FROM _meta WHERE key='schema_version'"
        ).fetchone()[0]
        assert version == _schema.SCHEMA_VERSION

    def test_v6_db_adds_agent_version_columns_as_null(self, tmp_db: Path) -> None:
        self._build_v6_db(tmp_db)
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        run = conn.execute("SELECT * FROM runs WHERE run_id='run-1'").fetchone()
        task = conn.execute(
            "SELECT * FROM tasks WHERE run_id='run-1'"
        ).fetchone()
        assert run[_schema.COL_RUN_AGENT_VERSION] is None
        assert task[_schema.COL_TASK_AGENT_VERSION] is None

    def test_v6_to_v7_creates_backup(self, tmp_db: Path) -> None:
        self._build_v6_db(tmp_db)
        SQLiteStore(str(tmp_db))
        backup = tmp_db.with_suffix(tmp_db.suffix + ".backup-v6")
        assert backup.exists()


class TestV7ToV8Migration:
    """Exercise the v7→v8 step: delivery audit columns on tasks (ADR-0002 #8)."""

    def _build_v7_db(self, tmp_db: Path) -> None:
        _build_v1_db(tmp_db)
        conn = sqlite3.connect(tmp_db)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            _apply_v1_to_v2(conn, tmp_db)
            _apply_v2_to_v3(conn, tmp_db)
            _apply_v3_to_v4(conn, tmp_db)
            _apply_v4_to_v5(conn, tmp_db)
            _apply_v5_to_v6(conn, tmp_db)
            _apply_v6_to_v7(conn, tmp_db)
        finally:
            conn.close()

    def test_v7_db_bumps_to_head(self, tmp_db: Path) -> None:
        self._build_v7_db(tmp_db)
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        version = conn.execute(
            "SELECT value FROM _meta WHERE key='schema_version'"
        ).fetchone()[0]
        assert version == _schema.SCHEMA_VERSION

    def test_v7_db_adds_delivery_audit_columns_as_null(self, tmp_db: Path) -> None:
        self._build_v7_db(tmp_db)
        SQLiteStore(str(tmp_db))
        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        task = conn.execute(
            "SELECT * FROM tasks WHERE run_id='run-1'"
        ).fetchone()
        assert task[_schema.COL_TASK_DELIVERY_ATTEMPTS] is None
        assert task[_schema.COL_TASK_JOURNALED_AT] is None

    def test_v7_to_v8_creates_backup(self, tmp_db: Path) -> None:
        self._build_v7_db(tmp_db)
        SQLiteStore(str(tmp_db))
        backup = tmp_db.with_suffix(tmp_db.suffix + ".backup-v7")
        assert backup.exists()


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
