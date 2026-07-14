"""SQLite-backed checkpoint store with step-level observability.

Implements CheckpointStore for durable execution compatibility.

Plan 34 noun consolidation (schema v12): the local ledger speaks the
product vocabulary directly —

    runs   = invocations (one map() call, one cron fire, or an implicit
             run-of-one wrapped around a direct call)   [was `batches`]
    items  = per-item records: outcome, trace, cost      [was `runs`]
    steps  = trace nodes written by run.step()/llm_step() [was `tasks`]

The CheckpointStore protocol method names (load/save_task/set_status/
create) and their ``run_id`` parameter names are intentionally UNCHANGED:
they are the internal contract shared with MemoryStore/FileStore/
CloudStore, and the CloudStore HTTP wire is frozen at the old names until
Plan 34 Unit 5. In this store, the protocol's ``run_id`` addresses an
ITEM row (its surrogate ``id`` column).

The schema is a cross-language contract — future SDKs (TypeScript, Go)
will write to the same tables and the dashboard reads from them unchanged.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import _serialize
from . import _schema
from .types import RunCheckpoint, TaskEntry, _outcome_severity


# Flag gate for all Slice-2 capture logic. Setting to "false" restores the
# Slice-1 schema-only behaviour: new columns stay null, no implicit
# run-of-one rows, no aggregate bumps. Intended as a safety hatch if capture
# logic produces bad data in the wild; not something users should normally
# touch.
_CAPTURE_V2_ENABLED = os.environ.get("PAPAYYA_LOCAL_CAPTURE_V2", "true").lower() != "false"


def _capture_enabled() -> bool:
    # Read on every call so tests can monkeypatch os.environ and get an
    # accurate answer without re-importing.
    return os.environ.get("PAPAYYA_LOCAL_CAPTURE_V2", "true").lower() != "false"


def _env_int(name: str, default: int) -> int:
    """Parse an int env knob, falling back to default on unset/garbage."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _single_run_id(item_id: str) -> str:
    """Sentinel run ID for the implicit run-of-one around a direct call.

    A direct ``@agent`` / ``papayya().item()`` call has no surrounding
    ``map()``/``iter()`` invocation, so the store wraps the item in a run
    of one. The ``single-`` prefix lets the UI filter run-of-one work in
    or out cleanly. (This is the pre-v12 implicit batch, renamed.)
    """
    return f"single-{item_id}"


_SCHEMA_VERSION = _schema.SCHEMA_VERSION


# Slice 6: snapshot payloads are JSON-encoded TEXT. Cap per column at 64 KB
# to keep the local dev DB small — if a user is passing megabyte payloads
# through step boundaries they should store a reference (e.g. an S3 key)
# instead. The truncation sentinel preserves a short preview so the
# dashboard can still show something meaningful.
_SNAPSHOT_BYTE_CAP = 64 * 1024
_SNAPSHOT_PREVIEW_CHARS = 256


def _encode_snapshot(value: Any) -> str | None:
    """JSON-encode a snapshot payload with size capping. Returns None when
    the caller didn't supply a snapshot at all (preserves null-vs-empty
    distinction in the DB)."""
    if value is None:
        return None
    encoded = _serialize.encode_user_value(value, strict=True)
    if len(encoded.encode("utf-8")) <= _SNAPSHOT_BYTE_CAP:
        return encoded
    preview = encoded[:_SNAPSHOT_PREVIEW_CHARS]
    return json.dumps(
        {
            "__truncated__": True,
            "bytes": len(encoded.encode("utf-8")),
            "preview": preview,
        }
    )


def _decode_snapshot(raw: str | None) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def _encode_metadata(value: dict[str, Any] | None) -> str | None:
    """JSON-encode the metadata dict for storage. None passes through.

    Uses the canonical user-value serializer so non-JSON-native values
    (datetimes, UUIDs, etc.) get the same treatment as snapshots.
    """
    if value is None:
        return None
    return _serialize.encode_user_value(value, strict=True)


def _decode_metadata(raw: str | None) -> dict[str, Any] | None:
    """Inverse of _encode_metadata. Returns None for null rows."""
    if raw is None:
        return None
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return decoded if isinstance(decoded, dict) else None


# v12 base schema. Fresh databases are created at the CURRENT version
# directly — they never walk the v1→v12 migration chain (that both wasted
# work and produced one backup file per migration step: the backup storm).
# Existing databases (any DB that already has a `_meta` table) never run
# this script; they go through `_migrate` instead.
_SCHEMA_SQL = f"""\
CREATE TABLE IF NOT EXISTS {_schema.TBL_META} (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT OR IGNORE INTO {_schema.TBL_META} (key, value)
    VALUES ('schema_version', '{_SCHEMA_VERSION}');

CREATE TABLE IF NOT EXISTS {_schema.TBL_RUNS} (
    {_schema.COL_RUN_ID}              TEXT PRIMARY KEY,
    {_schema.COL_RUN_AGENT}           TEXT NOT NULL,
    {_schema.COL_RUN_STATUS}          TEXT NOT NULL DEFAULT 'queued',
    {_schema.COL_RUN_TOTAL_ITEMS}     INTEGER NOT NULL,
    {_schema.COL_RUN_COMPLETED}       INTEGER NOT NULL DEFAULT 0,
    {_schema.COL_RUN_FAILED}          INTEGER NOT NULL DEFAULT 0,
    {_schema.COL_RUN_CONCURRENCY_CAP} INTEGER,
    {_schema.COL_RUN_CREATED_AT}      TEXT NOT NULL,
    {_schema.COL_RUN_COMPLETED_AT}    TEXT,
    {_schema.COL_RUN_REPLAYED_FROM}   TEXT
);

CREATE TABLE IF NOT EXISTS {_schema.TBL_ITEMS} (
    {_schema.COL_ITEM_ID}              TEXT PRIMARY KEY,
    {_schema.COL_ITEM_AGENT}           TEXT NOT NULL,
    {_schema.COL_ITEM_STATUS}          TEXT NOT NULL DEFAULT 'running',
    {_schema.COL_ITEM_OUTPUT}          TEXT,
    {_schema.COL_ITEM_CREATED_AT}      TEXT NOT NULL,
    {_schema.COL_ITEM_UPDATED_AT}      TEXT NOT NULL,
    {_schema.COL_ITEM_RUN_ID}          TEXT,
    {_schema.COL_ITEM_ERROR_CODE}      TEXT,
    {_schema.COL_ITEM_ITEM_ID}         TEXT,
    {_schema.COL_ITEM_INPUT_SNAPSHOT}  TEXT,
    {_schema.COL_ITEM_DLQ_DISPOSITION} TEXT,
    {_schema.COL_ITEM_DLQ_RESOLVED_AT} TEXT,
    {_schema.COL_ITEM_REPLAYED_FROM}   TEXT,
    {_schema.COL_ITEM_AGENT_VERSION}   TEXT,
    {_schema.COL_ITEM_METADATA}        TEXT,
    {_schema.COL_ITEM_PARTITION_KEY}   TEXT,
    {_schema.COL_ITEM_PARENT_RUN_ID}   TEXT,
    {_schema.COL_ITEM_WORST_OUTCOME_STATUS} TEXT NOT NULL DEFAULT 'ok',
    {_schema.COL_ITEM_DEGRADED_COUNT}  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS {_schema.TBL_STEPS} (
    id                                 INTEGER PRIMARY KEY AUTOINCREMENT,
    {_schema.COL_STEP_ITEM_ID}         TEXT NOT NULL REFERENCES {_schema.TBL_ITEMS}({_schema.COL_ITEM_ID}),
    {_schema.COL_STEP_LABEL}           TEXT NOT NULL,
    {_schema.COL_STEP_RESULT}          TEXT,
    {_schema.COL_STEP_DURATION_MS}     INTEGER NOT NULL DEFAULT 0,
    {_schema.COL_STEP_COMPLETED_AT}    TEXT NOT NULL,
    {_schema.COL_STEP_CUSTOMER_ITEM_ID} TEXT,
    {_schema.COL_STEP_INPUT_SNAPSHOT}  TEXT,
    {_schema.COL_STEP_OUTPUT_SNAPSHOT} TEXT,
    {_schema.COL_STEP_KIND}            TEXT,
    {_schema.COL_STEP_LLM_PROMPT_TOKENS}     INTEGER,
    {_schema.COL_STEP_LLM_COMPLETION_TOKENS} INTEGER,
    {_schema.COL_STEP_LLM_TOTAL_TOKENS}      INTEGER,
    {_schema.COL_STEP_LLM_MODEL}       TEXT,
    {_schema.COL_STEP_LLM_STOP_REASON} TEXT,
    {_schema.COL_STEP_LLM_PROVIDER_SHAPE} TEXT,
    {_schema.COL_STEP_ERROR_CATEGORY}  TEXT,
    {_schema.COL_STEP_AGENT_VERSION}   TEXT,
    {_schema.COL_STEP_DELIVERY_ATTEMPTS} INTEGER,
    {_schema.COL_STEP_JOURNALED_AT}    TEXT,
    {_schema.COL_STEP_METADATA}        TEXT,
    {_schema.COL_STEP_PARTITION_KEY}   TEXT,
    {_schema.COL_STEP_OUTCOME_STATUS}  TEXT NOT NULL DEFAULT 'ok',
    {_schema.COL_STEP_OUTCOME_REASON}  TEXT
);
"""

# v12 indexes, shared between fresh-create and the v11→v12 rebuild.
_V12_INDEXES: list[str] = [
    f"CREATE INDEX IF NOT EXISTS {_schema.IDX_ITEMS_RUN} "
    f"ON {_schema.TBL_ITEMS}({_schema.COL_ITEM_RUN_ID});",
    f"CREATE INDEX IF NOT EXISTS {_schema.IDX_ITEMS_ITEM} "
    f"ON {_schema.TBL_ITEMS}({_schema.COL_ITEM_ITEM_ID});",
    f"CREATE INDEX IF NOT EXISTS {_schema.IDX_ITEMS_DLQ} "
    f"ON {_schema.TBL_ITEMS}({_schema.COL_ITEM_RUN_ID}, {_schema.COL_ITEM_DLQ_DISPOSITION}) "
    f"WHERE {_schema.COL_ITEM_STATUS} = 'failed';",
    f"CREATE INDEX IF NOT EXISTS {_schema.IDX_ITEMS_PARTITION} "
    f"ON {_schema.TBL_ITEMS}({_schema.COL_ITEM_PARTITION_KEY});",
    f"CREATE INDEX IF NOT EXISTS {_schema.IDX_ITEMS_PARENT} "
    f"ON {_schema.TBL_ITEMS}({_schema.COL_ITEM_PARENT_RUN_ID});",
    f"CREATE INDEX IF NOT EXISTS {_schema.IDX_STEPS_ITEM} "
    f"ON {_schema.TBL_STEPS}({_schema.COL_STEP_ITEM_ID});",
    f"CREATE INDEX IF NOT EXISTS {_schema.IDX_STEPS_CUSTOMER_ITEM} "
    f"ON {_schema.TBL_STEPS}({_schema.COL_STEP_CUSTOMER_ITEM_ID});",
    f"CREATE INDEX IF NOT EXISTS {_schema.IDX_STEPS_PARTITION} "
    f"ON {_schema.TBL_STEPS}({_schema.COL_STEP_PARTITION_KEY});",
]


# --------------------------------------------------------------------------- #
#  Frozen migration history (v1 → v11).                                        #
#                                                                               #
#  Everything below this banner operates on the PRE-v12 table names            #
#  (`batches` / `runs`-as-per-item / `tasks` / legacy `steps`) because it      #
#  runs BEFORE the v12 rename. These are deliberately literal strings, not     #
#  `_schema` constants — the constants now carry the post-rename meanings.     #
#  Do not edit this history; add new migrations at the bottom.                 #
# --------------------------------------------------------------------------- #

# v2 additions: batch entity + denormalized columns for clustering and search.
_V2_CREATE_BATCHES = """\
CREATE TABLE IF NOT EXISTS batches (
    batch_id         TEXT PRIMARY KEY,
    agent            TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'queued',
    total_items      INTEGER NOT NULL,
    completed        INTEGER NOT NULL DEFAULT 0,
    failed           INTEGER NOT NULL DEFAULT 0,
    aggregate_cost_usd REAL NOT NULL DEFAULT 0.0,
    budget_limit_usd REAL,
    concurrency_cap  INTEGER,
    created_at       TEXT NOT NULL,
    completed_at     TEXT
);
"""

_V2_ADD_COLUMNS: list[tuple[str, str, str]] = [
    ("runs", "batch_id", "TEXT"),
    ("runs", "error_code", "TEXT"),
    ("steps", "tool_name", "TEXT"),
    ("steps", "error_code", "TEXT"),
    ("steps", "error_category", "TEXT"),
    ("steps", "input_hash", "TEXT"),
]

_V2_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_steps_tool ON steps(tool_name);",
    "CREATE INDEX IF NOT EXISTS idx_steps_error ON steps(error_code);",
    "CREATE INDEX IF NOT EXISTS idx_runs_batch ON runs(batch_id);",
]

_V3_ADD_COLUMNS: list[tuple[str, str, str]] = [
    ("tasks", "item_id", "TEXT"),
    ("tasks", "input_snapshot", "TEXT"),
    ("tasks", "output_snapshot", "TEXT"),
    ("runs", "item_id", "TEXT"),
]

_V3_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_tasks_item ON tasks(item_id);",
    "CREATE INDEX IF NOT EXISTS idx_runs_item ON runs(item_id);",
]

_V4_DROP_COLUMNS: list[tuple[str, str]] = [
    ("runs", "budget_limit_usd"),
    ("runs", "budget_consumed_usd"),
    ("runs", "total_input_tokens"),
    ("runs", "total_output_tokens"),
    ("runs", "budget_input_tokens"),
    ("runs", "budget_output_tokens"),
    ("tasks", "cost_usd"),
    ("tasks", "input_tokens"),
    ("tasks", "output_tokens"),
    ("steps", "input_tokens"),
    ("steps", "output_tokens"),
    ("steps", "cost_usd"),
    ("batches", "aggregate_cost_usd"),
    ("batches", "budget_limit_usd"),
]

_V5_ADD_COLUMNS: list[tuple[str, str, str]] = [
    ("tasks", "kind", "TEXT"),
    ("tasks", "llm_prompt_tokens", "INTEGER"),
    ("tasks", "llm_completion_tokens", "INTEGER"),
    ("tasks", "llm_total_tokens", "INTEGER"),
    ("tasks", "llm_model", "TEXT"),
    ("tasks", "llm_stop_reason", "TEXT"),
    ("tasks", "llm_provider_shape", "TEXT"),
    ("tasks", "error_category", "TEXT"),
]

_V6_ADD_COLUMNS: list[tuple[str, str, str]] = [
    ("runs", "input_snapshot", "TEXT"),
    ("runs", "dlq_disposition", "TEXT"),
    ("runs", "dlq_resolved_at", "TEXT"),
    ("runs", "replayed_from", "TEXT"),
]

_V6_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_runs_dlq "
    "ON runs(batch_id, dlq_disposition) WHERE status = 'failed';",
]

_V7_ADD_COLUMNS: list[tuple[str, str, str]] = [
    ("runs", "agent_version", "TEXT"),
    ("tasks", "agent_version", "TEXT"),
]

_V8_ADD_COLUMNS: list[tuple[str, str, str]] = [
    ("tasks", "delivery_attempts", "INTEGER"),
    ("tasks", "journaled_at", "TEXT"),
]

_V9_ADD_COLUMNS: list[tuple[str, str, str]] = [
    ("runs", "metadata", "TEXT"),
    ("runs", "partition_key", "TEXT"),
    ("tasks", "metadata", "TEXT"),
    ("tasks", "partition_key", "TEXT"),
]

_V9_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_runs_partition ON runs(partition_key);",
    "CREATE INDEX IF NOT EXISTS idx_tasks_partition ON tasks(partition_key);",
]

_V10_ADD_COLUMNS: list[tuple[str, str, str]] = [
    ("runs", "parent_run_id", "TEXT"),
]

_V10_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_runs_parent ON runs(parent_run_id);",
]

_V11_ADD_COLUMNS: list[tuple[str, str, str]] = [
    ("tasks", "outcome_status", "TEXT NOT NULL DEFAULT 'ok'"),
    ("tasks", "outcome_reason", "TEXT"),
    ("runs", "worst_outcome_status", "TEXT NOT NULL DEFAULT 'ok'"),
    ("runs", "degraded_count", "INTEGER NOT NULL DEFAULT 0"),
]

# Pre-v12 index names that the v12 rebuild drops (renamed tables keep their
# old-named indexes; the legacy `steps` table's indexes die with the table).
_PRE_V12_INDEXES = [
    "idx_tasks_run_id",
    "idx_runs_batch",
    "idx_tasks_item",
    "idx_runs_item",
    "idx_runs_dlq",
    "idx_runs_partition",
    "idx_tasks_partition",
    "idx_runs_parent",
]


def _existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r[1] for r in rows}


def _get_schema_version(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT value FROM _meta WHERE key = 'schema_version'"
    ).fetchone()
    return row[0] if row else "1"


def _backup_db(db_path: Path, from_version: str) -> Path | None:
    """Copy the DB to a sibling backup file before a destructive-ish migration.

    Returns the backup path on success, None if the DB doesn't exist yet
    (fresh install) or if the backup already exists (don't overwrite).

    The caller has already opened the connection (and so may have rewritten
    the journal_mode header) by the time we run, so the backup is not
    necessarily byte-identical to the pre-open file. Row data is preserved
    exactly; only the SQLite header may differ. SDK init is synchronous, so no
    concurrent writer can have touched the DB between open and backup.
    """
    if not db_path.exists():
        return None
    backup = db_path.with_suffix(db_path.suffix + f".backup-v{from_version}")
    if backup.exists():
        return backup
    shutil.copy2(db_path, backup)
    return backup


def _has_meta_table(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = '_meta'"
    ).fetchone()
    return row is not None


def _init_schema(conn: sqlite3.Connection, db_path: Path) -> None:
    """Create-or-migrate the schema on an open connection.

    Fresh databases (no ``_meta`` table yet) are created at
    ``_SCHEMA_VERSION`` directly — they never enter the migration chain,
    which is both faster and fixes the backup storm (a fresh DB used to be
    created at v1 and then chain-migrated, leaving one ``backup-vN`` file
    per migration step). Existing databases skip the base script entirely
    (its v12 CREATEs would collide with pre-v12 tables mid-rename) and go
    through ``_migrate``.
    """
    if _has_meta_table(conn):
        _migrate(conn, db_path)
        return
    with conn:
        conn.executescript(_SCHEMA_SQL)
        for index_sql in _V12_INDEXES:
            conn.execute(index_sql)


def ensure_migrated(db_path: str | Path) -> None:
    """Open ``db_path``, create the base schema, and run migrations.

    Safe entry point for any caller that needs the DB to be at the current
    schema version — the SDK writer path goes through ``SQLiteStore``, but
    the dashboard server opens the file read-only and needs the tables to
    exist already. Idempotent: no-op if the DB is already current.
    """
    file = Path(db_path)
    file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(file))
    try:
        # busy_timeout before any lock-taking pragma: flipping a previously-WAL
        # database back to DELETE needs an exclusive checkpoint, so wait out a
        # concurrent reader/writer instead of erroring with SQLITE_BUSY.
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA journal_mode=DELETE")
        _init_schema(conn, file)
    finally:
        conn.close()


def _set_schema_version(conn: sqlite3.Connection, version: str) -> None:
    conn.execute(
        "UPDATE _meta SET value = ? WHERE key = 'schema_version'",
        (version,),
    )


def _promote_partial_if_drained(
    conn: sqlite3.Connection,
    run_id: str,
    now: str,
) -> None:
    """Promote a 'partial' run to 'completed' when its DLQ is empty.

    Called after any event that could change the unresolved-dead-letter
    count: a DLQ disposition change, or a new item resolving inside a
    partial run. Only transitions 'partial' → 'completed'; 'failed' and
    'cancelled' runs stay terminal on their own terms.

    The NOT EXISTS clause treats each replay's fresh item as part of the
    run's active surface — a failed replay keeps the run 'partial'
    because there's a new unresolved dead letter, which is what the
    operator would expect.
    """
    conn.execute(
        f"""UPDATE {_schema.TBL_RUNS}
            SET {_schema.COL_RUN_STATUS} = 'completed',
                {_schema.COL_RUN_COMPLETED_AT} =
                    COALESCE({_schema.COL_RUN_COMPLETED_AT}, ?)
            WHERE {_schema.COL_RUN_ID} = ?
              AND {_schema.COL_RUN_STATUS} = 'partial'
              AND NOT EXISTS (
                  SELECT 1 FROM {_schema.TBL_ITEMS}
                  WHERE {_schema.COL_ITEM_RUN_ID} = ?
                    AND {_schema.COL_ITEM_STATUS} = 'failed'
                    AND {_schema.COL_ITEM_DLQ_DISPOSITION} IS NULL
              )""",
        (now, run_id, run_id),
    )


def _apply_v1_to_v2(conn: sqlite3.Connection, db_path: Path) -> None:
    _backup_db(db_path, "1")
    with conn:
        conn.execute(_V2_CREATE_BATCHES)
        for table, column, type_decl in _V2_ADD_COLUMNS:
            if column in _existing_columns(conn, table):
                continue
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {type_decl}")
        for index_sql in _V2_INDEXES:
            conn.execute(index_sql)
        _set_schema_version(conn, "2")


def _apply_v2_to_v3(conn: sqlite3.Connection, db_path: Path) -> None:
    _backup_db(db_path, "2")
    with conn:
        for table, column, type_decl in _V3_ADD_COLUMNS:
            if column in _existing_columns(conn, table):
                continue
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {type_decl}")
        for index_sql in _V3_INDEXES:
            conn.execute(index_sql)
        _set_schema_version(conn, "3")


def _apply_v3_to_v4(conn: sqlite3.Connection, db_path: Path) -> None:
    """Drop budget/cost/token columns. Requires SQLite 3.35+ (DROP COLUMN).

    SQLite 3.35 shipped March 2021; Python 3.12+ bundles 3.40+. Older
    Pythons with an older SQLite will raise on DROP COLUMN — the caller
    should upgrade Python rather than us falling back to table-recreation,
    which is substantially more error-prone.
    """
    _backup_db(db_path, "3")
    with conn:
        for table, column in _V4_DROP_COLUMNS:
            if column not in _existing_columns(conn, table):
                continue
            conn.execute(f'ALTER TABLE {table} DROP COLUMN "{column}"')
        _set_schema_version(conn, "4")


def _apply_v4_to_v5(conn: sqlite3.Connection, db_path: Path) -> None:
    _backup_db(db_path, "4")
    with conn:
        for table, column, type_decl in _V5_ADD_COLUMNS:
            if column in _existing_columns(conn, table):
                continue
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {type_decl}")
        _set_schema_version(conn, "5")


def _apply_v5_to_v6(conn: sqlite3.Connection, db_path: Path) -> None:
    _backup_db(db_path, "5")
    with conn:
        for table, column, type_decl in _V6_ADD_COLUMNS:
            if column in _existing_columns(conn, table):
                continue
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {type_decl}")
        for index_sql in _V6_INDEXES:
            conn.execute(index_sql)
        _set_schema_version(conn, "6")


def _apply_v6_to_v7(conn: sqlite3.Connection, db_path: Path) -> None:
    _backup_db(db_path, "6")
    with conn:
        for table, column, type_decl in _V7_ADD_COLUMNS:
            if column in _existing_columns(conn, table):
                continue
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {type_decl}")
        _set_schema_version(conn, "7")


def _apply_v7_to_v8(conn: sqlite3.Connection, db_path: Path) -> None:
    _backup_db(db_path, "7")
    with conn:
        for table, column, type_decl in _V8_ADD_COLUMNS:
            if column in _existing_columns(conn, table):
                continue
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {type_decl}")
        _set_schema_version(conn, "8")


def _apply_v8_to_v9(conn: sqlite3.Connection, db_path: Path) -> None:
    _backup_db(db_path, "8")
    with conn:
        for table, column, type_decl in _V9_ADD_COLUMNS:
            if column in _existing_columns(conn, table):
                continue
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {type_decl}")
        for index_sql in _V9_INDEXES:
            conn.execute(index_sql)
        _set_schema_version(conn, "9")


def _apply_v9_to_v10(conn: sqlite3.Connection, db_path: Path) -> None:
    _backup_db(db_path, "9")
    with conn:
        for table, column, type_decl in _V10_ADD_COLUMNS:
            if column in _existing_columns(conn, table):
                continue
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {type_decl}")
        for index_sql in _V10_INDEXES:
            conn.execute(index_sql)
        _set_schema_version(conn, "10")


def _apply_v10_to_v11(conn: sqlite3.Connection, db_path: Path) -> None:
    _backup_db(db_path, "10")
    with conn:
        for table, column, type_decl in _V11_ADD_COLUMNS:
            if column in _existing_columns(conn, table):
                continue
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {type_decl}")
        _set_schema_version(conn, "11")


def _apply_v11_to_v12(conn: sqlite3.Connection, db_path: Path) -> None:
    """Plan 34 noun consolidation. Order is load-bearing:

    1. drop the dead legacy ``steps`` table (no SDK writer since v2's
       record_step fell out of the production path; must go first so
       ``tasks`` can take the name)
    2. ``runs`` → ``items`` (must precede ``batches`` → ``runs``)
    3. on ``items``: ``run_id`` → ``id`` BEFORE ``batch_id`` → ``run_id``
       (reversed order would collide)
    4. ``batches`` → ``runs`` (+ ``batch_id`` → ``run_id``, + the new
       ``replayed_from`` column for slice-replay lineage)
    5. ``tasks`` → ``steps``: customer ``item_id`` → ``customer_item_id``
       BEFORE the ``run_id`` FK → ``item_id`` (same collision logic)
    6. drop old-named indexes, build the v12 set
    """
    _backup_db(db_path, "11")
    with conn:
        # 1. dead legacy steps table
        conn.execute("DROP TABLE IF EXISTS steps")
        # 2-3. per-item records
        conn.execute("ALTER TABLE runs RENAME TO items")
        conn.execute("ALTER TABLE items RENAME COLUMN run_id TO id")
        conn.execute("ALTER TABLE items RENAME COLUMN batch_id TO run_id")
        # 4. invocations
        conn.execute("ALTER TABLE batches RENAME TO runs")
        conn.execute("ALTER TABLE runs RENAME COLUMN batch_id TO run_id")
        if "replayed_from" not in _existing_columns(conn, "runs"):
            conn.execute("ALTER TABLE runs ADD COLUMN replayed_from TEXT")
        # 5. trace nodes
        conn.execute("ALTER TABLE tasks RENAME COLUMN item_id TO customer_item_id")
        conn.execute("ALTER TABLE tasks RENAME COLUMN run_id TO item_id")
        conn.execute("ALTER TABLE tasks RENAME TO steps")
        # 6. indexes
        for index_name in _PRE_V12_INDEXES:
            conn.execute(f"DROP INDEX IF EXISTS {index_name}")
        for index_sql in _V12_INDEXES:
            conn.execute(index_sql)
        _set_schema_version(conn, "12")


def _migrate(conn: sqlite3.Connection, db_path: Path) -> None:
    """Forward-only migrations. Idempotent: safe to call on any schema version.

    Each migration runs in a single transaction. A mid-migration crash leaves
    the DB at the prior version with no half-applied ALTERs. When more than
    one version gap separates the DB from the SDK, migrations chain in order
    (e.g. v1 → v2 → … → v11 → v12) so long-dormant local DBs catch up cleanly.
    """
    current = _get_schema_version(conn)
    while current != _SCHEMA_VERSION:
        if current == "1":
            _apply_v1_to_v2(conn, db_path)
            current = "2"
        elif current == "2":
            _apply_v2_to_v3(conn, db_path)
            current = "3"
        elif current == "3":
            _apply_v3_to_v4(conn, db_path)
            current = "4"
        elif current == "4":
            _apply_v4_to_v5(conn, db_path)
            current = "5"
        elif current == "5":
            _apply_v5_to_v6(conn, db_path)
            current = "6"
        elif current == "6":
            _apply_v6_to_v7(conn, db_path)
            current = "7"
        elif current == "7":
            _apply_v7_to_v8(conn, db_path)
            current = "8"
        elif current == "8":
            _apply_v8_to_v9(conn, db_path)
            current = "9"
        elif current == "9":
            _apply_v9_to_v10(conn, db_path)
            current = "10"
        elif current == "10":
            _apply_v10_to_v11(conn, db_path)
            current = "11"
        elif current == "11":
            _apply_v11_to_v12(conn, db_path)
            current = "12"
        else:
            raise RuntimeError(
                f"Unknown schema version {current!r}; expected {_SCHEMA_VERSION!r}. "
                "This usually means the DB was written by a newer SDK than this one. "
                "Upgrade the SDK to match, or point at a fresh database path."
            )


class SQLiteStore:
    """SQLite-backed checkpoint store with step-level observability."""

    def __init__(self, db_path: str = ".papayya/local.db") -> None:
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # journal_mode=DELETE (not WAL): the worker mints one short-lived store
        # connection per item, while `papayya dev`/the test harness keeps a
        # long-lived reader open. Under WAL that reader blocks checkpointing, so
        # committed frames strand in the -wal file and never reach main.db —
        # rows silently vanish on worker shutdown. DELETE checkpoints on each
        # commit; busy_timeout serializes the now-exclusive writer against
        # concurrent readers instead of failing them with SQLITE_BUSY.
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA journal_mode=DELETE")
        _init_schema(self._conn, db_file)

        # Plan 33 local fences (Decision 6) — so `papayya dev` demos auto-pause
        # with zero cloud dependency. Run-level: pause after K consecutive
        # degraded steps (env override wins; else the per-run kwarg via
        # set_run_fence; else 3; 0 disables). Workload-level: pause the agent
        # when >= Pct% of its last Window terminal items degraded, with a
        # MinDegraded floor. State lives in memory — a local demo, not a
        # durable control plane; a process restart clears it.
        self._pending_pause: dict[str, str] = {}    # run_id (item) -> reason
        self._workload_paused: dict[str, str] = {}  # agent -> reason
        self._run_fence: dict[str, int] = {}        # run_id -> K override
        _env_k = os.environ.get("PAPAYYA_PAUSE_AFTER_DEGRADED")
        self._env_pause_after_degraded: int | None = (
            _env_int("PAPAYYA_PAUSE_AFTER_DEGRADED", 3) if _env_k not in (None, "") else None
        )
        self._workload_pause_pct = _env_int("PAPAYYA_WORKLOAD_PAUSE_PCT", 50)
        self._workload_pause_window = _env_int("PAPAYYA_WORKLOAD_PAUSE_WINDOW", 20)
        self._workload_pause_min = _env_int("PAPAYYA_WORKLOAD_PAUSE_MIN_DEGRADED", 5)

    # --- Plan 33 auto-pause fences (local parity, Decision 6) ---

    def set_run_fence(self, run_id: str, pause_after_degraded: int) -> None:
        """Register a per-run run-level K (from DurableRunConfig). The env
        override, when set, still wins over this."""
        self._run_fence[run_id] = pause_after_degraded

    def _resolve_k(self, run_id: str) -> int:
        if self._env_pause_after_degraded is not None:
            return self._env_pause_after_degraded
        return self._run_fence.get(run_id, 3)

    def pending_pause(self, run_id: str) -> str | None:
        """Run-level pause reason for this item, or None. Read by
        PapayyaRun._pre_call before the next step."""
        return self._pending_pause.get(run_id)

    def clear_pending_pause(self, run_id: str) -> None:
        """Clear a run-level pause — the local resume surface for a paused run.
        After clearing, a replay of the same run_id skips saved steps and
        continues from exactly where the pause landed."""
        self._pending_pause.pop(run_id, None)

    def workload_paused(self, agent: str) -> str | None:
        """Workload-level pause reason for this agent, or None. Read by
        iter/map before minting the next item's run."""
        return self._workload_paused.get(agent)

    def resume_workload(self, agent: str) -> None:
        """Clear the workload-level pause — the local resume surface."""
        self._workload_paused.pop(agent, None)

    def _evaluate_workload_fence(self, agent: str | None) -> None:
        """Pause the agent if its recent terminal items degraded past the
        threshold. Called on each item completion; idempotent once paused."""
        if agent is None or agent in self._workload_paused:
            return
        rows = self._conn.execute(
            f"SELECT {_schema.COL_ITEM_WORST_OUTCOME_STATUS} AS w "
            f"FROM {_schema.TBL_ITEMS} "
            f"WHERE {_schema.COL_ITEM_AGENT} = ? "
            f"  AND {_schema.COL_ITEM_STATUS} IN ('completed', 'failed') "
            f"ORDER BY {_schema.COL_ITEM_CREATED_AT} DESC LIMIT ?",
            (agent, self._workload_pause_window),
        ).fetchall()
        total = len(rows)
        degraded = sum(1 for r in rows if r["w"] != "ok")
        if total == 0 or degraded < self._workload_pause_min:
            return
        if degraded * 100 < self._workload_pause_pct * total:
            return
        self._workload_paused[agent] = f"{degraded} of last {total} runs degraded"

    # --- CheckpointStore protocol ---
    #
    # Protocol note: the ``run_id`` parameter below is the shared
    # CheckpointStore vocabulary (frozen with the cloud wire until Unit 5).
    # In this store it addresses an ITEM row — the `items.id` surrogate.

    def load(self, run_id: str) -> RunCheckpoint | None:
        row = self._conn.execute(
            f"SELECT * FROM {_schema.TBL_ITEMS} WHERE {_schema.COL_ITEM_ID} = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None

        step_rows = self._conn.execute(
            f"SELECT * FROM {_schema.TBL_STEPS} "
            f"WHERE {_schema.COL_STEP_ITEM_ID} = ? ORDER BY id",
            (run_id,),
        ).fetchall()

        tasks = [
            TaskEntry(
                label=t["label"],
                result=json.loads(t["result"]) if t["result"] is not None else None,
                duration_ms=t["duration_ms"],
                completed_at=t["completed_at"],
                item_id=t[_schema.COL_STEP_CUSTOMER_ITEM_ID],
                input_snapshot=_decode_snapshot(t[_schema.COL_STEP_INPUT_SNAPSHOT]),
                output_snapshot=_decode_snapshot(t[_schema.COL_STEP_OUTPUT_SNAPSHOT]),
                kind=t[_schema.COL_STEP_KIND],
                llm_prompt_tokens=t[_schema.COL_STEP_LLM_PROMPT_TOKENS],
                llm_completion_tokens=t[_schema.COL_STEP_LLM_COMPLETION_TOKENS],
                llm_total_tokens=t[_schema.COL_STEP_LLM_TOTAL_TOKENS],
                llm_model=t[_schema.COL_STEP_LLM_MODEL],
                llm_stop_reason=t[_schema.COL_STEP_LLM_STOP_REASON],
                llm_provider_shape=t[_schema.COL_STEP_LLM_PROVIDER_SHAPE],
                error_category=t[_schema.COL_STEP_ERROR_CATEGORY],
                agent_version=t[_schema.COL_STEP_AGENT_VERSION],
                metadata=_decode_metadata(t[_schema.COL_STEP_METADATA]),
                partition_key=t[_schema.COL_STEP_PARTITION_KEY],
                outcome_status=t[_schema.COL_STEP_OUTCOME_STATUS],
                outcome_reason=t[_schema.COL_STEP_OUTCOME_REASON],
            )
            for t in step_rows
        ]

        return RunCheckpoint(
            run_id=row[_schema.COL_ITEM_ID],
            agent=row["agent"],
            tasks=tasks,
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            item_id=row[_schema.COL_ITEM_ITEM_ID],
            input_snapshot=_decode_snapshot(row[_schema.COL_ITEM_INPUT_SNAPSHOT]),
            agent_version=row[_schema.COL_ITEM_AGENT_VERSION],
            metadata=_decode_metadata(row[_schema.COL_ITEM_METADATA]),
            partition_key=row[_schema.COL_ITEM_PARTITION_KEY],
            parent_run_id=row[_schema.COL_ITEM_PARENT_RUN_ID],
            worst_outcome_status=row[_schema.COL_ITEM_WORST_OUTCOME_STATUS],
            degraded_count=row[_schema.COL_ITEM_DEGRADED_COUNT],
            invocation_id=row[_schema.COL_ITEM_RUN_ID],
        )

    def save_task(self, run_id: str, entry: TaskEntry) -> None:
        now = datetime.now(timezone.utc).isoformat()
        input_snapshot_json = _encode_snapshot(entry.input_snapshot)
        output_snapshot_json = _encode_snapshot(entry.output_snapshot)
        with self._conn:
            # Idempotency guard (parity with the control-plane SaveCheckpoint
            # xmax=0 fix): a re-delivery of the same (item, label) must not
            # insert a duplicate step row or double-count the item aggregates
            # below. The local step cache normally prevents re-execution, so
            # this is defensive; first-writer-wins matches the cloud path's
            # ON CONFLICT (run_id, label) semantics.
            already = self._conn.execute(
                f"SELECT 1 FROM {_schema.TBL_STEPS} "
                f"WHERE {_schema.COL_STEP_ITEM_ID} = ? AND label = ? LIMIT 1",
                (run_id, entry.label),
            ).fetchone()
            if already is not None:
                return
            self._conn.execute(
                f"""INSERT INTO {_schema.TBL_STEPS} ({_schema.COL_STEP_ITEM_ID},
                   label, result, duration_ms, completed_at,
                   {_schema.COL_STEP_CUSTOMER_ITEM_ID},
                   {_schema.COL_STEP_INPUT_SNAPSHOT},
                   {_schema.COL_STEP_OUTPUT_SNAPSHOT},
                   {_schema.COL_STEP_KIND},
                   {_schema.COL_STEP_LLM_PROMPT_TOKENS},
                   {_schema.COL_STEP_LLM_COMPLETION_TOKENS},
                   {_schema.COL_STEP_LLM_TOTAL_TOKENS},
                   {_schema.COL_STEP_LLM_MODEL},
                   {_schema.COL_STEP_LLM_STOP_REASON},
                   {_schema.COL_STEP_LLM_PROVIDER_SHAPE},
                   {_schema.COL_STEP_ERROR_CATEGORY},
                   {_schema.COL_STEP_AGENT_VERSION},
                   {_schema.COL_STEP_METADATA},
                   {_schema.COL_STEP_PARTITION_KEY},
                   {_schema.COL_STEP_OUTCOME_STATUS},
                   {_schema.COL_STEP_OUTCOME_REASON})
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    entry.label,
                    _serialize.encode_user_value(entry.result),
                    entry.duration_ms,
                    entry.completed_at,
                    entry.item_id,
                    input_snapshot_json,
                    output_snapshot_json,
                    entry.kind,
                    entry.llm_prompt_tokens,
                    entry.llm_completion_tokens,
                    entry.llm_total_tokens,
                    entry.llm_model,
                    entry.llm_stop_reason,
                    entry.llm_provider_shape,
                    entry.error_category,
                    entry.agent_version,
                    _encode_metadata(entry.metadata),
                    entry.partition_key,
                    entry.outcome_status,
                    entry.outcome_reason,
                ),
            )
            self._conn.execute(
                f"UPDATE {_schema.TBL_ITEMS} SET updated_at = ? "
                f"WHERE {_schema.COL_ITEM_ID} = ?",
                (now, run_id),
            )
            # Denormalize the customer item_id onto the item row on
            # first-writer-wins basis. Later steps with a different item_id
            # don't overwrite — the item-level item_id represents the primary
            # record flowing through this item, not a mutable state field.
            if entry.item_id is not None:
                self._conn.execute(
                    f"""UPDATE {_schema.TBL_ITEMS} SET {_schema.COL_ITEM_ITEM_ID} = ?
                       WHERE {_schema.COL_ITEM_ID} = ?
                         AND {_schema.COL_ITEM_ITEM_ID} IS NULL""",
                    (entry.item_id, run_id),
                )
            # Incremental aggregation of the item's worst-outcome severity.
            # save_task is INSERT-only (no in-place step rewrites) so an
            # incremental update is sound. Skips the UPDATE when neither
            # value changes to avoid touching the row on the all-'ok' path.
            item_row = self._conn.execute(
                f"""SELECT {_schema.COL_ITEM_WORST_OUTCOME_STATUS} AS worst,
                          {_schema.COL_ITEM_DEGRADED_COUNT} AS degraded
                   FROM {_schema.TBL_ITEMS} WHERE {_schema.COL_ITEM_ID} = ?""",
                (run_id,),
            ).fetchone()
            if item_row is not None:
                cur_status = item_row["worst"]
                cur_count = item_row["degraded"]
                next_status = (
                    entry.outcome_status
                    if _outcome_severity(entry.outcome_status) > _outcome_severity(cur_status)
                    else cur_status
                )
                next_count = cur_count + (0 if entry.outcome_status == "ok" else 1)
                if next_status != cur_status or next_count != cur_count:
                    self._conn.execute(
                        f"""UPDATE {_schema.TBL_ITEMS} SET
                               {_schema.COL_ITEM_WORST_OUTCOME_STATUS} = ?,
                               {_schema.COL_ITEM_DEGRADED_COUNT} = ?
                           WHERE {_schema.COL_ITEM_ID} = ?""",
                        (next_status, next_count, run_id),
                    )
            # Plan 33 local run-level fence: pause after K consecutive degraded
            # steps. A streak can only complete on a non-ok step, so only check
            # then; the last K step rows by id are this run's most recent steps
            # (save_task is INSERT-only). Sets pending_pause — PapayyaRun._pre_call
            # raises WorkloadPaused before the next step. Once set, don't recompute.
            k = self._resolve_k(run_id)
            if k > 0 and entry.outcome_status != "ok" and run_id not in self._pending_pause:
                recent = self._conn.execute(
                    f"SELECT {_schema.COL_STEP_OUTCOME_STATUS} AS s "
                    f"FROM {_schema.TBL_STEPS} WHERE {_schema.COL_STEP_ITEM_ID} = ? "
                    f"ORDER BY id DESC LIMIT ?",
                    (run_id, k),
                ).fetchall()
                if len(recent) == k and all(r["s"] != "ok" for r in recent):
                    detail = entry.outcome_reason or entry.outcome_status
                    self._pending_pause[run_id] = f"{k} consecutive degraded steps: {detail}"

    def set_status(self, run_id: str, status: str, output: Any = None) -> None:
        """Transition an item's status, and roll up terminal counts to its run."""
        now = datetime.now(timezone.utc).isoformat()
        # Capture prior status before we overwrite it — used below to decide
        # whether this transition bumps a run counter.
        row = self._conn.execute(
            f"SELECT status, {_schema.COL_ITEM_RUN_ID} AS run_id, {_schema.COL_ITEM_AGENT} AS agent "
            f"FROM {_schema.TBL_ITEMS} WHERE {_schema.COL_ITEM_ID} = ?",
            (run_id,),
        ).fetchone()

        with self._conn:
            self._conn.execute(
                f"UPDATE {_schema.TBL_ITEMS} SET status = ?, output = ?, updated_at = ? "
                f"WHERE {_schema.COL_ITEM_ID} = ?",
                (status, _serialize.encode_user_value(output) if output is not None else None, now, run_id),
            )
            if (
                _capture_enabled()
                and row is not None
                and row["run_id"] is not None
                and row["status"] not in ("completed", "failed")
                and status in ("completed", "failed")
            ):
                counter = (
                    _schema.COL_RUN_COMPLETED
                    if status == "completed"
                    else _schema.COL_RUN_FAILED
                )
                self._conn.execute(
                    f"""UPDATE {_schema.TBL_RUNS}
                        SET {counter} = {counter} + 1
                        WHERE {_schema.COL_RUN_ID} = ?""",
                    (row["run_id"],),
                )
                # Roll the run to its terminal status once every item has
                # resolved. Ternary outcome: zero failures → 'completed';
                # zero successes → 'failed'; mixed → 'partial'. The DLQ
                # surface acts on partial-terminal runs to re-drive the
                # failed items; once all dead letters are replayed or
                # skipped, a later pass promotes the run from 'partial'
                # to 'completed'.
                #
                # total_items > 0 guard: an OPEN run (minted by map()/iter()
                # before the item count is known) carries total_items=0 until
                # finalize_run seals it — the rollup must not fire while the
                # invocation is still producing items.
                self._conn.execute(
                    f"""UPDATE {_schema.TBL_RUNS}
                        SET {_schema.COL_RUN_STATUS} = CASE
                                WHEN {_schema.COL_RUN_FAILED} = 0 THEN 'completed'
                                WHEN {_schema.COL_RUN_COMPLETED} = 0 THEN 'failed'
                                ELSE 'partial'
                            END,
                            {_schema.COL_RUN_COMPLETED_AT} = ?
                        WHERE {_schema.COL_RUN_ID} = ?
                          AND {_schema.COL_RUN_COMPLETED_AT} IS NULL
                          AND {_schema.COL_RUN_TOTAL_ITEMS} > 0
                          AND ({_schema.COL_RUN_COMPLETED} + {_schema.COL_RUN_FAILED})
                              >= {_schema.COL_RUN_TOTAL_ITEMS}""",
                    (now, row["run_id"]),
                )
                # An item resolving inside an already-terminal 'partial' run
                # (e.g. a replay's fresh item completing) can drain the DLQ —
                # re-check and promote if so. No-op when the run is still
                # running or is already in a different terminal state.
                _promote_partial_if_drained(self._conn, row["run_id"], now)
                # Plan 33 local workload-level fence: this item just reached a
                # terminal state, so re-check the agent's recent degraded rate.
                # Enforced at the next run-open in iter/map, which reads
                # workload_paused before minting the next item's run.
                self._evaluate_workload_fence(row["agent"])

    def create(self, checkpoint: RunCheckpoint) -> None:
        """Create an item row.

        Invocation linkage (Plan 34 Unit 1):

        * ``checkpoint.invocation_id`` set → the caller (``papayya.map`` /
          ``papayya.iter`` / slice replay) already minted the run row via
          :meth:`create_run`; link this item to it.
        * unset → this is a direct call: wrap it in an implicit run-of-one
          whose id is ``single-{item id}`` (the pre-v12 implicit batch,
          renamed), when capture is enabled.
        """
        run_id: str | None = checkpoint.invocation_id
        if run_id is None and _capture_enabled():
            run_id = _single_run_id(checkpoint.run_id)
            self._conn.execute(
                f"""INSERT OR IGNORE INTO {_schema.TBL_RUNS}
                    ({_schema.COL_RUN_ID}, {_schema.COL_RUN_AGENT},
                     {_schema.COL_RUN_STATUS}, {_schema.COL_RUN_TOTAL_ITEMS},
                     {_schema.COL_RUN_CREATED_AT})
                    VALUES (?, ?, 'running', 1, ?)""",
                (run_id, checkpoint.agent, checkpoint.created_at),
            )

        self._conn.execute(
            f"""INSERT INTO {_schema.TBL_ITEMS} ({_schema.COL_ITEM_ID}, agent, status,
               created_at, updated_at,
               {_schema.COL_ITEM_RUN_ID}, {_schema.COL_ITEM_ITEM_ID},
               {_schema.COL_ITEM_INPUT_SNAPSHOT},
               {_schema.COL_ITEM_AGENT_VERSION},
               {_schema.COL_ITEM_METADATA}, {_schema.COL_ITEM_PARTITION_KEY},
               {_schema.COL_ITEM_PARENT_RUN_ID})
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                checkpoint.run_id,
                checkpoint.agent,
                checkpoint.status,
                checkpoint.created_at,
                checkpoint.updated_at,
                run_id,
                checkpoint.item_id,
                _encode_snapshot(checkpoint.input_snapshot),
                checkpoint.agent_version,
                _encode_metadata(checkpoint.metadata),
                checkpoint.partition_key,
                checkpoint.parent_run_id,
            ),
        )
        self._conn.commit()

    # --- Run entity (invocations) ---

    def create_run(
        self,
        run_id: str,
        agent: str,
        total_items: int = 0,
        *,
        concurrency_cap: int | None = None,
        replayed_from: str | None = None,
    ) -> None:
        """Create an explicit run row (one invocation). Items link via
        ``RunCheckpoint.invocation_id``.

        ``total_items=0`` creates an OPEN run: the caller doesn't know the
        item count up front (``map()``/``iter()`` over a generator). Open
        runs are exempt from the terminal-status rollup until
        :meth:`finalize_run` seals them with the real count.

        (Renamed from the pre-v12 ``create_batch``, which had no callers.)
        """
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            f"""INSERT INTO {_schema.TBL_RUNS}
                ({_schema.COL_RUN_ID}, {_schema.COL_RUN_AGENT},
                 {_schema.COL_RUN_STATUS}, {_schema.COL_RUN_TOTAL_ITEMS},
                 {_schema.COL_RUN_CONCURRENCY_CAP},
                 {_schema.COL_RUN_CREATED_AT},
                 {_schema.COL_RUN_REPLAYED_FROM})
                VALUES (?, ?, 'running', ?, ?, ?, ?)""",
            (run_id, agent, total_items, concurrency_cap, now, replayed_from),
        )
        self._conn.commit()

    def finalize_run(self, run_id: str) -> None:
        """Seal an OPEN run: set ``total_items`` to the actual item count and
        apply the terminal-status rollup if every item has already resolved.

        Called by ``map()``/``iter()`` when the iteration finishes (normally
        or via unwind). Idempotent; a run whose items are still in flight
        keeps status 'running' and rolls up on the last item's set_status.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                f"""UPDATE {_schema.TBL_RUNS}
                    SET {_schema.COL_RUN_TOTAL_ITEMS} = (
                        SELECT COUNT(*) FROM {_schema.TBL_ITEMS}
                        WHERE {_schema.COL_ITEM_RUN_ID} = ?
                    )
                    WHERE {_schema.COL_RUN_ID} = ?""",
                (run_id, run_id),
            )
            self._conn.execute(
                f"""UPDATE {_schema.TBL_RUNS}
                    SET {_schema.COL_RUN_STATUS} = CASE
                            WHEN {_schema.COL_RUN_FAILED} = 0 THEN 'completed'
                            WHEN {_schema.COL_RUN_COMPLETED} = 0 THEN 'failed'
                            ELSE 'partial'
                        END,
                        {_schema.COL_RUN_COMPLETED_AT} = ?
                    WHERE {_schema.COL_RUN_ID} = ?
                      AND {_schema.COL_RUN_COMPLETED_AT} IS NULL
                      AND {_schema.COL_RUN_TOTAL_ITEMS} > 0
                      AND ({_schema.COL_RUN_COMPLETED} + {_schema.COL_RUN_FAILED})
                          >= {_schema.COL_RUN_TOTAL_ITEMS}""",
                (now, run_id),
            )
            # A sealed zero-item run (map() over an empty iterable) has
            # nothing left to wait for — close it out as completed rather
            # than leaving it 'running' forever.
            self._conn.execute(
                f"""UPDATE {_schema.TBL_RUNS}
                    SET {_schema.COL_RUN_STATUS} = 'completed',
                        {_schema.COL_RUN_COMPLETED_AT} = ?
                    WHERE {_schema.COL_RUN_ID} = ?
                      AND {_schema.COL_RUN_TOTAL_ITEMS} = 0
                      AND {_schema.COL_RUN_COMPLETED_AT} IS NULL""",
                (now, run_id),
            )
            _promote_partial_if_drained(self._conn, run_id, now)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Read one run (invocation) row as a plain dict, or None."""
        row = self._conn.execute(
            f"SELECT * FROM {_schema.TBL_RUNS} WHERE {_schema.COL_RUN_ID} = ?",
            (run_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    # --- Dead letter queue (v6) ---

    def mark_dlq_disposition(
        self,
        run_id: str,
        disposition: str,
        *,
        replayed_from: str | None = None,
    ) -> None:
        """Record an operator's triage decision for a failed item.

        ``disposition`` must be one of the constants in ``_schema`` —
        ``DLQ_REPLAYED``, ``DLQ_SKIPPED``, ``DLQ_ACKNOWLEDGED``. A replay
        transitions the *old* item to ``DLQ_REPLAYED`` and should be paired
        with a freshly created item carrying ``replayed_from=<old id>``
        so the chain is discoverable.

        Write is a no-op if the item is already resolved — double-clicks
        on the dashboard should not silently overwrite disposition.
        """
        if disposition not in (
            _schema.DLQ_REPLAYED,
            _schema.DLQ_SKIPPED,
            _schema.DLQ_ACKNOWLEDGED,
        ):
            raise ValueError(
                f"disposition must be one of replayed/skipped/acknowledged; got {disposition!r}"
            )
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            parent_row = self._conn.execute(
                f"SELECT {_schema.COL_ITEM_RUN_ID} AS run_id FROM {_schema.TBL_ITEMS} "
                f"WHERE {_schema.COL_ITEM_ID} = ?",
                (run_id,),
            ).fetchone()
            self._conn.execute(
                f"""UPDATE {_schema.TBL_ITEMS}
                    SET {_schema.COL_ITEM_DLQ_DISPOSITION} = ?,
                        {_schema.COL_ITEM_DLQ_RESOLVED_AT} = ?,
                        {_schema.COL_ITEM_REPLAYED_FROM} = COALESCE(?, {_schema.COL_ITEM_REPLAYED_FROM}),
                        updated_at = ?
                    WHERE {_schema.COL_ITEM_ID} = ?
                      AND status = 'failed'
                      AND {_schema.COL_ITEM_DLQ_DISPOSITION} IS NULL""",
                (disposition, now, replayed_from, now, run_id),
            )
            if parent_row is not None and parent_row["run_id"] is not None:
                _promote_partial_if_drained(
                    self._conn, parent_row["run_id"], now
                )

    def set_item_replayed_from(self, item_id: str, source_item_id: str) -> None:
        """Link a freshly-created item to the item it re-drives (slice replay)."""
        with self._conn:
            self._conn.execute(
                f"""UPDATE {_schema.TBL_ITEMS}
                    SET {_schema.COL_ITEM_REPLAYED_FROM} = ?
                    WHERE {_schema.COL_ITEM_ID} = ?""",
                (source_item_id, item_id),
            )

    def close(self) -> None:
        self._conn.close()
