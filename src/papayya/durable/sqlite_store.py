"""SQLite-backed checkpoint store with step-level observability.

Implements CheckpointStore for durable execution compatibility.
Additionally supports recording individual LLM steps for the
local development dashboard.

The schema is a cross-language contract — future SDKs (TypeScript, Go)
will write to the same tables and the dashboard reads from them unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import _errors, _schema
from .types import RunCheckpoint, TaskEntry


# Flag gate for all Slice-2 capture logic. Setting to "false" restores the
# Slice-1 schema-only behaviour: new columns stay null, no implicit batches,
# no aggregate bumps. Intended as a safety hatch if capture logic produces
# bad data in the wild; not something users should normally touch.
_CAPTURE_V2_ENABLED = os.environ.get("PAPAYYA_LOCAL_CAPTURE_V2", "true").lower() != "false"


def _capture_enabled() -> bool:
    # Read on every call so tests can monkeypatch os.environ and get an
    # accurate answer without re-importing.
    return os.environ.get("PAPAYYA_LOCAL_CAPTURE_V2", "true").lower() != "false"


def _single_batch_id(run_id: str) -> str:
    """Sentinel batch ID for implicit batches-of-1 around single runs."""
    return f"single-{run_id}"


def _compute_input_hash(
    task_label: str | None,
    tool_calls: list[dict[str, Any]] | None,
) -> str | None:
    """BLAKE2b-64 hex of a stable identity for this step's input.

    Uses ``task_label + first_tool_call_JSON`` — stable across retries of
    the same logical call, so ``(error_code, input_hash)`` buckets cleanly
    for failure clustering.

    Deliberately does NOT include ``response_text``: the model's output
    varies even for identical inputs, and we want identical-input calls
    to hash the same.
    """
    if task_label is None and not tool_calls:
        return None
    label = task_label or ""
    first_tool = ""
    if tool_calls:
        try:
            first_tool = json.dumps(tool_calls[0], sort_keys=True)
        except (TypeError, ValueError):
            first_tool = str(tool_calls[0])
    material = (label + first_tool)[:200].encode("utf-8", errors="replace")
    return hashlib.blake2b(material, digest_size=8).hexdigest()


def _extract_tool_name(tool_calls: list[dict[str, Any]] | None) -> str | None:
    """Pull the first tool call's name out of the permissive dict shape.

    ``tool_calls`` is declared as ``list[dict[str, Any]]`` — the shape is
    whatever the engine interceptor emits, which is itself whatever the
    provider SDK returned. Tolerate missing keys by returning None.
    """
    if not tool_calls:
        return None
    first = tool_calls[0]
    if not isinstance(first, dict):
        return None
    name = first.get("name")
    if not isinstance(name, str):
        return None
    return name

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
    try:
        encoded = json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "snapshot must be JSON-encodable. Pass a dict/list/primitive, "
            "or store a reference (e.g. an S3 key) instead of the object. "
            f"Original error: {exc}"
        ) from exc
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

# v1 base schema. Kept verbatim with the original (pre-v4) columns so the
# migration chain still applies cleanly for long-dormant DBs upgrading from
# v1/v2/v3. The v3→v4 migration drops the budget/cost/token columns below;
# fresh DBs pay one create-then-drop round at init, which is fine.
_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS _meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT OR IGNORE INTO _meta (key, value) VALUES ('schema_version', '1');

CREATE TABLE IF NOT EXISTS runs (
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

CREATE TABLE IF NOT EXISTS tasks (
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
CREATE INDEX IF NOT EXISTS idx_tasks_run_id ON tasks(run_id);

CREATE TABLE IF NOT EXISTS steps (
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
CREATE INDEX IF NOT EXISTS idx_steps_run_id ON steps(run_id);
"""


# v2 additions: batch entity + denormalized columns for clustering and search.
# Every ALTER is idempotent-by-detection (see `_migrate`); do not add ALTERs
# directly to _SCHEMA_SQL — they'd fail on re-open once the columns exist.
#
# NOTE: This creates the batches table with the legacy aggregate_cost_usd /
# budget_limit_usd columns. v3→v4 drops them; kept here verbatim so DBs that
# stopped at v2 still migrate cleanly forward through v3 and then v4.
_V2_CREATE_BATCHES = f"""\
CREATE TABLE IF NOT EXISTS {_schema.TBL_BATCHES} (
    {_schema.COL_BATCH_ID}         TEXT PRIMARY KEY,
    {_schema.COL_BATCH_AGENT}      TEXT NOT NULL,
    {_schema.COL_BATCH_STATUS}     TEXT NOT NULL DEFAULT 'queued',
    {_schema.COL_BATCH_TOTAL_ITEMS} INTEGER NOT NULL,
    {_schema.COL_BATCH_COMPLETED}  INTEGER NOT NULL DEFAULT 0,
    {_schema.COL_BATCH_FAILED}     INTEGER NOT NULL DEFAULT 0,
    aggregate_cost_usd             REAL NOT NULL DEFAULT 0.0,
    budget_limit_usd               REAL,
    {_schema.COL_BATCH_CONCURRENCY_CAP} INTEGER,
    {_schema.COL_BATCH_CREATED_AT} TEXT NOT NULL,
    {_schema.COL_BATCH_COMPLETED_AT} TEXT
);
"""

# (table, column, type_decl) — applied via ALTER only if the column is absent.
# input_hash is BLAKE2b-64 over `task_label + first_tool_call_input` (see
# Slice 2 capture logic). 64 bits is fine for local-scale clustering; a
# hosted aggregation across projects may later want SHA-256.
_V2_ADD_COLUMNS: list[tuple[str, str, str]] = [
    (_schema.TBL_RUNS, _schema.COL_RUN_BATCH_ID, "TEXT"),
    (_schema.TBL_RUNS, _schema.COL_RUN_ERROR_CODE, "TEXT"),
    (_schema.TBL_STEPS, _schema.COL_STEP_TOOL_NAME, "TEXT"),
    (_schema.TBL_STEPS, _schema.COL_STEP_ERROR_CODE, "TEXT"),
    (_schema.TBL_STEPS, _schema.COL_STEP_ERROR_CATEGORY, "TEXT"),
    (_schema.TBL_STEPS, _schema.COL_STEP_INPUT_HASH, "TEXT"),
]

_V2_INDEXES = [
    f"CREATE INDEX IF NOT EXISTS {_schema.IDX_STEPS_TOOL} "
    f"ON {_schema.TBL_STEPS}({_schema.COL_STEP_TOOL_NAME});",
    f"CREATE INDEX IF NOT EXISTS {_schema.IDX_STEPS_ERROR} "
    f"ON {_schema.TBL_STEPS}({_schema.COL_STEP_ERROR_CODE});",
    f"CREATE INDEX IF NOT EXISTS {_schema.IDX_RUNS_BATCH} "
    f"ON {_schema.TBL_RUNS}({_schema.COL_RUN_BATCH_ID});",
]


# Slice 6 (v3): per-object state snapshots. Columns live on `tasks` (the
# row written by run.step()), with item_id denormalized onto `runs` so the
# dashboard can list items inside a batch without joining through tasks.
# Snapshot columns are TEXT + JSON-encoded at the Python layer (see
# `_encode_snapshot` below).
_V3_ADD_COLUMNS: list[tuple[str, str, str]] = [
    (_schema.TBL_TASKS, _schema.COL_TASK_ITEM_ID, "TEXT"),
    (_schema.TBL_TASKS, _schema.COL_TASK_INPUT_SNAPSHOT, "TEXT"),
    (_schema.TBL_TASKS, _schema.COL_TASK_OUTPUT_SNAPSHOT, "TEXT"),
    (_schema.TBL_RUNS, _schema.COL_RUN_ITEM_ID, "TEXT"),
]

_V3_INDEXES = [
    f"CREATE INDEX IF NOT EXISTS {_schema.IDX_TASKS_ITEM} "
    f"ON {_schema.TBL_TASKS}({_schema.COL_TASK_ITEM_ID});",
    f"CREATE INDEX IF NOT EXISTS {_schema.IDX_RUNS_ITEM} "
    f"ON {_schema.TBL_RUNS}({_schema.COL_RUN_ITEM_ID});",
]


# v4: drop budget/cost/token columns. Budget and cost are cloud-only
# concepts — enforced by the runtime shim at dispatch time, reconciled
# against provider usage, and fenced at step-insert on the control plane.
# The SDK no longer tracks cost locally, so these columns became dead.
# (table, column) pairs to drop if present.
_V4_DROP_COLUMNS: list[tuple[str, str]] = [
    (_schema.TBL_RUNS, "budget_limit_usd"),
    (_schema.TBL_RUNS, "budget_consumed_usd"),
    (_schema.TBL_RUNS, "total_input_tokens"),
    (_schema.TBL_RUNS, "total_output_tokens"),
    (_schema.TBL_RUNS, "budget_input_tokens"),
    (_schema.TBL_RUNS, "budget_output_tokens"),
    (_schema.TBL_TASKS, "cost_usd"),
    (_schema.TBL_TASKS, "input_tokens"),
    (_schema.TBL_TASKS, "output_tokens"),
    (_schema.TBL_STEPS, "input_tokens"),
    (_schema.TBL_STEPS, "output_tokens"),
    (_schema.TBL_STEPS, "cost_usd"),
    (_schema.TBL_BATCHES, "aggregate_cost_usd"),
    (_schema.TBL_BATCHES, "budget_limit_usd"),
]


# v5: BYOF observability columns on tasks. All nullable — a non-LLM step
# (kind is None) writes nulls; an LLM step whose provider shape wasn't
# recognised writes nulls for the token/model/stop_reason fields and
# "unknown" for provider_shape. error_category fills only on classified
# provider exceptions.
_V5_ADD_COLUMNS: list[tuple[str, str, str]] = [
    (_schema.TBL_TASKS, _schema.COL_TASK_KIND, "TEXT"),
    (_schema.TBL_TASKS, _schema.COL_TASK_LLM_PROMPT_TOKENS, "INTEGER"),
    (_schema.TBL_TASKS, _schema.COL_TASK_LLM_COMPLETION_TOKENS, "INTEGER"),
    (_schema.TBL_TASKS, _schema.COL_TASK_LLM_TOTAL_TOKENS, "INTEGER"),
    (_schema.TBL_TASKS, _schema.COL_TASK_LLM_MODEL, "TEXT"),
    (_schema.TBL_TASKS, _schema.COL_TASK_LLM_STOP_REASON, "TEXT"),
    (_schema.TBL_TASKS, _schema.COL_TASK_LLM_PROVIDER_SHAPE, "TEXT"),
    (_schema.TBL_TASKS, _schema.COL_TASK_ERROR_CATEGORY, "TEXT"),
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

    The caller has already opened the connection (and so flipped the
    journal_mode=WAL header) by the time we run, so the backup is not
    byte-identical to the pre-open file. Row data is preserved exactly;
    only the SQLite header differs. SDK init is synchronous, so no
    concurrent writer can have touched the DB between open and backup.
    """
    if not db_path.exists():
        return None
    backup = db_path.with_suffix(db_path.suffix + f".backup-v{from_version}")
    if backup.exists():
        return backup
    shutil.copy2(db_path, backup)
    return backup


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
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA_SQL)
        _migrate(conn, file)
    finally:
        conn.close()


def _set_schema_version(conn: sqlite3.Connection, version: str) -> None:
    conn.execute(
        "UPDATE _meta SET value = ? WHERE key = 'schema_version'",
        (version,),
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


def _migrate(conn: sqlite3.Connection, db_path: Path) -> None:
    """Forward-only migrations. Idempotent: safe to call on any schema version.

    Each migration runs in a single transaction. A mid-migration crash leaves
    the DB at the prior version with no half-applied ALTERs. When more than
    one version gap separates the DB from the SDK, migrations chain in order
    (e.g. v1 → v2 → v3 → v4 → v5) so long-dormant local DBs catch up cleanly.
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
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA_SQL)
        _migrate(self._conn, db_file)

    # --- CheckpointStore protocol ---

    def load(self, run_id: str) -> RunCheckpoint | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None

        task_rows = self._conn.execute(
            "SELECT * FROM tasks WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()

        tasks = [
            TaskEntry(
                label=t["label"],
                result=json.loads(t["result"]) if t["result"] is not None else None,
                duration_ms=t["duration_ms"],
                completed_at=t["completed_at"],
                item_id=t[_schema.COL_TASK_ITEM_ID],
                input_snapshot=_decode_snapshot(t[_schema.COL_TASK_INPUT_SNAPSHOT]),
                output_snapshot=_decode_snapshot(t[_schema.COL_TASK_OUTPUT_SNAPSHOT]),
                kind=t[_schema.COL_TASK_KIND],
                llm_prompt_tokens=t[_schema.COL_TASK_LLM_PROMPT_TOKENS],
                llm_completion_tokens=t[_schema.COL_TASK_LLM_COMPLETION_TOKENS],
                llm_total_tokens=t[_schema.COL_TASK_LLM_TOTAL_TOKENS],
                llm_model=t[_schema.COL_TASK_LLM_MODEL],
                llm_stop_reason=t[_schema.COL_TASK_LLM_STOP_REASON],
                llm_provider_shape=t[_schema.COL_TASK_LLM_PROVIDER_SHAPE],
                error_category=t[_schema.COL_TASK_ERROR_CATEGORY],
            )
            for t in task_rows
        ]

        return RunCheckpoint(
            run_id=row["run_id"],
            agent=row["agent"],
            tasks=tasks,
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            item_id=row[_schema.COL_RUN_ITEM_ID],
        )

    def save_task(self, run_id: str, entry: TaskEntry) -> None:
        now = datetime.now(timezone.utc).isoformat()
        input_snapshot_json = _encode_snapshot(entry.input_snapshot)
        output_snapshot_json = _encode_snapshot(entry.output_snapshot)
        with self._conn:
            self._conn.execute(
                f"""INSERT INTO tasks (run_id, label, result, duration_ms, completed_at,
                   {_schema.COL_TASK_ITEM_ID},
                   {_schema.COL_TASK_INPUT_SNAPSHOT},
                   {_schema.COL_TASK_OUTPUT_SNAPSHOT},
                   {_schema.COL_TASK_KIND},
                   {_schema.COL_TASK_LLM_PROMPT_TOKENS},
                   {_schema.COL_TASK_LLM_COMPLETION_TOKENS},
                   {_schema.COL_TASK_LLM_TOTAL_TOKENS},
                   {_schema.COL_TASK_LLM_MODEL},
                   {_schema.COL_TASK_LLM_STOP_REASON},
                   {_schema.COL_TASK_LLM_PROVIDER_SHAPE},
                   {_schema.COL_TASK_ERROR_CATEGORY})
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    entry.label,
                    json.dumps(entry.result),
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
                ),
            )
            self._conn.execute(
                "UPDATE runs SET updated_at = ? WHERE run_id = ?",
                (now, run_id),
            )
            # Denormalize item_id onto the run on first-writer-wins basis.
            # Later steps with a different item_id don't overwrite — the
            # run-level item_id represents the primary record flowing
            # through this run, not a mutable state field.
            if entry.item_id is not None:
                self._conn.execute(
                    f"""UPDATE runs SET {_schema.COL_RUN_ITEM_ID} = ?
                       WHERE run_id = ? AND {_schema.COL_RUN_ITEM_ID} IS NULL""",
                    (entry.item_id, run_id),
                )

    def set_status(self, run_id: str, status: str, output: Any = None) -> None:
        """Transition a run's status, and roll up terminal counts to the batch."""
        now = datetime.now(timezone.utc).isoformat()
        # Capture prior status before we overwrite it — used below to decide
        # whether this transition bumps a batch counter.
        row = self._conn.execute(
            f"SELECT status, {_schema.COL_RUN_BATCH_ID} FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()

        with self._conn:
            self._conn.execute(
                "UPDATE runs SET status = ?, output = ?, updated_at = ? WHERE run_id = ?",
                (status, json.dumps(output) if output is not None else None, now, run_id),
            )
            if (
                _capture_enabled()
                and row is not None
                and row["batch_id"] is not None
                and row["status"] not in ("completed", "failed")
                and status in ("completed", "failed")
            ):
                counter = (
                    _schema.COL_BATCH_COMPLETED
                    if status == "completed"
                    else _schema.COL_BATCH_FAILED
                )
                self._conn.execute(
                    f"""UPDATE {_schema.TBL_BATCHES}
                        SET {counter} = {counter} + 1
                        WHERE {_schema.COL_BATCH_ID} = ?""",
                    (row["batch_id"],),
                )
                # Roll the batch to its terminal status once every item has
                # resolved. Ternary outcome: zero failures → 'completed';
                # zero successes → 'failed'; mixed → 'partial'. The DLQ
                # surface acts on partial-terminal batches to re-drive the
                # failed items; once all dead letters are replayed or
                # skipped, a later pass should promote the batch from
                # 'partial' to 'completed'.
                self._conn.execute(
                    f"""UPDATE {_schema.TBL_BATCHES}
                        SET {_schema.COL_BATCH_STATUS} = CASE
                                WHEN {_schema.COL_BATCH_FAILED} = 0 THEN 'completed'
                                WHEN {_schema.COL_BATCH_COMPLETED} = 0 THEN 'failed'
                                ELSE 'partial'
                            END,
                            {_schema.COL_BATCH_COMPLETED_AT} = ?
                        WHERE {_schema.COL_BATCH_ID} = ?
                          AND {_schema.COL_BATCH_COMPLETED_AT} IS NULL
                          AND ({_schema.COL_BATCH_COMPLETED} + {_schema.COL_BATCH_FAILED})
                              >= {_schema.COL_BATCH_TOTAL_ITEMS}""",
                    (now, row["batch_id"]),
                )

    def create(self, checkpoint: RunCheckpoint) -> None:
        """Create a run. Auto-wraps in an implicit batch-of-1 when capture is on.

        The implicit batch ID is ``single-{run_id}`` so the UI can filter
        single-run work in or out cleanly. Batches created explicitly via
        ``create_batch`` have their own IDs and the run is linked via the
        optional ``batch_id`` kwarg path once that lands in the caller.
        """
        batch_id: str | None = None
        if _capture_enabled():
            batch_id = _single_batch_id(checkpoint.run_id)
            self._conn.execute(
                f"""INSERT OR IGNORE INTO {_schema.TBL_BATCHES}
                    ({_schema.COL_BATCH_ID}, {_schema.COL_BATCH_AGENT},
                     {_schema.COL_BATCH_STATUS}, {_schema.COL_BATCH_TOTAL_ITEMS},
                     {_schema.COL_BATCH_CREATED_AT})
                    VALUES (?, ?, 'running', 1, ?)""",
                (batch_id, checkpoint.agent, checkpoint.created_at),
            )

        self._conn.execute(
            f"""INSERT INTO runs (run_id, agent, status, created_at, updated_at,
               batch_id, {_schema.COL_RUN_ITEM_ID})
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                checkpoint.run_id,
                checkpoint.agent,
                checkpoint.status,
                checkpoint.created_at,
                checkpoint.updated_at,
                batch_id,
                checkpoint.item_id,
            ),
        )
        self._conn.commit()

    # --- Batch entity (Slice 2) ---

    def create_batch(
        self,
        batch_id: str,
        agent: str,
        total_items: int,
        *,
        concurrency_cap: int | None = None,
    ) -> None:
        """Create an explicit multi-item batch. Caller links runs via batch_id."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            f"""INSERT INTO {_schema.TBL_BATCHES}
                ({_schema.COL_BATCH_ID}, {_schema.COL_BATCH_AGENT},
                 {_schema.COL_BATCH_STATUS}, {_schema.COL_BATCH_TOTAL_ITEMS},
                 {_schema.COL_BATCH_CONCURRENCY_CAP},
                 {_schema.COL_BATCH_CREATED_AT})
                VALUES (?, ?, 'running', ?, ?, ?)""",
            (batch_id, agent, total_items, concurrency_cap, now),
        )
        self._conn.commit()

    # --- Step-level capture (beyond CheckpointStore) ---

    def record_step(
        self,
        run_id: str,
        *,
        task_label: str | None = None,
        model: str = "unknown",
        duration_ms: int = 0,
        tool_calls: list[dict[str, Any]] | None = None,
        response_text: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Record an individual LLM call step for observability.

        When ``error_message`` is supplied, it is classified via
        ``_errors.classify_error`` into ``(error_code, error_category)`` and
        persisted so the dashboard can cluster and colour failures.
        """
        now = datetime.now(timezone.utc).isoformat()
        step_index = self._conn.execute(
            "SELECT COALESCE(MAX(step_index), -1) + 1 FROM steps WHERE run_id = ?",
            (run_id,),
        ).fetchone()[0]

        capture_on = _capture_enabled()
        tool_name = _extract_tool_name(tool_calls) if capture_on else None
        input_hash = (
            _compute_input_hash(task_label, tool_calls) if capture_on else None
        )
        error_code, error_category = (
            _errors.classify_error(error_message) if capture_on else (None, None)
        )

        self._conn.execute(
            f"""INSERT INTO steps (run_id, task_label, step_index, model,
               duration_ms, tool_calls, response_text, created_at,
               {_schema.COL_STEP_TOOL_NAME},
               {_schema.COL_STEP_ERROR_CODE},
               {_schema.COL_STEP_ERROR_CATEGORY},
               {_schema.COL_STEP_INPUT_HASH})
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                task_label,
                step_index,
                model,
                duration_ms,
                json.dumps(tool_calls) if tool_calls is not None else None,
                response_text,
                now,
                tool_name,
                error_code,
                error_category,
                input_hash,
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
