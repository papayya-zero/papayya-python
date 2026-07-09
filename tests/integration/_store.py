"""SharedSQLiteStore — test wrapper over papayya.durable.SQLiteStore.

Backs an integration-test fixture with a real SQLite file at tmp_path.
"In memory" in spirit (lifetime = test); on disk in implementation
(workers in a subprocess need to read/write the same store as the test
process, and SQLite at a tmp file does that without IPC).

Adds the assertion helpers the acceptance test depends on.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from papayya.durable.sqlite_store import SQLiteStore
from papayya.durable.types import RunCheckpoint


class SharedSQLiteStore:
    """Thin facade over SQLiteStore with item-keyed lookup helpers.

    The underlying SQLiteStore is the same one the SDK already ships;
    we just add convenience accessors to keep the acceptance test
    readable and decoupled from raw SQL.

    The assertion helpers (``completed_run_count``, ``run_for_item``)
    open a fresh sqlite3 connection per query rather than reusing
    ``_store._conn``. The worker subprocess writes through its own
    short-lived connections; the long-lived test connection can hold
    on to a stale read snapshot under WAL when the worker pipelines
    many small commits, hiding the last few rows from a SELECT issued
    immediately after ``wait_until_drained``. A fresh connection per
    assertion sees the latest committed WAL state every time.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._store = SQLiteStore(db_path)

    @property
    def store(self) -> SQLiteStore:
        return self._store

    # --- assertion helpers -------------------------------------------- #

    def _fresh_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def completed_run_count(self) -> int:
        conn = self._fresh_conn()
        try:
            cur = conn.execute(
                "SELECT COUNT(*) AS n FROM items WHERE status = 'completed'"
            )
            return cur.fetchone()["n"]
        finally:
            conn.close()

    def run_for_item(self, item_id: str) -> RunCheckpoint | None:
        """Most recent completed run carrying this item_id (denormalized)."""
        conn = self._fresh_conn()
        try:
            row = conn.execute(
                "SELECT id AS run_id FROM items WHERE item_id = ? AND status = 'completed' "
                "ORDER BY updated_at DESC LIMIT 1",
                (item_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return self._store.load(row["run_id"])

    def all_runs(self) -> list[RunCheckpoint]:
        conn = self._fresh_conn()
        try:
            rows = conn.execute(
                "SELECT id AS run_id FROM items ORDER BY created_at"
            ).fetchall()
        finally:
            conn.close()
        return [self._store.load(r["run_id"]) for r in rows if r is not None]
