"""SharedSQLiteStore — test wrapper over papayya.durable.SQLiteStore.

Backs an integration-test fixture with a real SQLite file at tmp_path.
"In memory" in spirit (lifetime = test); on disk in implementation
(workers in a subprocess need to read/write the same store as the test
process, and SQLite at a tmp file does that without IPC).

Adds the assertion helpers the acceptance test depends on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from papayya.durable.sqlite_store import SQLiteStore
from papayya.durable.types import RunCheckpoint


class SharedSQLiteStore:
    """Thin facade over SQLiteStore with item-keyed lookup helpers.

    The underlying SQLiteStore is the same one the SDK already ships;
    we just add convenience accessors to keep the acceptance test
    readable and decoupled from raw SQL.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._store = SQLiteStore(db_path)

    @property
    def store(self) -> SQLiteStore:
        return self._store

    # --- assertion helpers -------------------------------------------- #

    def completed_run_count(self) -> int:
        cur = self._store._conn.execute(
            "SELECT COUNT(*) AS n FROM runs WHERE status = 'completed'"
        )
        return cur.fetchone()["n"]

    def run_for_item(self, item_id: str) -> RunCheckpoint | None:
        """Most recent completed run carrying this item_id (denormalized)."""
        row = self._store._conn.execute(
            "SELECT run_id FROM runs WHERE item_id = ? AND status = 'completed' "
            "ORDER BY updated_at DESC LIMIT 1",
            (item_id,),
        ).fetchone()
        if row is None:
            return None
        return self._store.load(row["run_id"])

    def all_runs(self) -> list[RunCheckpoint]:
        rows = self._store._conn.execute(
            "SELECT run_id FROM runs ORDER BY created_at"
        ).fetchall()
        return [self._store.load(r["run_id"]) for r in rows if r is not None]
