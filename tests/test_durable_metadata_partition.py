"""partition-key metadata convention (code-first).

Covers the metadata codec + store round-trip of the ``partition_key`` value
supplied to ``item()`` / ``map()``. Attribution is opt-in: a run carries a
partition_key only when the caller passes one — there is no declarative
(papayya.yaml) enforcement.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from papayya.durable.sqlite_store import SQLiteStore, _decode_metadata, _encode_metadata
from papayya.durable.types import RunCheckpoint, TaskEntry


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "local.db"


class TestMetadataCodec:
    def test_none_round_trips(self) -> None:
        assert _decode_metadata(_encode_metadata(None)) is None

    def test_dict_round_trips(self) -> None:
        encoded = _encode_metadata({"organization_id": "org_42", "user_id": "u_7"})
        assert _decode_metadata(encoded) == {
            "organization_id": "org_42",
            "user_id": "u_7",
        }

    def test_decode_invalid_returns_none(self) -> None:
        assert _decode_metadata("not-json") is None

    def test_decode_non_dict_returns_none(self) -> None:
        # The convention is dict-only — a list or scalar in the column
        # means the writer broke the invariant; defensively map to None.
        assert _decode_metadata("[1, 2, 3]") is None
        assert _decode_metadata("42") is None


class TestSQLiteStoreRoundTrip:
    def test_create_persists_metadata_and_partition_key(self, tmp_db: Path) -> None:
        store = SQLiteStore(str(tmp_db))
        checkpoint = RunCheckpoint(
            run_id="r1",
            agent="enrich",
            tasks=[],
            status="running",
            created_at="2026-05-01T00:00:00+00:00",
            updated_at="2026-05-01T00:00:00+00:00",
            metadata={"organization_id": "org_42", "user_id": "u_7"},
            partition_key="org_42",
        )
        store.create(checkpoint)

        loaded = store.load("r1")
        assert loaded is not None
        assert loaded.metadata == {"organization_id": "org_42", "user_id": "u_7"}
        assert loaded.partition_key == "org_42"

    def test_save_task_persists_metadata_and_partition_key(self, tmp_db: Path) -> None:
        store = SQLiteStore(str(tmp_db))
        store.create(
            RunCheckpoint(
                run_id="r1",
                agent="enrich",
                tasks=[],
                status="running",
                created_at="2026-05-01T00:00:00+00:00",
                updated_at="2026-05-01T00:00:00+00:00",
            )
        )
        entry = TaskEntry(
            label="enrich",
            result={"out": 1},
            duration_ms=42,
            completed_at="2026-05-01T00:00:01+00:00",
            metadata={"organization_id": "org_42"},
            partition_key="org_42",
        )
        store.save_task("r1", entry)

        loaded = store.load("r1")
        assert loaded is not None
        (task,) = loaded.tasks
        assert task.metadata == {"organization_id": "org_42"}
        assert task.partition_key == "org_42"

    def test_null_metadata_stays_null(self, tmp_db: Path) -> None:
        """Backward-compat: existing callers that don't pass metadata must
        end up with None in both columns, not an empty dict."""
        store = SQLiteStore(str(tmp_db))
        store.create(
            RunCheckpoint(
                run_id="r1",
                agent="enrich",
                tasks=[],
                status="running",
                created_at="2026-05-01T00:00:00+00:00",
                updated_at="2026-05-01T00:00:00+00:00",
            )
        )
        entry = TaskEntry(
            label="enrich",
            result=None,
            duration_ms=0,
            completed_at="2026-05-01T00:00:01+00:00",
        )
        store.save_task("r1", entry)

        loaded = store.load("r1")
        assert loaded is not None
        assert loaded.metadata is None
        assert loaded.partition_key is None
        (task,) = loaded.tasks
        assert task.metadata is None
        assert task.partition_key is None


class TestCodeFirstPartitionKey:
    """Attribution is opt-in via ``item(partition_key=...)`` — a non-empty
    string is used as-is; omitting it (or ``None``) records unattributed."""

    def test_explicit_partition_key_persists(self, tmp_path: Path) -> None:
        from papayya import Papayya

        store = SQLiteStore(str(tmp_path / "local.db"))
        run = Papayya(store=store).item(
            agent="enrich", run_id="r1", partition_key="org_42"
        )
        run.step("enrich", lambda: {"out": 1})()

        loaded = store.load("r1")
        assert loaded is not None
        assert loaded.partition_key == "org_42"
        (task,) = loaded.tasks
        assert task.partition_key == "org_42"

    def test_omitted_partition_key_is_unattributed(self, tmp_path: Path) -> None:
        from papayya import Papayya

        store = SQLiteStore(str(tmp_path / "local.db"))
        run = Papayya(store=store).item(agent="enrich", run_id="r1")
        run.step("enrich", lambda: {"out": 1})()

        loaded = store.load("r1")
        assert loaded is not None
        assert loaded.partition_key is None

    def test_empty_partition_key_raises(self, tmp_path: Path) -> None:
        from papayya import Papayya

        store = SQLiteStore(str(tmp_path / "local.db"))
        with pytest.raises(ValueError, match="non-empty"):
            Papayya(store=store).item(agent="enrich", partition_key="")

    def test_non_string_partition_key_raises(self, tmp_path: Path) -> None:
        from papayya import Papayya

        store = SQLiteStore(str(tmp_path / "local.db"))
        with pytest.raises(ValueError, match="non-empty"):
            Papayya(store=store).item(agent="enrich", partition_key=42)
