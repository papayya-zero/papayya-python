"""v9 multi-tenancy metadata convention — SQLiteStore round-trip.

End-to-end strict-when-declared enforcement and papayya.yaml integration
land alongside the PapayyaClient changes in a later commit. This file
covers persistence: the store reads back what it wrote, and the JSON
encoder used for snapshots round-trips dict metadata cleanly.
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
    def test_create_persists_metadata_and_tenant_key(self, tmp_db: Path) -> None:
        store = SQLiteStore(str(tmp_db))
        checkpoint = RunCheckpoint(
            run_id="r1",
            agent="enrich",
            tasks=[],
            status="running",
            created_at="2026-05-01T00:00:00+00:00",
            updated_at="2026-05-01T00:00:00+00:00",
            metadata={"organization_id": "org_42", "user_id": "u_7"},
            tenant_key="org_42",
        )
        store.create(checkpoint)

        loaded = store.load("r1")
        assert loaded is not None
        assert loaded.metadata == {"organization_id": "org_42", "user_id": "u_7"}
        assert loaded.tenant_key == "org_42"

    def test_save_task_persists_metadata_and_tenant_key(self, tmp_db: Path) -> None:
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
            tenant_key="org_42",
        )
        store.save_task("r1", entry)

        loaded = store.load("r1")
        assert loaded is not None
        (task,) = loaded.tasks
        assert task.metadata == {"organization_id": "org_42"}
        assert task.tenant_key == "org_42"

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
        assert loaded.tenant_key is None
        (task,) = loaded.tasks
        assert task.metadata is None
        assert task.tenant_key is None
