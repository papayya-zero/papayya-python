"""v9 multi-tenancy metadata convention.

Covers store round-trip + PapayyaClient strict-when-declared enforcement
against papayya.yaml. The yaml's tenant_key declaration is the contract:
when set, every run() call must include the named key in metadata.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from papayya.durable.client import PapayyaClient, PapayyaClientConfig
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


@pytest.fixture
def papayya_yaml_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run tests inside a temp cwd so PapayyaClient picks up our yaml."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write_yaml(dir: Path, body: str) -> None:
    (dir / "papayya.yaml").write_text(body)


class TestPapayyaClientStrictWhenDeclared:
    """PapayyaClient.run() reads papayya.yaml at first call and enforces
    that every run includes the declared tenant_key in its metadata."""

    def test_no_yaml_means_tenant_key_optional(
        self, tmp_path: Path, papayya_yaml_dir: Path
    ) -> None:
        # No papayya.yaml exists — runs without metadata are accepted.
        store = SQLiteStore(str(tmp_path / "local.db"))
        client = PapayyaClient(PapayyaClientConfig(store=store))
        run = client.run(agent="enrich")
        assert run is not None

    def test_yaml_without_tenant_key_means_optional(
        self, tmp_path: Path, papayya_yaml_dir: Path
    ) -> None:
        _write_yaml(papayya_yaml_dir, "version: 1\n")
        store = SQLiteStore(str(tmp_path / "local.db"))
        client = PapayyaClient(PapayyaClientConfig(store=store))
        run = client.run(agent="enrich", metadata={"any": "thing"})
        assert run is not None

    def test_declared_key_present_extracts_value(
        self, tmp_path: Path, papayya_yaml_dir: Path
    ) -> None:
        _write_yaml(
            papayya_yaml_dir, "version: 1\ntenant_key: organization_id\n"
        )
        store = SQLiteStore(str(tmp_path / "local.db"))
        client = PapayyaClient(PapayyaClientConfig(store=store))
        run = client.run(
            agent="enrich",
            run_id="r1",
            metadata={"organization_id": "org_42", "user_id": "u_7"},
        )
        run.step("enrich", lambda: {"out": 1})()
        # Round-trip via the store.
        loaded = store.load("r1")
        assert loaded is not None
        assert loaded.tenant_key == "org_42"
        assert loaded.metadata == {"organization_id": "org_42", "user_id": "u_7"}
        (task,) = loaded.tasks
        assert task.tenant_key == "org_42"

    def test_declared_key_missing_metadata_raises(
        self, tmp_path: Path, papayya_yaml_dir: Path
    ) -> None:
        _write_yaml(
            papayya_yaml_dir, "version: 1\ntenant_key: organization_id\n"
        )
        store = SQLiteStore(str(tmp_path / "local.db"))
        client = PapayyaClient(PapayyaClientConfig(store=store))
        with pytest.raises(ValueError, match="organization_id"):
            client.run(agent="enrich")

    def test_declared_key_metadata_missing_key_raises(
        self, tmp_path: Path, papayya_yaml_dir: Path
    ) -> None:
        _write_yaml(
            papayya_yaml_dir, "version: 1\ntenant_key: organization_id\n"
        )
        store = SQLiteStore(str(tmp_path / "local.db"))
        client = PapayyaClient(PapayyaClientConfig(store=store))
        with pytest.raises(ValueError, match="organization_id"):
            client.run(agent="enrich", metadata={"unrelated": "value"})

    def test_declared_key_empty_value_raises(
        self, tmp_path: Path, papayya_yaml_dir: Path
    ) -> None:
        _write_yaml(
            papayya_yaml_dir, "version: 1\ntenant_key: organization_id\n"
        )
        store = SQLiteStore(str(tmp_path / "local.db"))
        client = PapayyaClient(PapayyaClientConfig(store=store))
        with pytest.raises(ValueError, match="non-empty"):
            client.run(
                agent="enrich",
                metadata={"organization_id": ""},
            )

    def test_declared_key_non_string_raises(
        self, tmp_path: Path, papayya_yaml_dir: Path
    ) -> None:
        _write_yaml(
            papayya_yaml_dir, "version: 1\ntenant_key: organization_id\n"
        )
        store = SQLiteStore(str(tmp_path / "local.db"))
        client = PapayyaClient(PapayyaClientConfig(store=store))
        with pytest.raises(ValueError, match="non-empty"):
            client.run(
                agent="enrich",
                metadata={"organization_id": 42},
            )
