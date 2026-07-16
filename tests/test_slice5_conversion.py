"""Tests for Slice 5 — conversion plumbing.

Covers:
1. ``/api/tier-recommendation`` endpoint shape and math
2. ``papayya project export`` default excludes ``response_text``
3. ``--include-response-text`` flag round-trips the field
4. ``papayya project import`` validates the shape and counts records
"""

from __future__ import annotations

import json
import socket
import threading
import urllib.request
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import pytest
from click.testing import CliRunner

from papayya.cli import main as cli_main
from papayya.dev.server import DevHandler
from papayya.durable.sqlite_store import SQLiteStore
from papayya.durable.types import RunCheckpoint, TaskEntry

# Plan 37: this file's SUBJECT is a DEACTIVATED local surface (iter/map / local SQLite
# CLI / keyless demo). The code is retained in-repo for self-host / revival, so the
# file is skipped rather than deleted — unskip when the local surface is revived.
import pytest as _pytest
pytestmark = _pytest.mark.skip(reason="Plan 37: local surface deactivated")



def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _checkpoint(run_id: str) -> RunCheckpoint:
    now = datetime.now(timezone.utc).isoformat()
    return RunCheckpoint(
        run_id=run_id, agent="t", tasks=[], status="running",
        created_at=now, updated_at=now,
    )


@pytest.fixture
def populated_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "local.db"
    store = SQLiteStore(str(db_path))
    store.create(_checkpoint("run-1"))
    store.save_task("run-1", TaskEntry(
        label="t", result="some model response", duration_ms=100,
        completed_at=datetime.now(timezone.utc).isoformat(),
        item_id="co_1", input_snapshot={"q": "foo"},
        output_snapshot={"answer": "some model response"},
    ))
    store.set_status("run-1", "completed", output=None)
    store.close()
    return db_path


@pytest.fixture
def server(populated_db: Path) -> Iterator[str]:
    DevHandler.db_path = str(populated_db)
    port = _free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), DevHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.shutdown()
        srv.server_close()


def _get_json(base: str, path: str) -> dict:
    with urllib.request.urlopen(base + path, timeout=5) as resp:
        return json.loads(resp.read().decode())


# --------------------------------------------------------------------------- #
#  Tier-recommendation endpoint                                                #
# --------------------------------------------------------------------------- #


class TestTierEndpoint:
    def test_endpoint_returns_shape(self, server: str) -> None:
        body = _get_json(server, "/api/tier-recommendation")
        assert "projection" in body
        assert "peak_concurrency" in body
        assert "recommendation" in body
        rec = body["recommendation"]
        assert "primary" in rec
        assert "reason" in rec
        assert rec["primary"]["name"] in {"free", "starter", "pro", "scale"}

    def test_empty_db_recommends_free(self, tmp_path: Path) -> None:
        db_path = tmp_path / "empty.db"
        SQLiteStore(str(db_path)).close()
        DevHandler.db_path = str(db_path)
        port = _free_port()
        srv = ThreadingHTTPServer(("127.0.0.1", port), DevHandler)
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            body = _get_json(f"http://127.0.0.1:{port}", "/api/tier-recommendation")
            assert body["recommendation"]["primary"]["name"] == "free"
            assert body["peak_concurrency"] == 0
        finally:
            srv.shutdown()
            srv.server_close()


# --------------------------------------------------------------------------- #
#  Export CLI                                                                  #
# --------------------------------------------------------------------------- #


class TestProjectExport:
    def test_excludes_response_text_by_default(
        self, populated_db: Path, tmp_path: Path
    ) -> None:
        runner = CliRunner()
        out = tmp_path / "history.jsonl"
        result = runner.invoke(cli_main, [
            "project", "export", "--out", str(out), "--db", str(populated_db),
        ])
        assert result.exit_code == 0, result.output
        assert out.exists()

        records = [json.loads(line) for line in out.read_text().splitlines() if line]
        steps = [r for r in records if r["type"] == "step"]
        assert steps  # the seeded DB has at least one step
        for s in steps:
            # v12 privacy posture: raw results/snapshots (which carry model
            # output and customer payloads) are excluded by default.
            assert "result" not in s["data"]
            assert "input_snapshot" not in s["data"]
            assert "output_snapshot" not in s["data"]

    def test_include_response_text_flag(
        self, populated_db: Path, tmp_path: Path
    ) -> None:
        runner = CliRunner()
        out = tmp_path / "history.jsonl"
        result = runner.invoke(cli_main, [
            "project", "export",
            "--out", str(out), "--db", str(populated_db),
            "--include-response-text",
        ])
        assert result.exit_code == 0, result.output

        records = [json.loads(line) for line in out.read_text().splitlines() if line]
        steps = [r for r in records if r["type"] == "step"]
        assert any(s["data"].get("result") == '"some model response"' for s in steps)

    def test_missing_db_exits_nonzero(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli_main, [
            "project", "export",
            "--out", str(tmp_path / "out.jsonl"),
            "--db", str(tmp_path / "missing.db"),
        ])
        assert result.exit_code != 0
        assert "No local database" in result.output

    def test_export_record_counts(
        self, populated_db: Path, tmp_path: Path
    ) -> None:
        runner = CliRunner()
        out = tmp_path / "history.jsonl"
        runner.invoke(cli_main, [
            "project", "export", "--out", str(out), "--db", str(populated_db),
        ])
        records = [json.loads(line) for line in out.read_text().splitlines() if line]
        # v12 export nouns: run (invocation), item, step. The direct-call
        # item is wrapped in an implicit run-of-one, so all three appear.
        assert len([r for r in records if r["type"] == "run"]) >= 1
        assert len([r for r in records if r["type"] == "item"]) >= 1
        assert len([r for r in records if r["type"] == "step"]) >= 1


# --------------------------------------------------------------------------- #
#  Import CLI                                                                  #
# --------------------------------------------------------------------------- #


class TestProjectImport:
    def test_validates_good_file(
        self, populated_db: Path, tmp_path: Path
    ) -> None:
        runner = CliRunner()
        out = tmp_path / "history.jsonl"
        runner.invoke(cli_main, [
            "project", "export", "--out", str(out), "--db", str(populated_db),
        ])

        result = runner.invoke(cli_main, ["project", "import", str(out)])
        assert result.exit_code == 0, result.output
        assert "Validated import file" in result.output
        assert "runs:" in result.output
        assert "items:" in result.output

    def test_rejects_missing_file(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli_main, [
            "project", "import", str(tmp_path / "does-not-exist.jsonl"),
        ])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_rejects_malformed_json(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"type": "batch", "data": {}}\nnot json\n')
        runner = CliRunner()
        result = runner.invoke(cli_main, ["project", "import", str(bad)])
        assert result.exit_code != 0
        assert "invalid json" in result.output.lower()

    def test_rejects_unknown_type(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"type": "mystery", "data": {}}\n')
        runner = CliRunner()
        result = runner.invoke(cli_main, ["project", "import", str(bad)])
        assert result.exit_code != 0
        assert "unknown record type" in result.output.lower()
