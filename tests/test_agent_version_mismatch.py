"""Tests for the agent_version replay-mismatch gate (ADR-0002 #7).

The CLI's ``papayya dlq replay`` reads the agent_version captured on the
failed run and compares it to the registration's current value. A
mismatch must abort the replay unless ``--latest`` is passed; pre-#7
runs whose captured version is NULL replay without the gate (Q-1
permissive).

These tests stub the agent file as a tiny on-disk module so the CLI's
discovery path is exercised end-to-end without needing a real worker.
Mirrors ``test_dlq_cli.py``'s rhythm.
"""

from __future__ import annotations

import sqlite3
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

from papayya import cli as cli_module
from papayya.agent import _clear_agent_version_cache, _registry
from papayya.durable import _schema
from papayya.durable.sqlite_store import SQLiteStore
from papayya.durable.types import RunCheckpoint


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear module-level state between cases.

    The agent registry and the version resolver's env+git memoization
    both leak across tests otherwise — explicit clears here keep each
    case independent.
    """
    _registry.clear()
    _clear_agent_version_cache()
    monkeypatch.delenv("PAPAYYA_AGENT_VERSION", raising=False)


def _write_agent_module(dir: Path, version: str) -> Path:
    body = textwrap.dedent(f"""\
        from papayya import agent

        @agent(name="enricher", agent_version={version!r})
        def enricher(input_data):
            return {{"replayed": input_data}}
    """)
    path = dir / "agent.py"
    path.write_text(body)
    return path


def _seed_failed_run(
    db_path: Path,
    *,
    run_id: str = "dead-run",
    captured_version: str | None,
) -> None:
    """Write a failed run row into the SQLite DB with a chosen agent_version.

    Goes through SQLiteStore so the v7 migration runs end-to-end. The
    captured_version is stamped onto the RunCheckpoint at create time,
    matching what a real worker-driven run would persist.
    """
    store = SQLiteStore(str(db_path))
    now = datetime.now(timezone.utc).isoformat()
    store.create(RunCheckpoint(
        run_id=run_id,
        agent="enricher",
        tasks=[],
        status="running",
        created_at=now,
        updated_at=now,
        input_snapshot={"input_data": "co_42"},
        agent_version=captured_version,
    ))
    store.set_status(run_id, "failed", output="provider 5xx")
    store.close()


def _disposition(db_path: Path, run_id: str) -> str | None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,),
        ).fetchone()
    finally:
        conn.close()
    return row[_schema.COL_RUN_DLQ_DISPOSITION] if row is not None else None


def _invoke_replay(*, db_path: Path, agent_path: Path, latest: bool = False):
    args = [
        "dlq", "replay",
        "--run", "dead-run",
        "--db", str(db_path),
        "--file", str(agent_path),
    ]
    if latest:
        args.append("--latest")
    return CliRunner().invoke(cli_module.main, args, catch_exceptions=False)


def test_matching_versions_replay_succeeds(tmp_path: Path) -> None:
    """Sanity: same version on both sides → replay proceeds as today."""
    db_path = tmp_path / "local.db"
    _seed_failed_run(db_path, captured_version="v1")
    agent_path = _write_agent_module(tmp_path, version="v1")

    result = _invoke_replay(db_path=db_path, agent_path=agent_path)
    assert result.exit_code == 0, result.output
    assert "Replay returned" in result.output
    assert _disposition(db_path, "dead-run") == _schema.DLQ_REPLAYED


def test_mismatched_versions_block_replay(tmp_path: Path) -> None:
    """Captured 'v1' vs current 'v2' aborts with both versions in the message."""
    db_path = tmp_path / "local.db"
    _seed_failed_run(db_path, captured_version="v1")
    agent_path = _write_agent_module(tmp_path, version="v2")

    result = _invoke_replay(db_path=db_path, agent_path=agent_path)
    assert result.exit_code == 1, result.output
    assert "'v1'" in result.output
    assert "'v2'" in result.output
    assert "--latest" in result.output
    # Disposition must NOT have been set — replay never executed.
    assert _disposition(db_path, "dead-run") is None


def test_latest_flag_overrides_mismatch(tmp_path: Path) -> None:
    """--latest on a mismatch lets the replay proceed and resolves the run."""
    db_path = tmp_path / "local.db"
    _seed_failed_run(db_path, captured_version="v1")
    agent_path = _write_agent_module(tmp_path, version="v2")

    result = _invoke_replay(db_path=db_path, agent_path=agent_path, latest=True)
    assert result.exit_code == 0, result.output
    assert "Replay returned" in result.output
    assert _disposition(db_path, "dead-run") == _schema.DLQ_REPLAYED


def test_legacy_null_version_replays_freely(tmp_path: Path) -> None:
    """Pre-#7 runs (NULL captured version) replay without --latest (Q-1)."""
    db_path = tmp_path / "local.db"
    _seed_failed_run(db_path, captured_version=None)
    agent_path = _write_agent_module(tmp_path, version="v2")

    result = _invoke_replay(db_path=db_path, agent_path=agent_path)
    assert result.exit_code == 0, result.output
    assert _disposition(db_path, "dead-run") == _schema.DLQ_REPLAYED
