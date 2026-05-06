"""Subprocess-free SDK tests for ``papayya.durable.client.replay``.

The CLI integration path (``papayya dlq replay``) is covered by
``tests/test_dlq_cli.py``; these tests exercise the Python entry point
directly so callers using replay from a notebook / REPL / script have
the same coverage. DB path is steered via ``PAPAYYA_LOCAL_DB_PATH``
(monkeypatched per-test); ``agent_module=`` is passed explicitly so
no test depends on cwd.
"""

from __future__ import annotations

import sqlite3
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pytest

from papayya.agent import _clear_agent_version_cache
from papayya.durable import _schema
from papayya.durable.client import replay
from papayya.durable._replay import ReplayError
from papayya.durable.sqlite_store import SQLiteStore
from papayya.durable.types import RunCheckpoint


_DEFAULT_INPUT = object()


def _make_failed_run(
    db_path: Path,
    *,
    run_id: str = "dead-run",
    agent: str = "enricher",
    input_snapshot=_DEFAULT_INPUT,
    agent_version: str | None = None,
) -> None:
    snap = {"lead_id": "x"} if input_snapshot is _DEFAULT_INPUT else input_snapshot
    store = SQLiteStore(str(db_path))
    now = datetime.now(timezone.utc).isoformat()
    store.create(RunCheckpoint(
        run_id=run_id,
        agent=agent,
        tasks=[],
        status="running",
        created_at=now,
        updated_at=now,
        input_snapshot=snap,
        agent_version=agent_version,
    ))
    store.set_status(run_id, "failed", output="HTTP 429")
    store.close()


def _write_agent_module(
    dir: Path,
    *,
    name: str = "enricher",
    version: str | None = None,
) -> Path:
    version_arg = f", agent_version={version!r}" if version else ""
    src = textwrap.dedent(f"""\
        from papayya import agent

        @agent(name={name!r}{version_arg})
        def {name}(input_data):
            return {{"got": input_data}}
    """)
    path = dir / "agent.py"
    path.write_text(src)
    return path


@pytest.fixture
def db_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the SDK's DB resolver at a fresh tmp DB and reset the
    decoration-time agent_version cache so per-test version overrides
    aren't masked by an earlier test's resolution."""
    db_path = tmp_path / "local.db"
    monkeypatch.setenv("PAPAYYA_LOCAL_DB_PATH", str(db_path))
    _clear_agent_version_cache()
    return db_path


def test_replay_happy_path_returns_agent_value(
    tmp_path: Path, db_env: Path
) -> None:
    _make_failed_run(db_env, run_id="dead-run")
    agent_file = _write_agent_module(tmp_path)

    result = replay("dead-run", agent_module=agent_file)

    assert result == {"got": {"lead_id": "x"}}

    conn = sqlite3.connect(db_env)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM runs WHERE run_id='dead-run'").fetchone()
    conn.close()
    assert row[_schema.COL_RUN_DLQ_DISPOSITION] == _schema.DLQ_REPLAYED
    assert row[_schema.COL_RUN_DLQ_RESOLVED_AT] is not None


def test_replay_rejects_null_snapshot(tmp_path: Path, db_env: Path) -> None:
    _make_failed_run(db_env, run_id="legacy", input_snapshot=None)
    agent_file = _write_agent_module(tmp_path)

    # Sanity check: confirm the row is NULL on disk before asserting on
    # behavior. If the SQLiteStore later starts persisting a fallback
    # encoding the test would silently stop exercising the guard.
    conn = sqlite3.connect(db_env)
    snap = conn.execute(
        f"SELECT {_schema.COL_RUN_INPUT_SNAPSHOT} FROM runs WHERE run_id='legacy'"
    ).fetchone()[0]
    conn.close()
    if snap is not None:
        pytest.skip("fixture wrote a snapshot — null-snapshot path not exercised")

    with pytest.raises(ReplayError, match="no input_snapshot"):
        replay("legacy", agent_module=agent_file)

    # Original is untouched: gate fired before mark_dlq_disposition.
    conn = sqlite3.connect(db_env)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM runs WHERE run_id='legacy'").fetchone()
    conn.close()
    assert row[_schema.COL_RUN_DLQ_DISPOSITION] is None


def test_replay_blocks_version_mismatch_until_latest(
    tmp_path: Path, db_env: Path
) -> None:
    _make_failed_run(
        db_env,
        run_id="versioned",
        agent_version="v1-captured",
    )
    agent_file = _write_agent_module(tmp_path, version="v2-current")

    with pytest.raises(ReplayError, match="agent_version"):
        replay("versioned", agent_module=agent_file)

    conn = sqlite3.connect(db_env)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM runs WHERE run_id='versioned'"
    ).fetchone()
    conn.close()
    assert row[_schema.COL_RUN_DLQ_DISPOSITION] is None

    # latest=True opts out of the gate and completes the replay.
    result = replay("versioned", agent_module=agent_file, latest=True)
    assert result == {"got": {"lead_id": "x"}}

    conn = sqlite3.connect(db_env)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM runs WHERE run_id='versioned'"
    ).fetchone()
    conn.close()
    assert row[_schema.COL_RUN_DLQ_DISPOSITION] == _schema.DLQ_REPLAYED


def test_replay_rejects_missing_registration(
    tmp_path: Path, db_env: Path
) -> None:
    _make_failed_run(db_env, run_id="orphan", agent="does-not-exist")
    agent_file = _write_agent_module(tmp_path, name="enricher")

    with pytest.raises(ReplayError) as excinfo:
        replay("orphan", agent_module=agent_file)
    assert "does-not-exist" in str(excinfo.value)
    assert "enricher" in str(excinfo.value)
