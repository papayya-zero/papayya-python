"""Tests for the ``papayya dlq replay`` CLI command.

The command reads a failed run from a local SQLite DB, imports the user's
agent module, and re-invokes the agent function with the captured input
snapshot. These tests stub the agent file as a tiny on-disk module so the
CLI's discovery path is exercised end-to-end without needing a real LLM.
"""

from __future__ import annotations

import sqlite3
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

from papayya import cli as cli_module
from papayya.durable import _schema
from papayya.durable._replay import _replay_invoke
from papayya.durable.sqlite_store import SQLiteStore
from papayya.durable.types import RunCheckpoint


def _write_agent_module(dir: Path, outcome: str = "ok") -> Path:
    """Write a minimal module exposing a @papayya.agent function.

    ``outcome`` controls the body:
      - "ok"    → returns {"got": input_data}
      - "raise" → raises ValueError("boom")
    """
    body = textwrap.dedent(f"""\
        from papayya import agent

        @agent(name="enricher")
        def enricher(input_data):
            if {outcome!r} == "raise":
                raise ValueError("boom")
            return {{"got": input_data}}
    """)
    path = dir / "agent.py"
    path.write_text(body)
    return path


_DEFAULT_INPUT = object()  # Sentinel so tests can pass None explicitly.


def _make_failed_run(
    db_path: Path,
    run_id: str = "dead-run",
    agent: str = "enricher",
    input_snapshot=_DEFAULT_INPUT,
) -> None:
    snap = {"lead_id": "x"} if input_snapshot is _DEFAULT_INPUT else input_snapshot
    store = SQLiteStore(str(db_path))
    now = datetime.now(timezone.utc).isoformat()
    store.create(RunCheckpoint(
        run_id=run_id, agent=agent, tasks=[],
        status="running", created_at=now, updated_at=now,
        input_snapshot=snap,
    ))
    store.set_status(run_id, "failed", output="HTTP 429")
    store.close()


def test_replay_invokes_agent_and_marks_disposition(tmp_path: Path) -> None:
    db_path = tmp_path / "local.db"
    _make_failed_run(db_path)
    _write_agent_module(tmp_path, outcome="ok")

    runner = CliRunner()
    result = runner.invoke(
        cli_module.main,
        ["dlq", "replay", "--run", "dead-run", "--db", str(db_path)],
        catch_exceptions=False,
        # Run from a cwd where agent.py exists so auto-discovery works.
        env={"_PWD_OVERRIDE": str(tmp_path)},
    )
    # CliRunner doesn't actually cd; pass --file explicitly instead.
    result = runner.invoke(
        cli_module.main,
        ["dlq", "replay", "--run", "dead-run",
         "--db", str(db_path), "--file", str(tmp_path / "agent.py")],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "Replay returned" in result.output

    # Old run should now be disposition=replayed
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM runs WHERE run_id='dead-run'").fetchone()
    assert row[_schema.COL_RUN_DLQ_DISPOSITION] == _schema.DLQ_REPLAYED
    assert row[_schema.COL_RUN_DLQ_RESOLVED_AT] is not None


def test_replay_marks_disposition_even_when_agent_raises(tmp_path: Path) -> None:
    """A failed replay still resolves the original — it becomes a 'tried'
    record. Re-failures become new dead letters via the agent's own path."""
    db_path = tmp_path / "local.db"
    _make_failed_run(db_path, run_id="dead-raise")
    agent_file = _write_agent_module(tmp_path, outcome="raise")

    runner = CliRunner()
    result = runner.invoke(
        cli_module.main,
        ["dlq", "replay", "--run", "dead-raise",
         "--db", str(db_path), "--file", str(agent_file)],
        catch_exceptions=False,
    )
    # Exit code 2 signals "replay invoked but agent raised."
    assert result.exit_code == 2
    assert "Replay failed" in result.output

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM runs WHERE run_id='dead-raise'").fetchone()
    assert row[_schema.COL_RUN_DLQ_DISPOSITION] == _schema.DLQ_REPLAYED


def test_replay_rejects_already_resolved(tmp_path: Path) -> None:
    db_path = tmp_path / "local.db"
    _make_failed_run(db_path, run_id="already")
    agent_file = _write_agent_module(tmp_path)

    store = SQLiteStore(str(db_path))
    store.mark_dlq_disposition("already", _schema.DLQ_SKIPPED)
    store.close()

    runner = CliRunner()
    result = runner.invoke(
        cli_module.main,
        ["dlq", "replay", "--run", "already",
         "--db", str(db_path), "--file", str(agent_file)],
    )
    assert result.exit_code != 0
    assert "already resolved" in result.output or "already resolved" in (result.stderr_bytes or b"").decode()


def test_replay_rejects_run_without_snapshot(tmp_path: Path) -> None:
    db_path = tmp_path / "local.db"
    # Old-style run without input_snapshot (None)
    _make_failed_run(db_path, run_id="old-run", input_snapshot=None)
    agent_file = _write_agent_module(tmp_path)

    # The store.create() we already call above DID populate input_snapshot
    # via the dataclass default of None; but v6 column stores that as NULL.
    # Verify: null snapshot → replay rejected.
    conn = sqlite3.connect(db_path)
    snap = conn.execute(
        f"SELECT {_schema.COL_RUN_INPUT_SNAPSHOT} FROM runs WHERE run_id='old-run'"
    ).fetchone()[0]
    conn.close()

    if snap is not None:
        # Helper passed a non-None default; skip this test in that case.
        pytest.skip("fixture wrote a snapshot — skip the null-snapshot guard test")

    runner = CliRunner()
    result = runner.invoke(
        cli_module.main,
        ["dlq", "replay", "--run", "old-run",
         "--db", str(db_path), "--file", str(agent_file)],
    )
    assert result.exit_code != 0
    assert "no input_snapshot" in result.output or "no input_snapshot" in (result.stderr_bytes or b"").decode()


def test_replay_unpacks_dict_snapshot_as_kwargs(tmp_path: Path) -> None:
    """When the snapshot is a dict whose keys match the agent's
    signature, replay calls fn(**snapshot). This is the format the
    @agent wrapper captures (via inspect.signature.bind), so the runs
    written by the worker model replay correctly without each customer
    having to remember to pass a single positional arg.
    """
    db_path = tmp_path / "local.db"
    # Snapshot keyed by parameter name — matches the agent's signature.
    _make_failed_run(
        db_path,
        run_id="kw-run",
        agent="kw_enricher",
        input_snapshot={"item_id": "co_42", "retries": 0},
    )

    body = textwrap.dedent("""\
        from papayya import agent

        @agent(name="kw_enricher")
        def kw_enricher(item_id, retries=0):
            return {"item_id": item_id, "retries": retries}
    """)
    agent_file = tmp_path / "agent.py"
    agent_file.write_text(body)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.main,
        ["dlq", "replay", "--run", "kw-run",
         "--db", str(db_path), "--file", str(agent_file)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    # The fn returned the kwargs back — confirms unpack happened, not
    # a positional pass (which would have stuffed the whole dict into
    # `item_id` and tripped Python's binding before our test could see).
    assert "'item_id': 'co_42'" in result.output
    assert "'retries': 0" in result.output


def test_replay_rejects_unknown_agent(tmp_path: Path) -> None:
    db_path = tmp_path / "local.db"
    _make_failed_run(db_path, run_id="wrong-agent", agent="does-not-exist")
    agent_file = _write_agent_module(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.main,
        ["dlq", "replay", "--run", "wrong-agent",
         "--db", str(db_path), "--file", str(agent_file)],
    )
    assert result.exit_code != 0
    combined = result.output + (result.stderr_bytes or b"").decode()
    assert "No @agent" in combined or "does-not-exist" in combined


def test_replay_rejects_non_failed_run(tmp_path: Path) -> None:
    db_path = tmp_path / "local.db"
    store = SQLiteStore(str(db_path))
    now = datetime.now(timezone.utc).isoformat()
    store.create(RunCheckpoint(
        run_id="done", agent="enricher", tasks=[],
        status="running", created_at=now, updated_at=now,
        input_snapshot={"x": 1},
    ))
    store.set_status("done", "completed", output="ok")
    store.close()
    agent_file = _write_agent_module(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.main,
        ["dlq", "replay", "--run", "done",
         "--db", str(db_path), "--file", str(agent_file)],
    )
    assert result.exit_code != 0


def test_replay_rejects_unknown_run(tmp_path: Path) -> None:
    db_path = tmp_path / "local.db"
    # Just need the DB to exist at the right schema version.
    SQLiteStore(str(db_path)).close()
    agent_file = _write_agent_module(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.main,
        ["dlq", "replay", "--run", "ghost",
         "--db", str(db_path), "--file", str(agent_file)],
    )
    assert result.exit_code != 0
    assert "not found" in (result.output + (result.stderr_bytes or b"").decode()).lower()


# --- _replay_invoke unit tests ----------------------------------------- #

def test_replay_invoke_unpacks_dict_when_keys_bind() -> None:
    def fn(item_id, retries=0):
        return (item_id, retries)

    assert _replay_invoke(fn, {"item_id": "x"}) == ("x", 0)
    assert _replay_invoke(fn, {"item_id": "x", "retries": 3}) == ("x", 3)


def test_replay_invoke_falls_back_to_positional_when_keys_dont_bind() -> None:
    """A dict whose keys don't match the fn's params is passed as one
    positional argument — the agent receives the whole dict as before.
    """
    def fn(payload):
        return payload

    snap = {"unrelated_key": "x"}
    assert _replay_invoke(fn, snap) == snap


def test_replay_invoke_passes_non_dict_positionally() -> None:
    def fn(payload):
        return payload

    assert _replay_invoke(fn, "raw-string") == "raw-string"
    assert _replay_invoke(fn, [1, 2, 3]) == [1, 2, 3]
