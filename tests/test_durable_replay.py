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
from papayya.durable.types import RunCheckpoint, TaskEntry


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


# =========================================================================
# Phase 3 — from_step step-level rewind
# =========================================================================


def _make_failed_run_with_steps(
    db_path: Path,
    *,
    run_id: str,
    agent: str,
    steps: list[tuple[str, object]],
    input_snapshot: object | None = None,
) -> None:
    """Seed a failed run AND its cached step results.

    ``steps`` is a list of ``(label, result)`` pairs in execution order.
    Each pair becomes a TaskEntry row in the tasks table — the same
    shape PapayyaRun.init() reads when hydrating cache from a stored
    checkpoint. The run is marked status=failed so replay() accepts
    it.
    """
    if input_snapshot is None:
        input_snapshot = {"item_id": "co_seed"}
    store = SQLiteStore(str(db_path))
    now = datetime.now(timezone.utc).isoformat()
    store.create(RunCheckpoint(
        run_id=run_id,
        agent=agent,
        tasks=[],
        status="running",
        created_at=now,
        updated_at=now,
        input_snapshot=input_snapshot,
    ))
    for label, result in steps:
        store.save_task(run_id, TaskEntry(
            label=label,
            result=result,
            duration_ms=10,
            completed_at=now,
        ))
    store.set_status(run_id, "failed", output="step blew up")
    store.close()


def _write_pipeline_agent(
    dir: Path,
    *,
    db_path: Path,
    log_path: Path,
    name: str = "pipeline",
    summarize_return_expr: str = "{'summary': 'fresh-' + str(prev)}",
) -> Path:
    """Three-step durable agent (extract → enrich → summarize).

    Each step body appends its label to ``log_path`` so the test can
    assert which steps actually re-executed (cached steps must NOT
    log). ``summarize_return_expr`` is the only piece tests customise
    — it's evaluated inside ``do_summarize(prev)`` and is the natural
    place to inject the bounded-by-captured-input failure mode (Test
    6) by accessing a field absent from the cached predecessor's
    payload.
    """
    src = textwrap.dedent(f"""\
        import json
        from pathlib import Path

        from papayya import agent
        from papayya.durable.client import papayya
        from papayya.durable.sqlite_store import SQLiteStore

        _LOG = Path({str(log_path)!r})

        def log(label):
            with _LOG.open('a') as fh:
                fh.write(label + '\\n')

        @agent(name={name!r})
        def {name}(item_id):
            client = papayya(store=SQLiteStore({str(db_path)!r}))
            run = client.run({name!r}, item_id=item_id)
            run.init()

            def do_extract():
                log('extract')
                return {{'name': 'ACME'}}

            def do_enrich(prev):
                log('enrich')
                return {{'signals': ['a', 'b'], 'from_extract': prev}}

            def do_summarize(prev):
                log('summarize')
                return {summarize_return_expr}

            extract = run.step('extract', do_extract)
            extract_result = extract()
            enrich = run.step('enrich', lambda: do_enrich(extract_result))
            enrich_result = enrich()
            summarize = run.step('summarize', lambda: do_summarize(enrich_result))
            summarize_result = summarize()
            run.complete(summarize_result)
            return summarize_result
    """)
    path = dir / "agent.py"
    path.write_text(src)
    return path


def _read_log(log_path: Path) -> list[str]:
    if not log_path.exists():
        return []
    return [line for line in log_path.read_text().splitlines() if line]


def _new_run_ids(db_path: Path, *, exclude: str) -> list[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT run_id FROM runs WHERE run_id != ?", (exclude,)
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def _task_labels_for(db_path: Path, run_id: str) -> list[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT label FROM tasks WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def test_replay_from_step_label_uses_cached_earlier_steps(
    tmp_path: Path, db_env: Path
) -> None:
    log_path = tmp_path / "calls.log"
    _make_failed_run_with_steps(
        db_env,
        run_id="dead",
        agent="pipeline",
        steps=[
            ("extract", {"name": "ACME"}),
            ("enrich", {"signals": ["a", "b"], "from_extract": {"name": "ACME"}}),
        ],
    )
    agent_file = _write_pipeline_agent(tmp_path, db_path=db_env, log_path=log_path)

    result = replay("dead", agent_module=agent_file, from_step="summarize")

    # Only summarize re-executed; extract + enrich hit the seeded cache.
    assert _read_log(log_path) == ["summarize"]
    assert result == {"summary": "fresh-{'signals': ['a', 'b'], 'from_extract': {'name': 'ACME'}}"}


def test_replay_from_step_int_resolves_to_position(
    tmp_path: Path, db_env: Path
) -> None:
    log_path = tmp_path / "calls.log"
    _make_failed_run_with_steps(
        db_env,
        run_id="dead",
        agent="pipeline",
        steps=[
            ("extract", {"name": "ACME"}),
            ("enrich", {"signals": ["a", "b"], "from_extract": {"name": "ACME"}}),
        ],
    )
    agent_file = _write_pipeline_agent(tmp_path, db_path=db_env, log_path=log_path)

    replay("dead", agent_module=agent_file, from_step=3)

    # 1-indexed step 3 == "summarize" — same behaviour as the label form.
    assert _read_log(log_path) == ["summarize"]


def test_replay_from_step_unmatched_label_hydrates_all(
    tmp_path: Path, db_env: Path
) -> None:
    """Failure-replay case: the dead step's label isn't in stored tasks
    (it never completed). With nothing to validate against, hydrate
    every cached predecessor and let the agent fn pick up at the
    failure point. This is the *primary* use case for from_step — a
    run that died at step N, replayed to retry step N. Typo detection
    is not feasible without knowing the agent's full label set, which
    is only discoverable by executing it.
    """
    log_path = tmp_path / "calls.log"
    _make_failed_run_with_steps(
        db_env,
        run_id="dead",
        agent="pipeline",
        steps=[
            ("extract", {"name": "ACME"}),
            ("enrich", {"signals": ["a", "b"], "from_extract": {"name": "ACME"}}),
        ],
    )
    agent_file = _write_pipeline_agent(tmp_path, db_path=db_env, log_path=log_path)

    # Label "summarize" is not in stored tasks (the dead step). We
    # don't raise — we hydrate all stored and re-execute summarize.
    replay("dead", agent_module=agent_file, from_step="summarize")
    assert _read_log(log_path) == ["summarize"]


def test_replay_from_step_int_out_of_range_raises(
    tmp_path: Path, db_env: Path
) -> None:
    log_path = tmp_path / "calls.log"
    _make_failed_run_with_steps(
        db_env,
        run_id="dead",
        agent="pipeline",
        steps=[
            ("extract", {"name": "ACME"}),
            ("enrich", {"signals": ["a"]}),
            ("summarize", {"summary": "old"}),
        ],
    )
    agent_file = _write_pipeline_agent(tmp_path, db_path=db_env, log_path=log_path)

    # Valid range with 3 cached steps is 1..4 (4 = "all cached, then
    # re-execute the next uncached step"). 5 is out of range.
    with pytest.raises(ReplayError) as excinfo:
        replay("dead", agent_module=agent_file, from_step=5)
    msg = str(excinfo.value)
    assert "5" in msg
    assert "3 cached step(s)" in msg
    assert "1..4" in msg

    # No step body ran — validation fired before agent invoke.
    assert _read_log(log_path) == []
    # Original un-marked — gate fired before mark_dlq_disposition.
    conn = sqlite3.connect(db_env)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM runs WHERE run_id='dead'").fetchone()
    conn.close()
    assert row[_schema.COL_RUN_DLQ_DISPOSITION] is None


def test_replay_from_step_first_step_replays_everything(
    tmp_path: Path, db_env: Path
) -> None:
    log_path = tmp_path / "calls.log"
    _make_failed_run_with_steps(
        db_env,
        run_id="dead",
        agent="pipeline",
        steps=[
            ("extract", {"name": "OLD"}),
            ("enrich", {"signals": ["x"]}),
            ("summarize", {"summary": "old"}),
        ],
    )
    agent_file = _write_pipeline_agent(tmp_path, db_path=db_env, log_path=log_path)

    replay("dead", agent_module=agent_file, from_step="extract")

    # from_step on the first label means empty prepopulated list —
    # behaviour reduces to top-of-agent replay; every body runs fresh.
    assert _read_log(log_path) == ["extract", "enrich", "summarize"]


def test_replay_bounded_by_captured_input_propagates_keyerror(
    tmp_path: Path, db_env: Path
) -> None:
    log_path = tmp_path / "calls.log"
    # Original cached enrich result lacks the 'industry' field that the
    # new summarize body now reads. The natural failure mode of the
    # bounded-by-captured-input gotcha.
    _make_failed_run_with_steps(
        db_env,
        run_id="dead",
        agent="pipeline",
        steps=[
            ("extract", {"name": "ACME"}),
            ("enrich", {"signals": ["a"]}),  # no 'industry'
        ],
    )
    agent_file = _write_pipeline_agent(
        tmp_path,
        db_path=db_env,
        log_path=log_path,
        summarize_return_expr="{'industry': prev['industry']}",
    )

    with pytest.raises(KeyError) as excinfo:
        replay("dead", agent_module=agent_file, from_step="summarize")
    assert "industry" in str(excinfo.value)

    # Cached steps did not re-run; only summarize attempted (and failed).
    assert _read_log(log_path) == ["summarize"]
    # Original still marked replayed — Phase 1+2 mark-on-either-outcome
    # semantics preserved.
    conn = sqlite3.connect(db_env)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM runs WHERE run_id='dead'").fetchone()
    conn.close()
    assert row[_schema.COL_RUN_DLQ_DISPOSITION] == _schema.DLQ_REPLAYED


def test_replay_from_step_writes_only_reexecuted_steps_to_new_run(
    tmp_path: Path, db_env: Path
) -> None:
    log_path = tmp_path / "calls.log"
    _make_failed_run_with_steps(
        db_env,
        run_id="dead",
        agent="pipeline",
        steps=[
            ("extract", {"name": "ACME"}),
            ("enrich", {"signals": ["a", "b"], "from_extract": {"name": "ACME"}}),
        ],
    )
    agent_file = _write_pipeline_agent(tmp_path, db_path=db_env, log_path=log_path)

    replay("dead", agent_module=agent_file, from_step="summarize")

    new_ids = _new_run_ids(db_env, exclude="dead")
    assert len(new_ids) == 1
    new_run_id = new_ids[0]
    # Only the re-executed step is persisted; hydrated extract/enrich
    # live only in the in-memory cache for the duration of the run.
    assert _task_labels_for(db_env, new_run_id) == ["summarize"]


def test_replay_from_step_carries_input_snapshot(
    tmp_path: Path, db_env: Path
) -> None:
    log_path = tmp_path / "calls.log"
    _make_failed_run_with_steps(
        db_env,
        run_id="dead",
        agent="pipeline",
        steps=[
            ("extract", {"name": "ACME"}),
            ("enrich", {"signals": ["a"]}),
        ],
        input_snapshot={"item_id": "co_specific_value"},
    )
    agent_file = _write_pipeline_agent(tmp_path, db_path=db_env, log_path=log_path)

    replay("dead", agent_module=agent_file, from_step="summarize")

    # The new run's input_snapshot must equal what the original run
    # captured — from_step only changes intra-fn cache hydration, not
    # the agent's input args.
    new_ids = _new_run_ids(db_env, exclude="dead")
    assert len(new_ids) == 1
    conn = sqlite3.connect(db_env)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT input_snapshot FROM runs WHERE run_id = ?", (new_ids[0],)
    ).fetchone()
    conn.close()
    import json
    assert json.loads(row["input_snapshot"]) == {"item_id": "co_specific_value"}
