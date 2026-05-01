"""@agent input snapshot capture — wires call args into runs.input_snapshot.

This is the bridge `runs.replay()` / dlq replay / `papayya replay` rely on.
Before this bridge existed, every fresh run was created with
input_snapshot=NULL and replay surfaces error'd with "no input_snapshot —
cannot replay."

The capture happens in the @agent decorator's wrapper: it binds the call
args via inspect.signature, JSON-encodes them defensively, sets a
contextvar, and DurableRun.init() reads the contextvar when seeding the
RunCheckpoint.
"""

from __future__ import annotations

import pytest

from papayya import agent
from papayya._serialize import build_input_snapshot
from papayya.agent import (
    _AGENT_INPUT,
    consume_agent_input_snapshot,
)
from papayya.durable import papayya
from papayya.durable.sqlite_store import SQLiteStore


# --- build_input_snapshot — pure helper -------------------------------- #

def test_build_snapshot_captures_kwargs():
    def fn(item_id: str, retries: int = 0) -> None:
        ...

    import inspect
    sig = inspect.signature(fn)
    snap = build_input_snapshot(sig, ("co_42",), {})
    assert snap == {"item_id": "co_42", "retries": 0}


def test_build_snapshot_normalizes_positional_and_keyword():
    def fn(a: int, b: int, c: int = 3) -> None:
        ...

    import inspect
    sig = inspect.signature(fn)
    assert build_input_snapshot(sig, (1, 2), {}) == {"a": 1, "b": 2, "c": 3}
    assert build_input_snapshot(sig, (1,), {"b": 2}) == {"a": 1, "b": 2, "c": 3}


def test_build_snapshot_returns_none_for_non_json_args():
    """Custom class instances → None, so the run still executes."""
    class Custom:
        pass

    def fn(thing) -> None:
        ...

    import inspect
    sig = inspect.signature(fn)
    assert build_input_snapshot(sig, (Custom(),), {}) is None


def test_build_snapshot_returns_none_when_signature_missing():
    """Builtins / C callables — no introspectable signature."""
    assert build_input_snapshot(None, ("anything",), {}) is None


# --- @agent wrapper — contextvar lifecycle ------------------------------ #

def test_agent_wrapper_sets_contextvar_during_call():
    captured: list = []

    @agent(name="probe")
    def probe(item_id: str) -> None:
        captured.append(consume_agent_input_snapshot())

    probe("co_42")
    assert captured == [{"item_id": "co_42"}]


def test_agent_wrapper_clears_contextvar_after_return():
    @agent(name="probe-clear")
    def probe(item_id: str) -> None:
        ...

    probe("co_99")
    assert _AGENT_INPUT.get() is None


def test_agent_wrapper_clears_contextvar_after_exception():
    @agent(name="probe-raise")
    def probe(item_id: str) -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        probe("co_99")
    assert _AGENT_INPUT.get() is None


def test_agent_wrapper_with_non_encodable_arg_does_not_raise():
    """Non-JSON arg → snapshot=None, fn still runs."""
    class Custom:
        pass

    captured: list = []

    @agent(name="probe-noenc")
    def probe(thing) -> None:
        captured.append(consume_agent_input_snapshot())

    probe(Custom())
    assert captured == [None]


# --- end-to-end: @agent + papayya().run() populates runs.input_snapshot - #

def test_run_input_snapshot_populated_from_agent_args(tmp_path):
    """The whole point of this work — proves replay paths now have data."""
    db_path = str(tmp_path / "agent_snap.db")

    @agent(name="enrich-snap")
    def enrich(item_id: str) -> dict:
        run = papayya(store=SQLiteStore(db_path)).run(
            "enrich-snap", item_id=item_id
        )
        echo = run.step("echo", lambda x: x)
        echo(item_id)
        run.complete({"id": item_id})
        return {"id": item_id}

    enrich("co_seed")

    store = SQLiteStore(db_path)
    rows = store._conn.execute(
        "SELECT input_snapshot FROM runs WHERE agent = 'enrich-snap'"
    ).fetchall()
    assert len(rows) == 1
    import json
    assert json.loads(rows[0]["input_snapshot"]) == {"item_id": "co_seed"}


def test_run_without_agent_decorator_still_works(tmp_path):
    """Direct papayya().run(...) — no @agent, no contextvar, no snapshot.

    Preserves the existing behavior: input_snapshot stays None when
    nothing seeded it. No regression for scripts/tests/notebooks that
    bypass the decorator.
    """
    db_path = str(tmp_path / "no_agent.db")
    run = papayya(store=SQLiteStore(db_path)).run("bare", item_id="co_x")
    echo = run.step("echo", lambda x: x)
    echo("hi")
    run.complete("done")

    store = SQLiteStore(db_path)
    row = store._conn.execute(
        "SELECT input_snapshot FROM runs WHERE agent = 'bare'"
    ).fetchone()
    assert row["input_snapshot"] is None
