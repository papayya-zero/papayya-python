"""Layer 3 #9 — @agent ↔ run injection.

The documented agent shape is now ``def process_note(run, note): ...``
with ``run`` injected by the wrapper as the first positional argument.
Functions whose first param is not named ``run`` keep working under a
``DeprecationWarning`` for one release.

Detection is by parameter name (literal ``run``), so customers don't
need type annotations to opt in.
"""

from __future__ import annotations

import asyncio
import json
import warnings

import pytest

from papayya import agent
from papayya.agent import _LEGACY_AGENT_PATH_ACTIVE, legacy_agent_path_active
from papayya.durable import papayya
from papayya.durable.run import PapayyaRun
from papayya.durable.sqlite_store import SQLiteStore


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Keep the wrapper's ``papayya()`` factory away from real creds /
    on-disk DB files. The wrapper auto-selects SQLiteStore when no api
    key is resolvable; ``PAPAYYA_LOCAL_DB_PATH`` steers it to tmp."""
    monkeypatch.delenv("PAPAYYA_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PAPAYYA_LOCAL_DB_PATH", str(tmp_path / "store.db"))


# --- inject path -------------------------------------------------------- #

def test_inject_run_first_positional():
    captured: dict = {}

    @agent(name="inject-basic")
    def fn(run, item_id: str) -> None:
        captured["run"] = run
        captured["item_id"] = item_id

    fn("co_42")

    assert isinstance(captured["run"], PapayyaRun)
    assert captured["run"].agent == "inject-basic"
    assert captured["run"]._run_item_id == "co_42"
    assert captured["item_id"] == "co_42"


def test_inject_run_async():
    captured: dict = {}

    @agent(name="inject-async")
    async def fn(run, item_id: str) -> None:
        captured["run"] = run
        captured["item_id"] = item_id

    asyncio.run(fn("co_async"))

    assert isinstance(captured["run"], PapayyaRun)
    assert captured["run"].agent == "inject-async"
    assert captured["run"]._run_item_id == "co_async"
    assert captured["item_id"] == "co_async"


def test_inject_run_with_no_positional_args():
    """``def fn(run)`` invoked with no args → item_id resolves to None."""
    captured: dict = {}

    @agent(name="inject-no-args")
    def fn(run) -> None:
        captured["run"] = run

    fn()

    assert isinstance(captured["run"], PapayyaRun)
    assert captured["run"]._run_item_id is None


def test_inject_run_populates_input_snapshot(tmp_path, monkeypatch):
    """Snapshot is built from a sig view that excludes ``run``, so
    runs.input_snapshot reflects the user's args — not a binding error."""
    db_path = tmp_path / "snap.db"
    monkeypatch.setenv("PAPAYYA_LOCAL_DB_PATH", str(db_path))

    @agent(name="inject-snap")
    def fn(run, item_id: str) -> None:
        echo = run.step("echo", lambda x: x)
        echo(item_id)
        run.complete({"id": item_id})

    fn("co_seed")

    store = SQLiteStore(str(db_path))
    rows = store._conn.execute(
        "SELECT input_snapshot FROM runs WHERE agent = 'inject-snap'"
    ).fetchall()
    assert len(rows) == 1
    assert json.loads(rows[0]["input_snapshot"]) == {"item_id": "co_seed"}


def test_inject_run_does_not_warn():
    """New-pattern functions must not trip the legacy DeprecationWarning."""

    @agent(name="inject-silent")
    def fn(run, item_id: str) -> None:
        ...

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        fn("co_silent")


def test_inject_run_clears_legacy_flag_after_return():
    @agent(name="inject-flag-clear")
    def fn(run, item_id: str) -> None:
        ...

    fn("co_x")
    assert legacy_agent_path_active() is False


# --- legacy path -------------------------------------------------------- #

def test_legacy_signature_no_injection():
    """Legacy fn receives args verbatim; no ``run`` is prepended."""
    captured: list = []

    @agent(name="legacy-basic")
    def fn(item_id: str) -> None:
        captured.append(item_id)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        fn("co_legacy")

    assert captured == ["co_legacy"]


def test_legacy_path_sets_active_flag_during_call():
    seen: list[bool] = []

    @agent(name="legacy-flag")
    def fn(item_id: str) -> None:
        seen.append(legacy_agent_path_active())

    fn("co_a")
    assert seen == [True]
    assert legacy_agent_path_active() is False


def test_legacy_path_emits_deprecation_warning():
    """Customer calling ``papayya().run(...)`` inside fn body trips the
    warning; the message names the new pattern."""

    @agent(name="legacy-warn")
    def fn(item_id: str) -> None:
        run = papayya().run("legacy-warn", item_id=item_id)
        run.complete("done")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        fn("co_w")

    deprecations = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecations) == 1
    msg = str(deprecations[0].message)
    assert "@agent" in msg
    assert "first positional parameter" in msg


def test_legacy_path_resets_flag_after_exception():
    @agent(name="legacy-raise")
    def fn(item_id: str) -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        fn("co_q")
    assert legacy_agent_path_active() is False
    assert _LEGACY_AGENT_PATH_ACTIVE.get() is False
