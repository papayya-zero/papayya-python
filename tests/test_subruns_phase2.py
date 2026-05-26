"""Layer 3 #7 Phase 2 — sub-runs lineage carry.

Phase 2 closes the loop that Phase 1's schema opened: when a child run
is created from inside an outer @agent body, the child's
``parent_run_id`` is automatically set to the outer run's id. An
explicit ``parent_run_id=`` kwarg on ``papayya.run()`` lets callers
override the auto-detected value (or attach lineage out-of-band).

These tests target the local SQLite store path end-to-end. The hosted
CloudStore path threads the same field into a wire payload; the SDK
unit test covers wire intent, and the integration test against a live
control-pane runs in tests/integration/.
"""

from __future__ import annotations

import asyncio

import pytest

from papayya import agent
from papayya.agent import get_active_run_id
from papayya.durable import papayya
from papayya.durable.sqlite_store import SQLiteStore


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Local SQLite only — keep the wrapper's papayya() factory away
    from real creds. Mirrors test_agent_run_injection.py."""
    monkeypatch.delenv("PAPAYYA_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PAPAYYA_LOCAL_DB_PATH", str(tmp_path / "store.db"))


def _read_parent(db_path: str, run_id: str) -> str | None:
    store = SQLiteStore(db_path)
    rows = store._conn.execute(
        "SELECT parent_run_id FROM runs WHERE run_id = ?", (run_id,),
    ).fetchall()
    assert len(rows) == 1, f"expected one row for {run_id}, got {len(rows)}"
    return rows[0]["parent_run_id"]


# --- contextvar -------------------------------------------------------- #

def test_active_run_id_is_none_outside_agent_body():
    assert get_active_run_id() is None


def test_active_run_id_is_set_inside_inject_agent_body():
    seen: list[str | None] = []

    @agent(name="ctx-probe")
    def fn(run, item_id: str) -> None:
        seen.append(get_active_run_id())

    fn("co_ctx")
    assert len(seen) == 1
    assert seen[0] is not None
    # The contextvar should reset cleanly after the wrapper returns.
    assert get_active_run_id() is None


def test_active_run_id_is_unset_in_legacy_path():
    """Legacy fns don't pre-create the run, so the contextvar stays
    unset — same posture as before Phase 2 for those callers."""
    seen: list[str | None] = []

    @agent(name="ctx-legacy")
    def fn(item_id: str) -> None:
        seen.append(get_active_run_id())

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        fn("co_legacy")

    assert seen == [None]


# --- auto-carry on child via papayya().run(...) ------------------------ #

def test_child_run_inherits_parent_via_contextvar(tmp_path, monkeypatch):
    """Acceptance criterion from subruns_plan.md Phase 2:

    spawn child via client.run(...) from inside @agent body → child
    row's parent_run_id equals parent's id.
    """
    db_path = tmp_path / "subruns.db"
    monkeypatch.setenv("PAPAYYA_LOCAL_DB_PATH", str(db_path))

    captured: dict = {}

    @agent(name="parent-agent")
    def fn(run, item_id: str) -> None:
        captured["parent_id"] = run.run_id
        child = papayya().run(agent="child-agent", item_id=f"child-of-{item_id}")
        child.step("noop", lambda: None)()
        child.complete("done")
        captured["child_id"] = child.run_id
        run.complete("done")

    fn("co_root")

    # Parent row: top-level, no parent.
    assert _read_parent(str(db_path), captured["parent_id"]) is None
    # Child row: parent_run_id points at the outer run.
    assert _read_parent(str(db_path), captured["child_id"]) == captured["parent_id"]


def test_child_run_inherits_parent_via_contextvar_async(tmp_path, monkeypatch):
    db_path = tmp_path / "subruns_async.db"
    monkeypatch.setenv("PAPAYYA_LOCAL_DB_PATH", str(db_path))

    captured: dict = {}

    @agent(name="parent-async")
    async def fn(run, item_id: str) -> None:
        captured["parent_id"] = run.run_id
        child = papayya().run(agent="child-async", item_id=item_id)
        child.step("noop", lambda: None)()
        child.complete("done")
        captured["child_id"] = child.run_id
        run.complete("done")

    asyncio.run(fn("co_async"))

    assert _read_parent(str(db_path), captured["parent_id"]) is None
    assert _read_parent(str(db_path), captured["child_id"]) == captured["parent_id"]


def test_multiple_children_share_same_parent(tmp_path, monkeypatch):
    """The contextvar stays set across the entire fn body, so all child
    runs created within one @agent invocation inherit the same parent."""
    db_path = tmp_path / "fanout.db"
    monkeypatch.setenv("PAPAYYA_LOCAL_DB_PATH", str(db_path))

    captured: dict = {"children": []}

    @agent(name="parent-fanout")
    def fn(run, item_id: str) -> None:
        captured["parent_id"] = run.run_id
        for i in range(3):
            child = papayya().run(agent=f"child-{i}", item_id=f"item-{i}")
            child.step("work", lambda: i)()
            child.complete("done")
            captured["children"].append(child.run_id)
        run.complete("done")

    fn("co_fanout")

    for child_id in captured["children"]:
        assert _read_parent(str(db_path), child_id) == captured["parent_id"]


# --- explicit kwarg --------------------------------------------------- #

def test_explicit_parent_run_id_kwarg_outside_agent(tmp_path, monkeypatch):
    """Out-of-band spawn: caller is not inside an @agent body, so the
    contextvar is unset, but they pass an explicit parent."""
    db_path = tmp_path / "explicit.db"
    monkeypatch.setenv("PAPAYYA_LOCAL_DB_PATH", str(db_path))

    parent_id = "11111111-1111-1111-1111-111111111111"
    child = papayya().run(agent="explicit-child", parent_run_id=parent_id)
    child.step("noop", lambda: None)()
    child.complete("done")

    assert _read_parent(str(db_path), child.run_id) == parent_id


def test_explicit_parent_run_id_overrides_contextvar(tmp_path, monkeypatch):
    """Caller inside an @agent body can override the auto-detected
    outer-run id by passing an explicit parent_run_id=. The override
    wins."""
    db_path = tmp_path / "override.db"
    monkeypatch.setenv("PAPAYYA_LOCAL_DB_PATH", str(db_path))

    override_parent = "22222222-2222-2222-2222-222222222222"
    captured: dict = {}

    @agent(name="parent-override")
    def fn(run, item_id: str) -> None:
        child = papayya().run(
            agent="overridden-child",
            parent_run_id=override_parent,
        )
        child.step("noop", lambda: None)()
        child.complete("done")
        captured["child_id"] = child.run_id
        run.complete("done")

    fn("co_override")

    assert _read_parent(str(db_path), captured["child_id"]) == override_parent


# --- top-level invariants --------------------------------------------- #

def test_top_level_run_outside_agent_has_no_parent(tmp_path, monkeypatch):
    db_path = tmp_path / "toplevel.db"
    monkeypatch.setenv("PAPAYYA_LOCAL_DB_PATH", str(db_path))

    run = papayya().run(agent="top-level")
    run.step("noop", lambda: None)()
    run.complete("done")

    assert _read_parent(str(db_path), run.run_id) is None


def test_outer_run_in_inject_agent_has_no_parent(tmp_path, monkeypatch):
    """The outer run created by the @agent wrapper itself must be
    top-level. Only runs created INSIDE the fn body get the carry."""
    db_path = tmp_path / "outer.db"
    monkeypatch.setenv("PAPAYYA_LOCAL_DB_PATH", str(db_path))

    captured: dict = {}

    @agent(name="solo-parent")
    def fn(run, item_id: str) -> None:
        captured["parent_id"] = run.run_id
        run.step("noop", lambda: None)()
        run.complete("done")

    fn("co_solo")

    assert _read_parent(str(db_path), captured["parent_id"]) is None
