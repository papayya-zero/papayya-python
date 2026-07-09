"""Plan 34 Unit 1 — invocation minting.

One ``papayya.map()`` / ``papayya.iter()`` call is ONE run: a single run
row is minted per call and every processed item links to it. A direct
call (``papayya().item()`` / ``@agent``) is an implicit run-of-one.
Without this, a 1,000-item map() would render as 1,000 "runs" of one
item each — the vocabulary would ship hollow.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import papayya
from papayya.durable.sqlite_store import SQLiteStore


@pytest.fixture
def db(tmp_path: Path):
    path = tmp_path / "mint.db"
    store = SQLiteStore(str(path))
    yield store, path
    store.close()


def _rows(path: Path, table: str) -> list[dict]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table}")]
    finally:
        conn.close()


ITEMS = [
    {"id": "a", "tenant": "t1"},
    {"id": "b", "tenant": "t2"},
    {"id": "c", "tenant": "t1"},
]


def test_map_mints_one_run_row_and_n_item_rows(db) -> None:
    store, path = db
    results = papayya.map(
        lambda t: t["id"].upper(),
        ITEMS,
        agent="minty",
        item_id=lambda t: t["id"],
        partition_key=lambda t: t["tenant"],
        store=store,
    )
    assert results == ["A", "B", "C"]

    runs = _rows(path, "runs")
    items = _rows(path, "items")
    assert len(runs) == 1, f"expected ONE run row for one map() call, got {runs}"
    run = runs[0]
    assert run["agent"] == "minty"
    assert run["total_items"] == 3
    assert run["status"] == "completed"
    assert run["completed_at"] is not None

    assert len(items) == 3
    assert all(i["run_id"] == run["run_id"] for i in items)
    assert sorted(i["item_id"] for i in items) == ["a", "b", "c"]
    assert all(i["status"] == "completed" for i in items)


def test_iter_mints_one_run_row(db) -> None:
    store, path = db
    for _ in papayya.iter(
        ITEMS,
        agent="itery",
        item_id=lambda t: t["id"],
        partition_key=lambda t: t["tenant"],
        store=store,
    ):
        pass

    runs = _rows(path, "runs")
    assert len(runs) == 1
    assert runs[0]["total_items"] == 3
    assert runs[0]["status"] == "completed"


def test_two_map_calls_are_two_runs(db) -> None:
    store, path = db
    kw = dict(item_id=lambda t: t["id"], partition_key=lambda t: t["tenant"], store=store)
    papayya.map(lambda t: t, ITEMS, agent="w", **kw)
    papayya.map(lambda t: t, ITEMS, agent="w", **kw)
    runs = _rows(path, "runs")
    assert len(runs) == 2
    assert all(r["total_items"] == 3 for r in runs)


def test_workload_kwarg_still_accepted_as_alias(db) -> None:
    store, path = db
    papayya.map(
        lambda t: t,
        ITEMS[:1],
        workload="old-spelling",
        item_id=lambda t: t["id"],
        partition_key=lambda t: t["tenant"],
        store=store,
    )
    runs = _rows(path, "runs")
    assert len(runs) == 1
    assert runs[0]["agent"] == "old-spelling"


def test_iter_requires_agent_or_workload() -> None:
    with pytest.raises(TypeError, match="agent"):
        papayya.iter([1], item_id=str, partition_key=str)


def test_failing_item_rolls_run_to_partial(db) -> None:
    store, path = db

    def body(t):
        if t["id"] == "b":
            raise ValueError("boom")
        return t

    with pytest.raises(ValueError):
        for t in papayya.iter(
            ITEMS,
            agent="mixed",
            item_id=lambda t: t["id"],
            partition_key=lambda t: t["tenant"],
            store=store,
        ):
            body(t)

    runs = _rows(path, "runs")
    assert len(runs) == 1
    run = runs[0]
    # Items a completed, b failed, iteration stopped there.
    assert run["total_items"] == 2
    assert run["completed"] == 1
    assert run["failed"] == 1
    assert run["status"] == "partial"


def test_direct_call_is_implicit_run_of_one(db) -> None:
    store, path = db
    client = papayya.Papayya(store=store)
    item = client.item(agent="solo", partition_key=None)
    item.step("s", lambda: 1)()
    item.complete()

    runs = _rows(path, "runs")
    items = _rows(path, "items")
    assert len(runs) == 1
    assert runs[0]["run_id"] == f"single-{item.id}"
    assert runs[0]["total_items"] == 1
    assert runs[0]["status"] == "completed"
    assert items[0]["run_id"] == runs[0]["run_id"]


def test_item_handle_surface(db) -> None:
    """papayya().item() returns an Item; .id is the surrogate; the old
    names (.run(), PapayyaRun, .run_id, active_run_id) stay as aliases."""
    store, _ = db
    from papayya import Item, PapayyaRun

    assert PapayyaRun is Item
    client = papayya.Papayya(store=store)
    via_new = client.item(agent="x", partition_key=None)
    via_old = client.run(agent="x", partition_key=None)
    assert isinstance(via_new, Item)
    assert isinstance(via_old, Item)
    assert via_new.id == via_new.run_id


def test_active_item_returns_handle(db) -> None:
    store, _ = db
    seen: list = []
    for _ in papayya.iter(
        ITEMS[:1],
        agent="handles",
        item_id=lambda t: t["id"],
        partition_key=lambda t: t["tenant"],
        store=store,
    ):
        handle = papayya.active_item()
        seen.append((handle, papayya.active_run_id()))
    (handle, old_id), = seen
    assert handle is not None
    assert handle.id == old_id  # deprecated alias returns the same id
    assert papayya.active_item() is None
