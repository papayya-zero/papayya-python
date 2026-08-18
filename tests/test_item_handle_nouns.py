"""Plan 34's noun consolidation, on the types the SDK actually exports.

`Item` is the per-record handle; `PapayyaRun` is its pre-Plan-34 alias, kept
because customer code and the dashboard's older payloads both still say
"run" for the thing that is now an item. `.id` is the name; `.run_id` is the
alias. If those ever stop being the same object, every downstream reader
that switched to the new spelling silently addresses a different record.

WHY THIS FILE EXISTS (plan 53). These assertions lived in
test_invocation_minting.py under a file-level
`pytestmark = skip(reason="Plan 37: local surface deactivated")`. The skip is
right about that file's subject — papayya.map / papayya.iter minting run rows
in the local SQLite ledger — and wrong about these, which are about two names
in `papayya/__init__.py`'s export list. A file-level skip only covers what the
file is ABOUT, and files grow; this is the second instance found (plan 52's
was test_runs_cli.py, where the same shape had disabled four tests of live
hosted commands).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import papayya
from papayya.durable.sqlite_store import SQLiteStore


@pytest.fixture
def store(tmp_path: Path):
    s = SQLiteStore(str(tmp_path / "nouns.db"))
    yield s
    s.close()


def test_papayya_run_is_the_same_object_as_item():
    from papayya import Item, PapayyaRun

    assert PapayyaRun is Item


def test_both_constructors_return_an_item_and_id_aliases_run_id(store):
    from papayya import Item

    client = papayya.Papayya(store=store)
    via_new = client.item(agent="x", partition_key=None)
    via_old = client.run(agent="x", partition_key=None)

    assert isinstance(via_new, Item)
    assert isinstance(via_old, Item)
    assert via_new.id == via_new.run_id


def test_a_direct_call_is_an_implicit_run_of_one(store, tmp_path: Path):
    """One item addressed directly still belongs to a run — otherwise the
    vocabulary ships hollow, with records that have no invocation."""
    import sqlite3

    path = tmp_path / "nouns.db"
    client = papayya.Papayya(store=store)
    item = client.item(agent="solo", partition_key=None)
    item.step("s", lambda: 1)()
    item.complete()

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    runs = [dict(r) for r in conn.execute("SELECT * FROM runs")]
    items = [dict(r) for r in conn.execute("SELECT * FROM items")]
    conn.close()

    assert len(runs) == 1
    assert runs[0]["total_items"] == 1
    assert runs[0]["status"] == "completed"
    assert items[0]["run_id"] == runs[0]["run_id"]
