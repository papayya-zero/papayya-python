"""Plan 32 — the coherence core.

The silent-failure wedge must fire on the clean ``@papayya.durable def f(item)``
front door, not just inside a ``run.step`` / ``run.llm_step`` wrapper. Before
this, an ambient ``@papayya.llm`` / ``mark_degraded`` inside a clean body found
no active run (``@agent`` published only a run-id string, on a different
contextvar than the verbs read) and ran bare — no journal, no ran-vs-worked
verdict. These tests pin that the front door now records and inspects.
"""

from __future__ import annotations

import pytest

import papayya
from papayya.durable.sqlite_store import SQLiteStore


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Point the ambient store at a throwaway SQLite file.

    The clean-path run is minted via ``papayya().run()`` → ``_auto_store()``,
    which resolves ``PAPAYYA_LOCAL_DB_PATH``. Set it so the run lands somewhere
    we can read back.
    """
    path = str(tmp_path / "wedge.db")
    monkeypatch.setenv("PAPAYYA_LOCAL_DB_PATH", path)
    return path


def _runs(db_path):
    return SQLiteStore(db_path)._conn.execute(
        "SELECT run_id, item_id, partition_key, status, worst_outcome_status "
        "FROM runs ORDER BY item_id"
    ).fetchall()


def _tasks(db_path):
    return SQLiteStore(db_path)._conn.execute(
        "SELECT r.item_id, t.label, t.kind, t.outcome_status "
        "FROM tasks t JOIN runs r ON t.run_id = r.run_id "
        "ORDER BY r.item_id, t.label"
    ).fetchall()


def test_wedge_fires_on_clean_durable_path(db):
    """@papayya.llm inside a clean @papayya.durable body → journaled + inspected."""

    @papayya.llm
    def call_model(company):
        return ""  # a degraded shape (empty result) the inspector should flag

    @papayya.durable
    def enrich(company):
        call_model(company)
        return {"id": company["id"]}

    enrich({"id": "co_1"})

    runs = _runs(db)
    assert len(runs) == 1
    assert runs[0]["status"] == "completed"
    # The ran-vs-worked verdict fired — this is the whole point.
    assert runs[0]["worst_outcome_status"] == "degraded"

    tasks = _tasks(db)
    assert [(t["label"], t["kind"], t["outcome_status"]) for t in tasks] == [
        ("call_model", "llm", "degraded"),
    ]


def test_map_composes_one_run_per_item_with_attribution(db):
    """papayya.map over a decorated fn: one run per item, attribution from
    the lambdas, no double-open."""

    @papayya.llm
    def call_model(c):
        return {"summary": c["name"]}

    @papayya.durable
    def enrich(c):
        call_model(c)
        return {"id": c["id"]}

    companies = [
        {"id": "co_1", "name": "Stripe", "tenant": "t1"},
        {"id": "co_2", "name": "Vercel", "tenant": "t2"},
    ]
    out = papayya.map(
        enrich, companies,
        item_id=lambda c: c["id"],
        partition_key=lambda c: c["tenant"],
    )
    assert out == [{"id": "co_1"}, {"id": "co_2"}]

    runs = _runs(db)
    # Exactly one run per item — the decorated fn reused map's run rather than
    # opening a second.
    assert len(runs) == 2
    assert [(r["item_id"], r["partition_key"]) for r in runs] == [
        ("co_1", "t1"), ("co_2", "t2"),
    ]
    assert all(r["worst_outcome_status"] == "ok" for r in runs)


def test_mark_degraded_fires_under_durable(db):
    """mark_degraded resolves against the lazily-minted ambient run."""

    @papayya.durable
    def enrich(company):
        papayya.mark_degraded("manual_reason")
        return {"id": company["id"]}

    enrich({"id": "co_9"})

    runs = _runs(db)
    assert len(runs) == 1
    assert runs[0]["worst_outcome_status"] == "degraded"


def test_adoption_rewarded_not_required(db):
    """Bare @papayya.llm with no ambient decorator runs bare — no run minted."""

    @papayya.llm
    def call_model(x):
        return {"ok": True}

    assert call_model("hi") == {"ok": True}
    assert papayya.active_run_id() is None
    assert _runs(db) == []


def test_durable_bare_and_parameterized_forms(db):
    """@papayya.durable works both bare and with keyword args."""

    @papayya.durable
    def a(item):
        return {"a": item}

    @papayya.durable(name="beta", budget_usd=1.0)
    def b(item):
        return {"b": item}

    assert a("x") == {"a": "x"}
    assert b("y") == {"b": "y"}
    # The registration name honored the keyword.
    assert papayya.get_agent("beta") is not None


def test_durable_module_still_imports_execution_machinery():
    """papayya.durable serves double duty: callable decorator AND subpackage."""
    assert callable(papayya.durable)
    from papayya.durable import papayya as factory, PapayyaRun  # noqa: F401
    assert factory is not None
