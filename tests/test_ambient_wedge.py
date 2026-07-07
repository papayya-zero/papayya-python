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


async def test_gather_fanout_body_succeeds_and_journals_one_run(db):
    """Regression: the first ambient mint inside an asyncio.Task created its
    _ACTIVE_RUN token in the task's copied context; the wrapper's finally then
    reset that foreign token in the parent context and a SUCCEEDED body raised
    ``ValueError: token was created in a different Context``."""
    import asyncio

    @papayya.llm
    async def call_model(prompt):
        await asyncio.sleep(0)
        return {"content": f"answer:{prompt}", "stop_reason": "end_turn"}

    @papayya.durable
    async def enrich(item):
        return await asyncio.gather(call_model(f"{item}-a"), call_model(f"{item}-b"))

    out = await enrich("co_7")
    assert [r["content"] for r in out] == ["answer:co_7-a", "answer:co_7-b"]

    runs = _runs(db)
    assert len(runs) == 1
    assert runs[0]["status"] == "completed"
    # Both gathered calls journaled onto the single minted run.
    assert [(t["label"], t["outcome_status"]) for t in _tasks(db)] == [
        ("call_model", "ok"), ("call_model#2", "ok"),
    ]


def test_partition_key_kwarg_reaches_minted_run(db):
    """The decorator's partition_key kwarg must land on the run row, not just
    in OTel baggage."""

    @papayya.durable
    def enrich(item, *, partition_key=None):
        papayya.mark_degraded("check_attribution")
        return item

    enrich("co_3", partition_key="tenant-a")

    runs = _runs(db)
    assert len(runs) == 1
    assert runs[0]["partition_key"] == "tenant-a"


@pytest.fixture
def declared_partition_key_yaml(tmp_path, monkeypatch):
    """A project whose papayya.yaml declares a partition_key — the
    multi-tenant configuration. Chdir so ``papayya().run()`` resolves it."""
    (tmp_path / "papayya.yaml").write_text(
        "version: 1\n"
        "partition_key: tenant_id\n"
        "envs:\n"
        "  dev:\n"
        "    agents: {}\n"
    )
    monkeypatch.chdir(tmp_path)


def test_declared_partition_key_does_not_crash_clean_path(
    db, declared_partition_key_yaml
):
    """Regression: the lazy mint called papayya().run() with no metadata, so a
    declared partition_key hit strict-when-declared and the first ambient verb
    raised ValueError. The body can't pass metadata — record NULL instead."""

    @papayya.llm
    def call_model(item):
        return {"content": "x", "stop_reason": "end_turn"}

    @papayya.durable
    def enrich(item):
        return call_model(item)

    enrich("co_1")

    runs = _runs(db)
    assert len(runs) == 1
    assert runs[0]["status"] == "completed"
    assert runs[0]["partition_key"] is None


def test_declared_partition_key_does_not_crash_inject_path(
    db, declared_partition_key_yaml
):
    """Same regression on the ``def f(run, …)`` branch — the decorator's
    factory call passes no metadata either."""

    @papayya.durable
    def enrich(run, item):
        run.step("work", lambda: {"ok": True})()
        return run.complete()

    result = enrich("co_2")
    assert result.status == "completed"


def test_nested_durable_reuses_minted_run(db):
    """After the outer body's first verb mints the ambient run, a nested
    decorated fn must reuse it (Case C now peeks the isolate's minted run —
    the mint no longer publishes on _ACTIVE_RUN)."""

    @papayya.durable
    def inner(item):
        papayya.mark_degraded("from_inner")
        return item

    @papayya.durable
    def outer(item):
        papayya.mark_degraded("from_outer")  # mints the ambient run
        return inner(item)

    outer("co_5")

    runs = _runs(db)
    assert len(runs) == 1
    assert runs[0]["worst_outcome_status"] == "degraded"
