"""Integration tests for conservation contracts through PapayyaRun.

Plan 40 Unit 1. The pure-inspector tests live in ``test_outcomes.py``;
these drive real steps through ``_wrap``/``_post_call_success`` and assert
the declared relation is evaluated against the step's LIVE bound arguments
— the part that can't be tested at the inspector layer.

The case this unit exists for: a nightly batch step that returns 40 rows
against 500 inputs, HTTP 200, no exception. ``inspect_empty`` passes it.
A declared count does not.
"""

from __future__ import annotations

import logging

from papayya.durable.run import PapayyaRun
from papayya.durable.sqlite_store import SQLiteStore
from papayya.durable.types import DurableRunConfig


def _make_run(store=None) -> PapayyaRun:
    return PapayyaRun(DurableRunConfig(agent="test-agent", store=store))


# --- cardinality ------------------------------------------------------- #

def test_half_empty_batch_is_degraded_on_run_one():
    run = _make_run()
    docs = [{"doc_id": i} for i in range(500)]

    def extract(docs):
        return docs[:40]  # the silent 460-row shortfall

    run.step("extract", extract, expect_count=len(docs))(docs)

    entry = run._cache["extract"]
    assert entry.outcome_status == "degraded"
    assert entry.outcome_reason == "conservation:count_short"


def test_full_batch_is_ok():
    run = _make_run()
    docs = [{"doc_id": i} for i in range(10)]

    run.step("extract", lambda docs: list(docs), expect_count=len(docs))(docs)

    assert run._cache["extract"].outcome_status == "ok"


def test_callable_count_receives_the_steps_bound_arguments():
    run = _make_run()

    def extract(docs, *, limit=None):
        return docs[:2]

    run.step("extract", extract, expect_count=lambda docs, limit: len(docs))(
        [1, 2, 3, 4]
    )

    entry = run._cache["extract"]
    assert entry.outcome_reason == "conservation:count_short"


def test_filtering_step_without_a_contract_is_untouched():
    """D4: absence means no check. A step that legitimately drops rows is
    normal and must never be flagged."""
    run = _make_run()

    def only_matches(docs):
        return [d for d in docs if d["match"]]

    run.step("filter", only_matches)(
        [{"match": True}, {"match": False}, {"match": False}]
    )

    entry = run._cache["filter"]
    assert entry.outcome_status == "ok"
    assert entry.outcome_reason is None


# --- coverage ---------------------------------------------------------- #

def test_coverage_locates_the_input_collection_by_key_name():
    run = _make_run()
    docs = [{"doc_id": "a"}, {"doc_id": "b"}, {"doc_id": "c"}]

    def enrich(docs):
        return [d for d in docs if d["doc_id"] != "b"]

    run.step("enrich", enrich, expect_coverage="doc_id")(docs)

    entry = run._cache["enrich"]
    assert entry.outcome_status == "degraded"
    assert entry.outcome_reason == "conservation:coverage"


def test_coverage_complete_is_ok():
    run = _make_run()
    docs = [{"doc_id": "a"}, {"doc_id": "b"}]

    run.step("enrich", lambda docs: list(reversed(docs)), expect_coverage="doc_id")(docs)

    assert run._cache["enrich"].outcome_status == "ok"


# --- field-presence floor ---------------------------------------------- #

def test_half_populated_records_are_degraded():
    run = _make_run()

    def extract(raw):
        return [{"name": "Ada", "email": "", "phone": None} for _ in raw]

    run.step("extract", extract, expect_fields=["name", "email", "phone"])(["x"])

    entry = run._cache["extract"]
    assert entry.outcome_status == "degraded"
    assert entry.outcome_reason == "conservation:field:email"


def test_fully_populated_records_are_ok():
    run = _make_run()

    def extract(raw):
        return [{"name": "Ada", "email": "a@b.c"} for _ in raw]

    run.step("extract", extract, expect_fields=["name", "email"])(["x"])

    assert run._cache["extract"].outcome_status == "ok"


# --- observer invariant ------------------------------------------------ #

def test_a_raising_contract_never_fails_the_step(caplog):
    run = _make_run()

    def bad_count(**_kwargs):
        raise RuntimeError("contract is broken")

    with caplog.at_level(logging.ERROR, logger="papayya.outcomes"):
        out = run.step("extract", lambda docs: docs, expect_count=bad_count)([1, 2])

    assert out == [1, 2]
    assert run._cache["extract"].outcome_status == "ok"
    assert "treating as a pass" in caplog.text


def test_a_malformed_declaration_is_dropped_not_raised():
    run = _make_run()

    # expect_count as a string is a typo, not a contract. The step still runs.
    out = run.step("extract", lambda docs: docs, expect_count="500")([1])

    assert out == [1]
    assert run._cache["extract"].outcome_status == "ok"


def test_contract_on_an_unbindable_signature_is_a_pass():
    """Builtins have no introspectable signature, so there are no bound
    inputs to relate against. The step must still run."""
    run = _make_run()

    out = run.step("count", len, expect_count=lambda **kw: 99)([1, 2, 3])

    assert out == 3
    assert run._cache["count"].outcome_status == "ok"


# --- one pipeline (D2) -------------------------------------------------- #

def test_conservation_verdict_flows_into_the_run_rollup(tmp_path):
    store = SQLiteStore(str(tmp_path / "conservation.db"))
    try:
        run = _make_run(store=store)
        docs = [{"doc_id": i} for i in range(500)]

        run.step("ok-step", lambda: {"x": 1})()
        run.step("extract", lambda docs: docs[:40], expect_count=len(docs))(docs)

        loaded = store.load(run.run_id)
        assert loaded is not None
        assert loaded.worst_outcome_status == "degraded"
        assert loaded.degraded_count == 1

        by_label = {t.label: t for t in loaded.tasks}
        assert by_label["extract"].outcome_reason == "conservation:count_short"
    finally:
        store.close()


def test_llm_step_accepts_a_contract():
    run = _make_run()

    def extract_rows(docs):
        # A parsed-out shape, not the raw provider response.
        return [{"name": "Ada"}]

    run.llm_step("extract", extract_rows, expect_count=lambda docs: len(docs))(
        ["a", "b", "c"]
    )

    entry = run._cache["extract"]
    assert entry.outcome_status == "degraded"
    assert entry.outcome_reason == "conservation:count_short"


def test_stop_reason_still_outranks_conservation():
    """A truncated generation explains WHY the count is short — the
    stop-reason verdict stays the most informative one."""
    from types import SimpleNamespace

    run = _make_run()

    def call_llm(docs):
        return SimpleNamespace(
            model="gpt-4o-mini",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4096, total_tokens=4106),
            choices=[SimpleNamespace(finish_reason="length")],
        )

    run.llm_step("call-llm", call_llm, expect_count=3)(["a", "b", "c"])

    assert run._cache["call-llm"].outcome_reason == "llm_stop_reason:length"
