"""D6 taint — a degraded input makes the next output suspect.

Plan 44. ADR 0009 D6: "Record R was produced from A and B; a verdict on A
taints its descendants as suspect."

The incident this exists for (corpus case 5): a context fetch times out, the
tool returns empty, and the agent hallucinates confidently around the hole.
``inspect_empty`` already fires on the fetch. Nothing related that verdict to
the confident answer downstream — which is the precise mechanism of silent
failure.

Two of these tests are regression pins rather than feature tests: taint must
leave ``outcome_status`` and therefore both degraded-streak fences completely
alone. That is the design (§2), and it is the thing a future refactor toward
"just make it degraded" would quietly break.
"""

from __future__ import annotations

import papayya.outcomes as outcomes
from papayya.durable.run import PapayyaRun
from papayya.durable.sqlite_store import SQLiteStore
from papayya.durable.types import DurableRunConfig


def _make_run(store=None, run_id=None) -> PapayyaRun:
    cfg = DurableRunConfig(agent="test-agent", store=store)
    if run_id is not None:
        cfg.run_id = run_id
    return PapayyaRun(cfg)


# --- the incident ------------------------------------------------------ #

def test_confident_answer_over_an_empty_fetch_is_tainted():
    run = _make_run()

    docs = run.step("retrieve", lambda: [])()
    run.step("answer", lambda context: "Yes, per your policy.")(docs)

    fetch = run._cache["retrieve"]
    answer = run._cache["answer"]
    assert fetch.outcome_status == "degraded"
    assert fetch.outcome_reason == "empty_sequence"
    # The answer's own verdict is untouched — nobody judged the string.
    assert answer.outcome_status == "ok"
    assert answer.outcome_reason is None
    # Its provenance is not.
    assert answer.tainted_by == "retrieve"
    assert answer.tainted_reason == "empty_sequence"


def test_taint_is_transitive_across_two_hops():
    run = _make_run()

    a = run.step("fetch", lambda: [])()
    b = run.step("summarize", lambda docs: {"summary": "none found"})(a)
    run.step("publish", lambda payload: "ok")(b)

    assert run._cache["summarize"].tainted_by == "fetch"
    # The second hop taints off the first hop's own (tainted) result. It names
    # the ROOT, not its immediate parent: an operator asking "what did that bad
    # fetch touch?" gets the whole chain from one equality predicate.
    assert run._cache["publish"].tainted_by == "fetch"
    assert run._cache["publish"].tainted_reason == "empty_sequence"


def test_a_step_that_never_saw_the_value_is_clean():
    """Adjacency is not the test. Dataflow is."""
    run = _make_run()

    run.step("retrieve", lambda: [])()
    run.step("heartbeat", lambda: "beat")()

    assert run._cache["retrieve"].outcome_status == "degraded"
    assert run._cache["heartbeat"].tainted_by is None


def test_a_step_with_its_own_degradation_keeps_its_own_verdict():
    run = _make_run()

    docs = run.step("retrieve", lambda: [])()
    run.step("narrow", lambda d: [])(docs)

    narrow = run._cache["narrow"]
    assert narrow.outcome_status == "degraded"
    assert narrow.outcome_reason == "empty_sequence"
    # Taint is about the input's provenance; the output verdict is the more
    # specific signal and wins the status. We do not double-report.
    assert narrow.tainted_by is None


# --- sources we deliberately do not propagate from ---------------------- #

def test_empty_none_is_not_a_taint_source():
    """`CountedDegradedSteps` already calls a None-returning step
    "overwhelmingly a void side effect (send_email, write_row)" and excludes
    it from re-execution. Tainting off it would be backwards."""
    run = _make_run()

    sent = run.step("send_email", lambda: None)()
    run.step("log_it", lambda receipt: "logged")(sent)

    assert run._cache["send_email"].outcome_status == "degraded"
    assert run._cache["send_email"].outcome_reason == "empty_none"
    assert run._cache["log_it"].tainted_by is None


def test_interned_empties_are_not_taint_sources():
    """`"" is ""` and `() is ()` are True, so identity would fire on any
    unrelated argument holding the same singleton."""
    run = _make_run()

    blank = run.step("render", lambda: "")()
    run.step("ship", lambda body, footer="": body)(blank)

    assert run._cache["render"].outcome_status == "degraded"
    assert run._cache["ship"].tainted_by is None


def test_equal_but_unrelated_values_do_not_taint():
    """Identity, not equality."""
    run = _make_run()

    run.step("retrieve", lambda: [])()
    run.step("answer", lambda context: "hi")([])  # a DIFFERENT empty list

    assert run._cache["answer"].tainted_by is None


# --- shapes the value arrives in ---------------------------------------- #

def test_taint_finds_the_value_inside_varargs():
    run = _make_run()

    chunks = run.step("fetch", lambda: [])()
    run.step("join", lambda *parts: "joined")(chunks)

    assert run._cache["join"].tainted_by == "fetch"


def test_taint_finds_the_value_one_container_deep():
    run = _make_run()

    docs = run.step("fetch", lambda: [])()
    run.step("answer", lambda context: "sure")({"docs": docs})

    assert run._cache["answer"].tainted_by == "fetch"


# --- persistence and replay --------------------------------------------- #

def test_taint_round_trips_through_the_local_store(tmp_path):
    store = SQLiteStore(db_path=tmp_path / "papayya.db")
    run = _make_run(store=store)

    docs = run.step("retrieve", lambda: [])()
    run.step("answer", lambda context: "yes")(docs)

    loaded = store.load(run.run_id)
    by_label = {t.label: t for t in loaded.tasks}
    assert by_label["answer"].tainted_by == "retrieve"
    assert by_label["answer"].tainted_reason == "empty_sequence"
    assert by_label["answer"].outcome_status == "ok"


def test_taint_survives_replay_hydration(tmp_path):
    """`_pre_call` returns the cached result BY REFERENCE, so identity still
    holds on a resumed run and transitivity is not lost at the seam."""
    store = SQLiteStore(db_path=tmp_path / "papayya.db")
    run = _make_run(store=store)
    docs = run.step("retrieve", lambda: [])()
    assert docs == []

    resumed = _make_run(store=store, run_id=run.run_id)
    resumed.init()
    replayed = resumed.step("retrieve", lambda: ["should not run"])()
    resumed.step("answer", lambda context: "yes")(replayed)

    assert resumed._cache["answer"].tainted_by == "retrieve"


def test_taint_cannot_cross_items(tmp_path):
    """`_cache` belongs to a PapayyaRun and iterators mint one run per record,
    so isolation needs no scoping code — but it needs a test."""
    store = SQLiteStore(db_path=tmp_path / "papayya.db")
    dirty = _make_run(store=store)
    dirty.step("retrieve", lambda: [])()

    clean = _make_run(store=store)
    clean.step("answer", lambda context: "yes")([1, 2, 3])

    assert clean._cache["answer"].tainted_by is None


# --- regression pins: taint must not move anything else ------------------ #

def test_taint_does_not_arm_the_local_degraded_streak_fence(tmp_path):
    """The pin for plan 44's first blocking finding.

    `sqlite_store` runs its own K-fence (default 3) on
    `entry.outcome_status != "ok"`, independent of the control pane's. If a
    taint ever becomes a 'degraded' status, ONE benign empty fetch pauses the
    run two steps later — every time, in every customer's pipeline.
    """
    store = SQLiteStore(db_path=tmp_path / "papayya.db")
    run = _make_run(store=store)

    docs = run.step("retrieve", lambda: [])()          # 1 real degradation
    summary = run.step("summarize", lambda d: {"s": 1})(docs)   # taint
    run.step("answer", lambda s: "yes")(summary)                # taint

    assert run._cache["answer"].tainted_by == "retrieve"
    assert store.pending_pause(run.run_id) is None


def test_taint_does_not_move_the_run_rollups(tmp_path):
    """A tainted step must leave `worst_outcome_status` / `degraded_count`
    reading exactly as the single real degradation left them — the drift
    rates and cluster keys downstream are learned against those numbers."""
    store = SQLiteStore(db_path=tmp_path / "papayya.db")
    run = _make_run(store=store)

    docs = run.step("retrieve", lambda: [])()
    run.step("answer", lambda context: "yes")(docs)

    loaded = store.load(run.run_id)
    assert loaded.worst_outcome_status == "degraded"
    assert loaded.degraded_count == 1  # the fetch alone, not the answer


# --- the inspector, directly -------------------------------------------- #

def test_inspect_taint_is_inert_without_sources():
    assert outcomes.inspect_taint({"x": [1]}, []) == (None, None)
    assert outcomes.inspect_taint(None, []) == (None, None)
