"""Plan 41 R7 step 4 — a resumed run re-executes what the fence objected to.

R6 made a paused run resumable. It resumed **on the exact output it paused
over**: ``_pre_call`` returns the cached result for any hydrated label with no
outcome filter, so the degraded steps the fence stopped the run for were cache
hits and never ran again. The operator fixed the prompt, resumed, and got the
same run back.

The set of executions to re-run is recorded server-side at pause time and
arrives on the loaded ``RunCheckpoint``. These tests pin the client half: the
filter, the seam R3 built for it, and the property that makes the set spend
itself exactly once.
"""

from __future__ import annotations

from datetime import datetime, timezone

from papayya.durable.run import PapayyaRun
from papayya.durable.store import MemoryStore
from papayya.durable.types import DurableRunConfig, RunCheckpoint, TaskEntry


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _entry(label: str, result, attempt: int = 1, outcome: str = "ok") -> TaskEntry:
    return TaskEntry(
        label=label,
        result=result,
        duration_ms=1,
        completed_at=_now(),
        attempt=attempt,
        outcome_status=outcome,
    )


class _LoadedStore(MemoryStore):
    """A store whose ``load`` returns a checkpoint we control completely,
    including ``pause_invalidated`` — which is what the real CloudStore reads
    off the run GET after a resume."""

    def __init__(self, tasks, invalidated):
        super().__init__()
        self._loaded = RunCheckpoint(
            run_id="run", agent="a", tasks=tasks, status="running",
            created_at=_now(), updated_at=_now(),
            pause_invalidated=invalidated,
        )
        self.saved: list[TaskEntry] = []

    def load(self, run_id):  # type: ignore[override]
        return self._loaded

    def save_task(self, run_id, entry):  # type: ignore[override]
        self.saved.append(entry)


def _run(tasks, invalidated):
    store = _LoadedStore(tasks, invalidated)
    run = PapayyaRun(DurableRunConfig(agent="a", run_id="run", store=store))
    run.init()
    return run, store


def test_an_invalidated_step_re_executes():
    """The bug, fixed. Without the filter `draft` is a cache hit and the
    operator's fix never runs."""
    run, store = _run(
        tasks=[_entry("keep", "kept"), _entry("draft", "", outcome="degraded")],
        invalidated=[("draft", 1)],
    )

    calls = []

    def body(label, value):
        calls.append(label)
        return value

    assert run.step("keep", lambda: body("keep", "recomputed"))() == "kept", (
        "an untouched step must still be a cache hit — R7 invalidates the "
        "fence's steps, not the whole run"
    )
    assert run.step("draft", lambda: body("draft", "better"))() == "better", (
        "the invalidated step returned its CACHED result — it did not re-execute"
    )
    assert calls == ["draft"]


def test_a_step_that_already_re_executed_is_not_invalidated_again():
    """W2, and the reason the set carries the attempt.

    The set has to survive the resume — the SDK reads it on the load() that
    happens after. If it were labels only it could never be cleared safely,
    and a LATER unrelated recovery (a lease reap, a worker death) would
    re-execute these steps a second time and bill the provider for it.

    The recorded pair was attempt 1. The step has since run again, so the
    stored row is attempt 2, and it no longer matches.
    """
    run, _ = _run(
        tasks=[_entry("draft", "already-rerun", attempt=2)],
        invalidated=[("draft", 1)],
    )
    calls = []
    result = run.step("draft", lambda: calls.append("draft") or "third-run")()

    assert result == "already-rerun", (
        "a step that already re-executed was invalidated AGAIN — the set is "
        "not spending itself, and every later crash re-runs it"
    )
    assert calls == []


def test_attempts_are_seeded_from_the_raw_list_not_the_filtered_cache():
    """R3's contract with R7, now driven through the real path rather than by
    popping the cache by hand.

    ``_seed_attempts`` reads the RAW loaded list, before and independently of
    the invalidation filter. If it read the filtered cache instead, an
    invalidated label would have no entry, the next attempt would compute as 1,
    and the re-execution would reuse the recorded attempt's provider
    idempotency key — the exact bug R3 exists to fix, on the exact path R3 was
    built for.
    """
    run, _ = _run(
        tasks=[_entry("draft", "", outcome="degraded")],
        invalidated=[("draft", 1)],
    )
    assert "draft" not in run._cache, "the entry should have been invalidated"
    assert run.idempotency_key("draft") == "run:draft:2", (
        "the re-execution would have reused attempt 1's provider key — the "
        "provider replays its stored response and the fix never reaches it"
    )


def test_the_call_order_lists_a_step_only_once_it_has_run():
    """An invalidated entry is skipped in `_task_call_order` too. The step
    appends itself when it re-executes, so the order stays honest instead of
    listing a step that has not run."""
    run, _ = _run(
        tasks=[_entry("a", "A"), _entry("b", "", outcome="degraded")],
        invalidated=[("b", 1)],
    )
    assert run._task_call_order == ["a"]
    run.step("a", lambda: "A2")()
    run.step("b", lambda: "B2")()
    assert run._task_call_order == ["a", "b"]


def test_an_empty_set_changes_nothing():
    """A budget pause records nothing, and a run that never paused has nothing
    recorded. Both must hydrate exactly as before R7."""
    run, _ = _run(
        tasks=[_entry("draft", "", outcome="degraded")],
        invalidated=[],
    )
    calls = []
    assert run.step("draft", lambda: calls.append(1) or "new")() == ""
    assert calls == []


def test_a_control_pane_predating_r7_is_a_no_op():
    """An older control pane sends no `pause_invalidated` key, and the SDK
    must resume the way that binary's own resume expects."""
    store = MemoryStore()
    store.create(RunCheckpoint(
        run_id="run", agent="a", tasks=[_entry("draft", "cached")],
        status="running", created_at=_now(), updated_at=_now(),
    ))
    run = PapayyaRun(DurableRunConfig(agent="a", run_id="run", store=store))
    run.init()
    assert run.step("draft", lambda: "new")() == "cached"


# ─── W9: the run-scoped fence kwarg is not silent ────────────────────

def test_pause_after_degraded_warns_when_the_store_cannot_honour_it(caplog):
    """`papayya().run(..., pause_after_degraded=5)` reaches a duck-typed
    `set_run_fence` that only SQLiteStore implements. CloudStore has none, and
    hosted is the only execution path — so on the path that matters the kwarg
    has always done nothing, and said nothing.

    Measured before this changed: zero log output at DEBUG. A kwarg that
    quietly does nothing is the failure class this product exists to remove.
    """
    import logging

    with caplog.at_level(logging.WARNING, logger="papayya.durable"):
        run = PapayyaRun(DurableRunConfig(
            agent="a", run_id="r", store=MemoryStore(), pause_after_degraded=5,
        ))
        run.init()

    assert caplog.records, "the ignored kwarg was silent"
    msg = caplog.records[-1].getMessage()
    assert "IGNORED" in msg
    assert "pause_after_degraded" in msg
    # It must name the control that DOES work, not just complain.
    assert "agents update" in msg


def test_a_store_that_honours_it_does_not_warn(caplog):
    import logging

    class _FenceStore(MemoryStore):
        def __init__(self):
            super().__init__()
            self.fence = None

        def set_run_fence(self, run_id, pause_after_degraded):
            self.fence = (run_id, pause_after_degraded)

    store = _FenceStore()
    with caplog.at_level(logging.WARNING, logger="papayya.durable"):
        run = PapayyaRun(DurableRunConfig(
            agent="a", run_id="r", store=store, pause_after_degraded=5,
        ))
        run.init()

    assert store.fence == ("r", 5)
    assert not caplog.records
