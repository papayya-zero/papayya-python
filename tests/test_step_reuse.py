"""Plan 58 U — a re-drive stops redoing work it already has.

`papayya replay` was never a replay. It mints a NEW run_id; `checkpoints` are
keyed by run_id; so the new run started on an empty namespace and every
`run.step` cache-missed. Nothing was skipped because nothing could be FOUND.
Measured on plan 57's workload: re-driving a 40-page document to fix one page
re-executed all 40 and wrote every one of them downstream a second time.

The server copies the source run's still-good completed steps onto the new run
id (CopyReusableCheckpoints — the policy is one SQL WHERE clause). These tests
are the other half: what the SDK is allowed to MATCH, which is where the
correctness lives.

The rule, and the reason it is not simply "carry the old run_id across":
a reused entry is matched on ``(item_id, label)`` and NEVER positionally.
`label#N` means what it means only while the loop runs from the top — page 17
is `read-photo#17` because the customer's `for page in range(...)` produced it
seventeenth. Carry that key across runs and a customer who filters or reorders
their pages gets page 17's text returned for page 22, silently, with the record
claiming success. Temporal buys alignment with a determinism sandbox; Papayya
buys it with a noun the customer already passes.
"""

from __future__ import annotations

import pytest

from papayya.durable.run import PapayyaRun
from papayya.durable.store import MemoryStore
from papayya.durable.types import DurableRunConfig, RunCheckpoint, TaskEntry

SOURCE_RUN = "run-source-0001"
REPLAY_RUN = "run-replay-0002"
DOC = "DOC-2001"


def _page_entry(page: int, *, reused: bool = True, text: str | None = None) -> TaskEntry:
    """One copied page checkpoint, as the server's INSERT ... SELECT writes it."""
    return TaskEntry(
        label="read-photo" if page == 1 else f"read-photo#{page}",
        result={"page": page, "text": text if text is not None else f"page {page} body text"},
        duration_ms=0,
        completed_at="2026-08-19T17:00:00Z",
        item_id=f"{DOC}#photo-{page}",
        reused_from=SOURCE_RUN if reused else None,
    )


def _replay(entries: list[TaskEntry], *, run_item_id: str | None = DOC) -> tuple[PapayyaRun, MemoryStore]:
    """A replay run whose checkpoints were pre-copied — the hosted shape.

    The hosted path ALWAYS looks like this: the submission pre-creates the
    durable_run row, so `store.load(run_id)` returns non-None and `init()` takes
    the resume branch. That is why U rides the ordinary hydration path and not
    `prepopulated_tasks`, which only the local (`--from-step`) replay ever
    reached and which is unreachable once the row exists.
    """
    store = MemoryStore()
    store.create(RunCheckpoint(
        run_id=REPLAY_RUN, agent="docproc", tasks=list(entries),
        status="running", item_id=run_item_id,
    ))
    run = PapayyaRun(DurableRunConfig(
        agent="docproc", run_id=REPLAY_RUN, store=store, item_id=run_item_id))
    return run, store


# ── The headline ──────────────────────────────────────────────────────────

def test_fixing_one_page_executes_one_page() -> None:
    """Plan 58's gate, in a unit test: SIDEEFFECT 40 -> 1, not 40 -> 40.

    Pages 1-16 and 18-40 were copied from the source run. Page 17 was not (it
    was the one that failed, so the server's `outcome_status = 'ok'` clause
    excluded it). Only page 17 may execute.
    """
    copied = [_page_entry(p) for p in range(1, 41) if p != 17]
    run, _ = _replay(copied)

    executed: list[int] = []

    def ocr(page: int) -> dict:
        executed.append(page)
        return {"page": page, "text": f"page {page} body text"}

    results = [
        run.step("read-photo", ocr, item_id=f"{DOC}#photo-{p}")(p)
        for p in range(1, 41)
    ]

    assert executed == [17], "a re-drive to fix page 17 must execute page 17 and nothing else"
    assert len(results) == 40, "and the document still has all forty pages"
    assert results[4]["page"] == 5, "each reused page returns its OWN result"


# ── Why the key is (item_id, label) and not the position ──────────────────

def test_a_reordered_loop_does_not_get_its_neighbours_answer() -> None:
    """The bug the positional key would have shipped.

    The customer re-drives with their pages in a different order — sorted
    differently, filtered, whatever. Positionally, `read-photo#2` is now page
    40. If the reused entry were reachable by position, page 40 would silently
    return page 2's text and the record would call it a success. This is the
    single reason U keys on the item and not the suffix.
    """
    copied = [_page_entry(p) for p in range(1, 41)]
    run, _ = _replay(copied)

    for page in [1, 40, 2, 39]:
        got = run.step("read-photo", lambda p: {"page": p, "text": "FRESH"},
                       item_id=f"{DOC}#photo-{page}")(page)
        assert got["page"] == page, (
            f"reused entry for page {page} came back as page {got['page']} — "
            "the positional key leaked across runs"
        )


def test_a_reused_row_is_unreachable_positionally() -> None:
    """Same label, an item the source run never saw: MISS, and it executes.

    The positional lookup still exists and still serves this run's own prior
    work (a resume, a re-lease). It must not be a second way to reach a reused
    row, or tier 1's guarantee is decorative.
    """
    run, _ = _replay([_page_entry(1)])
    executed: list[str] = []

    def ocr(page: int) -> dict:
        executed.append(f"page-{page}")
        return {"page": page, "text": "FRESH"}

    # Position 1 — exactly where the copied `read-photo` row sits — but a
    # different record.
    got = run.step("read-photo", ocr, item_id=f"{DOC}#photo-99")(99)

    assert executed == ["page-99"]
    assert got["text"] == "FRESH"


def test_a_step_with_no_item_id_reuses_nothing() -> None:
    """No item, no key, no reuse — and the degradation is to TODAY'S behaviour.

    A run with no item_id anywhere cannot form the reuse key, so every step
    executes exactly as it did before plan 58 U. That is the whole safety
    posture of this unit: every exclusion degrades to re-executing, never to a
    wrong answer.
    """
    orphan = TaskEntry(label="fetch", result={"v": "OLD"}, duration_ms=0,
                       completed_at="2026-08-19T17:00:00Z",
                       item_id=None, reused_from=SOURCE_RUN)
    run, _ = _replay([orphan], run_item_id=None)

    got = run.step("fetch", lambda: {"v": "FRESH"})()
    assert got["v"] == "FRESH"


# ── The aggregate gate: the silently-wrong case ───────────────────────────

def test_an_aggregate_step_is_reused_when_nothing_ran() -> None:
    """A clean re-drive reuses the verdict too — nothing under it changed."""
    copied = [_page_entry(p) for p in range(1, 4)]
    copied.append(TaskEntry(
        label="validate", result={"ok": True, "reason": ""}, duration_ms=0,
        completed_at="2026-08-19T17:00:00Z", item_id=DOC, reused_from=SOURCE_RUN))
    run, _ = _replay(copied)

    for p in range(1, 4):
        run.step("read-photo", lambda x: {"page": x, "text": "FRESH"},
                 item_id=f"{DOC}#photo-{p}")(p)
    verdict = run.step("validate", lambda: {"ok": False, "reason": "RECOMPUTED"})()

    assert verdict["ok"] is True, "nothing executed, so the aggregate still holds"


def test_an_aggregate_step_re_runs_once_anything_under_it_did() -> None:
    """The headline safety rule, and the one no SQL clause could enforce.

    `validate` read forty findings. If ANY page re-executed, its verdict is
    stale — reusing it returns yesterday's answer for today's document, and
    reports success. That is exactly the failure class this product exists to
    catch, so an aggregate is reusable only while nothing has run.

    An aggregate is recognised structurally: its item_id is the RUN's own id
    rather than a sub-record's. `validate` sits under DOC-2001; `read-photo`
    sits under DOC-2001#photo-17. The customer already draws that line and does
    not have to be told they are drawing it.
    """
    copied = [_page_entry(p) for p in range(1, 4) if p != 2]
    copied.append(TaskEntry(
        label="validate", result={"ok": True, "reason": ""}, duration_ms=0,
        completed_at="2026-08-19T17:00:00Z", item_id=DOC, reused_from=SOURCE_RUN))
    run, _ = _replay(copied)

    for p in range(1, 4):
        run.step("read-photo", lambda x: {"page": x, "text": "FRESH"},
                 item_id=f"{DOC}#photo-{p}")(p)
    verdict = run.step("validate", lambda: {"ok": False, "reason": "page 2 is blank"})()

    assert verdict["reason"] == "page 2 is blank", (
        "page 2 re-executed under it, so the reused verdict is stale and must "
        "not be returned"
    )


def test_a_step_that_raised_still_counts_as_having_run() -> None:
    """An exception is a change like any other.

    The flag is set BEFORE the call, not after, so an aggregate that follows a
    step which raised is not reused either. Getting this backwards would reuse
    a verdict computed over work that has since failed.
    """
    copied = [TaskEntry(
        label="validate", result={"ok": True, "reason": ""}, duration_ms=0,
        completed_at="2026-08-19T17:00:00Z", item_id=DOC, reused_from=SOURCE_RUN)]
    run, _ = _replay(copied)

    def boom(_: int) -> dict:
        raise RuntimeError("ocr sidecar returned 503")

    with pytest.raises(RuntimeError):
        run.step("read-photo", boom, item_id=f"{DOC}#photo-1", retries=0)(1)

    verdict = run.step("validate", lambda: {"ok": False, "reason": "RECOMPUTED"})()
    assert verdict["reason"] == "RECOMPUTED"


# ── No regression on the paths that already worked ────────────────────────

def test_a_runs_own_prior_work_still_resumes_positionally() -> None:
    """Not every hydrated row is a reused one.

    A resume, a re-lease and a redelivery all re-find this run's OWN
    checkpoints, positionally, and that behaviour predates U and is already
    correct. Reused rows are indexed separately precisely so this path is
    untouched — including for a step that names no item at all.
    """
    own = TaskEntry(label="classify", result="photo_report", duration_ms=3,
                    completed_at="2026-08-19T17:00:00Z", item_id=DOC,
                    reused_from=None)
    run, _ = _replay([own])

    got = run.step("classify", lambda: "RECOMPUTED")()
    assert got == "photo_report", "this run already did this step; it must not run again"
