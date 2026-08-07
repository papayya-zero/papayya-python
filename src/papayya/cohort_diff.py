"""The cohort diff — did the fix work, on which records, at what cost.

Plan 41 R4 C7; ADR 0009 D7. This is the answer `release` exists to give, and
the reason the whole loop is worth building: every other tool in the corpus
answers *"did the fix work?"* with **"I re-ran it and I think it's better."**

The per-record comparison is a pair of facts the platform already holds — the
source record's verdict and cost, and the re-driven record's — so the diff is
arithmetic over the release manifest rather than a new engine. The one column
that earns it is **newly broken**:

    An operator who re-drives 500 records after a fix wants to know the fix
    did not break 30 that were fine.

A note on the cost columns: they are ESTIMATES (tokens x the customer's own
project rate card, Plan 41 R4 C9), not bills. Every renderer of this data
carries the caveat the server sends on `cost_note`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Iterable


# Per-record verdicts. The first four are the answer; PENDING is the honest
# state of a record whose re-drive has not finished, and it is never folded
# into one of the others.
RECOVERED = "recovered"
STILL_NOT_OK = "still_not_ok"
NEWLY_BROKEN = "newly_broken"
STILL_OK = "still_ok"
PENDING = "pending"

_TERMINAL = ("completed", "failed", "quarantined")


@dataclass
class RecordDiff:
    """One record, before and after the re-drive."""

    source_id: str
    new_run_id: str
    item_id: str | None
    agent: str
    before_status: str
    after_status: str | None
    after_run_status: str | None
    before_cost_usd: float
    after_cost_usd: float
    verdict: str
    agent_version: str | None = None


@dataclass
class CohortDiff:
    """The roll-up. This is what an operator reads."""

    records: list[RecordDiff] = field(default_factory=list)
    # Carried straight through from the server so the caveat cannot be lost
    # in the middle of the pipeline.
    cost_note: str = ""
    # Named skips from the release manifest — they are part of the answer
    # ("40 selected, 38 released") and folding them away would make a
    # partial release read as a whole one.
    skipped_not_terminal: int = 0
    skipped_agent_missing: int = 0
    cohort_total: int = 0

    def count(self, verdict: str) -> int:
        return sum(1 for r in self.records if r.verdict == verdict)

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.records:
            out[r.verdict] = out.get(r.verdict, 0) + 1
        return out

    @property
    def before_cost_usd(self) -> float:
        return sum(r.before_cost_usd for r in self.records)

    @property
    def after_cost_usd(self) -> float:
        return sum(r.after_cost_usd for r in self.records)

    @property
    def pending(self) -> int:
        return self.count(PENDING)

    def to_dict(self) -> dict[str, Any]:
        return {
            "released": len(self.records),
            "cohort_total": self.cohort_total,
            "counts": self.counts,
            "pending": self.pending,
            "before_cost_usd": self.before_cost_usd,
            "after_cost_usd": self.after_cost_usd,
            "skipped_not_terminal": self.skipped_not_terminal,
            "skipped_agent_missing": self.skipped_agent_missing,
            "cost_note": self.cost_note,
            "records": [asdict(r) for r in self.records],
        }


def _verdict(before: str, after: str | None, run_status: str | None) -> str:
    """Classify one record.

    A re-drive that has not reached a terminal run status has NO verdict yet
    — `worst_outcome_status` on a queued run is its initial 'ok', and reading
    that as "recovered" would report a fix that has not run.
    """
    if run_status not in _TERMINAL or after is None:
        return PENDING
    was_ok = before == "ok"
    now_ok = after == "ok" and run_status != "failed"
    if was_ok:
        return STILL_OK if now_ok else NEWLY_BROKEN
    return RECOVERED if now_ok else STILL_NOT_OK


def diff_from_release(
    release: dict[str, Any],
    new_runs: dict[str, dict[str, Any]] | None = None,
) -> CohortDiff:
    """Build the diff from a release manifest plus the re-driven runs.

    ``release`` is the response of ``POST /v1/durable/cohorts/replay``; it
    already carries each source's outcome and cost, which is the "before"
    half. ``new_runs`` maps a new run id to that run as fetched from
    ``GET /v1/durable/runs/{id}`` — the "after" half. Omitting a run leaves
    its record PENDING, which is the correct reading of "we have not looked
    yet" and of "it has not finished".
    """
    new_runs = new_runs or {}
    out = CohortDiff(
        cost_note=release.get("cost_note", ""),
        skipped_not_terminal=release.get("skipped_not_terminal", 0),
        skipped_agent_missing=release.get("skipped_agent_missing", 0),
        cohort_total=release.get("cohort_total", 0),
    )
    for member in release.get("members", []):
        new_id = member.get("new_run_id", "")
        fresh = new_runs.get(new_id) or {}
        run_status = fresh.get("status")
        before = member.get("source_outcome", "ok")
        verdict = _verdict(before, fresh.get("worst_outcome_status"), run_status)
        # A pending record has NO after-verdict. `worst_outcome_status` on a
        # queued run is its initial 'ok', and rendering "degraded -> ok" next
        # to "still running" reads as a recovery that has not happened.
        after = None if verdict == PENDING else fresh.get("worst_outcome_status")
        out.records.append(
            RecordDiff(
                source_id=member.get("source_id", ""),
                new_run_id=new_id,
                item_id=member.get("item_id"),
                agent=member.get("agent", ""),
                agent_version=member.get("agent_version"),
                before_status=before,
                after_status=after,
                after_run_status=run_status,
                before_cost_usd=float(member.get("source_cost_usd") or 0.0),
                after_cost_usd=float(fresh.get("budget_consumed_usd") or 0.0),
                verdict=verdict,
            )
        )
    return out


def pending_run_ids(diff: CohortDiff) -> list[str]:
    """New run ids still awaiting a terminal status — what a poller re-reads."""
    return [r.new_run_id for r in diff.records if r.verdict == PENDING]


def merge_runs(
    diff: CohortDiff, new_runs: Iterable[tuple[str, dict[str, Any]]]
) -> CohortDiff:
    """Fold freshly-polled runs into an existing diff, in place."""
    fetched = dict(new_runs)
    for rec in diff.records:
        fresh = fetched.get(rec.new_run_id)
        if fresh is None:
            continue
        rec.after_run_status = fresh.get("status")
        rec.after_cost_usd = float(fresh.get("budget_consumed_usd") or 0.0)
        rec.verdict = _verdict(
            rec.before_status, fresh.get("worst_outcome_status"), rec.after_run_status
        )
        # See diff_from_release: a still-queued run's outcome is not a verdict.
        rec.after_status = (
            None if rec.verdict == PENDING else fresh.get("worst_outcome_status")
        )
    return diff


__all__ = [
    "RECOVERED",
    "STILL_NOT_OK",
    "NEWLY_BROKEN",
    "STILL_OK",
    "PENDING",
    "RecordDiff",
    "CohortDiff",
    "diff_from_release",
    "merge_runs",
    "pending_run_ids",
]
