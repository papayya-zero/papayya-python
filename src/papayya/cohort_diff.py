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
    # How many completed steps this re-drive inherited from its source rather
    # than re-executing (plan 58 U). The server computes it per record and the
    # CLI dropped it (plan 59 D5) in favour of a dollar figure that is $0.00 in
    # any unpriced project — and unpriced is not free.
    reused_steps: int = 0
    # Whether the re-drive produced a DIFFERENT ANSWER than the run it replaced
    # — plan 64 D3, the same axis `verify` reports and computed by the same
    # rule (the step trace, in `(attempt, completed_at, seq)` order).
    #
    # None is UNKNOWN: still running, or neither run wrote a checkpoint. It is
    # never "the answer did not move".
    output_changed: bool | None = None
    # Steps ON the re-drive, total. `after_steps - reused_steps` is what this
    # re-drive actually EXECUTED, which is the number plan 59 D2 measured at
    # "150 → 300 step executions to fix one line" and the number no surface
    # printed. Zero until the run is polled.
    after_steps: int = 0


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
    # The cohort-wide reuse sum, straight off the release manifest. The one
    # honest measure of what a re-drive cost: `papayya release` is the command
    # that multiplies the version gate by the size of the cohort, and before
    # plan 58 U this was always zero and never said so.
    reused_steps: int = 0

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

    @property
    def moved(self) -> int:
        """Records that produced a different answer than the run they replaced.

        THE ONLY COUNT THAT CAN BE NON-ZERO ON A COHORT A HUMAN FLAGGED. A
        `recovered` requires the inspectors to change their mind, and a
        customer fixing their own extraction logic does not change an
        inspector's mind — plan 59 D7 measured a re-drive that corrected page
        17 reported as `ok … still ok` inside a summary reading `0 recovered`.
        """
        return sum(1 for r in self.records if r.output_changed is True)

    @property
    def unmoved(self) -> int:
        """Terminal records this re-drive did NOT change. Excludes unknowns."""
        return sum(1 for r in self.records if r.output_changed is False)

    @property
    def executed_steps(self) -> int:
        """Steps this re-drive actually ran, cohort-wide.

        Total steps on the new runs minus what reuse handed them. This is the
        number plan 59 D2 put at "150 → 300 step executions to fix one line",
        and the one no surface printed: `release` was handed `reused_steps` and
        printed a dollar figure instead, which reads $0.00 in any unpriced
        project (plan 59 D5).

        Never negative: a re-drive that somehow carries fewer steps than reuse
        claims is a bug in one of them, and a negative count in a summary line
        would be the first anyone heard of it — clamped and left to the
        reused/total pair to expose.
        """
        return max(0, sum(r.after_steps for r in self.records) - self.reused_steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "released": len(self.records),
            "cohort_total": self.cohort_total,
            "counts": self.counts,
            "pending": self.pending,
            "before_cost_usd": self.before_cost_usd,
            "after_cost_usd": self.after_cost_usd,
            "moved": self.moved,
            "unmoved": self.unmoved,
            "reused_steps": self.reused_steps,
            "executed_steps": self.executed_steps,
            "skipped_not_terminal": self.skipped_not_terminal,
            "skipped_agent_missing": self.skipped_agent_missing,
            "cost_note": self.cost_note,
            "records": [asdict(r) for r in self.records],
        }


def record_verdict(status: str | None, worst_outcome: str | None) -> str:
    """Collapse a record's TWO verdict columns into the one an operator reads.

    ``status`` is the run's; ``worst_outcome_status`` is the roll-up of its
    steps. A record that raises before its first checkpoint has no step to
    inspect, so it keeps the ``ok`` it was born with while ``status`` says
    ``failed`` — the commonest way a first batch fails, and invisible to the
    step column alone. That was plan 48 R5.

    The Go twin is ``handler.recordVerdict`` and the SQL twin is
    ``store.cohortWhere``'s outcome switch. Three expressions of one
    definition, because one has to run in the database and one has to run
    against runs this SDK polls for itself.
    """
    if status == "failed":
        return "failed"
    if worst_outcome and worst_outcome != "ok":
        return worst_outcome
    return "ok"


def _verdict(before: str, after: str | None, run_status: str | None) -> str:
    """Classify one record.

    A re-drive that has not reached a terminal run status has NO verdict yet
    — `worst_outcome_status` on a queued run is its initial 'ok', and reading
    that as "recovered" would report a fix that has not run.

    ``before`` arrives ALREADY COLLAPSED: it is the server's ``source_outcome``,
    which the release handler puts through its own ``recordVerdict``. The after
    side collapses here, because these runs are polled directly from
    ``GET /v1/durable/runs/{id}`` and arrive as the two raw columns.

    Both sides have to ask the same question. They did not, and the asymmetry
    made ``recovered`` unreachable for a record that crashed: a real release of
    two crashed records onto a fixed version printed "0 recovered, 2 still ok"
    while both went from KeyError to a classification.
    """
    if run_status not in _TERMINAL or after is None:
        return PENDING
    was_ok = before == "ok"
    now_ok = record_verdict(run_status, after) == "ok"
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
        reused_steps=int(release.get("reused_steps") or 0),
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
        #
        # Collapsed, so the transition the CLI prints agrees with the verdict
        # printed beside it. The released cohort that exposed R5 printed
        # `NEWLY BROKEN … [ok -> ok]`: a verdict from one column, a transition
        # from another, contradicting each other on the same line.
        after = (
            None
            if verdict == PENDING
            else record_verdict(run_status, fresh.get("worst_outcome_status"))
        )
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
                reused_steps=int(member.get("reused_steps") or 0),
                output_changed=fresh.get("output_changed"),
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
        # Only becomes non-None once the run is terminal — the server refuses
        # to compare a partial trace against a complete one, because that
        # difference is about the clock rather than about the answer.
        rec.output_changed = fresh.get("output_changed")
        rec.after_steps = len(fresh.get("checkpoints") or [])
        rec.verdict = _verdict(
            rec.before_status, fresh.get("worst_outcome_status"), rec.after_run_status
        )
        # See diff_from_release: a still-queued run's outcome is not a verdict.
        rec.after_status = (
            None
            if rec.verdict == PENDING
            else record_verdict(rec.after_run_status, fresh.get("worst_outcome_status"))
        )
    return diff


__all__ = [
    "record_verdict",
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
