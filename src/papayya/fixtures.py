"""Cohort fixtures — a production incident, materialized as files on disk.

Plan 41 R4; ADR 0009 D7b, which argues this is the verb that pays first:

    The cohort's records already carry input, the real bad output, verdict,
    tenant and timestamp — which is a fixture set. Written to disk, the
    production failure becomes reproducible locally, offline, deterministically,
    at zero API spend: fix against it, verify against it before spending a cent
    of inference, then re-drive only that cohort. The fixture then stays in the
    customer's CI forever, so the shape cannot ship again.

Two properties make this worth building early rather than last: it is the only
capability on that board **worth something on incident one with no retained
history**, and it turns each incident into a permanent, zero-maintenance
regression case — the accumulation of ADR 0009 D3 arriving as files rather than
as distributions.

**Where the input comes from, and why it is two places.** A fixture's `input` is
the thing the agent was asked to process, and the platform stores it differently
on each path:

* Local (SQLite ledger): the `items` table has its own ``input_snapshot``,
  written at the item boundary by ``papayya.iter``.
* Hosted: ``durable_runs.input`` is the submitted request, verbatim (migration
  089). It is what the worker hands the ``@agent`` function and what a replay
  re-executes, so a fixture built from it re-runs the real thing.
* Hosted, before 089: there **was** no input column. The worker invoked
  ``fn(item_id)`` and the request was never persisted, so the closest recorded
  thing was the ``input_snapshot`` of the item's FIRST checkpoint — the bound
  call args of its first durable step. Records from that era still resolve this
  way, and a bare ``@agent`` that never checkpointed has nothing at all.

The third is a *reconstruction*, not the original request, and a fixture that
quietly claimed otherwise would be the same class of lie plan 41 R2 spent a step
removing. So every fixture records ``input_source`` and every consumer can say
what it is replaying.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# Bumped only on a breaking change to the on-disk shape. Fixtures are meant to
# live in a customer's repo for years — a reader must be able to say "I don't
# understand this file" rather than silently mis-read a newer one.
FIXTURE_VERSION = 2

# Where the input came from. Named on every fixture rather than inferred.
INPUT_FROM_ITEM = "item_snapshot"
# The submitted request itself (durable_runs.input, migration 089) — the only
# source that is the original rather than a reconstruction of it.
INPUT_FROM_SUBMISSION = "submitted_input"
INPUT_FROM_FIRST_CHECKPOINT = "first_checkpoint_snapshot"
INPUT_MISSING = "none"

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FixtureStep:
    """One recorded step execution inside a fixture's trace."""

    label: str
    attempt: int = 1
    outcome_status: str = "ok"
    outcome_reason: str | None = None
    duration_ms: int = 0
    cost_usd: float = 0.0
    input_snapshot: Any = None
    output_snapshot: Any = None
    result: Any = None


@dataclass
class Fixture:
    """One record's production failure, reproducible offline.

    Carries the whole step trace rather than only the item's boundary. The
    boundary is all the acceptance sentence needs, but D7b's argument is that
    the fixture becomes a permanent CI case — and a boundary-only fixture
    cannot catch a regression that changes *which steps run*.
    """

    item_id: str | None
    record_id: str
    agent: str
    input: Any
    input_source: str
    recorded_output: Any
    worst_outcome_status: str = "ok"
    # The RUN's status — 'failed' / 'completed' — as distinct from the step
    # verdict above. A record that raised before its first checkpoint keeps
    # worst_outcome_status='ok' forever, so a fixture carrying only that
    # column described three crashes as healthy and `verify` duly reported
    # `[ok -> ok] · 0 fixed` on the fix that repaired them (plan 48 R2/R4).
    status: str = "completed"
    # Why it failed, one line: "KeyError: 'body'". None on anything that did
    # not fail. This is what makes a pulled fixture answer "what am I fixing?"
    # offline, which is the entire point of pulling one.
    error: str | None = None
    error_category: str | None = None
    agent_version: str | None = None
    partition_key: str | None = None
    created_at: str | None = None
    steps: list[FixtureStep] = field(default_factory=list)
    # The predicate that selected this record, carried so a fixture set is
    # self-describing months later: "which incident was this?"
    cohort: dict[str, Any] = field(default_factory=dict)
    papayya_fixture_version: int = FIXTURE_VERSION
    pulled_at: str = field(default_factory=_utcnow_iso)

    # --- serialization ------------------------------------------------- #

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=False, default=str)

    @classmethod
    def from_json(cls, raw: str) -> "Fixture":
        data = json.loads(raw)
        version = data.get("papayya_fixture_version")
        # OLDER fixtures load; NEWER ones do not.
        #
        # This was a strict `!=`, which is right for a removal and wrong for
        # an addition — and plan 50 is an addition. Under the strict gate,
        # adding `status` and `error` would have invalidated every fixture
        # already on disk with "Re-pull the cohort", i.e. broken exactly the
        # use case fixtures exist for: a permanent, committed CI case (ADR
        # 0009 D7b). A field this SDK has a default for is a field it can
        # read without one.
        #
        # The forward direction stays hard, because there the missing fields
        # are genuinely unknown — a newer SDK's file may encode meaning this
        # one has no way to honour, and guessing would be the silent-wrong
        # answer this product exists to catch.
        if not isinstance(version, int) or version > FIXTURE_VERSION:
            raise FixtureError(
                f"fixture is version {version!r}, this SDK reads up to version "
                f"{FIXTURE_VERSION}. Upgrade the SDK to read it."
            )
        steps = [FixtureStep(**s) for s in data.pop("steps", [])]
        # Drop keys this SDK does not know rather than TypeError on them: a
        # forward-compatible read is only useful if it actually reads.
        known = {f.name for f in fields(cls)}
        return cls(steps=steps, **{k: v for k, v in data.items() if k in known})

    def filename(self) -> str:
        """Stable, filesystem-safe name for this fixture.

        Keyed on the CUSTOMER's item_id when there is one, because that is the
        name the failure has in their world ("co_007.json" is greppable; a
        surrogate uuid is not). Falls back to the record id.
        """
        base = self.item_id or self.record_id
        safe = _SAFE_NAME.sub("_", str(base)).strip("_") or "item"
        # Disambiguate: two records can share one customer item_id (a retry
        # of the same record is a second row). Suffix with a short record
        # discriminator so a pull never silently overwrites its own output.
        return f"{safe}.{str(self.record_id)[:8]}.json"


class FixtureError(Exception):
    """Raised for an unreadable or unusable fixture."""


# --- assembly ---------------------------------------------------------- #


def _first_and_last(checkpoints: list[dict[str, Any]]) -> tuple[dict | None, dict | None]:
    """First and last checkpoint by EXECUTION order.

    Ordered on ``(attempt, completed_at, seq)`` — plan 41 R3's ordering, reused
    for the same reason it exists there: ``seq`` is assigned at INSERT, so a
    write journaled during an outage drains late and lands the OLDER execution
    at the HIGHER seq. Ordering on it alone hands back the stale execution.
    """
    if not checkpoints:
        return None, None

    def key(c: dict[str, Any]) -> tuple:
        return (
            c.get("attempt") or 1,
            str(c.get("completed_at") or c.get("created_at") or ""),
            c.get("seq") or 0,
        )

    ordered = sorted(checkpoints, key=key)
    return ordered[0], ordered[-1]


def fixture_from_record(
    record: dict[str, Any],
    checkpoints: list[dict[str, Any]] | None = None,
    *,
    cohort: dict[str, Any] | None = None,
) -> Fixture:
    """Build a fixture from a cohort member plus its recorded steps.

    ``record`` is a cohort member as returned by ``GET /v1/durable/cohorts``
    (hosted) or a row off the local ledger. ``checkpoints`` is that record's
    step rows; an empty list yields a boundary-only fixture rather than an
    error, because an item that failed before its first checkpoint is exactly
    the kind of thing an operator wants in the cohort.
    """
    checkpoints = checkpoints or []
    first, last = _first_and_last(checkpoints)

    # Input, best source first: what was actually submitted (hosted, 089+),
    # then the local ledger's own snapshot, then the first step's bound args —
    # a reconstruction, and named as one.
    #
    # `"input" in record` rather than a truthiness test: a submitted input is
    # legitimately null, 0, "" or [], and every one of them is falsy.
    if "input" in record:
        raw_input, source = record["input"], INPUT_FROM_SUBMISSION
    elif record.get("input_snapshot") is not None:
        raw_input, source = record["input_snapshot"], INPUT_FROM_ITEM
    elif first is not None and first.get("input_snapshot") is not None:
        raw_input, source = first["input_snapshot"], INPUT_FROM_FIRST_CHECKPOINT
    else:
        raw_input, source = None, INPUT_MISSING

    # Recorded output: the item's final output when present; else the last
    # step's snapshot, which is the closest thing to "what it produced".
    recorded = record.get("output")
    if recorded is None and last is not None:
        recorded = last.get("output_snapshot", last.get("result"))

    steps = [
        FixtureStep(
            label=c.get("label", ""),
            attempt=c.get("attempt") or 1,
            outcome_status=c.get("outcome_status") or "ok",
            outcome_reason=c.get("outcome_reason"),
            duration_ms=c.get("duration_ms") or 0,
            cost_usd=c.get("cost_usd") or 0.0,
            input_snapshot=c.get("input_snapshot"),
            output_snapshot=c.get("output_snapshot"),
            result=c.get("result"),
        )
        for c in sorted(
            checkpoints,
            key=lambda c: (
                c.get("attempt") or 1,
                str(c.get("completed_at") or c.get("created_at") or ""),
                c.get("seq") or 0,
            ),
        )
    ]

    return Fixture(
        item_id=record.get("item_id"),
        record_id=record.get("id") or record.get("run_id") or "",
        agent=record.get("agent", ""),
        input=raw_input,
        input_source=source,
        recorded_output=recorded,
        worst_outcome_status=record.get("worst_outcome_status") or "ok",
        status=record.get("status") or "completed",
        error=record.get("error"),
        error_category=record.get("error_category"),
        agent_version=record.get("agent_version"),
        partition_key=record.get("partition_key"),
        created_at=str(record.get("created_at")) if record.get("created_at") else None,
        steps=steps,
        cohort=cohort or {},
    )


# --- disk -------------------------------------------------------------- #


def write_fixtures(fixtures: Iterable[Fixture], out_dir: str | Path) -> list[Path]:
    """Write fixtures into ``out_dir``, one file each. Returns the paths.

    Atomic per file (temp + replace) so an interrupted pull leaves whole
    fixtures rather than truncated JSON — a half-written fixture that still
    parses is worse than a missing one.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fx in fixtures:
        target = out / fx.filename()
        fd, tmp = tempfile.mkstemp(dir=str(out), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(fx.to_json())
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        written.append(target)
    return written


def read_fixtures(path: str | Path) -> list[Fixture]:
    """Read a fixture file, or every ``*.json`` fixture in a directory."""
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.json"))
    elif p.exists():
        files = [p]
    else:
        raise FixtureError(f"no fixture path at {p}")

    out: list[Fixture] = []
    for f in files:
        try:
            out.append(Fixture.from_json(f.read_text(encoding="utf-8")))
        except FixtureError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise FixtureError(f"{f}: {exc}") from exc
    if not out:
        raise FixtureError(f"no fixtures found in {p}")
    return out


__all__ = [
    "FIXTURE_VERSION",
    "INPUT_FROM_ITEM",
    "INPUT_FROM_SUBMISSION",
    "INPUT_FROM_FIRST_CHECKPOINT",
    "INPUT_MISSING",
    "Fixture",
    "FixtureStep",
    "FixtureError",
    "fixture_from_record",
    "write_fixtures",
    "read_fixtures",
]
