"""Offline verification — run the patched function against pulled fixtures.

Plan 41 R4 C8; ADR 0009 D7b. This is the middle verb of the recovery loop::

    papayya pull    --agent enrich --tenant acme --since …   # incident → disk
    papayya verify  --fixtures ./fixtures                    # fix → prove it
    papayya release --agent enrich --tenant acme --since …   # re-drive it

``verify`` answers *"would the fix have worked?"* before a single record is
re-driven in production. It reads fixtures from disk, resolves the customer's
callable the same way :func:`papayya.durable._replay.replay` does, runs it over
each fixture's recorded input, and re-derives the verdict with the **same
inspector pipeline production used** — so "fixed" here means what "ok" means in
the dashboard, not a second opinion computed by different code.

**What it does not touch.** No control-plane call, no API key, no durable store,
nothing written to disk, no `usage_events`, no checkpoint rows anywhere a human
or a bill can see. The one concession is a
:class:`~papayya.durable.store.MemoryStore`: the verdict pipeline lives in
``Item``'s step wrapper, so re-deriving a verdict needs a run to hang steps off.
That store is a dict in this process and is discarded on return. The alternative
— re-implementing ``inspect_result`` over the function's return value — would
produce a verdict from *different code than production ran*, which is the exact
failure this product exists to catch.

**The caveat, stated rather than implied (C8).** ``verify`` stops *Papayya*
from spending and stops the re-drive from touching production. Every claim
above is about Papayya, and none of them is a claim about the customer's own
code: verify runs their function in this process, so **every side effect that
function has still fires, once per fixture** — the database write, the queue
publish, the email, the call to their LLM provider. Plan 59 D4 measured it at
190 downstream writes and twenty-one minutes for a two-fixture cohort, from a
command whose help then opened with the word "offline".

The exposure is not only the bill. A command sold as offline wrote rows into a
claims database whose owner had been told nothing was sent. Naming the tokens
and not the writes named the smaller of the two.

**Two deliberate divergences from ``replay``**, both because verify's purpose is
the opposite of replay's:

* **No version gate.** ``replay`` refuses when the captured ``agent_version``
  differs from the registration's, because re-driving on different code than the
  original silently changes what happened. ``verify`` exists *precisely* to run
  changed code against an old failure — gating on the mismatch would refuse
  every real use. The shift is **reported** per fixture
  (``recorded_agent_version`` → ``current_agent_version``) instead of refused.
* **A run id derived from the fixture, not minted fresh.** ``uuid5`` over the
  record id, so re-running verify over the same fixtures makes the same
  decisions — including :func:`papayya.checks.run_checks`' per-run sampling gate,
  which keys on ``run_id``. D7b's claim is that a fixture is a permanent CI case;
  a case whose sampled checks fire on a different subset every run is not one.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

from papayya.cohort_diff import record_verdict
from papayya.fixtures import (
    INPUT_FROM_FIRST_CHECKPOINT,
    INPUT_MISSING,
    Fixture,
    read_fixtures,
)

# Stable namespace for verify's derived run ids. A fixed uuid literal, never
# generated at import — the whole point is that the same fixture yields the
# same run id in every process, on every machine, forever.
_VERIFY_NS = uuid.UUID("6f1d0a2e-9b3c-4f7a-8d51-2c0a7e4b91d3")


# --- per-fixture verdicts ---------------------------------------------- #
#
# The first four are the answer an operator came for. The last three are
# reasons verify could not answer, kept distinct from "still broken" because
# conflating them would report a setup problem as a failed fix.

FIXED = "fixed"                          # was not-ok, now ok — the fix worked
STILL_NOT_OK = "still_not_ok"            # was not-ok, still not-ok
NEWLY_BROKEN = "newly_broken"            # was ok, now not-ok — the fix broke it
STILL_OK = "still_ok"                    # was ok, still ok

SKIPPED_NO_INPUT = "skipped_no_input"    # nothing recorded to re-run
UNRESOLVED_AGENT = "unresolved_agent"    # no registration matches fixture.agent
UNRUNNABLE_INPUT = "unrunnable_input"    # input doesn't fit the signature
NO_VERDICT = "no_verdict"                # ran, but inspected nothing

_ANSWERED = (FIXED, STILL_NOT_OK, NEWLY_BROKEN, STILL_OK)
_PASSING = (FIXED, STILL_OK)
_UNANSWERED = (SKIPPED_NO_INPUT, UNRESOLVED_AGENT, UNRUNNABLE_INPUT, NO_VERDICT)

_SEVERITY = {"ok": 0, "degraded": 1, "failed": 2}


class VerifyError(Exception):
    """Raised for setup failures — no fixtures, no agent module, both modes."""


@dataclass
class VerifyResult:
    """One fixture, re-run."""

    record_id: str
    item_id: str | None
    agent: str
    verdict: str
    recorded_status: str
    new_status: str | None = None
    new_reason: str | None = None
    # True when this record produced something different from what production
    # recorded. A `still_not_ok` whose output is UNCHANGED says the fix did not
    # touch this path at all; one whose output changed says it moved and did
    # not move far enough. Different next actions, so they are different facts.
    #
    # Compared on the STEP TRACE — each step's label and output_snapshot,
    # in execution order — because that is the like-for-like pair. The
    # function's return value is not: `recorded_output` is the ITEM's output
    # (R4 step 3a made it so deliberately), and on the clean `@agent` path
    # nothing sets that from the return value at all, so comparing the two
    # would report "changed" on every fixture that has one and never fire the
    # signal that earns the field. Falls back to return-vs-recorded_output
    # only for a fixture with no trace to compare.
    output_changed: bool | None = None
    # Whether a human flagged the item this fixture was pulled from — read off
    # the fixture's own cohort predicate, not re-queried. It is what makes
    # `still_ok` readable: on a cohort the INSPECTORS called ok and a PERSON
    # called wrong, `ok -> ok` is the expected verdict and carries no
    # information, so the only evidence a fix did anything is `output_changed`.
    # Without this flag verify cannot tell "fine before, fine now" from "the
    # thing I was asked to fix, unfixed".
    flagged: bool = False
    raised: str | None = None
    # Carried so a reader can tell a hosted reconstruction from a real request
    # (C4) without re-opening the fixture.
    input_source: str = INPUT_MISSING
    recorded_agent_version: str | None = None
    current_agent_version: str | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    note: str | None = None

    @property
    def passed(self) -> bool:
        return self.verdict in _PASSING

    @property
    def answered(self) -> bool:
        return self.verdict in _ANSWERED

    @property
    def stalled(self) -> bool:
        """A flagged item this code answers exactly as the deployed code did.

        The one thing verify can say about a `still_ok` fixture, and the reason
        `--strict` has anything to fail on over a flagged cohort. `None` is not
        `False`: a fixture with no trace to compare is UNKNOWN, and reporting
        unknown as "did not move" would be the silent wrong answer.
        """
        return self.flagged and self.answered and self.output_changed is False


@dataclass
class VerifySummary:
    """The cohort-level answer, which is the one the operator acts on."""

    results: list[VerifyResult] = field(default_factory=list)
    strict: bool = False

    def count(self, verdict: str) -> int:
        return sum(1 for r in self.results if r.verdict == verdict)

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.results:
            out[r.verdict] = out.get(r.verdict, 0) + 1
        return out

    @property
    def answered(self) -> int:
        return sum(1 for r in self.results if r.answered)

    @property
    def unanswered(self) -> int:
        return len(self.results) - self.answered

    @property
    def flagged(self) -> int:
        """Fixtures pulled from items a human flagged."""
        return sum(1 for r in self.results if r.flagged)

    @property
    def moved(self) -> int:
        """Answered fixtures whose output differs from what production recorded.

        On a flagged cohort this is the whole answer. The verdict axis is
        `ok -> ok` there by construction — `pull --flagged` selects items the
        inspectors called ok, and verify re-derives the inspectors' verdict —
        so movement is the only evidence available that the new code does
        anything at all.
        """
        return sum(1 for r in self.results if r.answered and r.output_changed is True)

    @property
    def stalled(self) -> int:
        """Flagged fixtures this code answers exactly as the deployed code did."""
        return sum(1 for r in self.results if r.stalled)

    @property
    def ok(self) -> bool:
        """Whether this run should be treated as a pass.

        A fixture verify could not *answer* is not a pass and not a failure —
        it is a gap, and ``pull`` already said so at write time for the
        commonest cause. Under ``strict`` (the CI setting) any gap fails.

        A set where NOTHING was answered fails either way. "Verified" over a
        run that verified nothing is the silent partial success this product
        exists to catch, and printing it would be us committing the failure we
        sell against.

        And under ``strict``, a FLAGGED cohort where nothing moved fails. Plan
        59 D3 measured the alternative: byte-identical output and exit 0 for
        the broken code and the fix, on the one incident class where the fix is
        the customer's own code. `ok -> ok` cannot distinguish them; movement
        can, and it was already being computed. A cohort that moved on no item
        is a re-drive guaranteed to reproduce what a person complained about —
        at this product's own measured price, 300 step executions to change
        nothing.
        """
        if any(r.verdict in (STILL_NOT_OK, NEWLY_BROKEN) for r in self.results):
            return False
        if self.strict and self.unanswered:
            return False
        if self.strict and self.flagged and self.moved == 0:
            return False
        return self.answered > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixtures": len(self.results),
            "answered": self.answered,
            "counts": self.counts,
            "flagged": self.flagged,
            "moved": self.moved,
            "stalled": self.stalled,
            "ok": self.ok,
            "results": [asdict(r) for r in self.results],
        }


# --- callable resolution ------------------------------------------------ #


def _injects_run(fn: Any) -> bool:
    """Whether the @agent wrapper injects ``run`` as fn's first argument.

    Same rule as ``papayya.agent`` applies at decoration time (agent.py, the
    ``inject_run`` block): first positional parameter literally named ``run``.
    Re-derived here rather than read off the registration because the
    registration does not carry it, and calling the wrapper with the wrapped
    function's arity would bind one argument too many.
    """
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
    positional = [
        p for p in params
        if p.name not in ("self", "cls")
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                       inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return bool(positional) and positional[0].name == "run"


def _call_signature(fn: Any) -> inspect.Signature | None:
    """The signature verify must satisfy — the wrapper's, minus an injected
    ``run``. ``functools.wraps`` makes ``inspect.signature`` report the
    *wrapped* function, which still declares ``run``; the wrapper supplies it."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    if _injects_run(fn):
        return sig.replace(
            parameters=[p for p in sig.parameters.values() if p.name != "run"]
        )
    return sig


def _plan_invocation(fn: Any, payload: Any) -> tuple[tuple, dict] | None:
    """Decide how ``payload`` binds to ``fn``: kwargs, positional, or neither.

    Mirrors ``_replay._replay_invoke`` — a dict whose keys bind to the
    parameters is unpacked as kwargs, everything else is passed positionally —
    but *pre-checks* the positional form too. ``_replay_invoke`` calls
    ``fn(snapshot)`` unguarded, so an input that does not fit surfaces as the
    customer's ``TypeError`` and reads as "the fix failed". It didn't; the
    fixture didn't fit. Returns None when neither form binds.
    """
    sig = _call_signature(fn)
    if sig is None:  # builtin / C-level — no introspection, try positional
        return (payload,), {}
    if isinstance(payload, dict):
        try:
            sig.bind(**payload)
        except (TypeError, ValueError):
            pass
        else:
            return (), dict(payload)
    try:
        sig.bind(payload)
    except (TypeError, ValueError):
        return None
    return (payload,), {}


def _resolve_registrations(
    agent_module: str | Path | None,
) -> dict[str, Any]:
    """Discover ``@agent`` registrations, keyed by agent name.

    Defaults to ``agent.py`` in cwd exactly as ``replay_slice`` does. Unlike
    replay, resolution is per FIXTURE rather than once for the whole set: a
    cohort selected by predicate without ``--agent`` legitimately spans agents,
    so a set can carry more than one.
    """
    from papayya.durable._replay import ReplayError, _discover_agents

    if agent_module is None:
        if Path("agent.py").exists():
            agent_module = "agent.py"
        else:
            raise VerifyError(
                "No agent.py in cwd. Pass --agent-module (CLI) / agent_module= "
                "to point at the module your @agent lives in, or handler= (SDK) "
                "for papayya.iter-style bodies."
            )
    try:
        registrations = _discover_agents(agent_module)
    except ReplayError as exc:
        raise VerifyError(str(exc)) from exc
    return {r.name: r for r in registrations}


# --- the run ------------------------------------------------------------ #


def _worst(entries: list[Any]) -> tuple[str, str | None]:
    """Worst outcome across a run's step entries.

    The local and hosted stores compute this aggregate inside ``save_task``;
    MemoryStore does not, so verify folds it here on the same severity
    ordering (``papayya.checks._SEVERITY``) rather than a second one.
    """
    status, reason = "ok", None
    for entry in entries:
        entry_status = getattr(entry, "outcome_status", "ok") or "ok"
        if _SEVERITY.get(entry_status, 0) > _SEVERITY.get(status, 0):
            status, reason = entry_status, getattr(entry, "outcome_reason", None)
    return status, reason


def _decode(payload: Any) -> Any:
    """Local ledger rows keep ``input_snapshot`` as a JSON string; the hosted
    API returns it already decoded. Decode a string that is JSON *and* decodes
    to a container — a plain string input stays a plain string, because
    ``"hello"`` is not a JSON document a customer meant us to unwrap."""
    if not isinstance(payload, str):
        return payload
    try:
        decoded = json.loads(payload)
    except (TypeError, ValueError):
        return payload
    return decoded if isinstance(decoded, (dict, list)) else payload


def _run_one(
    fx: Fixture,
    invoke: Callable[..., Any],
    fn_for_binding: Any,
    *,
    body_exception_reason: str,
) -> VerifyResult:
    """Drive one fixture through the customer's callable, offline."""
    from papayya import iterators
    from papayya.durable.run import Item
    from papayya.durable.store import MemoryStore
    from papayya.durable.types import DurableRunConfig
    from papayya.outcomes import OutcomeVerdict

    result = VerifyResult(
        record_id=fx.record_id,
        item_id=fx.item_id,
        agent=fx.agent,
        verdict=STILL_NOT_OK,
        recorded_status=record_verdict(fx.status, fx.worst_outcome_status),
        input_source=fx.input_source,
        recorded_agent_version=fx.agent_version,
    )

    payload = _decode(fx.input)
    plan = _plan_invocation(fn_for_binding, payload)
    if plan is None:
        result.verdict = UNRUNNABLE_INPUT
        result.note = (
            "the recorded input does not fit the function's signature"
            + (
                " — this fixture's input is a RECONSTRUCTION of the first "
                "step's bound arguments (hosted records persist no request "
                "input), so it may not be shaped like the agent's own "
                "parameters"
                if fx.input_source == INPUT_FROM_FIRST_CHECKPOINT
                else ""
            )
        )
        return result
    args, kwargs = plan

    store = MemoryStore()
    run_id = str(uuid.uuid5(_VERIFY_NS, fx.record_id or fx.item_id or fx.agent))
    item = Item(
        DurableRunConfig(
            agent=fx.agent,
            run_id=run_id,
            item_id=fx.item_id,
            partition_key=fx.partition_key,
            store=store,
            input_snapshot=payload,
        )
    )
    item.init()

    # Publish the run so the @agent wrapper's Case C branch reuses it rather
    # than minting one of its own — which is what keeps verify off the cloud
    # store. Same mechanism replay_slice uses (_replay.py, `_ACTIVE_RUN.set`).
    token = iterators._ACTIVE_RUN.set(item)
    returned: Any = None
    try:
        returned = invoke(*args, **kwargs)
        if inspect.isawaitable(returned):
            returned = asyncio.run(_await(returned))
    except Exception as exc:  # noqa: BLE001 — customer code, class unknown
        # Exception, not BaseException: a fixture that fails is a result, but
        # KeyboardInterrupt is the operator stopping a 500-fixture run and
        # swallowing it would report their Ctrl-C as a failed fix.
        result.raised = f"{type(exc).__name__}: {exc}"
        # Production records a body exception as a synthetic failed entry, and
        # the reason differs by path: `agent_body_exception` from
        # drive_ambient_sync (@agent), `loop_body_exception` from _iter_gen
        # (papayya.iter). Reproduce the one that matches the mode being
        # verified — a different reason would make the before/after comparison
        # of `outcome_reason` a lie.
        iterators._write_synthetic_entry(
            item, OutcomeVerdict("failed", body_exception_reason)
        )
    finally:
        iterators._ACTIVE_RUN.reset(token)

    checkpoint = store.load(run_id)
    entries = list(checkpoint.tasks) if checkpoint is not None else []
    new_status, new_reason = _worst(entries)
    # Left as None when nothing was inspected. `_worst([])` is "ok" — the same
    # default production applies to a record with no steps — but reporting
    # `degraded -> ok` for a run that inspected nothing would read as a fix.
    if entries:
        result.new_status = new_status
        result.new_reason = new_reason
    result.steps = [
        {
            "label": e.label,
            "outcome_status": e.outcome_status,
            "outcome_reason": e.outcome_reason,
        }
        for e in entries
    ]
    if fx.steps:
        result.output_changed = _trace_of(entries) != _recorded_trace(fx)
    elif result.raised is None:
        result.output_changed = _differs(returned, fx.recorded_output)

    if not entries:
        # Nothing was inspected, so there is nothing to compare. A verdict in
        # this product is derived from STEPS — `run.step` / `papayya.llm` /
        # `papayya.mark_*` — and never from a function's return value; that is
        # true in `_iter_gen` and in `drive_ambient_sync` alike, and it is why
        # a record whose body ran no steps is `ok` in production.
        #
        # Calling this `fixed` would be the worst available answer: the
        # operator would ship on the strength of an inspection that never ran.
        # It is a gap, and it is reported as one.
        result.verdict = NO_VERDICT
        result.note = (
            "the function ran but produced no inspected step, so there is no "
            "verdict to compare. A fixture is only verifiable through the "
            "verbs that recorded the failure — run.step / papayya.llm / "
            "papayya.mark_*"
        )
        return result

    # The recorded verdict comes from BOTH columns, through the one definition
    # that collapses them (plan 48 R5). Reading worst_outcome_status alone was
    # plan 48 R4: a record that raised before its first checkpoint keeps the
    # 'ok' it was born with, so `verify` called three crashed fixtures healthy,
    # reported `[ok -> ok] · 0 fixed` on the fix that repaired them, and exited
    # 0 — under --strict, "the CI setting". A gate that cannot fail on the
    # commonest class of failure is worse than no gate, because it is quoted as
    # evidence.
    #
    # The `status` this reads was added to the fixture by plan 50 for exactly
    # this call site, and until now nothing consumed it.
    was_ok = record_verdict(fx.status, fx.worst_outcome_status) == "ok"
    now_ok = new_status == "ok" and result.raised is None
    if was_ok:
        result.verdict = STILL_OK if now_ok else NEWLY_BROKEN
    else:
        result.verdict = FIXED if now_ok else STILL_NOT_OK
    return result


async def _await(awaitable: Any) -> Any:
    return await awaitable


def _norm(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(value)


def _trace_of(entries: list[Any]) -> list[tuple[str, str]]:
    """(label, output) pairs from a freshly-run trace, in execution order."""
    return [(e.label, _norm(e.output_snapshot)) for e in entries]


def _recorded_trace(fx: Fixture) -> list[tuple[str, str]]:
    """The same pairs off the fixture. ``pull`` already wrote the steps in
    execution order — ``(attempt, completed_at, seq)``, plan 41 R3's ordering —
    so no re-sort here would be an improvement, and re-sorting on ``seq``
    alone would undo it."""
    return [(s.label, _norm(s.output_snapshot)) for s in fx.steps]


def _differs(new: Any, recorded: Any) -> bool:
    """Compare through JSON so a dataclass/Decimal/datetime return value is
    compared on the shape that was recorded, not on Python identity. Anything
    unserializable falls back to ``repr`` rather than reporting "unchanged",
    which would be the wrong default."""
    def norm(value: Any) -> Any:
        try:
            return json.dumps(value, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return repr(value)

    return norm(new) != norm(recorded)


# --- entry points -------------------------------------------------------- #


def verify_fixtures(
    fixtures: list[Fixture],
    *,
    handler: Callable[..., Any] | None = None,
    agent_module: str | Path | None = None,
    strict: bool = False,
) -> VerifySummary:
    """Run every fixture through the customer's function, offline.

    ``handler=`` and ``agent_module=`` are mutually exclusive and resolve the
    callable exactly as :func:`papayya.durable._replay.replay` does: a handler
    for ``papayya.iter``-style bodies that have no registration to discover,
    ``@agent`` discovery otherwise.

    A fixture that raises does not abort the set — the same discipline
    ``replay_slice`` applies, for the same reason: recovery over N records
    keeps going and reports N answers.
    """
    if handler is not None and agent_module is not None:
        raise VerifyError(
            "Pass either handler= (run fixtures through your own callable) or "
            "agent_module= (discover an @agent registration), not both."
        )
    if not fixtures:
        raise VerifyError("No fixtures to verify.")

    registrations: dict[str, Any] | None = None
    if handler is None:
        registrations = _resolve_registrations(agent_module)

    summary = VerifySummary(strict=strict)

    def add(result: VerifyResult, fx: Fixture) -> None:
        """Every result carries the predicate that selected its fixture.

        Stamped here rather than at each construction site so a new verdict
        cannot be added without it — a result with `flagged` left False is
        indistinguishable from one a human never complained about, and that is
        precisely the distinction `--strict` now rests on.
        """
        result.flagged = bool(fx.cohort.get("flagged"))
        summary.results.append(result)

    for fx in fixtures:
        if fx.input is None or fx.input_source == INPUT_MISSING:
            add(
                VerifyResult(
                    record_id=fx.record_id,
                    item_id=fx.item_id,
                    agent=fx.agent,
                    verdict=SKIPPED_NO_INPUT,
                    recorded_status=record_verdict(fx.status, fx.worst_outcome_status),
                    input_source=fx.input_source,
                    recorded_agent_version=fx.agent_version,
                    note="no input was recorded for this record; it is kept "
                         "for its trace and cannot be re-run",
                ),
                fx,
            )
            continue

        if handler is not None:
            add(
                _run_one(fx, handler, handler,
                         body_exception_reason="loop_body_exception"),
                fx,
            )
            continue

        assert registrations is not None
        reg = registrations.get(fx.agent)
        if reg is None:
            known = ", ".join(sorted(registrations)) or "(none)"
            add(
                VerifyResult(
                    record_id=fx.record_id,
                    item_id=fx.item_id,
                    agent=fx.agent,
                    verdict=UNRESOLVED_AGENT,
                    recorded_status=record_verdict(fx.status, fx.worst_outcome_status),
                    input_source=fx.input_source,
                    recorded_agent_version=fx.agent_version,
                    note=f"no @agent named {fx.agent!r} in the module. "
                         f"Registered: {known}",
                ),
                fx,
            )
            continue
        outcome = _run_one(fx, reg.fn, reg.fn,
                           body_exception_reason="agent_body_exception")
        outcome.current_agent_version = getattr(reg, "agent_version", None)
        add(outcome, fx)

    return summary


def verify(
    path: str | Path = "./fixtures",
    *,
    handler: Callable[..., Any] | None = None,
    agent_module: str | Path | None = None,
    strict: bool = False,
) -> VerifySummary:
    """Read fixtures from ``path`` (file or directory) and verify them.

    The SDK entry point behind ``papayya verify``. See the module docstring
    for what this does and does not spend.
    """
    from papayya.fixtures import FixtureError

    try:
        fixtures = read_fixtures(path)
    except FixtureError as exc:
        raise VerifyError(str(exc)) from exc
    return verify_fixtures(
        fixtures, handler=handler, agent_module=agent_module, strict=strict
    )


__all__ = [
    "FIXED",
    "STILL_NOT_OK",
    "NEWLY_BROKEN",
    "STILL_OK",
    "SKIPPED_NO_INPUT",
    "UNRESOLVED_AGENT",
    "UNRUNNABLE_INPUT",
    "NO_VERDICT",
    "VerifyError",
    "VerifyResult",
    "VerifySummary",
    "verify",
    "verify_fixtures",
]
