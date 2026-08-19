"""Item — the durable per-item execution wrapper (formerly PapayyaRun).

Plan 34 noun consolidation: one *item* is one record a run processed —
outcome, trace, cost; replayable. The class wraps functions as
checkpoint-able steps exactly as before; only the noun changed.
``PapayyaRun`` remains available as a deprecated alias.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time as _time
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar, overload

from papayya import outcomes
from papayya._serialize import bind_arguments, build_input_snapshot
from papayya.classify import classify_provider_error, is_retryable
from papayya.errors import CreditExhausted, WorkloadPaused
from papayya.llm_extract import LlmUsage, extract_llm_usage
from papayya.runtime_context import get_current_reporter

from .store import MemoryStore
from .types import (
    _UNSET,
    CheckpointStore,
    DurableRunConfig,
    DurableRunResult,
    RunCheckpoint,
    TaskEntry,
    latest_per_label,
)


log = logging.getLogger("papayya.durable")

# ── Step retry defaults (plan 58 R1) ──────────────────────────────────────
#
# FIVE ATTEMPTS: waits of 1s, 2s, 4s and 8s, so a step gets ~15s of ladder.
#
# BOUNDED, NOT UNLIMITED, and that is a scoping decision rather than a tuning
# one. Temporal retries an activity until the workflow timeout; Papayya's
# `max_duration_seconds` watchdog lives in the WORKER (SIGALRM around the whole
# invocation, worker.py:1557) and the run object cannot see it, so the SDK has
# no deadline to retry against. A fixed ladder is what can be made correct
# without inventing that channel first.
#
# THE NUMBER WAS MEASURED, not picked. The first cut was 3 attempts / 3s, and
# driving walk 57's workload against a 9-second sidecar outage showed the
# ladder giving up while the outage was still going:
#
#     step read-photo failed on attempt 1/3 ...; retrying in 1.0s
#     step read-photo failed on attempt 2/3 ...; retrying in 2.0s
#     step read-photo failed on attempt 3/3, giving up: OcrUnavailable: ...
#
# 15s is ~0.8% of the 1800s default run budget (dispatch.DefaultMaxDurationSeconds),
# so one bad step cannot eat the run, and it bridges the outage class this unit
# exists for: a sidecar restart, a deploy blip, a rate limit, a dropped socket.
#
# WHAT IT STILL DOES NOT COVER, said plainly because the difference is the
# customer's: a four-minute provider outage. That one wants the run to PARK and
# resume rather than spin — the pause covenant's shape, which `CreditExhausted`
# already implements for an unfunded account. Separate unit.
#
# THE COST SIDE IS REAL TOO. A deterministic failure — a KeyError on a cached
# checkpoint result, where the arguments are byte-identical every attempt — now
# buys four extra executions and 15s for nothing. That is the price of one rule
# instead of a hand-made taxonomy of which exception classes are deterministic,
# and the two opt-outs (`NonRetriable`, `retries=0`) are how a customer who
# knows better says so. Revisit with production data, not with a guess.
DEFAULT_STEP_RETRIES = 4
STEP_RETRY_BASE_SECONDS = 1.0
STEP_RETRY_MAX_SECONDS = 8.0
# Matches durable_runs.error's storage bound in the control pane.
_RETRY_REASON_MAX = 2000


# Module-level seams so a test can observe the backoff WITHOUT reaching
# `time.sleep`. `run.py` does `import time as _time`, so `_time` IS the time
# module and monkeypatching `_time.sleep` replaces it for the whole process —
# which broke the heartbeat harness, a duration assertion and an llm-judge
# timeout in three unrelated files, order-dependently, the one time it was
# tried. One indirection is cheaper than that class of leak.
def _sleep(seconds: float) -> None:
    _time.sleep(seconds)


async def _async_sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)

T = TypeVar("T")

# Sentinel: distinguishes "snapshot kwarg not provided" (auto-capture)
# from "snapshot=None" (explicit None) and "snapshot=False" (opt-out).
_AUTO = object()

# Sentinel returned from ``_pre_call`` when the cache MISSED — the
# wrapper proceeds to invoke ``fn``. A cache hit returns the cached
# value instead and the wrapper short-circuits.
_MISS = object()


@dataclass
class _CallCtx:
    """Per-call state shared between ``_pre_call`` and ``_post_call_*``.

    Lives on the stack of one wrapper invocation. Carries the dedupe
    handle for the runtime reporter so the post-call helpers can ask
    "did the interceptor handle this call?" via the per-call token
    (new path) or the legacy pre/post counter (back-compat with old
    shims that don't expose ``begin_call``).
    """

    effective_label: str
    effective_item_id: str | None
    runtime_reporter: Any | None
    call_token: object | None
    legacy_pre_count: int | None
    start: float
    cleanup_done: bool = False
    # Plan 58 R4 — what this call cost. Incremented by the retry loop, read by
    # _post_call_success onto the one row this call writes. On the ctx rather
    # than a closure variable because the ctx is already the thing that spans
    # the call boundary, and a step invoked twice in a loop must not inherit
    # the previous invocation's tally.
    retry_count: int = 0
    retry_reason: str | None = None


def _label_for_warning(label_or_fn: Any, fn: Any) -> str:
    """Best-effort label string for deprecation messages.

    The deprecation tracker keys per-(run, label) to dedupe noise. When
    the caller used a legacy form, we may not yet have resolved the
    final label — this helper picks the most informative string we can
    surface in the warning message.
    """
    if isinstance(label_or_fn, str):
        return label_or_fn
    if callable(label_or_fn):
        return getattr(label_or_fn, "__name__", "<anonymous>")
    return "<unknown>"


def _store_pending_pause(store: Any, run_id: str) -> str | None:
    """Ask the store whether a fence has paused this run (Plan 33).

    Duck-typed on ``pending_pause(run_id)`` so the check stays store-agnostic:
    CloudStore sets it from the SaveCheckpoint response, SQLiteStore from its
    local run-level fence, and stores without the method (MemoryStore) never
    pause. Any error reading it is swallowed — telemetry must never crash a
    step, and a reliability product keeps working under a flaky signal.
    """
    fn = getattr(store, "pending_pause", None)
    if not callable(fn):
        return None
    try:
        return fn(run_id)
    except Exception:
        return None


class Item:
    """A durable per-item record that wraps functions as checkpoint-able steps.

    **Execution guarantee:** at-least-once. If a crash occurs between
    executing a step and saving its checkpoint, the step will re-execute
    on resume. Design steps to be idempotent (safe to run more than once).

    **Repeated labels:** each *call* is its own durable step. Calling a
    step with the same label more than once in a run (e.g. inside an
    agent loop) keys the first call on the clean label and call *N* on
    ``label#N`` — call 2 never silently replays call 1's cached result.
    Replay relies on deterministic re-execution: the resumed code must
    make same-label calls in the same order, which is the same contract
    every durable step already carries.

    Usage::

        run = PapayyaRun(DurableRunConfig(agent="my-agent"))

        search = run.step("search", search_web)
        summarize = run.step("summarize", summarize_results)

        results = search(query)       # cached on replay
        summary = summarize(results)  # cached on replay

        run.complete(summary)

    Or with decorators::

        @run.step("search")
        def search(query: str) -> list[str]:
            return search_web(query)

    ``run.task(...)`` is kept as an alias of ``run.step(...)`` for existing
    code — identical behavior, same call conventions.
    """

    def __init__(self, config: DurableRunConfig) -> None:
        self.agent = config.agent
        self.run_id = config.run_id or str(uuid.uuid4())
        # Plan 34: id of the run row (invocation) this item belongs to.
        # Set by papayya.map()/iter()/slice-replay; None for direct calls
        # (the store wraps those in an implicit run-of-one at create time).
        self._invocation_id: str | None = config.invocation_id
        self._store: CheckpointStore = config.store or MemoryStore()
        # Plan 33: per-run override for the local run-level auto-pause fence
        # (None = store default). Registered with the store in init().
        self._pause_after_degraded: int | None = config.pause_after_degraded
        # Plan 35: custom outcome checks. Config-supplied checks (run-scoped);
        # the @agent registration's checks are merged in init(). Both run in
        # _post_call_success alongside the built-in inspectors.
        self._config_checks: list | None = config.checks
        self._checks: list = []
        self._cache: dict[str, TaskEntry] = {}
        self._task_call_order: list[str] = []
        # Plan 58 U — the reuse index, and the reason it is SEPARATE from
        # _cache rather than replacing it.
        #
        # _cache is positional (`label#N`) and every existing consumer depends
        # on that: _build_result walks _task_call_order, the taint inspector
        # reads _cache.values() as "the steps before this one", and a resumed
        # run re-finds its own work by recomputing the same sequence. None of
        # that changes.
        #
        # What changes is how a REUSED entry — one copied here from another
        # run, which never executed in this one — may be found. A positional
        # key means what it means only while the loop runs from the top; page
        # 17 is `read-photo#17` because the customer's `for page in range(...)`
        # produced it seventeenth. Carry that key across runs and a customer
        # who filters or reorders their pages gets page 17's text returned for
        # page 22, silently, with the record claiming success. So reused
        # entries are keyed on ``(item_id, base label)`` — the customer's own
        # id for the record, which they already pass — and the positional
        # lookup is not allowed to match them at all.
        self._reuse_index: dict[tuple[str, str], TaskEntry] = {}
        # The subset of _reuse_index whose steps are AGGREGATE — keyed to the
        # run's own record rather than a sub-record's, so their inputs are
        # everything that came before them. Computed at HYDRATION, against the
        # run id the store handed back, rather than at call time against
        # _run_item_id: that attribute is seeded from config and can legitimately
        # still be None when the first step runs, and "None == None" would
        # classify every anonymous step as item-scoped — the unsafe direction.
        self._reuse_aggregate: set[tuple[str, str]] = set()
        # Has anything actually EXECUTED in this pass yet? Gates aggregate
        # reuse — see _pre_call.
        self._executed_any = False
        self._initialized = False
        self._finished = False
        # Run-level item_id. Seeded from config; the first step that passes
        # item_id= also seeds it if still unset. Subsequent steps inherit
        # unless they pass an explicit override (which applies to that step
        # only — the run-level id does not change mid-run).
        self._run_item_id: str | None = config.item_id
        # ADR-0002 #7: agent version pinned at run creation. On replay this
        # is read from the loaded checkpoint, NOT recomputed — otherwise
        # replay would silently rewrite the version onto rows that were
        # produced under a different code version.
        self._agent_version: str | None = None
        # v9 partition-key: metadata blob and the extracted partition_key
        # value. Both denormalize onto every TaskEntry written by this run.
        self._metadata: dict[str, Any] | None = config.metadata
        self._partition_key: str | None = config.partition_key
        # v10 / Layer 3 #7 Phase 2: outer run id for sub-runs lineage.
        # Set by Papayya.run() (explicit kwarg or @agent contextvar).
        # Pinned at create time; on replay it's read from the loaded
        # checkpoint, not re-derived.
        self._parent_run_id: str | None = config.parent_run_id
        # Replay snapshot supplied by the caller (the iter() wrapper passes
        # the per-item payload). Left as the _UNSET sentinel by the @agent
        # path, in which case init() falls back to the call args captured
        # on the contextvar. Held verbatim so an explicit None is preserved.
        self._config_input_snapshot: Any = config.input_snapshot
        # Live-call occurrence counter per bare step label. A loop calling
        # run.step("think", ...) once per iteration must produce one durable
        # step per call — not silently hand iteration 2 the cached result of
        # iteration 1. Call 1 keeps the clean label; call N keys "label#N".
        # Never seeded from hydrated cache entries: replay re-executes the
        # body from the top, recomputing the same sequence, so computed keys
        # line up with stored labels positionally.
        self._label_occurrences: dict[str, int] = {}
        # Per-EFFECTIVE-label re-execution counter (Plan 41 R3). Distinct
        # from _label_occurrences above and easy to confuse with it: that
        # one separates call N of a repeated label within a single pass (a
        # loop — "draft", "draft#2"), this one separates re-executions of
        # the same logical step ACROSS passes. They compose, so a loop's
        # second iteration re-executed once is ("draft#2", attempt=2) —
        # which only holds because this is keyed on the effective label
        # rather than the bare one.
        #
        # Their seeding rules are OPPOSITE, which is the reason they are
        # two dicts. _label_occurrences is never seeded from stored
        # entries (replay re-executes the body from the top and recomputes
        # the same sequence). _attempts is seeded from stored entries and
        # from nothing else — see init().
        self._attempts: dict[str, int] = {}
        # Track which (label, deprecation-kind) pairs already emitted a
        # warning this run, so repeated calls don't spam the log.
        self._deprecation_seen: set[str] = set()
        # Replay Phase 3 hydration. When non-empty, init() seeds _cache
        # with these TaskEntry rows before invoking store.create() so
        # the wrapped agent fn's first step() calls find cache hits
        # for labels < from_step. Hydrated rows are NOT persisted to
        # the new run's tasks table — only steps the replay actually
        # re-executes get written. Populated by Papayya.run() reading
        # the one-shot _REPLAY_HYDRATION contextvar.
        self._prepopulated_tasks: list[TaskEntry] | None = config.prepopulated_tasks

    def init(self) -> None:
        """Load any existing checkpoint from the store."""
        if self._initialized:
            return
        self._initialized = True

        # Plan 33: register the per-run fence threshold with a store that
        # supports it (local SQLite). Duck-typed so cloud/memory stores, whose
        # K is server-side or absent, are unaffected.
        #
        # Plan 41 R7 §W9: a store that does NOT support it now says so. The
        # hosted CloudStore has no set_run_fence, and hosted is the only
        # execution path — so `papayya().run(..., pause_after_degraded=5)` has
        # always been a silent no-op there. A kwarg that quietly does nothing
        # is the failure class this whole product exists to remove, so it is
        # not left silent.
        #
        # NOT plumbed to the server, deliberately. A per-run fence the SDK can
        # set is a policy the operator cannot see: the dashboard reads agent
        # config, and the fence is an operator control. Two sources of truth
        # for one policy is worse than one inconvenient source, so the warning
        # names the key that does work instead.
        if self._pause_after_degraded is not None:
            _setter = getattr(self._store, "set_run_fence", None)
            if callable(_setter):
                _setter(self.run_id, self._pause_after_degraded)
            else:
                log.warning(
                    "pause_after_degraded=%s is not supported by %s and was "
                    "IGNORED. On the hosted path the run-level fence is an "
                    "operator control set per agent, not per run — set "
                    "pause_after_degraded in the agent's config "
                    "(`papayya agents update <agent-id> --config "
                    "'{\"pause_after_degraded\": %s}'`).",
                    self._pause_after_degraded,
                    type(self._store).__name__,
                    self._pause_after_degraded,
                )

        # Plan 35: resolve custom outcome checks — run-scoped config checks
        # first, then the @agent registration's checks. Resolved on both fresh
        # and replay paths, since a step re-executed after a resume point still
        # runs checks in _post_call_success.
        from papayya.agent import get_agent as _get_agent

        self._checks = list(self._config_checks or [])
        _reg = _get_agent(self.agent)
        if _reg is not None:
            _reg_checks = getattr(_reg, "checks", None)
            if _reg_checks:
                self._checks.extend(_reg_checks)

        existing = self._store.load(self.run_id)
        if existing is not None:
            self._agent_version = existing.agent_version
            # v9: partition_key/metadata pin at create time. On replay,
            # trust the stored values rather than rederiving — same
            # posture as agent_version (#7).
            if existing.metadata is not None:
                self._metadata = existing.metadata
            if existing.partition_key is not None:
                self._partition_key = existing.partition_key
            # v10: parent_run_id pins at create time too. Trust the
            # stored value on replay rather than the current invocation
            # context (the @agent body that's re-executing might not
            # be inside the same outer-run as when this child was
            # originally spawned).
            if existing.parent_run_id is not None:
                self._parent_run_id = existing.parent_run_id
            # Plan 41 R3. Seed _attempts from the RAW loaded list, before
            # and independently of any cache filter.
            #
            # This is R3's contract with R7. R7's entire mechanism is
            # removing invalidated entries from _cache at hydration so the
            # step re-executes. If _attempts were derived from _cache, an
            # invalidated label would have no entry, the next attempt
            # would compute as 1, and the re-execution would reuse the
            # recorded attempt's provider idempotency key — the exact bug
            # this unit exists to fix, on the exact path it was built for.
            # Pinned by test_attempts_survive_a_suppressed_cache_entry.
            self._seed_attempts(existing.tasks)

            # Plan 41 R7 — the whole point of this unit.
            #
            # Without this filter a resumed run cache-hits the very steps the
            # fence stopped it for: _pre_call returns cached.result for any
            # hydrated label, with no outcome filter, so the operator fixes
            # the prompt, resumes, and gets the same degraded outputs back.
            #
            # The set is (label, attempt) pairs the server recorded AT PAUSE
            # TIME — not a client-side walk back over the loaded rows, which
            # plan 40's A4 killed for three separate reasons (arbitrary
            # concurrent save order, void side-effect steps being permanently
            # degraded, and synthetic papayya.mark/* labels that can never
            # re-execute).
            #
            # Matching on the ATTEMPT is what makes it spend itself exactly
            # once. After the step re-runs it carries attempt+1, the recorded
            # pair stops matching, and a later unrelated recovery — a lease
            # reap, a worker death — re-hydrates without re-executing it a
            # second time and billing the provider twice for it.
            invalidated = set(getattr(existing, "pause_invalidated", None) or ())
            for entry in self._latest_per_label(existing.tasks):
                if (entry.label, entry.attempt) in invalidated:
                    # Deliberately skips _task_call_order too: the step
                    # appends itself there when it re-executes, so the order
                    # stays honest instead of listing a step that hasn't run.
                    log.info(
                        "re-executing %s (attempt %d): the fence that paused "
                        "this run counted it",
                        entry.label, entry.attempt,
                    )
                    continue
                self._cache[entry.label] = entry
                self._task_call_order.append(entry.label)
                self._index_if_reused(entry, existing.item_id)
        else:
            # Read the @agent wrapper's captured call args. None when the
            # caller bypassed the decorator (scripts, tests). Stays as-is
            # — we never inject a synthetic snapshot here.
            from papayya.agent import consume_agent_input_snapshot, get_agent

            registration = get_agent(self.agent)
            self._agent_version = (
                registration.agent_version if registration is not None else None
            )

            # Replay Phase 3: hydrate cache from prepopulated TaskEntry
            # rows before store.create(). Order matters — _wrap reads
            # _cache.get(label) on every invocation; entries seeded here
            # short-circuit the wrapped fn for labels < from_step. The
            # rows live only in memory; the new run's tasks table starts
            # empty and only fills with steps the replay re-executes.
            if self._prepopulated_tasks:
                self._seed_attempts(self._prepopulated_tasks)
                for entry in self._latest_per_label(self._prepopulated_tasks):
                    self._cache[entry.label] = entry
                    self._task_call_order.append(entry.label)
                    self._index_if_reused(entry, self._run_item_id)

            # Caller-supplied snapshot (iter() passes the item) wins; the
            # @agent path leaves it _UNSET and we read the captured call
            # args. This is what makes iter-runs replayable — without it
            # iter rows are created with input_snapshot=NULL because there's
            # no decorator above them to populate the contextvar.
            if self._config_input_snapshot is not _UNSET:
                input_snapshot = self._config_input_snapshot
            else:
                input_snapshot = consume_agent_input_snapshot()

            now = datetime.now(timezone.utc).isoformat()
            checkpoint = RunCheckpoint(
                run_id=self.run_id,
                agent=self.agent,
                tasks=[],
                status="running",
                created_at=now,
                updated_at=now,
                item_id=self._run_item_id,
                input_snapshot=input_snapshot,
                agent_version=self._agent_version,
                metadata=self._metadata,
                partition_key=self._partition_key,
                parent_run_id=self._parent_run_id,
                invocation_id=self._invocation_id,
            )
            self._store.create(checkpoint)

    @property
    def id(self) -> str:
        """The item's surrogate uuid (Plan 34 canonical name).

        ``item_id`` stays reserved for CUSTOMER identity (the value passed
        via ``item_id=``, e.g. ``"co_007"``); this is Papayya's own row id.
        ``run_id`` is the deprecated pre-consolidation alias.
        """
        return self.run_id

    # ------------------------------------------------------------------ #
    #  task() — supports both higher-order function and decorator usage   #
    # ------------------------------------------------------------------ #

    @overload
    def task(self, label: str, fn: Callable[..., T]) -> Callable[..., T]: ...

    @overload
    def task(self, label: str) -> Callable[[Callable[..., T]], Callable[..., T]]: ...

    @overload
    def task(self, fn: Callable[..., T]) -> Callable[..., T]: ...

    def task(  # type: ignore[no-untyped-def]
        self,
        label_or_fn=None,
        fn=None,
        *,
        item_id: str | None = None,
        snapshot: Any = _AUTO,
        kind: str | None = None,
        retries: int | None = None,
        expect_count: Any = None,
        expect_coverage: str | None = None,
        expect_fields: Any = None,
    ):
        """Wrap a function as a durable step. (Alias: ``run.step``.)

        Preferred call shape::

            run.step("label", some_fn)

        Labels may repeat: each call is its own durable step. In a loop,
        call 1 of ``"think"`` is stored as ``think`` and call N as
        ``think#N`` — later iterations never replay an earlier
        iteration's cached result. See the class docstring for the
        determinism contract this puts on replay.

        For LLM calls, use the explicit ``run.llm_step("label", fn)``
        method — it's equivalent to passing ``kind="llm"`` here but
        makes the intent visible in the type signature.

        Two legacy call shapes are still accepted for one release and
        emit ``DeprecationWarning`` — migrate to the canonical form
        before they're removed:

        * ``run.step(some_fn)`` — derives the label from ``fn.__name__``.
        * ``@run.step("label")`` — decorator form.

        Optional kwargs:

        * ``item_id`` — identifier of the record this step acts on. If set,
          the step row gets tagged with it; the first step to pass one also
          seeds the run-level item_id for later steps to inherit.
        * ``snapshot`` — controls input-state capture for the step row.
          Defaults to auto-capture: when an item_id is in effect, the
          wrapped fn's call args are bound against its signature and
          encoded as the input snapshot (same path ``@agent`` uses). Pass
          ``snapshot=False`` to opt out, or pass any other value to
          override the captured payload (escape hatch for args that
          aren't JSON-encodable). The fn's return value is captured as
          the output snapshot whenever an item_id is in effect.
        * ``kind`` — DEPRECATED. Pass ``kind="llm"`` triggers the LLM
          observability path (tokens, model, stop_reason, credit-error
          classification). Use ``run.llm_step(label, fn)`` instead;
          ``kind=`` will be removed in the next minor release.
        * ``retries`` — how many EXTRA attempts a raising step gets, on top of
          the first (plan 58 R1). Defaults to ``DEFAULT_STEP_RETRIES``. Pass
          ``retries=0`` for a step whose side effect must happen at most once
          and that you have no idempotency key for::

              run.step("charge-card", charge, retries=0)(order)

          Per-raise control is :class:`papayya.NonRetriable`, which is usually
          the better tool: it keeps the retry for the failures that deserve
          one and skips it for the failure you can already rule out.

        Conservation contracts — what this step's output must conserve
        about its input. All three default to absent, and absence means no
        check: a step that legitimately drops records is normal, so nothing
        is ever inferred. A declared contract that can't be evaluated is
        logged and treated as a pass; contracts mark a step ``degraded``,
        they never raise::

            rows = run.step("extract", extract_rows,
                            expect_count=len(docs),
                            expect_fields=["name", "email"])(docs)

        * ``expect_count`` — how many records the output must hold. Pass the
          integer when you have it (you usually do, three lines up); a
          callable is the escape hatch and is invoked with the step's bound
          arguments as keywords. Under-delivery reads
          ``conservation:count_short``, over-delivery ``count_over``.
        * ``expect_coverage`` — the name of a record-id key. Every input
          record carrying that key must come back out under it, so you
          learn *which* records went missing, not just how many. The input
          collection is located by the key itself; nothing extra to declare.
        * ``expect_fields`` — keys that must be present and non-blank on
          every output record, so a dict with three of eight fields
          populated is degraded instead of passing. Numeric ``0`` counts as
          populated; ``None``/``False``/``""``/``[]``/``{}`` do not.

        All kwargs are additive and optional.
        """
        if kind == "llm":
            self._warn_kind_llm_deprecated(
                _label_for_warning(label_or_fn, fn)
            )
        contract = outcomes.build_contract(
            expect_count=expect_count,
            expect_coverage=expect_coverage,
            expect_fields=expect_fields,
        )

        # Case 1: run.task("label", fn) — canonical, silent.
        if isinstance(label_or_fn, str) and fn is not None:
            return self._wrap(
                label_or_fn, fn, item_id=item_id, snapshot=snapshot, kind=kind,
                contract=contract, retries=retries,
            )

        # Case 2: run.task(fn) — DEPRECATED, label derived from fn.__name__.
        if callable(label_or_fn):
            label = label_or_fn.__name__
            if not label or label == "<lambda>":
                raise ValueError(
                    "Anonymous/lambda functions require an explicit label: "
                    "run.step('myLabel', lambda: ...)"
                )
            self._warn_legacy_step_form("fn-only", label)
            return self._wrap(
                label, label_or_fn, item_id=item_id, snapshot=snapshot, kind=kind,
                contract=contract, retries=retries,
            )

        # Case 3: @run.task("label") — DEPRECATED decorator form.
        if isinstance(label_or_fn, str):
            label = label_or_fn
            self._warn_legacy_step_form("decorator", label)
            _item_id = item_id
            _snapshot = snapshot
            _kind = kind
            _contract = contract
            _retries = retries

            def decorator(f: Callable[..., T]) -> Callable[..., T]:
                return self._wrap(
                    label, f, item_id=_item_id, snapshot=_snapshot, kind=_kind,
                    contract=_contract, retries=_retries,
                )

            return decorator

        raise TypeError("task() requires a label string or a callable")

    # Preferred public name. Matches the vocabulary used by peer durable
    # execution frameworks (Temporal, Inngest, DBOS); `task` is retained
    # as an alias so existing user code keeps working unchanged.
    step = task

    def _resolve_step_label(self, label: str) -> str:
        """Consume the next occurrence of ``label`` and return its cache key.

        Mutates the per-run occurrence counter — call exactly once per
        step invocation, from ``_pre_call``.
        """
        n = self._label_occurrences.get(label, 0) + 1
        self._label_occurrences[label] = n
        return label if n == 1 else f"{label}#{n}"

    def _peek_step_label(self, label: str) -> str:
        """Key the *next* call of ``label`` would get, without consuming it."""
        n = self._label_occurrences.get(label, 0) + 1
        return label if n == 1 else f"{label}#{n}"

    @staticmethod
    def _base_label(label: str) -> str:
        """Strip the positional occurrence suffix: ``read-photo#17`` → ``read-photo``.

        The suffix is exactly what U exists to stop depending on, so the reuse
        key is built from the customer's own string. The server guarantees the
        pair is unique per run before it copies anything — two calls of one
        label under one item_id collide here, and BOTH are excluded server-side
        rather than one silently winning.
        """
        base, sep, tail = label.rpartition("#")
        return base if sep and tail.isdigit() else label

    def _index_if_reused(self, entry: TaskEntry, run_item_id: str | None) -> None:
        """Index a hydrated entry for (item_id, label) lookup, if it was reused.

        Entries this run executed are deliberately NOT indexed: they are found
        positionally, exactly as before plan 58 U, and adding a second way to
        match them would change resume/re-lease behaviour that is already
        correct and already tested.

        AGGREGATE STEPS ARE INDEXED TOO, and _pre_call gates them separately.
        An aggregate step is one whose item_id is the RUN's id rather than a
        sub-record's — ``validate`` reading forty findings under ``DOC-2001``,
        against ``read-photo`` under ``DOC-2001#photo-17``. It is reusable only
        while nothing has re-executed, because its inputs are everything that
        came before it. Deciding that needs runtime state, so it cannot live in
        the server's WHERE clause; recording WHICH entries are aggregate needs
        only the two ids, so it happens here.
        """
        if entry.reused_from is None or entry.item_id is None:
            return
        key = (entry.item_id, self._base_label(entry.label))
        self._reuse_index[key] = entry
        if run_item_id is not None and entry.item_id == run_item_id:
            self._reuse_aggregate.add(key)

    def _seed_attempts(self, tasks: list[TaskEntry]) -> None:
        """Seed the per-label attempt counter from stored task rows.

        MAX, not count (Plan 41 R3 §T1). The two diverge in the case this
        unit exists for: after a journal collision two rows both carry
        attempt 1, so a third execution is 2 under max and 3 under count.
        Max is right and cheaper — the only requirement on the derived
        provider key is that it differ from every key already used, and
        both rows already used ``:1``.
        """
        for entry in tasks:
            prior = self._attempts.get(entry.label, 0)
            if entry.attempt > prior:
                self._attempts[entry.label] = entry.attempt

    # Collapse happens at HYDRATION and never at read: a read-side
    # collapse would leave _cache and _task_call_order disagreeing about
    # how many times a label appears.
    _latest_per_label = staticmethod(latest_per_label)

    def _next_execution(self, effective_label: str) -> tuple[str, int]:
        """Mint the identity for a new execution of ``effective_label``.

        Returns ``(execution_token, attempt)``. Attempt 1's token is the
        deterministic ``"{label}|1"`` — byte-identical to what the server
        derives for a client that sends no token at all — so a pre-R3 and
        a post-R3 SDK writing the same first execution land on the SAME
        derived row id. Without that, ADR-0002 #8's re-delivery
        idempotency would hold inside each SDK generation and never across
        it, and the worker service rolls at
        deployment_minimum_healthy_percent = 50, so mixed-SDK workers
        overlap by design.

        Attempt >= 2 is a random uuid4 *string*. It must be a str and not
        a uuid.UUID: the lineage journal's to_json_line raises on the
        object, and only on the journal path — i.e. only under an outage,
        which is the one time the write must not fail.
        """
        attempt = self._attempts.get(effective_label, 0) + 1
        self._attempts[effective_label] = attempt
        token = f"{effective_label}|1" if attempt == 1 else str(uuid.uuid4())
        return token, attempt

    def _version_for_attempt(self, attempt: int) -> str | None:
        """Agent version to stamp on a step row (ADR-0002 #7).

        #7 promises every step row records the agent version it ran on.
        The run-level version is *pinned* across a resume
        (``self._agent_version = existing.agent_version`` in init), which
        is right for ``--latest`` gating and wrong here: a step
        re-executed after the customer deployed a fix — the exact loop R7
        exists to create — would record the ORIGINAL version, making
        attempt 2 indistinguishable from attempt 1 by version, precisely
        where the distinction matters most.

        So a re-execution records the LIVE registration's version. The
        first execution keeps the pinned one, because that genuinely is
        the version the run started on. Falls back to the pin when no
        registration is in scope (scripts, tests, MemoryStore use).
        """
        if attempt < 2:
            return self._agent_version
        from papayya.agent import get_agent as _get_agent

        registration = _get_agent(self.agent)
        live = getattr(registration, "agent_version", None) if registration else None
        return live if live is not None else self._agent_version

    def idempotency_key(self, label: str) -> str:
        """Return a per-execution idempotency token for this step.

        The durable runtime is **at-least-once**: if a worker crashes
        between executing a step's side effect and persisting its
        checkpoint, the step re-executes on resume (see the class
        docstring). For a non-idempotent side effect — most importantly a
        billed LLM call — pass this token to your provider's own
        idempotency mechanism so the *provider* dedupes the retry::

            key = run.idempotency_key("draft")
            resp = run.llm_step("draft", lambda: client.messages.create(
                ..., extra_headers={"Idempotency-Key": key}))

        **What the token discriminates.** It is
        ``(run_id, effective_label, attempt)``, where ``attempt`` is one
        more than the highest attempt this run can *see* recorded for the
        step. So the key turns on whether the previous execution left a
        record, which is the distinction that matters:

        - A crash before the checkpoint persisted leaves nothing recorded,
          so the resumed execution sends the **same** key and the provider
          replays its stored response. That is the point of the key — it
          is what stops the retry billing your provider account twice.
        - A **deliberate** re-execution of a step that *was* recorded — an
          operator re-running it — sends a **new** key, so the provider
          does the work again. Asking for a step to be re-run and getting
          the old response replayed back would be the wrong outcome.

        Two executions can legitimately share an attempt number: a write
        journaled during an outage is by definition invisible to the
        execution that replaces it. So this is a provider key and not a
        unique record id — the record's identity is a separate token the
        SDK mints per execution (Plan 41 R3).

        Composes with the ``label#N`` occurrence suffix rather than
        colliding with it: iteration 2 of a loop, re-executed once, keys
        on ``"<run_id>:draft#2:2"``.

        Call it **immediately before** the step call it protects — it
        reads the counter as of now, so a call made after that step
        returns yields the *next* execution's key. This is a seam, not
        exactly-once: Papayya cannot dedupe a side effect it does not own.
        """
        effective = self._peek_step_label(label)
        return f"{self.run_id}:{effective}:{self._attempts.get(effective, 0) + 1}"

    def llm_step(
        self,
        label: str,
        fn: Callable[..., T],
        *,
        item_id: str | None = None,
        snapshot: Any = _AUTO,
        retries: int | None = None,
        expect_count: Any = None,
        expect_coverage: str | None = None,
        expect_fields: Any = None,
    ) -> Callable[..., T]:
        """Wrap an LLM-call function as a durable step.

        Equivalent to ``run.step(label, fn, kind="llm")`` but makes the
        intent explicit. The wrapper runs shape-based usage extraction
        on the returned response (tokens, model, stop_reason) and
        classifies any raised provider exception via
        ``classify_provider_error`` — credit-shaped exceptions are
        re-raised as ``CreditExhausted`` so the runtime pauses instead
        of failing.

        ``retries`` behaves as it does on ``run.step``, with one interaction
        worth naming: credit exhaustion is promoted to ``CreditExhausted``
        BEFORE the retry decision, so an unfunded account pauses the run rather
        than burning the ladder against a condition no retry can change.

        The ``expect_*`` conservation contracts behave exactly as they do on
        ``run.step`` — see its docstring. They read the value the step
        returned, so on an LLM step they apply to the parsed records when
        the wrapped fn returns them, not to the raw provider response.

        Canonical signature only — no ``__name__``-derived label or
        decorator form. ``run.step(..., kind="llm")`` keeps working for
        one release with a deprecation warning.
        """
        return self._wrap(
            label, fn, item_id=item_id, snapshot=snapshot, kind="llm",
            contract=outcomes.build_contract(
                expect_count=expect_count,
                expect_coverage=expect_coverage,
                expect_fields=expect_fields,
            ),
            retries=retries,
        )

    def _warn_kind_llm_deprecated(self, label: str) -> None:
        """Fire ``DeprecationWarning`` once per (run, label) for kind='llm'."""
        token = f"kind=llm:{label}"
        if token in self._deprecation_seen:
            return
        self._deprecation_seen.add(token)
        warnings.warn(
            "run.step(kind='llm') is deprecated; use run.llm_step(label, fn) "
            "instead. The kind= kwarg will be removed in the next minor release.",
            DeprecationWarning,
            stacklevel=3,
        )

    def _warn_legacy_step_form(self, form: str, label: str) -> None:
        """Fire ``DeprecationWarning`` once per (run, label, form)."""
        token = f"{form}:{label}"
        if token in self._deprecation_seen:
            return
        self._deprecation_seen.add(token)
        if form == "fn-only":
            warnings.warn(
                "run.step(fn) (label derived from fn.__name__) is deprecated; "
                "pass an explicit label: run.step('label', fn). The fn-only "
                "form will be removed in the next minor release.",
                DeprecationWarning,
                stacklevel=3,
            )
        elif form == "decorator":
            warnings.warn(
                "@run.step('label') decorator form is deprecated; rewrite as "
                "fn = run.step('label', fn). The decorator form will be removed "
                "in the next minor release.",
                DeprecationWarning,
                stacklevel=3,
            )

    def _wrap(
        self,
        label: str,
        fn: Callable[..., T],
        *,
        item_id: str | None = None,
        snapshot: Any = _AUTO,
        kind: str | None = None,
        contract: outcomes.ConservationContract | None = None,
        retries: int | None = None,
    ) -> Callable[..., Any]:
        """Build the durable wrapper around ``fn``.

        Returns an ``async def`` wrapper iff ``fn`` is a coroutine
        function (per :func:`inspect.iscoroutinefunction`, which
        unwraps ``functools.wraps``); otherwise returns a sync
        wrapper. The async path mirrors the sync path through
        ``await fn(...)`` — same pre/post helpers, same dedupe
        semantics. Async generators fall through to the sync branch
        (their wrapper returns the async-generator object — same as
        today's behavior).
        """
        try:
            sig: inspect.Signature | None = inspect.signature(fn)
        except (TypeError, ValueError):
            # Builtins / C-level callables — no introspectable signature.
            # Auto-capture skipped for these; the step still runs.
            sig = None

        is_async = inspect.iscoroutinefunction(fn)

        def _pre_call() -> tuple[Any, _CallCtx | None]:
            """Common pre-call work. Returns ``(cache_hit, None)`` on
            replay or ``(_MISS, ctx)`` when the wrapper should invoke
            ``fn``. Splitting this out keeps the sync and async wrappers
            byte-identical above the actual call boundary.
            """
            self.init()
            self._throw_if_finished()

            # Plan 33: a fence may have paused this run on the previous save
            # (server-signalled for the cloud store, locally-evaluated for
            # SQLite). The just-completed step is already checkpointed; stop
            # here before starting the next one. Raising a named, catchable
            # exception unwinds the body cleanly; on resume, replay skips every
            # saved step and picks up exactly here. Store-agnostic: any store
            # exposing pending_pause(run_id) participates; those that don't
            # (MemoryStore) simply never pause.
            _pending = _store_pending_pause(self._store, self.run_id)
            if _pending is not None:
                raise WorkloadPaused(_pending, self.run_id)

            # Each call consumes an occurrence of the bare label: call 1
            # keys the clean label, call N keys "label#N". Cache hits
            # consume too — on replay the recomputed sequence must walk
            # the hydrated entries in the same order it wrote them.
            effective_label = self._resolve_step_label(label)

            # Resolve effective item_id: explicit per-step kwarg wins; else
            # inherit the run-level id. First step to supply an explicit id
            # also seeds the run-level id for later inheritance.
            #
            # RESOLVED BEFORE THE CACHE LOOKUP since plan 58 U, where it used
            # to come after. The reuse key IS (item_id, label), so the lookup
            # cannot happen until the id is known. The move is safe because
            # nothing between the two lines reads it and the seeding side
            # effect is idempotent.
            effective_item_id = item_id if item_id is not None else self._run_item_id
            if item_id is not None and self._run_item_id is None:
                self._run_item_id = item_id

            # ── The two-tier lookup (plan 58 U) ───────────────────────────
            #
            # Tier 1: reused work, matched on the customer's own id for the
            # record. This is what makes a re-drive cost one page instead of
            # forty, and what makes it safe to do so without a determinism
            # sandbox — the key does not depend on the loop running from the
            # top, so filtering or reordering pages produces a MISS and a
            # re-execution rather than another page's answer.
            cached = None
            if effective_item_id is not None:
                reuse_key = (effective_item_id, label)
                candidate = self._reuse_index.get(reuse_key)
                if candidate is not None:
                    # An AGGREGATE step — one keyed to the run's own record
                    # rather than a sub-record — consumed everything before it.
                    # `validate` read forty findings; if any page re-executed,
                    # its verdict is stale and reusing it returns yesterday's
                    # answer for today's document. That is precisely the
                    # silently-wrong failure this product exists to catch, so
                    # aggregates are reusable only while nothing has run.
                    #
                    # Note this is strictly weaker than "nothing CHANGED": a
                    # re-executed step may well produce an identical result.
                    # Re-running an aggregate is cheap and always correct;
                    # keeping a stale one is neither.
                    if not (reuse_key in self._reuse_aggregate and self._executed_any):
                        cached = candidate
                        log.debug(
                            "reusing %s for %s from run %s",
                            label, effective_item_id, candidate.reused_from,
                        )

            # Tier 2: this run's OWN prior work, matched positionally, exactly
            # as before U — a resume, a re-lease, a redelivery. A reused entry
            # must never be reachable this way, or the positional key is back
            # in force across runs and tier 1's whole guarantee is decorative.
            if cached is None:
                positional = self._cache.get(effective_label)
                if positional is not None and positional.reused_from is None:
                    cached = positional

            if cached is not None:
                return cached.result, None

            # Anything that reaches here is about to EXECUTE. Recorded before
            # the call rather than after, because the flag gates aggregate
            # reuse and an aggregate that follows a step which raised must not
            # be reused either — the exception is a change like any other.
            self._executed_any = True

            runtime_reporter = get_current_reporter() if kind == "llm" else None
            call_token: object | None = None
            legacy_pre_count: int | None = None
            if runtime_reporter is not None:
                # New shims expose begin_call → use the per-call token
                # path (correct under asyncio.gather). Old shims only
                # expose intercepted_call_count → fall back to the
                # legacy pre/post snapshot. Both branches stay alive
                # one release.
                if hasattr(runtime_reporter, "begin_call"):
                    try:
                        call_token = runtime_reporter.begin_call(effective_label)
                    except Exception:
                        # Telemetry must never crash a step. Falling
                        # back to "no dedupe" means a duplicate emission
                        # at worst, never a missed step.
                        call_token = None
                if call_token is None:
                    try:
                        legacy_pre_count = runtime_reporter.intercepted_call_count()
                    except Exception:
                        legacy_pre_count = 0

            return _MISS, _CallCtx(
                effective_label=effective_label,
                effective_item_id=effective_item_id,
                runtime_reporter=runtime_reporter,
                call_token=call_token,
                legacy_pre_count=legacy_pre_count,
                start=_time.monotonic(),
            )

        def _interceptor_already_emitted(ctx: _CallCtx) -> bool:
            """Ask the reporter whether the interceptor handled this call.

            Closes the per-token dedupe scope on the new path (so the
            shim can release its bookkeeping); falls back to the
            legacy pre/post counter compare on old shims.
            """
            reporter = ctx.runtime_reporter
            if reporter is None:
                return False
            ctx.cleanup_done = True
            if ctx.call_token is not None:
                try:
                    return reporter.was_emitted_for(ctx.call_token)
                except Exception:
                    return False
            try:
                return reporter.intercepted_call_count() != ctx.legacy_pre_count
            except Exception:
                return False

        def _ensure_cleanup(ctx: _CallCtx) -> None:
            """Final-chance reset for the per-call token.

            Runs in the wrapper's ``finally`` so cancellation
            (``asyncio.CancelledError`` extends ``BaseException``,
            bypassing our ``except Exception``) doesn't leak the
            contextvar entry on the shim side.
            """
            if ctx.cleanup_done or ctx.call_token is None:
                return
            reporter = ctx.runtime_reporter
            if reporter is None:
                return
            try:
                reporter.was_emitted_for(ctx.call_token)
            except Exception:
                pass
            ctx.cleanup_done = True

        def _post_call_success(result: Any, ctx: _CallCtx, args: tuple, kwargs: dict) -> Any:
            """Common post-success work: usage extraction, dedupe,
            snapshot resolution, ``TaskEntry`` build, ``save_task``.
            """
            duration_ms = int((_time.monotonic() - ctx.start) * 1000)

            # LLM usage extraction runs on the returned response when this
            # step was wrapped with kind="llm". Extraction never raises —
            # unknown shapes fall through to provider_shape="unknown".
            usage: LlmUsage | None = None
            if kind == "llm":
                usage = extract_llm_usage(result)
                llm_prompt_tokens = usage.prompt_tokens
                llm_completion_tokens = usage.completion_tokens
                llm_total_tokens = usage.total_tokens
                llm_model = usage.model
                llm_stop_reason = usage.stop_reason
                llm_provider_shape = usage.provider_shape

                # Emit through the runtime reporter only when the
                # interceptor did not already emit for this call. This
                # avoids double-counting when the user wraps a patched
                # provider (openai / anthropic) in run.llm_step(...).
                if ctx.runtime_reporter is not None:
                    if not _interceptor_already_emitted(ctx):
                        ctx.runtime_reporter.report_llm_call(
                            label=ctx.effective_label,
                            usage=usage,
                            duration_ms=duration_ms,
                            error_category=None,
                        )
            else:
                llm_prompt_tokens = None
                llm_completion_tokens = None
                llm_total_tokens = None
                llm_model = None
                llm_stop_reason = None
                llm_provider_shape = None

            # Structural outcome inspection. Defaults to OK; the inspectors
            # overwrite when a known degraded shape (empty result, zero
            # embedding, degenerate LLM stop reason) is detected. The
            # parent run's worst_outcome/degraded_count aggregate updates
            # automatically inside the store on save_task.
            #
            # Plan 40 Unit 1: when the step declared a conservation
            # contract, the inspectors also get the step's LIVE bound
            # arguments — the input side of the relation. Bound here rather
            # than reusing input_snapshot below because the snapshot is
            # JSON-encoded, is skipped entirely without an item_id, and
            # counting rows off a round-trip would be both wasteful and
            # wrong for inputs that aren't JSON-encodable.
            if outcomes.ENABLE_STRUCTURAL_DETECTION:
                inputs = (
                    bind_arguments(sig, args, kwargs) if contract is not None else None
                )
                verdict = outcomes.inspect_result(
                    result, usage=usage, contract=contract, inputs=inputs
                )
            else:
                verdict = outcomes.OK
            # Plan 35: fold customer checks into the same verdict — worst
            # severity across built-in + custom wins (one pipeline, Decision 3).
            # run_checks contains every check (a raise/timeout is a pass) and
            # namespaces custom reasons under user: for the dashboard histogram.
            if self._checks:
                from papayya.checks import run_checks

                verdict = run_checks(self._checks, result, verdict, self.run_id)
            outcome_status = verdict.status
            outcome_reason = verdict.reason

            # ADR 0009 D6: did this step consume a degraded step's output?
            #
            # Only when this step's own verdict came back clean. A step that
            # degraded on its own output keeps that verdict — it is the more
            # specific signal, and taint is about the INPUT's provenance, not
            # a second opinion on the output.
            #
            # Deliberately its own pair of fields rather than a status value or
            # a reason token: outcome_status is read by two independent
            # degraded-streak fences, by drift rates learned against a floor,
            # by the failure-cluster key and by the reason histogram, and a
            # taint is signal to none of them. See plans/44-D6-taint.md §2.
            #
            # bind_arguments is computed here only when there is an eligible
            # source to match against — i.e. never on a clean run.
            tainted_by = tainted_reason = None
            if outcomes.ENABLE_STRUCTURAL_DETECTION and outcome_status == "ok":
                prior = list(self._cache.values())
                if any(e.outcome_status != "ok" for e in prior):
                    tainted_by, tainted_reason = outcomes.inspect_taint(
                        inputs if contract is not None else bind_arguments(sig, args, kwargs),
                        prior,
                    )

            # Snapshots only populate when an item_id is in effect — the
            # lineage view has no home for snapshots that aren't attached
            # to an item, and we'd rather not bloat the DB with noise.
            #
            # Resolution for input_snapshot when an item_id is in effect:
            #   - snapshot=False  → opt-out (None)
            #   - snapshot=_AUTO  → introspect args (matches @agent path)
            #   - any other value → explicit override (escape hatch for
            #     args that aren't JSON-encodable)
            if ctx.effective_item_id is not None:
                if snapshot is False:
                    input_snapshot = None
                elif snapshot is _AUTO:
                    input_snapshot = build_input_snapshot(sig, args, kwargs)
                else:
                    input_snapshot = snapshot
                output_snapshot = result
            else:
                input_snapshot = None
                output_snapshot = None

            execution_token, attempt = self._next_execution(ctx.effective_label)
            entry = TaskEntry(
                label=ctx.effective_label,
                result=result,
                duration_ms=duration_ms,
                completed_at=datetime.now(timezone.utc).isoformat(),
                item_id=ctx.effective_item_id,
                input_snapshot=input_snapshot,
                output_snapshot=output_snapshot,
                kind=kind,
                llm_prompt_tokens=llm_prompt_tokens,
                llm_completion_tokens=llm_completion_tokens,
                llm_total_tokens=llm_total_tokens,
                llm_model=llm_model,
                llm_stop_reason=llm_stop_reason,
                llm_provider_shape=llm_provider_shape,
                agent_version=self._version_for_attempt(attempt),
                metadata=self._metadata,
                partition_key=self._partition_key,
                outcome_status=outcome_status,
                outcome_reason=outcome_reason,
                tainted_by=tainted_by,
                tainted_reason=tainted_reason,
                execution_token=execution_token,
                attempt=attempt,
                retry_count=ctx.retry_count,
                retry_reason=ctx.retry_reason,
            )

            self._cache[ctx.effective_label] = entry
            self._task_call_order.append(ctx.effective_label)
            self._store.save_task(self.run_id, entry)

            return result

        def _post_call_exception(exc: BaseException, ctx: _CallCtx) -> None:
            """Common exception-path work for LLM steps.

            Mirrors the success path's dedupe choice: emit a failed
            step row only if the interceptor didn't already record
            this call. ``CreditExhausted`` promotion happens in the
            wrapper itself (we need to know whether to ``raise from``).
            Non-LLM steps don't call this — they propagate unchanged.
            """
            if kind != "llm" or ctx.runtime_reporter is None:
                return
            if _interceptor_already_emitted(ctx):
                return
            duration_ms_exc = int((_time.monotonic() - ctx.start) * 1000)
            try:
                ctx.runtime_reporter.report_llm_call(
                    label=ctx.effective_label,
                    usage=LlmUsage(None, None, None, None, None, "unknown"),
                    duration_ms=duration_ms_exc,
                    error_category=classify_provider_error(exc),
                )
            except Exception:
                pass


        # ── Retry (plan 58 R1) ────────────────────────────────────────────
        #
        # THE LOOP GOES AROUND `fn` ONLY, INSIDE ONE _pre_call ENVELOPE, and
        # that placement is the whole design. Two existing contracts decide it,
        # and the first draft of plan 58 got both backwards:
        #
        # 1. A FAILED EXECUTION WRITES NO CHECKPOINT. `_next_execution` — the
        #    only thing that mints an `attempt` — is called from
        #    `_post_call_success`. So `attempt` counts re-executions ACROSS
        #    PASSES (a resume, a replay, a re-lease), seeded from STORED rows.
        #    An in-process retry stores nothing, so it must not advance it, or
        #    the number an operator reads stops meaning what plan 41 R3 built
        #    it to mean.
        #
        # 2. `run.idempotency_key` MUST NOT ADVANCE EITHER, and its own
        #    docstring already says why: "a crash before the checkpoint
        #    persisted leaves nothing recorded, so the resumed execution sends
        #    the SAME key and the provider replays its stored response. That is
        #    the point of the key — it is what stops the retry billing your
        #    provider account twice." A retry after a raise is exactly that
        #    case. If the provider had already done the work when the
        #    connection dropped, the same key gets the stored answer back
        #    instead of paying for it again.
        #
        # Everything above the call boundary — the pause fence, the occurrence
        # counter, the cache lookup, the LLM interceptor token — therefore runs
        # once, which is also the only way `label#N` alignment survives.
        attempts_allowed = 1 + (
            DEFAULT_STEP_RETRIES if retries is None else max(0, int(retries))
        )

        def _retry_wait(attempt: int) -> float:
            """Seconds to sleep before attempt ``attempt + 1`` (1-indexed).

            Exponential from a 1s base, capped. Deliberately NOT the api.py
            ladder (0.1s → 2.0s): that one guards a control-plane round trip
            where the far side is either up or down. This one guards a
            PROVIDER outage, which is measured in seconds at best — a
            sub-second ladder would spend the whole budget before the sidecar
            has finished restarting.
            """
            return min(STEP_RETRY_BASE_SECONDS * (2 ** (attempt - 1)),
                       STEP_RETRY_MAX_SECONDS)

        def _should_retry(exc: BaseException, attempt: int, ctx: _CallCtx) -> bool:
            """Decide, log the decision, and sleep. Returns True to go again.

            THE LOG LINE IS NOT DECORATION. A retry that leaves no trace is the
            platform quietly spending the customer's money and their wall
            clock — the same rule that makes cohort truncation say so out loud.
            Until a failed attempt has somewhere to live on the record, this
            line is the only place it exists.
            """
            if attempt >= attempts_allowed:
                if attempts_allowed > 1:
                    log.warning(
                        "step %s failed on attempt %d/%d, giving up: %s: %s",
                        label, attempt, attempts_allowed,
                        type(exc).__name__, exc,
                    )
                return False
            if not is_retryable(exc):
                log.info(
                    "step %s failed on attempt %d/%d and will not be retried "
                    "(%s): %s",
                    label, attempt, attempts_allowed, type(exc).__name__, exc,
                )
                return False
            wait = _retry_wait(attempt)
            log.warning(
                "step %s failed on attempt %d/%d (%s: %s); retrying in %.1fs",
                label, attempt, attempts_allowed,
                type(exc).__name__, exc, wait,
            )
            ctx.retry_count += 1
            # The LAST failure wins, not the first: a step that hit a 429 and
            # then a 503 is better described by the one that was still true
            # when it finally worked. Truncated on the same rule as
            # durable_runs.error — a provider that returns a stack trace in
            # its message must not push a row over a column bound.
            ctx.retry_reason = f"{type(exc).__name__}: {exc}"[:_RETRY_REASON_MAX]
            return True

        if is_async:

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                cache_hit, ctx = _pre_call()
                if ctx is None:
                    return cache_hit
                try:
                    for attempt in range(1, attempts_allowed + 1):
                        try:
                            result = await fn(*args, **kwargs)
                        except Exception as exc:
                            _post_call_exception(exc, ctx)
                            if kind == "llm":
                                category = classify_provider_error(exc)
                                if category == "credit" and not isinstance(exc, CreditExhausted):
                                    raise CreditExhausted(
                                        f"{label}: provider credits exhausted ({exc})"
                                    ) from exc
                            if _should_retry(exc, attempt, ctx):
                                # asyncio.sleep, not _time.sleep — blocking the
                                # loop here would stall every other coroutine
                                # in the same @agent body, including the ones
                                # that are working.
                                await _async_sleep(_retry_wait(attempt))
                                continue
                            raise
                        return _post_call_success(result, ctx, args, kwargs)
                    raise AssertionError("unreachable: retry loop fell through")
                finally:
                    _ensure_cleanup(ctx)

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            cache_hit, ctx = _pre_call()
            if ctx is None:
                return cache_hit
            try:
                for attempt in range(1, attempts_allowed + 1):
                    try:
                        result = fn(*args, **kwargs)
                    except Exception as exc:
                        _post_call_exception(exc, ctx)
                        if kind == "llm":
                            category = classify_provider_error(exc)
                            if category == "credit" and not isinstance(exc, CreditExhausted):
                                # Promoted BEFORE the retry decision, so the
                                # pause path wins over the retry loop — you
                                # cannot retry an unfunded account.
                                raise CreditExhausted(
                                    f"{ctx.effective_label}: provider credits exhausted ({exc})"
                                ) from exc
                        if _should_retry(exc, attempt, ctx):
                            _sleep(_retry_wait(attempt))
                            continue
                        raise
                    return _post_call_success(result, ctx, args, kwargs)
                raise AssertionError("unreachable: retry loop fell through")
            finally:
                _ensure_cleanup(ctx)

        return sync_wrapper

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def complete(self, output: Any = None) -> DurableRunResult:
        """Mark the run as successfully completed."""
        self.init()
        self._throw_if_finished()
        self._finished = True
        self._store.set_status(self.run_id, "completed", output)
        return self._build_result("completed")

    def fail(self, error: Any = None) -> DurableRunResult:
        """Mark the run as failed."""
        self.init()
        if self._finished:
            return self._build_result("failed")
        self._finished = True
        self._store.set_status(self.run_id, "failed", error)
        return self._build_result("failed")

    @property
    def completed_tasks(self) -> list[str]:
        """Labels of completed tasks in execution order."""
        return list(self._task_call_order)

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _throw_if_finished(self) -> None:
        if self._finished:
            raise RuntimeError(
                f"Run {self.run_id} is already finished. Create a new run to continue."
            )

    def _build_result(self, status: str) -> DurableRunResult:
        tasks = [
            self._cache[label]
            for label in self._task_call_order
            if label in self._cache
        ]
        return DurableRunResult(
            run_id=self.run_id,
            agent=self.agent,
            status=status,
            tasks=tasks,
            total_duration_ms=sum(t.duration_ms for t in tasks),
        )


# Deprecated pre-Plan-34 alias. Existing ``PapayyaRun`` imports, type hints
# and isinstance checks keep working; new code should say ``Item``.
PapayyaRun = Item
