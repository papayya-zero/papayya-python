"""Item — the durable per-item execution wrapper (formerly PapayyaRun).

Plan 34 noun consolidation: one *item* is one record a run processed —
outcome, trace, cost; replayable. The class wraps functions as
checkpoint-able steps exactly as before; only the noun changed.
``PapayyaRun`` remains available as a deprecated alias.
"""

from __future__ import annotations

import functools
import inspect
import time as _time
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar, overload

from papayya import outcomes
from papayya._serialize import build_input_snapshot
from papayya.classify import classify_provider_error
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
)

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
        self._cache: dict[str, TaskEntry] = {}
        self._task_call_order: list[str] = []
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
        if self._pause_after_degraded is not None:
            _setter = getattr(self._store, "set_run_fence", None)
            if callable(_setter):
                _setter(self.run_id, self._pause_after_degraded)

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
            for entry in existing.tasks:
                self._cache[entry.label] = entry
                self._task_call_order.append(entry.label)
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
                for entry in self._prepopulated_tasks:
                    self._cache[entry.label] = entry
                    self._task_call_order.append(entry.label)

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

        All kwargs are additive and optional.
        """
        if kind == "llm":
            self._warn_kind_llm_deprecated(
                _label_for_warning(label_or_fn, fn)
            )
        # Case 1: run.task("label", fn) — canonical, silent.
        if isinstance(label_or_fn, str) and fn is not None:
            return self._wrap(label_or_fn, fn, item_id=item_id, snapshot=snapshot, kind=kind)

        # Case 2: run.task(fn) — DEPRECATED, label derived from fn.__name__.
        if callable(label_or_fn):
            label = label_or_fn.__name__
            if not label or label == "<lambda>":
                raise ValueError(
                    "Anonymous/lambda functions require an explicit label: "
                    "run.step('myLabel', lambda: ...)"
                )
            self._warn_legacy_step_form("fn-only", label)
            return self._wrap(label, label_or_fn, item_id=item_id, snapshot=snapshot, kind=kind)

        # Case 3: @run.task("label") — DEPRECATED decorator form.
        if isinstance(label_or_fn, str):
            label = label_or_fn
            self._warn_legacy_step_form("decorator", label)
            _item_id = item_id
            _snapshot = snapshot
            _kind = kind

            def decorator(f: Callable[..., T]) -> Callable[..., T]:
                return self._wrap(label, f, item_id=_item_id, snapshot=_snapshot, kind=_kind)

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

    def idempotency_key(self, label: str) -> str:
        """Return a stable per-step idempotency token for this run.

        The durable runtime is **at-least-once**: if a worker crashes
        between executing a step's side effect and persisting its
        checkpoint, the step re-executes on resume (see the class
        docstring). For a non-idempotent side effect — most importantly a
        billed LLM call — pass this token to your provider's own
        idempotency mechanism so the *provider* dedupes the retry::

            key = run.idempotency_key("draft")
            resp = run.llm_step("draft", lambda: client.messages.create(
                ..., extra_headers={"Idempotency-Key": key}))

        The token is deterministic in ``(run_id, label, occurrence)`` so
        the same logical step yields the same key across re-executions,
        while call N of a repeated label (an agent loop) gets its own key
        — matching the ``label#N`` keying of the step itself. Call it
        immediately before the step call it protects. This is a seam, not
        exactly-once: Papayya cannot dedupe a side effect it does not own.
        """
        return f"{self.run_id}:{self._peek_step_label(label)}"

    def llm_step(
        self,
        label: str,
        fn: Callable[..., T],
        *,
        item_id: str | None = None,
        snapshot: Any = _AUTO,
    ) -> Callable[..., T]:
        """Wrap an LLM-call function as a durable step.

        Equivalent to ``run.step(label, fn, kind="llm")`` but makes the
        intent explicit. The wrapper runs shape-based usage extraction
        on the returned response (tokens, model, stop_reason) and
        classifies any raised provider exception via
        ``classify_provider_error`` — credit-shaped exceptions are
        re-raised as ``CreditExhausted`` so the runtime pauses instead
        of failing.

        Canonical signature only — no ``__name__``-derived label or
        decorator form. ``run.step(..., kind="llm")`` keeps working for
        one release with a deprecation warning.
        """
        return self._wrap(
            label, fn, item_id=item_id, snapshot=snapshot, kind="llm"
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

            cached = self._cache.get(effective_label)
            if cached is not None:
                return cached.result, None

            # Resolve effective item_id: explicit per-step kwarg wins; else
            # inherit the run-level id. First step to supply an explicit id
            # also seeds the run-level id for later inheritance.
            effective_item_id = item_id if item_id is not None else self._run_item_id
            if item_id is not None and self._run_item_id is None:
                self._run_item_id = item_id

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
            if outcomes.ENABLE_STRUCTURAL_DETECTION:
                verdict = outcomes.inspect_result(result, usage=usage)
                outcome_status = verdict.status
                outcome_reason = verdict.reason
            else:
                outcome_status = "ok"
                outcome_reason = None

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
                agent_version=self._agent_version,
                metadata=self._metadata,
                partition_key=self._partition_key,
                outcome_status=outcome_status,
                outcome_reason=outcome_reason,
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

        if is_async:

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                cache_hit, ctx = _pre_call()
                if ctx is None:
                    return cache_hit
                try:
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
                        raise
                    return _post_call_success(result, ctx, args, kwargs)
                finally:
                    _ensure_cleanup(ctx)

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            cache_hit, ctx = _pre_call()
            if ctx is None:
                return cache_hit
            try:
                try:
                    result = fn(*args, **kwargs)
                except Exception as exc:
                    _post_call_exception(exc, ctx)
                    if kind == "llm":
                        category = classify_provider_error(exc)
                        if category == "credit" and not isinstance(exc, CreditExhausted):
                            raise CreditExhausted(
                                f"{ctx.effective_label}: provider credits exhausted ({exc})"
                            ) from exc
                    raise
                return _post_call_success(result, ctx, args, kwargs)
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
