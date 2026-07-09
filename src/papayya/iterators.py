"""Iterator wrappers for adopting Papayya around an existing for-loop.

The customer replaces ``for item in items:`` with
``for item in papayya.iter(items, workload=..., item_id=..., partition_key=...):``.
Each yielded item opens a per-item :class:`PapayyaRun` that is auto-closed
when the loop body returns or raises. Module-level :func:`mark_degraded` and
:func:`mark_outcome` read the active run from a contextvar and write a
synthetic :class:`TaskEntry` through the run's store, so callers can flag
outcomes the structural inspectors (Plan 02) can't catch.

This is the wrapper-shaped adoption path (Plan 10). The decorator-shaped
``@agent`` path remains the entrypoint for hosted execution; the two
coexist.
"""

from __future__ import annotations

import builtins
import functools
import inspect
import logging
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Iterator, TypeVar

from papayya.durable import Item, PapayyaRun
from papayya.durable.types import DurableRunConfig, TaskEntry
from papayya.outcomes import OutcomeVerdict

log = logging.getLogger("papayya.iter")

T = TypeVar("T")


# Per-context active run. Set by ``iter`` before yielding each item; reset
# after the loop body returns or raises. ``mark_degraded`` / ``mark_outcome``
# read this to find the run they should write against. Distinct from
# ``agent._ACTIVE_RUN_ID`` (which carries just a string id for sub-run
# lineage); this carries the run object itself.
_ACTIVE_RUN: ContextVar["PapayyaRun | None"] = ContextVar(
    "papayya_active_run", default=None
)


def iter(
    items: Iterable[T],
    *,
    agent: str | None = None,
    item_id: Callable[[T], str],
    partition_key: Callable[[T], str],
    store: Any | None = None,
    workload: str | None = None,
) -> Iterator[T]:
    """Wrap an iterable so each yielded item runs inside its own :class:`Item`.

    Required kwargs:
      ``agent`` — agent slug for the loop (string, not a lambda).
        (``workload=`` is the pre-Plan-34 spelling, accepted as a silent
        alias for one release.)
      ``item_id`` — callable extracting a stable per-item identifier.
        Required; explicit attribution is the whole point.
      ``partition_key`` — callable extracting the per-item partition key
        (typically tenant id). Required for multi-tenant visibility.
      ``store`` — optional checkpoint store. When omitted, the same store
        the ``@agent`` / client path uses is resolved automatically
        (SQLiteStore locally, CloudStore when an API key is present), so
        iter-items persist and show up in ``papayya dev`` like any other.
        Pass an explicit store to point an invocation at a specific database.

    Behavior:

    * One ``iter()`` call is ONE RUN: a single run row is minted up front
      (on stores that support it) and every item this loop processes links
      to it, so a 1,000-item loop shows up as one run of 1,000 items —
      not 1,000 separate runs.
    * For each item: open an ``Item`` record, install it as the active
      item via contextvar, yield the item to the caller's loop body.
    * On body return: ``complete()``, reset the contextvar.
    * On body exception: write a synthetic ``failed`` TaskEntry, mark the
      item failed, reset the contextvar, re-raise the original exception.
    * The active-item contextvar is :func:`mark_degraded` /
      :func:`mark_outcome` / :func:`active_item`'s source of truth.
    """
    resolved_agent = agent or workload
    if resolved_agent is None:
        raise TypeError("papayya.iter() requires the agent= keyword argument")
    return _iter_gen(items, resolved_agent, item_id, partition_key, store)


def map(
    fn: Callable[[T], Any],
    items: Iterable[T],
    *,
    item_id: Callable[[T], str],
    partition_key: Callable[[T], str],
    agent: str | None = None,
    store: Any | None = None,
    workload: str | None = None,
) -> list:
    """Eager fan-out: run ``fn`` once per item, all inside ONE run.

    The eager sibling of :func:`iter` and the documented lead entrypoint.
    Equivalent to
    ``[fn(x) for x in papayya.iter(items, agent=…, item_id=…, partition_key=…)]``.
    One ``map()`` call mints one run row; each processed item is one item
    row inside it.

    ``fn`` may be an ordinary function — its ambient ``@papayya.llm`` /
    ``mark_degraded`` calls resolve against the per-item record ``map``
    opens — or an ``@papayya.durable`` / ``@agent`` function, which detects
    the active item and reuses it rather than opening a second. Either way
    attribution comes from ``map``'s explicit ``item_id`` / ``partition_key``
    (callables over the item), not the decorator's weaker first-argument
    guess — which is why ``map`` is the correct-attribution path for rich
    items.

    ``agent`` defaults to the function's registered agent name, else its
    ``__name__``. (``workload=`` is the pre-Plan-34 spelling, accepted as a
    silent alias for one release.) As with :func:`iter`, a failing item is
    marked failed and the exception propagates; wrap the body for per-item
    error isolation.
    """
    reg = getattr(fn, "_papayya_agent", None)
    resolved_agent = (
        agent or workload or getattr(reg, "name", None) or getattr(fn, "__name__", "agent")
    )
    results: list = []
    for item in _iter_gen(items, resolved_agent, item_id, partition_key, store):
        results.append(fn(item))
    return results


def _resolve_default_store() -> Any:
    """Resolve the same store the @agent/client path uses (SQLite or Cloud).

    Lazy import keeps package-init ordering intact (see the note in
    ``papayya/__init__.py``); resolution failures fall back to ``None`` so a
    misconfigured environment degrades to the in-memory default rather than
    crashing the loop.
    """
    try:
        from papayya.papayya import Papayya

        return Papayya()._auto_store()
    except Exception:
        log.exception("papayya.iter: could not resolve default store; using in-memory")
        return None


def _mint_invocation(store: Any, agent: str) -> str | None:
    """Mint ONE run row (invocation) for a map()/iter() call.

    Returns the run row's id, or None when the store has no invocation
    surface (MemoryStore/FileStore, and CloudStore until the Unit 5 wire
    lands — the durable HTTP contract is frozen at the old shape, so
    hosted invocation rows are minted server-side later).
    """
    if store is None or not hasattr(store, "create_run"):
        return None
    run_id = str(uuid.uuid4())
    try:
        # total_items=0 marks the run OPEN: the item count isn't known up
        # front (the iterable may be a generator). finalize_run seals it.
        store.create_run(run_id, agent, 0)
    except Exception:
        log.exception("papayya.iter: could not mint the run row; items run unlinked")
        return None
    return run_id


def _finalize_invocation(store: Any, run_id: str | None) -> None:
    if run_id is None:
        return
    try:
        store.finalize_run(run_id)
    except Exception:
        log.exception("papayya.iter: finalize_run raised; run row left open")


def _iter_gen(
    items: Iterable[T],
    agent: str,
    item_id_fn: Callable[[T], str],
    partition_key_fn: Callable[[T], str],
    store: Any | None = None,
    invocation_id: str | None = None,
) -> Iterator[T]:
    resolved_store = store if store is not None else _resolve_default_store()
    # Plan 34 Unit 1: one map()/iter() call is ONE run. Mint the run row up
    # front and thread its id into every per-item create() below, so a
    # 1,000-item loop is one run of 1,000 items rather than 1,000 implicit
    # runs-of-one. Slice replay passes an already-minted invocation_id in.
    owns_invocation = invocation_id is None
    if owns_invocation:
        invocation_id = _mint_invocation(resolved_store, agent)
    try:
        # Use builtins.iter inside this module so the public name doesn't
        # shadow it for our own implementation. The for-loop below already
        # uses iteration protocol, so this is defense-in-depth — anything
        # that accepts Iterable[T] is fine.
        for item in builtins.iter(items):
            run = Item(
                DurableRunConfig(
                    agent=agent,
                    item_id=str(item_id_fn(item)),
                    partition_key=str(partition_key_fn(item)),
                    store=resolved_store,
                    invocation_id=invocation_id,
                    # Capture the item as the record's input_snapshot at the
                    # item boundary we already cross. This is the only write
                    # needed for item-replay: the item row now carries the
                    # exact payload that produced it, so a failed iter-item
                    # can be re-driven from its id. Zero added hot-path
                    # latency — store.create already writes the row; this
                    # just fills a column that was NULL before (the @agent
                    # path captures args via decorator, which iter has no
                    # equivalent of).
                    input_snapshot=item,
                )
            )
            run.init()
            token = _ACTIVE_RUN.set(run)
            try:
                try:
                    yield item
                except BaseException:
                    # When the for-loop body raises, Python sends
                    # ``GeneratorExit`` into the suspended yield as part of
                    # generator cleanup — the original exception propagates
                    # independently up the stack (it is NOT chained onto
                    # GeneratorExit.__context__). So ``BaseException as exc``
                    # here gets GeneratorExit, not the customer's exception,
                    # and we can't carry the original message into run.fail.
                    #
                    # What we CAN do — and what matters for the wedge — is
                    # mark the item as failed (status flip in the store) and
                    # write a synthetic TaskEntry whose ``outcome_reason``
                    # makes the cause visible in the audit trail. Operators
                    # see "this item failed because the loop body raised"
                    # without us inventing a fake error string.
                    _write_synthetic_entry(
                        run,
                        OutcomeVerdict("failed", "loop_body_exception"),
                    )
                    try:
                        run.fail(error="loop_body_exception")
                    except Exception:
                        log.exception(
                            "papayya.iter: run.fail() raised while handling loop-body exception"
                        )
                    # Re-raise the GeneratorExit so the generator unwinds
                    # cleanly; the original customer exception continues
                    # propagating to the caller via Python's normal stack
                    # unwind, untouched.
                    raise
                else:
                    try:
                        run.complete()
                    except Exception:
                        log.exception(
                            "papayya.iter: run.complete() raised at item boundary"
                        )
            finally:
                _ACTIVE_RUN.reset(token)
    finally:
        # Seal the run row (set the real total_items, roll up terminal
        # status) whether the loop finished, raised, or was abandoned.
        if owns_invocation:
            _finalize_invocation(resolved_store, invocation_id)


class _LazyIsolate:
    """Deferred per-item run for an ``@papayya.durable`` clean-path body.

    The decorator publishes one of these (not a run) before invoking the
    body. A run is minted only when the body actually touches an ambient
    verb (``@papayya.llm`` / ``@papayya.step`` / ``mark_degraded``) — which
    is precisely when the body is NOT managing its own ``papayya().run()``.
    Legacy bodies that call ``run()`` themselves never trip this, so their
    bootstrap-id / replay-hydration adoption is untouched.
    """

    __slots__ = ("agent", "item_id", "partition_key", "own_completion", "run")

    def __init__(
        self, agent: str, item_id: Any, partition_key: Any, own_completion: bool
    ) -> None:
        self.agent = agent
        self.item_id = item_id
        self.partition_key = partition_key
        self.own_completion = own_completion
        self.run: "PapayyaRun | None" = None


# Set by the @agent/@papayya.durable clean-path wrapper. Read by _resolve_run
# to lazily mint the ambient run on first in-body verb use.
_AMBIENT_ISOLATE: ContextVar["_LazyIsolate | None"] = ContextVar(
    "papayya_ambient_isolate", default=None
)


def _coerce_item_id(value: Any) -> str | None:
    """Best-effort item_id for a bare-decorator lazy run.

    The clean ``def f(item)`` path extracts ``args[0]`` as the item id, which
    is only a stable identifier when the first arg IS one (a str/int — the
    worker's ``fn(item_id)`` convention). A rich object (``def f(company)``
    called with a dataclass/dict) has no stable id here, so we record None
    rather than binding an unstable ``repr``. Correct per-item attribution for
    rich items is what ``papayya.map(..., item_id=lambda c: c.id)`` is for.
    """
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return str(value)
    return None


def _coerce_partition_key(value: Any) -> str | None:
    """Best-effort partition_key for a bare-decorator lazy run.

    Same posture as :func:`_coerce_item_id`: a str/int (a tenant id) coerces,
    anything richer records None. ``papayya.map(partition_key=lambda c: …)``
    is the correct-attribution path.
    """
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return str(value)
    return None


def _peek_run() -> "PapayyaRun | None":
    """The run ambient work would attach to, WITHOUT minting one.

    Checks ``_ACTIVE_RUN`` (set by ``iter`` and the ``def f(run, …)`` inject
    path), then an enclosing isolate's already-minted run. Used by the
    ``@papayya.durable`` wrapper to detect an owning run it should reuse.
    """
    run = _ACTIVE_RUN.get()
    if run is not None:
        return run
    iso = _AMBIENT_ISOLATE.get()
    return iso.run if iso is not None else None


def _resolve_run() -> "PapayyaRun | None":
    """Return the run the ambient verbs should write against.

    Preference order: an explicit run installed by ``iter`` or the
    ``def f(run, …)`` inject path (``_ACTIVE_RUN``); else a lazily-minted run
    for an ``@papayya.durable`` clean-path body (``_AMBIENT_ISOLATE``); else
    ``None`` (call runs bare — adoption stays rewarded, never required).
    """
    run = _ACTIVE_RUN.get()
    if run is not None:
        return run
    iso = _AMBIENT_ISOLATE.get()
    if iso is None:
        return None
    if iso.run is None:
        from papayya.durable import papayya as _factory
        from papayya.agent import _LEGACY_AGENT_PATH_ACTIVE

        # Minting the ambient run is the NEW recommended path — it must not
        # fire the "you called papayya().run() inside @agent" deprecation
        # warning, which keys off this flag. Suppress it just for the mint.
        tok = _LEGACY_AGENT_PATH_ACTIVE.set(False)
        try:
            # partition_key is passed explicitly (possibly None): the body
            # never sees run(), so strict-when-declared metadata extraction
            # can't apply here — an unattributed run beats a crash.
            iso.run = _factory().run(
                agent=iso.agent,
                item_id=_coerce_item_id(iso.item_id),
                partition_key=_coerce_partition_key(iso.partition_key),
            )
        finally:
            _LEGACY_AGENT_PATH_ACTIVE.reset(tok)
        # Create the run row now (same as iter does before the body) so a verb
        # that writes directly — e.g. mark_degraded's synthetic entry — has a
        # run to aggregate onto. run.step/llm_step self-init, but mark_* don't.
        iso.run.init()
        # Deliberately NOT published on _ACTIVE_RUN: this may execute inside
        # an asyncio.Task (a copied context), where a set() token could never
        # be reset from drive_ambient_*'s parent context. Later verbs resolve
        # through the isolate itself, which is shared across those contexts.
    return iso.run


def drive_ambient_sync(
    agent: str,
    item_id: Any,
    partition_key: Any,
    body: Callable[[], T],
    *,
    own_completion: bool,
) -> T:
    """Run an ``@papayya.durable`` clean-path body under a lazy isolate.

    Publishes an :class:`_LazyIsolate` so ambient verbs resolve (minting the
    run on first use), then — if a run was minted and ``own_completion`` —
    marks it completed/failed around the body. ``own_completion=False`` on the
    hosted worker path, where the worker/control-plane own terminal status.
    """
    iso = _LazyIsolate(agent, item_id, partition_key, own_completion)
    iso_token = _AMBIENT_ISOLATE.set(iso)
    try:
        try:
            result = body()
        except BaseException as exc:
            if iso.run is not None and own_completion:
                _write_synthetic_entry(iso.run, OutcomeVerdict("failed", "agent_body_exception"))
                try:
                    iso.run.fail(error=str(exc))
                except Exception:
                    log.exception("papayya: run.fail() raised handling an agent-body exception")
            raise
        else:
            if iso.run is not None and own_completion:
                try:
                    iso.run.complete()
                except Exception:
                    log.exception("papayya: run.complete() raised at agent-body return")
            return result
    finally:
        _AMBIENT_ISOLATE.reset(iso_token)


async def drive_ambient_async(
    agent: str,
    item_id: Any,
    partition_key: Any,
    body: Callable[[], Any],
    *,
    own_completion: bool,
) -> Any:
    """Async sibling of :func:`drive_ambient_sync`. ``body`` returns an awaitable."""
    iso = _LazyIsolate(agent, item_id, partition_key, own_completion)
    iso_token = _AMBIENT_ISOLATE.set(iso)
    try:
        try:
            result = await body()
        except BaseException as exc:
            if iso.run is not None and own_completion:
                _write_synthetic_entry(iso.run, OutcomeVerdict("failed", "agent_body_exception"))
                try:
                    iso.run.fail(error=str(exc))
                except Exception:
                    log.exception("papayya: run.fail() raised handling an agent-body exception")
            raise
        else:
            if iso.run is not None and own_completion:
                try:
                    iso.run.complete()
                except Exception:
                    log.exception("papayya: run.complete() raised at agent-body return")
            return result
    finally:
        _AMBIENT_ISOLATE.reset(iso_token)


def mark_degraded(reason: str) -> None:
    """Mark the active ``PapayyaRun``'s outcome as degraded.

    Writes a synthetic ``TaskEntry`` with ``outcome_status='degraded'``
    and the given ``reason``. The run-level ``worst_outcome_status`` /
    ``degraded_count`` aggregates update via Plan 01's ``save_task``
    aggregation logic on stores that support it (SQLiteStore, CloudStore).

    No-op (with a warning) if called outside an active ``papayya.iter`` run.
    """
    _mark(OutcomeVerdict("degraded", reason))


def mark_outcome(status: str, reason: str | None = None) -> None:
    """Mark the active ``PapayyaRun``'s outcome explicitly.

    ``status`` must be one of ``'ok'`` / ``'degraded'`` / ``'failed'``.
    ``'ok'`` is a no-op on aggregation but still writes a row so the
    audit trail shows the explicit assertion. Unknown statuses raise
    ``ValueError`` at the helper boundary — typos should not silently
    degrade behavior.
    """
    if status not in ("ok", "degraded", "failed"):
        raise ValueError(
            f"mark_outcome: status must be 'ok' | 'degraded' | 'failed', got {status!r}"
        )
    _mark(OutcomeVerdict(status, reason))


# --------------------------------------------------------------------------- #
#  Leaf-level adoption (L1): decorate the function that calls your model, not   #
#  your orchestration code. The wrap sits on the unit that owns execution —     #
#  the provider call — and reads the active run from the contextvar that        #
#  ``papayya.iter`` already sets per item. Your business functions never see a  #
#  ``run`` object and never gain a ``partition_key`` parameter.                 #
# --------------------------------------------------------------------------- #


def llm(fn: Callable | None = None, *, label: str | None = None):
    """Record each call to a provider-calling leaf function as an LLM step.

    Apply it to the function that actually calls your model::

        @papayya.llm
        def call_model(prompt: str) -> dict:
            return client.messages.create(...)

    Then drive items with :func:`iter` — your orchestration stays unchanged
    and Papayya-unaware::

        for item in papayya.iter(items, workload="triage",
                                 item_id=lambda i: i["id"],
                                 partition_key=lambda i: i["tenant"]):
            triage(item)        # call_model is recorded + outcome-inspected

    Every call inside an active ``papayya.iter`` run is captured with automatic
    ran-vs-worked detection (refusal / empty / degenerate stop-reason) and
    tagged with the item's ``partition_key``. Called **outside** an active run,
    the function runs bare with no recording — adoption is rewarded, never
    required (so the same code still works in tests and standalone scripts).
    """
    return _leaf_decorator(fn, label=label, kind="llm")


def step(fn: Callable | None = None, *, label: str | None = None):
    """Leaf decorator for a non-LLM step (retrieval, a tool call, a parse).

    Same contract as :func:`llm` — records the call against the active
    ``papayya.iter`` run and runs the empty/degenerate-result inspectors —
    but skips LLM usage/stop-reason extraction.
    """
    return _leaf_decorator(fn, label=label, kind="step")


def _leaf_decorator(fn: Callable | None, *, label: str | None, kind: str):
    def decorate(f: Callable) -> Callable:
        base_label = label or getattr(f, "__name__", None) or "step"

        def _record(run: "PapayyaRun") -> Callable:
            # run.step / run.llm_step return an async wrapper iff f is a
            # coroutine function, so the sync/async split below lines up.
            # Repeated calls per run are handled inside PapayyaRun: each
            # call of the same label becomes its own step (label#N).
            return (
                run.llm_step(base_label, f)
                if kind == "llm"
                else run.step(base_label, f)
            )

        if inspect.iscoroutinefunction(f):
            # Keep the wrapper a coroutine function too, so callers (and any
            # framework that introspects with iscoroutinefunction) still see
            # an awaitable.
            @functools.wraps(f)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                run = _resolve_run()
                if run is None:
                    return await f(*args, **kwargs)
                return await _record(run)(*args, **kwargs)

            return async_wrapper

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            run = _resolve_run()
            if run is None:
                # No ambient run (no iter / @papayya.durable in scope) — run
                # the function unchanged. Adoption is rewarded, never required.
                return f(*args, **kwargs)
            return _record(run)(*args, **kwargs)

        return wrapper

    # Support both bare ``@papayya.llm`` and parameterized ``@papayya.llm(label=…)``.
    return decorate(fn) if fn is not None else decorate


def active_item() -> "Item | None":
    """Return the active :class:`Item` handle, or ``None`` outside one.

    The handle exposes ``.id`` (the record's surrogate uuid — useful for
    correlating an item with the outcome Papayya recorded for it, e.g.
    ``store.load(papayya.active_item().id).worst_outcome_status``) plus the
    full step surface (``.step`` / ``.llm_step``).

    :func:`active_run_id` is the pre-Plan-34 spelling; it returns just the
    id string and is kept as a deprecated alias.
    """
    return _resolve_run()


def active_run_id() -> str | None:
    """Deprecated pre-Plan-34 alias: id of the active item, or ``None``.

    Use ``active_item().id`` instead — "run" now names the whole
    invocation, not the per-item record this returns.
    """
    run = _resolve_run()
    return run.run_id if run is not None else None


def _mark(verdict: OutcomeVerdict) -> None:
    run = _resolve_run()
    if run is None:
        log.warning(
            "papayya.mark_%s(%r) called outside an active papayya.iter run; ignored",
            verdict.status,
            verdict.reason,
        )
        return
    _write_synthetic_entry(run, verdict)


def _write_synthetic_entry(run: "PapayyaRun", verdict: OutcomeVerdict) -> None:
    """Write a synthetic ``TaskEntry`` carrying the outcome verdict.

    Plan 01's ``save_task`` aggregation picks up the entry and updates
    the parent run's ``worst_outcome_status`` / ``degraded_count``
    automatically on stores that support it.
    """
    label = f"papayya.mark/{uuid.uuid4().hex[:8]}"
    entry = TaskEntry(
        label=label,
        result=None,
        duration_ms=0,
        completed_at=datetime.now(timezone.utc).isoformat(),
        item_id=run._run_item_id,
        partition_key=run._partition_key,
        outcome_status=verdict.status,
        outcome_reason=verdict.reason,
    )
    try:
        run._store.save_task(run.run_id, entry)
    except Exception:
        log.exception(
            "papayya.mark_%s: store.save_task raised; outcome not persisted",
            verdict.status,
        )


# Public names for ``from papayya.iterators import *``. We intentionally do
# NOT include ``iter`` here — re-exporting it via star-import would shadow
# the builtin in any callsite using ``from papayya.iterators import *``,
# which is a surprise the customer should opt into via the qualified
# ``papayya.iter`` form.
__all__ = ["mark_degraded", "mark_outcome", "llm", "step", "active_item", "active_run_id"]
