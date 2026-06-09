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

from papayya.durable import PapayyaRun
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
    workload: str,
    item_id: Callable[[T], str],
    partition_key: Callable[[T], str],
    store: Any | None = None,
) -> Iterator[T]:
    """Wrap an iterable so each yielded item runs inside its own PapayyaRun.

    Required kwargs:
      ``workload`` — workload slug for the loop (string, not a lambda).
        Maps to ``DurableRunConfig.agent`` in v1; Plan 11 introduces a
        parallel workload field with explicit semantics.
      ``item_id`` — callable extracting a stable per-item identifier.
        Required; explicit attribution is the whole point.
      ``partition_key`` — callable extracting the per-item partition key
        (typically tenant id). Required for multi-tenant visibility.
      ``store`` — optional checkpoint store. When omitted, the same store
        the ``@agent`` / client path uses is resolved automatically
        (SQLiteStore locally, CloudStore when an API key is present), so
        iter-runs persist and show up in ``papayya dev`` like any other run.
        Pass an explicit store to point a batch at a specific database.

    Behavior:

    * For each item: open a ``PapayyaRun``, install it as the active run
      via contextvar, yield the item to the caller's loop body.
    * On body return: ``run.complete()``, reset the contextvar.
    * On body exception: write a synthetic ``failed`` TaskEntry, call
      ``run.fail(error=str(exc))``, reset the contextvar, re-raise the
      original exception.
    * The active-run contextvar is :func:`mark_degraded` /
      :func:`mark_outcome`'s source of truth.
    """
    return _iter_gen(items, workload, item_id, partition_key, store)


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


def _iter_gen(
    items: Iterable[T],
    workload: str,
    item_id_fn: Callable[[T], str],
    partition_key_fn: Callable[[T], str],
    store: Any | None = None,
) -> Iterator[T]:
    resolved_store = store if store is not None else _resolve_default_store()
    # Use builtins.iter inside this module so the public name doesn't
    # shadow it for our own implementation. The for-loop below already
    # uses iteration protocol, so this is defense-in-depth — anything
    # that accepts Iterable[T] is fine.
    for item in builtins.iter(items):
        run = PapayyaRun(
            DurableRunConfig(
                agent=workload,
                item_id=str(item_id_fn(item)),
                partition_key=str(partition_key_fn(item)),
                store=resolved_store,
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
                # What we CAN do — and what matters for the wedge — is mark
                # the run as failed (status flip in the store) and write a
                # synthetic TaskEntry whose ``outcome_reason`` makes the
                # cause visible in the audit trail. Operators see "this
                # item's run failed because the loop body raised" without
                # us inventing a fake error string.
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
            effective_label = _next_label(run, base_label)
            return (
                run.llm_step(effective_label, f)
                if kind == "llm"
                else run.step(effective_label, f)
            )

        if inspect.iscoroutinefunction(f):
            # Keep the wrapper a coroutine function too, so callers (and any
            # framework that introspects with iscoroutinefunction) still see
            # an awaitable.
            @functools.wraps(f)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                run = _ACTIVE_RUN.get()
                if run is None:
                    return await f(*args, **kwargs)
                return await _record(run)(*args, **kwargs)

            return async_wrapper

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            run = _ACTIVE_RUN.get()
            if run is None:
                # No ambient papayya.iter run — run the function unchanged.
                return f(*args, **kwargs)
            return _record(run)(*args, **kwargs)

        return wrapper

    # Support both bare ``@papayya.llm`` and parameterized ``@papayya.llm(label=…)``.
    return decorate(fn) if fn is not None else decorate


def _next_label(run: "PapayyaRun", base_label: str) -> str:
    """Give each call its own step label within a run.

    Durable steps are keyed by ``(run_id, label)``; a leaf decorated function
    called more than once per item would otherwise collide on one label and
    only execute once. The first call keeps the clean name; later calls in the
    same run get a ``#n`` suffix so each is its own step.
    """
    counts = getattr(run, "_leaf_step_counts", None)
    if counts is None:
        counts = {}
        try:
            run._leaf_step_counts = counts  # type: ignore[attr-defined]
        except Exception:
            # Run forbids attribute assignment — fall back to the bare label
            # (multiple same-label calls per item will dedupe, acceptable).
            return base_label
    n = counts.get(base_label, 0)
    counts[base_label] = n + 1
    return base_label if n == 0 else f"{base_label}#{n}"


def active_run_id() -> str | None:
    """Return the id of the active ``papayya.iter`` run, or ``None`` outside one.

    Useful for correlating an item with the outcome Papayya recorded for it
    (e.g. ``store.load(papayya.active_run_id()).worst_outcome_status``).
    """
    run = _ACTIVE_RUN.get()
    return run.run_id if run is not None else None


def _mark(verdict: OutcomeVerdict) -> None:
    run = _ACTIVE_RUN.get()
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
__all__ = ["mark_degraded", "mark_outcome", "llm", "step", "active_run_id"]
