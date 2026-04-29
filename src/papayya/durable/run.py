"""PapayyaRun — durable execution wrapper for any function."""

from __future__ import annotations

import functools
import time as _time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar, overload

from papayya.classify import classify_provider_error
from papayya.errors import CreditExhausted
from papayya.llm_extract import extract_llm_usage
from papayya.runtime_context import get_current_reporter

from .store import MemoryStore
from .types import (
    CheckpointStore,
    DurableRunConfig,
    DurableRunResult,
    RunCheckpoint,
    TaskEntry,
)

T = TypeVar("T")


class PapayyaRun:
    """A durable run that wraps functions as checkpoint-able steps.

    **Execution guarantee:** at-least-once. If a crash occurs between
    executing a step and saving its checkpoint, the step will re-execute
    on resume. Design steps to be idempotent (safe to run more than once).

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
        self._store: CheckpointStore = config.store or MemoryStore()
        self._cache: dict[str, TaskEntry] = {}
        self._task_call_order: list[str] = []
        self._initialized = False
        self._finished = False
        # Run-level item_id. Seeded from config; the first step that passes
        # item_id= also seeds it if still unset. Subsequent steps inherit
        # unless they pass an explicit override (which applies to that step
        # only — the run-level id does not change mid-run).
        self._run_item_id: str | None = config.item_id

    def init(self) -> None:
        """Load any existing checkpoint from the store."""
        if self._initialized:
            return
        self._initialized = True

        existing = self._store.load(self.run_id)
        if existing is not None:
            for entry in existing.tasks:
                self._cache[entry.label] = entry
                self._task_call_order.append(entry.label)
        else:
            # Read the @agent wrapper's captured call args. None when the
            # caller bypassed the decorator (scripts, tests). Stays as-is
            # — we never inject a synthetic snapshot here.
            from papayya.agent import consume_agent_input_snapshot

            now = datetime.now(timezone.utc).isoformat()
            checkpoint = RunCheckpoint(
                run_id=self.run_id,
                agent=self.agent,
                tasks=[],
                status="running",
                created_at=now,
                updated_at=now,
                item_id=self._run_item_id,
                input_snapshot=consume_agent_input_snapshot(),
            )
            self._store.create(checkpoint)

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
        snapshot: Any = None,
        kind: str | None = None,
    ):
        """Wrap a function as a durable step. (Alias: ``run.step``.)

        Three calling conventions:

        1. ``run.step("label", some_fn)``  — higher-order, explicit label
        2. ``run.step(some_fn)``           — higher-order, label = fn.__name__
        3. ``@run.step("label")``          — decorator with explicit label

        Optional kwargs:

        * ``item_id`` — identifier of the record this step acts on. If set,
          the step row gets tagged with it; the first step to pass one also
          seeds the run-level item_id for later steps to inherit.
        * ``snapshot`` — arbitrary JSON-encodable payload captured as the
          step's *input* state. The function's return value is captured as
          the step's *output* state whenever an item_id is in effect.
        * ``kind`` — optional step-kind hint. Pass ``"llm"`` to wrap an LLM
          call; the wrapper runs shape-based usage extraction on the
          returned response (tokens, model, stop_reason) and classifies
          any raised provider exception via ``classify_provider_error`` —
          credit-shaped exceptions are re-raised as ``CreditExhausted``
          so the runtime pauses instead of failing. Unrecognized shapes
          still record that the step ran; they just lose token granularity.

        All kwargs are additive and optional.
        """
        # Case 1: run.task("label", fn)
        if isinstance(label_or_fn, str) and fn is not None:
            return self._wrap(label_or_fn, fn, item_id=item_id, snapshot=snapshot, kind=kind)

        # Case 2: run.task(fn)
        if callable(label_or_fn):
            label = label_or_fn.__name__
            if not label or label == "<lambda>":
                raise ValueError(
                    "Anonymous/lambda functions require an explicit label: "
                    "run.task('myLabel', lambda: ...)"
                )
            return self._wrap(label, label_or_fn, item_id=item_id, snapshot=snapshot, kind=kind)

        # Case 3: @run.task("label") — return decorator
        if isinstance(label_or_fn, str):
            label = label_or_fn
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

    def _wrap(
        self,
        label: str,
        fn: Callable[..., T],
        *,
        item_id: str | None = None,
        snapshot: Any = None,
        kind: str | None = None,
    ) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            self.init()
            self._throw_if_finished()

            # Check cache for replay
            cached = self._cache.get(label)
            if cached is not None:
                return cached.result  # type: ignore[return-value]

            # Resolve effective item_id: explicit per-step kwarg wins; else
            # inherit the run-level id. First step to supply an explicit id
            # also seeds the run-level id for later inheritance.
            effective_item_id = item_id if item_id is not None else self._run_item_id
            if item_id is not None and self._run_item_id is None:
                self._run_item_id = item_id

            # Snapshot the runtime reporter's intercepted-call count before
            # running the fn. If it goes up during the call, the interceptor
            # already reported this LLM call and the wrapper must NOT emit a
            # second step row (double-counting cost + tokens).
            runtime_reporter = get_current_reporter() if kind == "llm" else None
            pre_intercepted = (
                runtime_reporter.intercepted_call_count()
                if runtime_reporter is not None
                else 0
            )

            start = _time.monotonic()
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                # For LLM-kind steps, classify the provider exception shape
                # and promote credit-exhaustion errors so the runtime pauses
                # the run (same behavior the interceptor produces for
                # patched providers). Non-LLM steps propagate unchanged.
                if kind == "llm":
                    category = classify_provider_error(exc)
                    if runtime_reporter is not None:
                        post = runtime_reporter.intercepted_call_count()
                        if post == pre_intercepted:
                            # Interceptor didn't see the failure; emit it
                            # so the dashboard has a step row for the
                            # unpatched provider's error.
                            duration_ms_exc = int((_time.monotonic() - start) * 1000)
                            from papayya.llm_extract import LlmUsage as _Usage
                            runtime_reporter.report_llm_call(
                                label=label,
                                usage=_Usage(None, None, None, None, None, "unknown"),
                                duration_ms=duration_ms_exc,
                                error_category=category,
                            )
                    if category == "credit" and not isinstance(exc, CreditExhausted):
                        raise CreditExhausted(
                            f"{label}: provider credits exhausted ({exc})"
                        ) from exc
                raise
            duration_ms = int((_time.monotonic() - start) * 1000)

            # LLM usage extraction runs on the returned response when this
            # step was wrapped with kind="llm". Extraction never raises —
            # unknown shapes fall through to provider_shape="unknown".
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
                # provider (openai / anthropic) in run.step(kind="llm").
                if runtime_reporter is not None:
                    post_intercepted = runtime_reporter.intercepted_call_count()
                    if post_intercepted == pre_intercepted:
                        runtime_reporter.report_llm_call(
                            label=label,
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

            # Snapshots only populate when an item_id is in effect — the
            # lineage view has no home for snapshots that aren't attached
            # to an item, and we'd rather not bloat the DB with noise.
            if effective_item_id is not None:
                input_snapshot = snapshot
                output_snapshot = result
            else:
                input_snapshot = None
                output_snapshot = None

            entry = TaskEntry(
                label=label,
                result=result,
                duration_ms=duration_ms,
                completed_at=datetime.now(timezone.utc).isoformat(),
                item_id=effective_item_id,
                input_snapshot=input_snapshot,
                output_snapshot=output_snapshot,
                kind=kind,
                llm_prompt_tokens=llm_prompt_tokens,
                llm_completion_tokens=llm_completion_tokens,
                llm_total_tokens=llm_total_tokens,
                llm_model=llm_model,
                llm_stop_reason=llm_stop_reason,
                llm_provider_shape=llm_provider_shape,
            )

            self._cache[label] = entry
            self._task_call_order.append(label)
            self._store.save_task(self.run_id, entry)

            return result  # type: ignore[return-value]

        return wrapper  # type: ignore[return-value]

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
