"""PapayyaRun — durable execution wrapper for any function."""

from __future__ import annotations

import functools
import time as _time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar, overload

from .store import MemoryStore
from .types import (
    BudgetExceededError,
    CheckpointStore,
    DurableRunConfig,
    DurableRunResult,
    RunCheckpoint,
    TaskEntry,
)

T = TypeVar("T")


class PapayyaRun:
    """A durable run that wraps functions as checkpoint-able tasks.

    **Execution guarantee:** at-least-once. If a crash occurs between
    executing a task and saving its checkpoint, the task will re-execute
    on resume. Design tasks to be idempotent (safe to run more than once).

    Usage::

        run = PapayyaRun(DurableRunConfig(agent="my-agent", budget_usd=1.0))

        search = run.task("search", search_web)
        summarize = run.task("summarize", summarize_results)

        results = search(query)       # cached on replay
        summary = summarize(results)  # cached on replay

        run.complete(summary)

    Or with decorators::

        @run.task("search")
        def search(query: str) -> list[str]:
            return search_web(query)
    """

    def __init__(self, config: DurableRunConfig) -> None:
        self.agent = config.agent
        self.run_id = config.run_id or str(uuid.uuid4())
        self._store: CheckpointStore = config.store or MemoryStore()
        self._budget_limit_usd = config.budget_usd
        self._budget_consumed_usd = 0.0
        self._cache: dict[str, TaskEntry] = {}
        self._task_call_order: list[str] = []
        self._initialized = False
        self._finished = False

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
            self._budget_consumed_usd = existing.budget_consumed_usd
        else:
            now = datetime.now(timezone.utc).isoformat()
            checkpoint = RunCheckpoint(
                run_id=self.run_id,
                agent=self.agent,
                tasks=[],
                status="running",
                budget_consumed_usd=0,
                budget_limit_usd=self._budget_limit_usd,
                created_at=now,
                updated_at=now,
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

    def task(self, label_or_fn=None, fn=None):  # type: ignore[no-untyped-def]
        """Wrap a function as a durable task.

        Three calling conventions:

        1. ``run.task("label", some_fn)``  — higher-order, explicit label
        2. ``run.task(some_fn)``           — higher-order, label = fn.__name__
        3. ``@run.task("label")``          — decorator with explicit label
        """
        # Case 1: run.task("label", fn)
        if isinstance(label_or_fn, str) and fn is not None:
            return self._wrap(label_or_fn, fn)

        # Case 2: run.task(fn)
        if callable(label_or_fn):
            label = label_or_fn.__name__
            if not label or label == "<lambda>":
                raise ValueError(
                    "Anonymous/lambda functions require an explicit label: "
                    "run.task('myLabel', lambda: ...)"
                )
            return self._wrap(label, label_or_fn)

        # Case 3: @run.task("label") — return decorator
        if isinstance(label_or_fn, str):
            label = label_or_fn

            def decorator(f: Callable[..., T]) -> Callable[..., T]:
                return self._wrap(label, f)

            return decorator

        raise TypeError("task() requires a label string or a callable")

    def _wrap(self, label: str, fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            self.init()
            self._throw_if_finished()

            # Check cache for replay
            cached = self._cache.get(label)
            if cached is not None:
                return cached.result  # type: ignore[return-value]

            # Check budget
            self._throw_if_budget_exceeded()

            # Execute. Cost tracking is the caller's responsibility — invoke
            # `run.record_cost(...)` from inside `fn` after your LLM call, or
            # pass a cost via the returned value and record it explicitly.
            # Papayya does not observe LLM calls automatically.
            cost_usd = 0.0
            start = _time.monotonic()
            result = fn(*args, **kwargs)
            duration_ms = int((_time.monotonic() - start) * 1000)

            entry = TaskEntry(
                label=label,
                result=result,
                cost_usd=cost_usd,
                duration_ms=duration_ms,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )

            self._budget_consumed_usd += cost_usd
            self._cache[label] = entry
            self._task_call_order.append(label)
            self._store.save_task(self.run_id, entry)

            return result  # type: ignore[return-value]

        return wrapper  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    #  Budget                                                              #
    # ------------------------------------------------------------------ #

    def record_cost(self, cost_usd: float) -> None:
        """Record a cost against this run's budget."""
        self._budget_consumed_usd += cost_usd

    @property
    def budget(self) -> dict[str, Any]:
        """Current budget state."""
        exceeded = (
            self._budget_limit_usd is not None
            and self._budget_consumed_usd >= self._budget_limit_usd
        )
        remaining = (
            max(0, self._budget_limit_usd - self._budget_consumed_usd)
            if self._budget_limit_usd is not None
            else None
        )
        return {
            "consumed_usd": self._budget_consumed_usd,
            "limit_usd": self._budget_limit_usd,
            "remaining": remaining,
            "exceeded": exceeded,
        }

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

    def _throw_if_budget_exceeded(self) -> None:
        if (
            self._budget_limit_usd is not None
            and self._budget_consumed_usd >= self._budget_limit_usd
        ):
            raise BudgetExceededError(self._budget_consumed_usd, self._budget_limit_usd)

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
            total_cost_usd=self._budget_consumed_usd,
            total_duration_ms=sum(t.duration_ms for t in tasks),
        )
