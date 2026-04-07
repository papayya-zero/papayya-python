"""Core types for the durable execution wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class BudgetExceededError(Exception):
    """Raised when a run exceeds its budget limit."""

    def __init__(self, consumed_usd: float, limit_usd: float) -> None:
        self.consumed_usd = consumed_usd
        self.limit_usd = limit_usd
        super().__init__(
            f"Budget exceeded: ${consumed_usd:.4f} consumed, limit is ${limit_usd:.2f}"
        )


@dataclass
class TaskEntry:
    """A cached task result stored by the checkpoint store."""

    label: str
    result: Any
    cost_usd: float
    duration_ms: int
    completed_at: str


@dataclass
class RunCheckpoint:
    """Snapshot of a full run's checkpoint state."""

    run_id: str
    agent: str
    tasks: list[TaskEntry]
    status: str  # "running" | "completed" | "failed"
    budget_consumed_usd: float
    budget_limit_usd: float | None
    created_at: str
    updated_at: str


@dataclass
class DurableRunConfig:
    """Configuration for a durable run."""

    agent: str
    run_id: str | None = None
    budget_usd: float | None = None
    metadata: dict[str, Any] | None = None
    store: CheckpointStore | None = None


@dataclass
class DurableRunResult:
    """Summary returned when a durable run completes."""

    run_id: str
    agent: str
    status: str
    tasks: list[TaskEntry]
    total_cost_usd: float
    total_duration_ms: int


@runtime_checkable
class CheckpointStore(Protocol):
    """Interface for checkpoint persistence backends."""

    def load(self, run_id: str) -> RunCheckpoint | None: ...
    def save_task(self, run_id: str, entry: TaskEntry) -> None: ...
    def set_status(self, run_id: str, status: str, output: Any = None) -> None: ...
    def create(self, checkpoint: RunCheckpoint) -> None: ...
