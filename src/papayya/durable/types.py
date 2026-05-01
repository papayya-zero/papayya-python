"""Core types for the durable execution wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class TaskEntry:
    """A cached task result stored by the checkpoint store."""

    label: str
    result: Any
    duration_ms: int
    completed_at: str
    # Slice 6: per-object state snapshots. `item_id` is the user-supplied
    # record identifier; snapshots are arbitrary JSON-encodable payloads
    # captured at step boundaries for lineage/drift/replay.
    item_id: str | None = None
    input_snapshot: Any = None
    output_snapshot: Any = None
    # BYOF observability: populated when the step was wrapped with
    # ``kind="llm"``. All fields are ``None`` for non-LLM steps and for
    # LLM steps whose provider shape was not recognized.
    kind: str | None = None
    llm_prompt_tokens: int | None = None
    llm_completion_tokens: int | None = None
    llm_total_tokens: int | None = None
    llm_model: str | None = None
    llm_stop_reason: str | None = None
    llm_provider_shape: str | None = None
    error_category: str | None = None
    # ADR-0002 #7: agent version that produced this task row. Denormalized
    # from the parent run so the dashboard can display it on a step without
    # joining. Stays None for runs created before the v7 migration and for
    # in-process MemoryStore use where no registration is in scope.
    agent_version: str | None = None
    # v9: multi-tenancy metadata. metadata is the JSON blob captured at
    # run() time; tenant_key is the value extracted from metadata using
    # the path declared in papayya.yaml. Both denormalize from the run
    # so the dashboard can filter steps by tenant without joining.
    metadata: dict[str, Any] | None = None
    tenant_key: str | None = None


@dataclass
class RunCheckpoint:
    """Snapshot of a full run's checkpoint state."""

    run_id: str
    agent: str
    tasks: list[TaskEntry]
    status: str  # "running" | "completed" | "failed" | "partial" (batch-only)
    created_at: str = ""
    updated_at: str = ""
    item_id: str | None = None  # Slice 6: run-level item identifier.
    # DLQ replay source — captured at run creation. When a run enters the
    # dead letter queue (status=failed, disposition=null), an operator
    # invokes replay which creates a new run using this payload as input.
    # Opaque JSON-encodable value; papayya does not inspect the shape.
    input_snapshot: Any = None
    # ADR-0002 #7: agent version this run executed under. Source of truth for
    # the replay-mismatch gate; the same value denormalizes onto every task
    # row written for this run.
    agent_version: str | None = None
    # v9: multi-tenancy metadata convention. metadata is the user-supplied
    # JSON captured at run() time. tenant_key is the value extracted at
    # the path declared by `tenant_key:` in papayya.yaml — populated only
    # when the project config opts in. Both fields stay None for runs
    # created before v9 and for projects with no tenant_key declaration.
    metadata: dict[str, Any] | None = None
    tenant_key: str | None = None


@dataclass
class DurableRunConfig:
    """Configuration for a durable run."""

    agent: str
    run_id: str | None = None
    metadata: dict[str, Any] | None = None
    store: CheckpointStore | None = None
    # Slice 6: run-level item_id. If set, every step inherits this item_id
    # unless a step overrides it via run.step(..., item_id=...).
    item_id: str | None = None
    # v9: tenant key value extracted from metadata at the path declared in
    # papayya.yaml. Resolved at PapayyaClient.run() construction time so
    # PapayyaRun never has to re-read the project config.
    tenant_key: str | None = None


@dataclass
class DurableRunResult:
    """Summary returned when a durable run completes."""

    run_id: str
    agent: str
    status: str
    tasks: list[TaskEntry]
    total_duration_ms: int


@runtime_checkable
class CheckpointStore(Protocol):
    """Interface for checkpoint persistence backends."""

    def load(self, run_id: str) -> RunCheckpoint | None: ...
    def save_task(self, run_id: str, entry: TaskEntry) -> None: ...
    def set_status(self, run_id: str, status: str, output: Any = None) -> None: ...
    def create(self, checkpoint: RunCheckpoint) -> None: ...
