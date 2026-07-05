"""Core types for the durable execution wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

# Sentinel for DurableRunConfig.input_snapshot. Distinguishes "caller did
# not supply a snapshot" (fall back to the @agent contextvar in
# PapayyaRun.init) from "caller supplied None" (an explicit empty snapshot).
# The iter() wrapper supplies the item itself; the @agent path leaves this
# unset so init() reads consume_agent_input_snapshot().
_UNSET: Any = object()


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
    # v9: partition-key metadata. metadata is the JSON blob captured at
    # run() time; partition_key is the value extracted from metadata
    # using the path declared in papayya.yaml. Both denormalize from
    # the run so the dashboard can filter steps by partition without
    # joining (the most common use case is per-tenant filtering).
    metadata: dict[str, Any] | None = None
    partition_key: str | None = None
    # v11: structural outcome accountability. status defaults to 'ok'; the
    # @agent wrapper (Plan 02) overwrites it via structural inspectors.
    # 'failed' is reserved for the future failed-row write path and for
    # control-pane writes; this plan only writes 'ok' / 'degraded'.
    outcome_status: str = "ok"
    outcome_reason: str | None = None


_OUTCOME_SEVERITY = {"ok": 0, "degraded": 1, "failed": 2}


def _outcome_severity(status: str) -> int:
    """Numeric severity for an outcome status. Unknown statuses are 'ok'-equivalent
    (fail-safe: don't escalate a run's worst-outcome on a typo). Both stores
    and Plan 02's inspectors use this to maintain a consistent severity order."""
    return _OUTCOME_SEVERITY.get(status, 0)


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
    # v9: partition-key metadata convention. metadata is the
    # user-supplied JSON captured at run() time. partition_key is the
    # value extracted at the path declared by `partition_key:` in
    # papayya.yaml — populated only when the project config opts in.
    # Both fields stay None for runs created before v9 and for projects
    # with no partition_key declaration.
    metadata: dict[str, Any] | None = None
    partition_key: str | None = None
    # v10: sub-runs lineage (Layer 3 #7). run_id of the outer run that
    # spawned this one. None on top-level runs; populated by the
    # dispatcher (Phase 2) when a run is created from inside another
    # run's lifetime.
    parent_run_id: str | None = None
    # v11: denormalized worst-outcome across this run's task entries.
    # worst_outcome_status is the max severity seen so far (see
    # _outcome_severity); degraded_count is how many task entries are
    # not 'ok'. Both update incrementally on each save_task and round-trip
    # through the local + cloud stores.
    worst_outcome_status: str = "ok"
    degraded_count: int = 0


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
    # v9: partition key value extracted from metadata at the path
    # declared in papayya.yaml. Resolved at PapayyaClient.run()
    # construction time so PapayyaRun never has to re-read the project
    # config.
    partition_key: str | None = None
    # Replay snapshot supplied at construction. The @agent decorator leaves
    # this _UNSET and lets PapayyaRun.init() read the call args from the
    # contextvar (consume_agent_input_snapshot); the iter() wrapper, which
    # has no decorator to capture args, passes the per-item payload here so
    # the run row carries an input_snapshot and `replay(run_id)` has
    # something to re-drive. _UNSET (not None) is the "unset" marker so an
    # explicit None snapshot is distinguishable from "fall back to @agent".
    input_snapshot: Any = _UNSET
    # v10 / Layer 3 #7 Phase 2: sub-runs lineage. The outer run's id
    # when this run was spawned from inside an @agent body; None for
    # top-level runs. Resolved by Papayya.run() (explicit kwarg wins,
    # else _ACTIVE_RUN_ID contextvar set by the @agent wrapper).
    parent_run_id: str | None = None
    # Replay Phase 3 hydration transport. When set, PapayyaRun.init()
    # seeds its in-memory _cache with these TaskEntry rows before the
    # normal store.create() path so the wrapped agent fn's first
    # step() calls find cache hits for labels < from_step. The list is
    # never persisted to the new run's tasks table — it lives only in
    # memory for this run's lifetime, so the on-disk artifact for the
    # new run contains only steps it actually re-executed. Populated
    # by Papayya.run() reading the one-shot _REPLAY_HYDRATION
    # contextvar that papayya.durable._replay sets before invoking the
    # agent fn. Stays None for non-replay callers.
    prepopulated_tasks: list[TaskEntry] | None = None


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
