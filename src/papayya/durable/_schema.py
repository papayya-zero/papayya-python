"""Shared column-name and table-name constants for the local SQLite store.

These names are a contract with the hosted control-plane schema (see
`control-pane/migrations/031_batches.up.sql` when it lands). Any divergence
between local and hosted column names breaks the mental-model transfer that
is the whole point of the local dev dashboard — so every new column goes
here first, and the migration / queries reference it by constant.

Renames must happen in lockstep with the control-plane side. When in doubt,
match the hosted name exactly.
"""

from __future__ import annotations


# Table names
TBL_BATCHES = "batches"
TBL_RUNS = "runs"
TBL_STEPS = "steps"
TBL_TASKS = "tasks"
TBL_META = "_meta"


# Batches columns
COL_BATCH_ID = "batch_id"
COL_BATCH_AGENT = "agent"
COL_BATCH_STATUS = "status"
COL_BATCH_TOTAL_ITEMS = "total_items"
COL_BATCH_COMPLETED = "completed"
COL_BATCH_FAILED = "failed"
COL_BATCH_CONCURRENCY_CAP = "concurrency_cap"
COL_BATCH_CREATED_AT = "created_at"
COL_BATCH_COMPLETED_AT = "completed_at"


# New columns on runs (Slice 1 adds these)
COL_RUN_BATCH_ID = "batch_id"
COL_RUN_ERROR_CODE = "error_code"


# New columns on steps (Slice 1 adds these)
COL_STEP_TOOL_NAME = "tool_name"
COL_STEP_ERROR_CODE = "error_code"
COL_STEP_ERROR_CATEGORY = "error_category"
COL_STEP_INPUT_HASH = "input_hash"


# Slice 6 columns — per-object state snapshots at step boundaries.
# item_id is user-supplied; snapshots are JSON-encoded payloads. Denormalized
# onto runs so the dashboard can list items within a batch without joining
# through every task.
COL_TASK_ITEM_ID = "item_id"
COL_TASK_INPUT_SNAPSHOT = "input_snapshot"
COL_TASK_OUTPUT_SNAPSHOT = "output_snapshot"
COL_RUN_ITEM_ID = "item_id"


# v5 columns — BYOF observability fields captured by run.step(kind="llm").
# Nullable: only populate when the caller passes kind="llm" and the provider
# shape is recognized. error_category fills on provider exceptions classified
# by the SDK's shared classifier (provider/timeout/tool/logic).
COL_TASK_KIND = "kind"
COL_TASK_LLM_PROMPT_TOKENS = "llm_prompt_tokens"
COL_TASK_LLM_COMPLETION_TOKENS = "llm_completion_tokens"
COL_TASK_LLM_TOTAL_TOKENS = "llm_total_tokens"
COL_TASK_LLM_MODEL = "llm_model"
COL_TASK_LLM_STOP_REASON = "llm_stop_reason"
COL_TASK_LLM_PROVIDER_SHAPE = "llm_provider_shape"
COL_TASK_ERROR_CATEGORY = "error_category"


# v6 columns — dead-letter-queue primitive on the runs table.
# input_snapshot is the payload captured at run creation; it's the replay
# source when an operator re-drives a failed run from the DLQ. dlq_disposition
# is null while the run is pending triage and transitions to 'replayed',
# 'skipped', or 'acknowledged' once an operator acts. replayed_from links a
# new run to the dead letter it re-drove, forming a chain the UI can follow.
COL_RUN_INPUT_SNAPSHOT = "input_snapshot"
COL_RUN_DLQ_DISPOSITION = "dlq_disposition"
COL_RUN_DLQ_RESOLVED_AT = "dlq_resolved_at"
COL_RUN_REPLAYED_FROM = "replayed_from"

# Disposition values — contract with the UI's DLQ section and the hosted CP.
DLQ_REPLAYED = "replayed"
DLQ_SKIPPED = "skipped"
DLQ_ACKNOWLEDGED = "acknowledged"


# v7 columns — version-tagged lineage (ADR-0002 #7). Every run records the
# agent_version it ran on; the same value denormalizes onto each task row so
# the dashboard can display it on a step without an extra join. Replay reads
# the run's agent_version and refuses to use a registration with a different
# value unless the operator passes --latest.
COL_RUN_AGENT_VERSION = "agent_version"
COL_TASK_AGENT_VERSION = "agent_version"


# v8 columns — lineage delivery audit (ADR-0002 #8). When a CloudStore POST
# exhausts retries the SDK appends to a local journal sidecar; on the next
# successful POST the reconciler drains the journal and reissues the original
# request with these two fields populated. NULL means the row landed on the
# first delivery attempt — the common case. A non-NULL journaled_at is the
# signal the dashboard uses to render a "late delivery" badge on the step.
#
# Local SQLiteStore never writes these columns: synchronous disk writes have
# no journal-backed delivery path. They exist on the local schema purely for
# parity with the hosted side, so the same dashboard query shape works.
COL_TASK_DELIVERY_ATTEMPTS = "delivery_attempts"
COL_TASK_JOURNALED_AT = "journaled_at"


# v9 columns — partition-key metadata convention (most common use case:
# multi-tenancy). metadata is the full user-supplied JSON blob captured
# at run() time; partition_key is the value extracted from metadata at
# the path declared by `partition_key:` in papayya.yaml. Both denormalize
# onto every task row written under the run so the dashboard can
# filter/aggregate by partition without joining through runs.
# partition_key is the indexed column; metadata stays opaque JSON for
# queryability via json_extract when needed.
COL_RUN_METADATA = "metadata"
COL_RUN_PARTITION_KEY = "partition_key"
COL_TASK_METADATA = "metadata"
COL_TASK_PARTITION_KEY = "partition_key"


# v10 column — sub-runs lineage (Layer 3 #7). Mirrors the hosted
# control-pane runs.parent_run_id column from migration 054. NULL on
# top-level runs; the dispatcher sets this in Phase 2 when a run is
# created from inside another run's lifetime, so dashboards can roll
# up children under their parent.
COL_RUN_PARENT_RUN_ID = "parent_run_id"


# v11 columns — structural outcome accountability (Plan 01 of the
# workload-layer redesign). outcome_status / outcome_reason on tasks
# carry the verdict produced by structural inspectors (Plan 02 wires
# the writer). worst_outcome_status / degraded_count on runs are
# denormalized aggregates maintained on each task insert so dashboards
# can answer "which runs had any degraded step" without joining tasks.
# 'failed' is part of the severity order but no SDK write path produces
# it in this slice; reserved for the future failed-row path and for
# control-pane writes.
COL_TASK_OUTCOME_STATUS = "outcome_status"
COL_TASK_OUTCOME_REASON = "outcome_reason"
COL_RUN_WORST_OUTCOME_STATUS = "worst_outcome_status"
COL_RUN_DEGRADED_COUNT = "degraded_count"


# Schema version bumps — update both sides when adding a migration
SCHEMA_VERSION = "11"


# Indexes — named explicitly so we can check for their presence in tests
IDX_STEPS_TOOL = "idx_steps_tool"
IDX_STEPS_ERROR = "idx_steps_error"
IDX_RUNS_BATCH = "idx_runs_batch"
IDX_TASKS_ITEM = "idx_tasks_item"
IDX_RUNS_ITEM = "idx_runs_item"
IDX_RUNS_DLQ = "idx_runs_dlq"
IDX_RUNS_PARTITION = "idx_runs_partition"
IDX_TASKS_PARTITION = "idx_tasks_partition"
IDX_RUNS_PARENT = "idx_runs_parent"
