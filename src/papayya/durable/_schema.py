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


# Schema version bumps — update both sides when adding a migration
SCHEMA_VERSION = "6"


# Indexes — named explicitly so we can check for their presence in tests
IDX_STEPS_TOOL = "idx_steps_tool"
IDX_STEPS_ERROR = "idx_steps_error"
IDX_RUNS_BATCH = "idx_runs_batch"
IDX_TASKS_ITEM = "idx_tasks_item"
IDX_RUNS_ITEM = "idx_runs_item"
IDX_RUNS_DLQ = "idx_runs_dlq"
