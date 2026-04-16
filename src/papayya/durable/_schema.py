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


# Schema version bumps — update both sides when adding a migration
SCHEMA_VERSION = "4"


# Indexes — named explicitly so we can check for their presence in tests
IDX_STEPS_TOOL = "idx_steps_tool"
IDX_STEPS_ERROR = "idx_steps_error"
IDX_RUNS_BATCH = "idx_runs_batch"
IDX_TASKS_ITEM = "idx_tasks_item"
IDX_RUNS_ITEM = "idx_runs_item"
