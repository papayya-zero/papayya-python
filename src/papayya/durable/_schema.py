"""Shared column-name and table-name constants for the local SQLite store.

Plan 34 noun consolidation (schema v12). The local ledger speaks the
product vocabulary directly:

    agent (deployable unit) -> run (one invocation) -> item (one record
    processed) -> step (one node in an item's trace)

Shift-by-one from the pre-v12 schema:

    old table   new table   meaning
    ---------   ---------   -------
    batches     runs        one invocation (one map() call, one cron fire,
                            or an implicit run-of-one around a direct call)
    runs        items       one record processed: outcome, trace, cost
    tasks       steps       one trace node written by run.step()/llm_step()
    steps       (dropped)   dead legacy LLM-call log; no SDK writer since v2

These names are a contract with the hosted control-plane schema. The
hosted side renames at Plan 34 Unit 5; until then the HTTP wire contract
(cloud_store.py, resources/items.py) is FROZEN at the old names — do not
"fix" the wire to match these constants.
"""

from __future__ import annotations


# Table names (v12)
TBL_RUNS = "runs"      # invocations (was `batches`)
TBL_ITEMS = "items"    # per-item records (was `runs`)
TBL_STEPS = "steps"    # trace nodes (was `tasks`)
TBL_META = "_meta"


# Runs (invocation) columns — was the batches table.
COL_RUN_ID = "run_id"                  # was batch_id
COL_RUN_AGENT = "agent"
COL_RUN_STATUS = "status"
COL_RUN_TOTAL_ITEMS = "total_items"
COL_RUN_COMPLETED = "completed"
COL_RUN_FAILED = "failed"
COL_RUN_CONCURRENCY_CAP = "concurrency_cap"
COL_RUN_CREATED_AT = "created_at"
COL_RUN_COMPLETED_AT = "completed_at"
# v12: slice replay lineage — the run this run was minted to re-drive.
COL_RUN_REPLAYED_FROM = "replayed_from"


# Items columns — was the runs table. The surrogate PK renamed run_id -> id;
# the invocation FK renamed batch_id -> run_id. `item_id` stays reserved for
# CUSTOMER identity (e.g. "co_007"), exactly as before.
COL_ITEM_ID = "id"                     # surrogate uuid (was run_id)
COL_ITEM_RUN_ID = "run_id"             # invocation FK (was batch_id)
COL_ITEM_AGENT = "agent"
COL_ITEM_STATUS = "status"
COL_ITEM_OUTPUT = "output"
COL_ITEM_CREATED_AT = "created_at"
COL_ITEM_UPDATED_AT = "updated_at"
COL_ITEM_ERROR_CODE = "error_code"
COL_ITEM_ITEM_ID = "item_id"           # customer identity — unchanged
COL_ITEM_INPUT_SNAPSHOT = "input_snapshot"
COL_ITEM_DLQ_DISPOSITION = "dlq_disposition"
COL_ITEM_DLQ_RESOLVED_AT = "dlq_resolved_at"
COL_ITEM_REPLAYED_FROM = "replayed_from"
COL_ITEM_AGENT_VERSION = "agent_version"
COL_ITEM_METADATA = "metadata"
COL_ITEM_PARTITION_KEY = "partition_key"
# Sub-item lineage. Still named parent_run_id in v12 — the parent_id rename
# is a Unit 5/6 decision (hosted mirror + taste-check), not taken locally yet.
COL_ITEM_PARENT_RUN_ID = "parent_run_id"
COL_ITEM_WORST_OUTCOME_STATUS = "worst_outcome_status"
COL_ITEM_DEGRADED_COUNT = "degraded_count"


# Steps columns — was the tasks table. The FK to the parent item renamed
# run_id -> item_id (it references items.id). The old denormalized CUSTOMER
# item_id column renamed to customer_item_id — the plan reserves the bare
# `item_id` name for "which item row does this step belong to" at the step
# level, while customer identity keeps the bare name at the item level.
COL_STEP_ITEM_ID = "item_id"           # FK -> items.id (was tasks.run_id)
COL_STEP_LABEL = "label"
COL_STEP_RESULT = "result"
COL_STEP_DURATION_MS = "duration_ms"
COL_STEP_COMPLETED_AT = "completed_at"
COL_STEP_CUSTOMER_ITEM_ID = "customer_item_id"  # was tasks.item_id
COL_STEP_INPUT_SNAPSHOT = "input_snapshot"
COL_STEP_OUTPUT_SNAPSHOT = "output_snapshot"
COL_STEP_KIND = "kind"
COL_STEP_LLM_PROMPT_TOKENS = "llm_prompt_tokens"
COL_STEP_LLM_COMPLETION_TOKENS = "llm_completion_tokens"
COL_STEP_LLM_TOTAL_TOKENS = "llm_total_tokens"
COL_STEP_LLM_MODEL = "llm_model"
COL_STEP_LLM_STOP_REASON = "llm_stop_reason"
COL_STEP_LLM_PROVIDER_SHAPE = "llm_provider_shape"
COL_STEP_ERROR_CATEGORY = "error_category"
COL_STEP_AGENT_VERSION = "agent_version"
COL_STEP_DELIVERY_ATTEMPTS = "delivery_attempts"
COL_STEP_JOURNALED_AT = "journaled_at"
COL_STEP_METADATA = "metadata"
COL_STEP_PARTITION_KEY = "partition_key"
COL_STEP_OUTCOME_STATUS = "outcome_status"
COL_STEP_OUTCOME_REASON = "outcome_reason"


# Disposition values — contract with the UI's DLQ section and the hosted CP.
DLQ_REPLAYED = "replayed"
DLQ_SKIPPED = "skipped"
DLQ_ACKNOWLEDGED = "acknowledged"


# Schema version bumps — update both sides when adding a migration
SCHEMA_VERSION = "12"


# Indexes (v12 names) — named explicitly so tests can check their presence.
IDX_ITEMS_RUN = "idx_items_run"                # was idx_runs_batch
IDX_ITEMS_ITEM = "idx_items_item"              # was idx_runs_item (customer id)
IDX_ITEMS_DLQ = "idx_items_dlq"                # was idx_runs_dlq
IDX_ITEMS_PARTITION = "idx_items_partition"    # was idx_runs_partition
IDX_ITEMS_PARENT = "idx_items_parent"          # was idx_runs_parent
IDX_STEPS_ITEM = "idx_steps_item"              # was idx_tasks_run_id
IDX_STEPS_CUSTOMER_ITEM = "idx_steps_customer_item"  # was idx_tasks_item
IDX_STEPS_PARTITION = "idx_steps_partition"    # was idx_tasks_partition
