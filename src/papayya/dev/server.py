"""Local development dashboard server.

Serves a static dashboard UI and a JSON API that reads from the local
SQLite database. Uses only Python stdlib — no frameworks.

Plan 34 noun consolidation, Unit 3: routes, JSON field names, and the UI
all speak the v12 vocabulary (agent → run → item → step). The two item
keyspaces are deliberate and must not be merged:

  * RECORD keyspace — ``items.id``, the surrogate uuid of one processed
    record. Per-record detail lives at ``/api/runs/:run_id/items/:id``
    (and the ``/record`` page). "This item, in this run."
  * CUSTOMER keyspace — ``items.item_id`` / ``steps.customer_item_id``,
    the caller-declared identity (e.g. ``co_007``). Lineage across runs
    lives at ``/api/items/:item_id`` (and the ``/item`` page). "This
    item, over time."

Legacy wire (one release, for curl muscle-memory and old deep links):
``/api/batches*`` aliases the run routes; ``/api/runs/<record-id>``
falls back to per-record detail when the id matches an item instead of
a run (mirrors the CLI's ``replay --run`` fallback). Item rows now emit
``run_id`` with the NEW meaning (the invocation); the pre-0.3.0 dev API
used ``run_id`` for the record uuid — that key could not carry both
meanings, so the record uuid is ``id`` everywhere (CHANGELOG documents
the break; the bundled UI was this API's only consumer).

The route table maps URL paths to handler functions. Handlers are short,
take a ``(conn, params)`` pair, and return a JSON-serialisable value.
Errors bubble up as ``_ApiError(status, message)`` and are translated to
clean 4xx/5xx responses — endpoints must never leak a 500 on malformed
input.

The server uses ``ThreadingHTTPServer`` so a slow query on one tab does
not block other requests. Writes remain single-writer via the SDK; the
dashboard's state-mutating endpoints (run cancel, DLQ dispositions) are
localhost-gated.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from ..durable import _schema
from ..durable.sqlite_store import _promote_partial_if_drained
from . import _tier

STATIC_DIR = Path(__file__).parent / "static"

# Coarse cap on any single query result. Keeps a 500k-row local DB from
# rendering the dashboard unusable — real users will never notice this.
_DEFAULT_LIMIT = 10000
_MAX_LIMIT = 10000

# Bind clean-URL paths to static files. Anything outside this map either
# resolves to a real file in STATIC_DIR or falls back to index.html.
#
# Legacy aliases (one release): /batches → the runs list, /batch → run
# detail (a batch id IS a run id post-v12). The old /run page showed
# per-RECORD detail; /run now serves run (invocation) detail — its JS
# redirects to /record when the ?id= turns out to be a record uuid.
_PAGE_ROUTES: dict[str, str] = {
    "/": "runs.html",
    "/runs": "runs.html",
    "/run": "run.html",
    "/agents": "agents.html",
    "/items": "items.html",
    "/item": "item.html",
    "/record": "record.html",
    "/search": "search.html",
    "/upgrade": "upgrade.html",
    # Legacy paths.
    "/batches": "runs.html",
    "/batch": "run.html",
}


class _ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message


def _require_int(params: dict[str, str], key: str, default: int, *, maximum: int | None = None) -> int:
    raw = params.get(key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise _ApiError(400, f"{key} must be an integer")
    if value < 0:
        raise _ApiError(400, f"{key} must be non-negative")
    if maximum is not None and value > maximum:
        value = maximum
    return value


# --------------------------------------------------------------------------- #
#  Row translation                                                             #
# --------------------------------------------------------------------------- #


def _item_json(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    """v12 items row → wire shape.

    ``id`` = record uuid, ``run_id`` = the invocation it belongs to (NEW
    meaning), ``item_id`` = customer identity. ``batch_id`` is the legacy
    alias for the invocation (kept one release).
    """
    d = dict(row)
    d["batch_id"] = d.get("run_id")
    return d


def _run_json(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    """v12 runs (invocation) row → wire shape (+ legacy ``batch_id`` alias).

    Rollup rows additionally carry item_count / degraded_items /
    failed_items / degraded_tenants / total_tokens and the derived
    ``worst_outcome_status`` — the ran-vs-worked verdict for the run.
    """
    d = dict(row)
    d["batch_id"] = d.get("run_id")
    if "failed_items" in d:
        if (d.get("failed_items") or 0) > 0:
            d["worst_outcome_status"] = "failed"
        elif (d.get("degraded_items") or 0) > 0:
            d["worst_outcome_status"] = "degraded"
        else:
            d["worst_outcome_status"] = "ok"
    return d


def _step_json(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    """v12 steps row → wire shape.

    ``item_id`` = FK to the parent record (items.id); ``customer_item_id``
    = the caller-declared identity. These are the two keyspaces, at the
    step level.
    """
    return dict(row)


# Shared rollup over items (+ per-item step token sums), grouped by run.
# Cost note: the local ledger has no $ column (rate cards are hosted-side);
# llm_total_tokens is the local cost signal, summed here per run.
_RUN_AGG_SQL = f"""
    SELECT i.{_schema.COL_ITEM_RUN_ID} AS rid,
           COUNT(*) AS item_count,
           SUM(CASE WHEN i.{_schema.COL_ITEM_WORST_OUTCOME_STATUS} = 'degraded'
                    THEN 1 ELSE 0 END) AS degraded_items,
           SUM(CASE WHEN i.status = 'failed'
                     OR i.{_schema.COL_ITEM_WORST_OUTCOME_STATUS} = 'failed'
                    THEN 1 ELSE 0 END) AS failed_items,
           COUNT(DISTINCT CASE WHEN i.{_schema.COL_ITEM_WORST_OUTCOME_STATUS} = 'degraded'
                               THEN i.{_schema.COL_ITEM_PARTITION_KEY} END)
               AS degraded_tenants,
           SUM(t.tokens) AS total_tokens
    FROM {_schema.TBL_ITEMS} i
    LEFT JOIN (
        SELECT {_schema.COL_STEP_ITEM_ID} AS sid,
               SUM({_schema.COL_STEP_LLM_TOTAL_TOKENS}) AS tokens
        FROM {_schema.TBL_STEPS} GROUP BY {_schema.COL_STEP_ITEM_ID}
    ) t ON t.sid = i.{_schema.COL_ITEM_ID}
    GROUP BY i.{_schema.COL_ITEM_RUN_ID}
"""

_RUN_ROLLUP_SQL = f"""
    SELECT r.*,
           COALESCE(a.item_count, 0) AS item_count,
           COALESCE(a.degraded_items, 0) AS degraded_items,
           COALESCE(a.failed_items, 0) AS failed_items,
           COALESCE(a.degraded_tenants, 0) AS degraded_tenants,
           a.total_tokens AS total_tokens
    FROM {_schema.TBL_RUNS} r
    LEFT JOIN ({_RUN_AGG_SQL}) a ON a.rid = r.{_schema.COL_RUN_ID}
"""

# Item rows with per-record step counts and token sums attached.
_ITEM_ROLLUP_SQL = f"""
    SELECT i.*, COALESCE(t.step_count, 0) AS step_count,
           t.tokens AS total_tokens
    FROM {_schema.TBL_ITEMS} i
    LEFT JOIN (
        SELECT {_schema.COL_STEP_ITEM_ID} AS sid, COUNT(*) AS step_count,
               SUM({_schema.COL_STEP_LLM_TOTAL_TOKENS}) AS tokens
        FROM {_schema.TBL_STEPS} GROUP BY {_schema.COL_STEP_ITEM_ID}
    ) t ON t.sid = i.{_schema.COL_ITEM_ID}
"""


# --------------------------------------------------------------------------- #
#  Handlers                                                                    #
# --------------------------------------------------------------------------- #


def _h_stats(conn: sqlite3.Connection, _: dict[str, str]) -> dict[str, Any]:
    items_total = conn.execute(f"SELECT COUNT(*) FROM {_schema.TBL_ITEMS}").fetchone()[0]
    items_completed = conn.execute(
        f"SELECT COUNT(*) FROM {_schema.TBL_ITEMS} WHERE status='completed'"
    ).fetchone()[0]
    items_failed = conn.execute(
        f"SELECT COUNT(*) FROM {_schema.TBL_ITEMS} WHERE status='failed'"
    ).fetchone()[0]
    items_degraded = conn.execute(
        f"SELECT COUNT(*) FROM {_schema.TBL_ITEMS} "
        f"WHERE {_schema.COL_ITEM_WORST_OUTCOME_STATUS}='degraded'"
    ).fetchone()[0]
    runs_total = conn.execute(f"SELECT COUNT(*) FROM {_schema.TBL_RUNS}").fetchone()[0]
    runs_completed = conn.execute(
        f"SELECT COUNT(*) FROM {_schema.TBL_RUNS} WHERE status='completed'"
    ).fetchone()[0]
    runs_running = conn.execute(
        f"SELECT COUNT(*) FROM {_schema.TBL_RUNS} WHERE status='running'"
    ).fetchone()[0]
    return {
        "items_total": items_total,
        "items_completed": items_completed,
        "items_failed": items_failed,
        "items_degraded": items_degraded,
        "runs_total": runs_total,
        "runs_completed": runs_completed,
        "runs_in_progress": runs_running,
        # Pre-0.3.0 keys (old "run" = item, old "batch" = run). One release.
        "total_runs": items_total,
        "completed_runs": items_completed,
        "failed_runs": items_failed,
        "total_batches": runs_total,
        "batches_completed": runs_completed,
        "batches_in_progress": runs_running,
    }


def _h_agents_list(conn: sqlite3.Connection, params: dict[str, str]) -> list[dict[str, Any]]:
    """One row per agent: run/item rollups so the wedge reads at the top level."""
    limit = _require_int(params, "limit", 200, maximum=_MAX_LIMIT)
    rows = conn.execute(
        f"""SELECT r.agent,
                   COUNT(*) AS run_count,
                   MAX(r.{_schema.COL_RUN_CREATED_AT}) AS last_run_at,
                   COALESCE(SUM(a.item_count), 0) AS item_count,
                   COALESCE(SUM(a.degraded_items), 0) AS degraded_items,
                   COALESCE(SUM(a.failed_items), 0) AS failed_items,
                   SUM(a.total_tokens) AS total_tokens
            FROM {_schema.TBL_RUNS} r
            LEFT JOIN ({_RUN_AGG_SQL}) a ON a.rid = r.{_schema.COL_RUN_ID}
            GROUP BY r.agent
            ORDER BY last_run_at DESC
            LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def _h_runs_list(conn: sqlite3.Connection, params: dict[str, str]) -> list[dict[str, Any]]:
    """Invocations, newest first, with the per-run outcome rollup inline —
    a degradation incident ("8 of 17 items degraded") must be visible on
    this list without clicking anything."""
    limit = _require_int(params, "limit", 100, maximum=_MAX_LIMIT)
    agent = params.get("agent")
    query = _RUN_ROLLUP_SQL
    args: list[Any] = []
    if agent:
        query += " WHERE r.agent = ?"
        args.append(agent)
    query += f" ORDER BY r.{_schema.COL_RUN_CREATED_AT} DESC LIMIT ?"
    args.append(limit)
    return [_run_json(r) for r in conn.execute(query, args).fetchall()]


def _h_run_detail(conn: sqlite3.Connection, run_id: str, _: dict[str, str]) -> dict[str, Any]:
    row = conn.execute(
        _RUN_ROLLUP_SQL + f" WHERE r.{_schema.COL_RUN_ID} = ?", (run_id,)
    ).fetchone()
    if row is not None:
        return _run_json(row)

    # Legacy fallback (one release): a pre-0.3.0 "run id" was a record
    # uuid. When the id matches an item instead, serve per-record detail
    # so old deep links / muscle memory land somewhere useful. Mirrors
    # `papayya replay --run <old id>`. The JS on /run redirects to /record.
    item = conn.execute(
        _ITEM_ROLLUP_SQL + f" WHERE i.{_schema.COL_ITEM_ID} = ?", (run_id,)
    ).fetchone()
    if item is None:
        raise _ApiError(404, "run not found")
    return _item_json(item)


def _require_run_exists(conn: sqlite3.Connection, run_id: str) -> None:
    exists = conn.execute(
        f"SELECT 1 FROM {_schema.TBL_RUNS} WHERE {_schema.COL_RUN_ID} = ?",
        (run_id,),
    ).fetchone()
    if exists is None:
        raise _ApiError(404, "run not found")


def _h_run_items(conn: sqlite3.Connection, run_id: str, params: dict[str, str]) -> list[dict[str, Any]]:
    """The items one run processed, with per-item outcome + token rollups."""
    # Ensure the run exists so a typo returns 404, not an empty array
    _require_run_exists(conn, run_id)

    limit = _require_int(params, "limit", 500, maximum=_MAX_LIMIT)
    offset = _require_int(params, "offset", 0)
    status = params.get("status")
    outcome = params.get("outcome")

    query = _ITEM_ROLLUP_SQL + f" WHERE i.{_schema.COL_ITEM_RUN_ID} = ?"
    args: list[Any] = [run_id]
    if status:
        query += " AND i.status = ?"
        args.append(status)
    if outcome:
        query += f" AND i.{_schema.COL_ITEM_WORST_OUTCOME_STATUS} = ?"
        args.append(outcome)
    query += " ORDER BY i.created_at DESC LIMIT ? OFFSET ?"
    args.extend([limit, offset])
    return [_item_json(r) for r in conn.execute(query, args).fetchall()]


def _h_run_item_record(
    conn: sqlite3.Connection, run_id: str, record_id: str, _: dict[str, str]
) -> dict[str, Any]:
    """Per-RECORD detail: one item row + its step trace.

    Keyed by the record uuid (``items.id``) scoped under its run — the
    record keyspace. Customer-identity lineage is ``/api/items/:item_id``.
    """
    item = conn.execute(
        _ITEM_ROLLUP_SQL
        + f" WHERE i.{_schema.COL_ITEM_ID} = ? AND i.{_schema.COL_ITEM_RUN_ID} = ?",
        (record_id, run_id),
    ).fetchone()
    if item is None:
        raise _ApiError(404, "item not found in this run")

    steps = conn.execute(
        f"SELECT * FROM {_schema.TBL_STEPS} WHERE {_schema.COL_STEP_ITEM_ID} = ? ORDER BY id",
        (record_id,),
    ).fetchall()
    return {"item": _item_json(item), "steps": [_step_json(s) for s in steps]}


def _h_run_tenants(conn: sqlite3.Connection, run_id: str, _: dict[str, str]) -> list[dict[str, Any]]:
    """Per-tenant blast radius for one run: items / degraded / failed by
    partition_key. This is the table that shows *which tenants* a
    degradation incident hit."""
    _require_run_exists(conn, run_id)
    rows = conn.execute(
        f"""SELECT COALESCE({_schema.COL_ITEM_PARTITION_KEY}, '—') AS tenant,
                   COUNT(*) AS item_count,
                   SUM(CASE WHEN {_schema.COL_ITEM_WORST_OUTCOME_STATUS} = 'degraded'
                            THEN 1 ELSE 0 END) AS degraded_items,
                   SUM(CASE WHEN status = 'failed'
                             OR {_schema.COL_ITEM_WORST_OUTCOME_STATUS} = 'failed'
                            THEN 1 ELSE 0 END) AS failed_items
            FROM {_schema.TBL_ITEMS}
            WHERE {_schema.COL_ITEM_RUN_ID} = ?
            GROUP BY {_schema.COL_ITEM_PARTITION_KEY}
            ORDER BY degraded_items DESC, failed_items DESC, item_count DESC""",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _h_record_steps(conn: sqlite3.Connection, record_id: str, _: dict[str, str]) -> list[dict[str, Any]]:
    """Step trace for one record. Legacy path (`/api/runs/<record>/tasks`)
    kept one release; the canonical read is /api/runs/:run/items/:id."""
    rows = conn.execute(
        f"SELECT * FROM {_schema.TBL_STEPS} WHERE {_schema.COL_STEP_ITEM_ID} = ? ORDER BY id",
        (record_id,),
    ).fetchall()
    return [_step_json(r) for r in rows]


def _h_items_list(conn: sqlite3.Connection, params: dict[str, str]) -> list[dict[str, Any]]:
    """Latest item RECORDS across all runs (the nav 'Items' page).

    Collection rows are records; the `/api/items/<item_id>` detail below
    is the customer-identity lineage view. Two keyspaces, stated.
    """
    limit = _require_int(params, "limit", 200, maximum=_MAX_LIMIT)
    outcome = params.get("outcome")
    agent = params.get("agent")

    query = _ITEM_ROLLUP_SQL + " WHERE 1=1"
    args: list[Any] = []
    if outcome:
        query += f" AND i.{_schema.COL_ITEM_WORST_OUTCOME_STATUS} = ?"
        args.append(outcome)
    if agent:
        query += " AND i.agent = ?"
        args.append(agent)
    query += " ORDER BY i.created_at DESC LIMIT ?"
    args.append(limit)
    return [_item_json(r) for r in conn.execute(query, args).fetchall()]


def _h_item_lineage(conn: sqlite3.Connection, item_id: str, _: dict[str, str]) -> dict[str, Any]:
    """Timeline of everything that happened to one CUSTOMER item.

    Keyed by customer item_id (``items.item_id`` / ``steps.customer_item_id``)
    across all runs — the lineage view ("this item over time"). Per-record
    detail is ``/api/runs/:run_id/items/:record_id``. Two endpoints, two
    keyspaces — that split is deliberate (Plan 34 Unit 3 keyspace decision).
    """
    records = conn.execute(
        _ITEM_ROLLUP_SQL
        + f"""
            WHERE i.{_schema.COL_ITEM_ITEM_ID} = ?
               OR i.{_schema.COL_ITEM_ID} IN (
                   SELECT DISTINCT {_schema.COL_STEP_ITEM_ID}
                   FROM {_schema.TBL_STEPS}
                   WHERE {_schema.COL_STEP_CUSTOMER_ITEM_ID} = ?
               )
            ORDER BY i.created_at""",
        (item_id, item_id),
    ).fetchall()

    steps = conn.execute(
        f"""SELECT s.*, i.agent AS record_agent,
                   i.{_schema.COL_ITEM_RUN_ID} AS run_id
            FROM {_schema.TBL_STEPS} s
            JOIN {_schema.TBL_ITEMS} i ON s.{_schema.COL_STEP_ITEM_ID} = i.{_schema.COL_ITEM_ID}
            WHERE s.{_schema.COL_STEP_CUSTOMER_ITEM_ID} = ?
               OR (i.{_schema.COL_ITEM_ITEM_ID} = ?
                   AND s.{_schema.COL_STEP_CUSTOMER_ITEM_ID} IS NULL)
            ORDER BY s.completed_at, s.id""",
        (item_id, item_id),
    ).fetchall()

    if not records and not steps:
        raise _ApiError(404, "item not found")

    return {
        "item_id": item_id,
        "records": [_item_json(r) for r in records],
        "steps": [_step_json(t) for t in steps],
        # Pre-0.3.0 keys. One release.
        "runs": [_item_json(r) for r in records],
        "tasks": [_step_json(t) for t in steps],
    }


def _h_run_customer_items(conn: sqlite3.Connection, run_id: str, _: dict[str, str]) -> list[dict[str, Any]]:
    """Per-customer-item aggregates within a run (legacy 'Items' tab shape).

    Groups by the CUSTOMER item_id. Records without a customer item_id
    are omitted — this view is the lineage surface, not a general list.
    """
    _require_run_exists(conn, run_id)

    rows = conn.execute(
        f"""SELECT i.{_schema.COL_ITEM_ITEM_ID} AS item_id,
                   COUNT(DISTINCT i.{_schema.COL_ITEM_ID}) AS record_count,
                   COUNT(s.id) AS step_count,
                   MIN(i.created_at) AS first_seen,
                   MAX(i.updated_at) AS last_seen,
                   SUM(CASE WHEN i.status = 'failed' THEN 1 ELSE 0 END) AS failed_records
            FROM {_schema.TBL_ITEMS} i
            LEFT JOIN {_schema.TBL_STEPS} s
                ON s.{_schema.COL_STEP_ITEM_ID} = i.{_schema.COL_ITEM_ID}
            WHERE i.{_schema.COL_ITEM_RUN_ID} = ?
              AND i.{_schema.COL_ITEM_ITEM_ID} IS NOT NULL
            GROUP BY i.{_schema.COL_ITEM_ITEM_ID}
            ORDER BY last_seen DESC
            LIMIT ?""",
        (run_id, _DEFAULT_LIMIT),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        # Pre-0.3.0 keys. One release.
        d["run_count"] = d["record_count"]
        d["failed_runs"] = d["failed_records"]
        out.append(d)
    return out


def _h_run_dlq(conn: sqlite3.Connection, run_id: str, _: dict[str, str]) -> list[dict[str, Any]]:
    """Dead letter queue: unresolved failed items for this run.

    A dead letter is an item with ``status='failed'`` whose
    ``dlq_disposition`` is still null — the operator hasn't decided what
    to do with it yet. The response carries the replay source
    (``input_snapshot``) and the recorded error so the UI can present a
    triage row without additional round-trips.
    """
    _require_run_exists(conn, run_id)

    rows = conn.execute(
        f"""SELECT {_schema.COL_ITEM_ID} AS id, agent, created_at, updated_at,
                   output AS error,
                   {_schema.COL_ITEM_ITEM_ID} AS item_id,
                   {_schema.COL_ITEM_PARTITION_KEY} AS partition_key,
                   {_schema.COL_ITEM_INPUT_SNAPSHOT} AS input_snapshot
            FROM {_schema.TBL_ITEMS}
            WHERE {_schema.COL_ITEM_RUN_ID} = ?
              AND status = 'failed'
              AND {_schema.COL_ITEM_DLQ_DISPOSITION} IS NULL
            ORDER BY created_at DESC""",
        (run_id,),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["run_id"] = d["id"]  # pre-0.3.0 key for the record uuid. One release.
        out.append(d)
    return out


def _h_run_clusters(conn: sqlite3.Connection, run_id: str, _: dict[str, str]) -> list[dict[str, Any]]:
    """Failure clusters for a run: the run's step rows grouped by
    ``error_category``. (Pre-v12 this clustered a dead LLM-call log; the
    v12 trace table is the real source.)"""
    rows = conn.execute(
        f"""SELECT s.{_schema.COL_STEP_ERROR_CATEGORY} AS error_category,
                   s.{_schema.COL_STEP_ERROR_CATEGORY} AS error_code,
                   COUNT(*) AS count,
                   MIN(s.label) AS sample_label,
                   MIN(s.{_schema.COL_STEP_OUTCOME_REASON}) AS sample_reason
            FROM {_schema.TBL_STEPS} s
            JOIN {_schema.TBL_ITEMS} i ON s.{_schema.COL_STEP_ITEM_ID} = i.{_schema.COL_ITEM_ID}
            WHERE i.{_schema.COL_ITEM_RUN_ID} = ?
              AND s.{_schema.COL_STEP_ERROR_CATEGORY} IS NOT NULL
            GROUP BY s.{_schema.COL_STEP_ERROR_CATEGORY}
            ORDER BY count DESC
            LIMIT ?""",
        (run_id, _DEFAULT_LIMIT),
    ).fetchall()
    return [dict(r) for r in rows]


def _h_run_outliers(conn: sqlite3.Connection, run_id: str, _: dict[str, str]) -> list[dict[str, Any]]:
    """Top 10 longest-running items in a run — the local-dash outlier view.

    Ranked by wall-clock duration (``updated_at - created_at``) across all
    items in the run. In-flight items count toward the top of the list
    because their ``updated_at`` keeps advancing until terminal.
    """
    rows = conn.execute(
        f"""SELECT {_schema.COL_ITEM_ID} AS id, agent, status, created_at, updated_at,
                   {_schema.COL_ITEM_ITEM_ID} AS item_id,
                   {_schema.COL_ITEM_WORST_OUTCOME_STATUS} AS worst_outcome_status,
                   (julianday(updated_at) - julianday(created_at)) * 24 * 60 * 60 * 1000
                     AS duration_ms
            FROM {_schema.TBL_ITEMS}
            WHERE {_schema.COL_ITEM_RUN_ID} = ?
            ORDER BY duration_ms DESC
            LIMIT 10""",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _h_steps_search(conn: sqlite3.Connection, params: dict[str, str]) -> list[dict[str, Any]]:
    """Search step rows across every item: by label substring, error
    category, outcome status, and completion date range.

    ``error_code`` is the pre-0.3.0 spelling of ``error_category`` (kept
    one release); ``tool_name`` searched a dead pre-v12 table and matches
    nothing.
    """
    label = params.get("label")
    error_category = params.get("error_category") or params.get("error_code")
    outcome = params.get("outcome")
    tool_name = params.get("tool_name")
    date_from = params.get("from")
    date_to = params.get("to")
    limit = _require_int(params, "limit", 200, maximum=_MAX_LIMIT)

    if tool_name:
        return []

    query = f"SELECT * FROM {_schema.TBL_STEPS} WHERE 1=1"
    args: list[Any] = []
    if label:
        query += " AND label LIKE ?"
        args.append(f"%{label}%")
    if error_category:
        query += f" AND {_schema.COL_STEP_ERROR_CATEGORY} = ?"
        args.append(error_category)
    if outcome:
        query += f" AND {_schema.COL_STEP_OUTCOME_STATUS} = ?"
        args.append(outcome)
    if date_from:
        query += " AND completed_at >= ?"
        args.append(date_from)
    if date_to:
        query += " AND completed_at <= ?"
        args.append(date_to)
    query += " ORDER BY completed_at DESC LIMIT ?"
    args.append(limit)
    return [_step_json(r) for r in conn.execute(query, args).fetchall()]


def _h_thrashing(conn: sqlite3.Connection, params: dict[str, str]) -> list[dict[str, Any]]:
    """Repeated-identical-call detection, rebuilt on v12 step rows.

    A thrash is the same step re-journaled with the same input snapshot
    3+ times inside one record. Repeated live calls of a step are stored
    as ``label``, ``label#2``, ``label#3`` (the occurrence suffix from the
    agent-loop label fix), so we group on the BARE label — the part before
    the first ``#``. Scoped to one record (``item``; ``run_id`` is the
    pre-0.3.0 spelling of the record uuid, kept one release) or to a
    whole run (``run`` / legacy ``batch_id``), grouped per record.
    """
    record_id = params.get("item") or params.get("run_id")
    run_scope = params.get("run") or params.get("batch_id")
    if not record_id and not run_scope:
        raise _ApiError(400, "thrashing requires item (record id) or run")

    where = (
        f"s.{_schema.COL_STEP_ITEM_ID} = ?"
        if record_id
        else f"i.{_schema.COL_ITEM_RUN_ID} = ?"
    )
    bare_label = (
        "CASE WHEN instr(s.label, '#') > 0 "
        "THEN substr(s.label, 1, instr(s.label, '#') - 1) ELSE s.label END"
    )
    rows = conn.execute(
        f"""SELECT s.{_schema.COL_STEP_ITEM_ID} AS item_id,
                   {bare_label} AS label,
                   MIN(s.{_schema.COL_STEP_INPUT_SNAPSHOT}) AS input_snapshot,
                   COUNT(*) AS repeat_count
            FROM {_schema.TBL_STEPS} s
            JOIN {_schema.TBL_ITEMS} i ON s.{_schema.COL_STEP_ITEM_ID} = i.{_schema.COL_ITEM_ID}
            WHERE {where}
              AND s.{_schema.COL_STEP_INPUT_SNAPSHOT} IS NOT NULL
            GROUP BY s.{_schema.COL_STEP_ITEM_ID}, {bare_label},
                     s.{_schema.COL_STEP_INPUT_SNAPSHOT}
            HAVING COUNT(*) >= 3
            ORDER BY repeat_count DESC
            LIMIT 20""",
        (record_id or run_scope,),
    ).fetchall()
    return [dict(r) for r in rows]


def _h_projection(conn: sqlite3.Connection, _: dict[str, str]) -> dict[str, Any]:
    """Rolling 30-day usage summary, used by the upgrade page's tier recommender.

    Duration is derived from ``updated_at - created_at`` on completed items
    only — in-flight items have a meaningless ``updated_at``. Timestamps
    are UTC ISO strings; we compare lexicographically against a 30-day
    cutoff the server computes once per call.

    ``julianday()`` below assumes TZ-suffixed ISO8601. All current writers
    emit ``datetime.now(timezone.utc).isoformat()`` — if a future writer
    emits a naive timestamp, duration math goes silently wrong.
    """
    from datetime import datetime, timedelta, timezone

    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

    totals = conn.execute(
        f"""SELECT COUNT(*) AS total_items
           FROM {_schema.TBL_ITEMS}
           WHERE created_at >= ?""",
        (cutoff,),
    ).fetchone()

    # Compute-minutes: only completed items have a meaningful duration.
    duration = conn.execute(
        f"""SELECT COALESCE(SUM(
             (julianday(updated_at) - julianday(created_at)) * 24 * 60
           ), 0) AS compute_minutes
           FROM {_schema.TBL_ITEMS}
           WHERE created_at >= ? AND status = 'completed'""",
        (cutoff,),
    ).fetchone()

    runs = conn.execute(
        f"""SELECT COUNT(*) AS total_runs,
                   COALESCE(MAX({_schema.COL_RUN_TOTAL_ITEMS}), 0) AS largest_run
            FROM {_schema.TBL_RUNS}
            WHERE {_schema.COL_RUN_CREATED_AT} >= ?""",
        (cutoff,),
    ).fetchone()

    return {
        "window_days": 30,
        "total_items": totals["total_items"],
        "runs_total": runs["total_runs"],
        "largest_run": runs["largest_run"],
        "compute_minutes": duration["compute_minutes"],
        # Pre-0.3.0 keys (old "run" = item, old "batch" = run). One release.
        "total_runs": totals["total_items"],
        "total_batches": runs["total_runs"],
        "largest_batch": runs["largest_run"],
    }


def _h_tier_recommendation(conn: sqlite3.Connection, _: dict[str, str]) -> dict[str, Any]:
    """Pair the 30-day projection with a tier recommendation.

    Peak concurrency locally is a crude proxy: the max number of items
    started within any single minute of the window. At laptop scale this
    is usually 1–2 and matches the experience a user would have on the
    hosted product's concurrency ceiling.
    """
    from datetime import datetime, timedelta, timezone

    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

    # Minute-bucketed item starts — max count across buckets is our proxy.
    # ISO 8601 strings sort lexicographically, so substr to the minute works.
    row = conn.execute(
        f"""SELECT COALESCE(MAX(c), 0) AS peak
           FROM (
             SELECT COUNT(*) AS c
             FROM {_schema.TBL_ITEMS}
             WHERE created_at >= ?
             GROUP BY substr(created_at, 1, 16)
           )""",
        (cutoff,),
    ).fetchone()
    peak_concurrency = int(row["peak"] if row["peak"] is not None else 0)

    # Reuse the projection handler rather than duplicating the SQL.
    projection = _h_projection(conn, {})
    compute_min = float(projection.get("compute_minutes", 0) or 0)

    rec = _tier.recommend(compute_min=compute_min, peak_concurrency=peak_concurrency)
    return {
        "projection": projection,
        "peak_concurrency": peak_concurrency,
        "recommendation": rec.to_dict(),
    }


def _h_dlq_disposition(
    conn: sqlite3.Connection,
    run_id: str,
    record_id: str,
    disposition: str,
) -> dict[str, Any]:
    """Mark a dead letter as skipped / acknowledged / replayed.

    Replay is only half-done here: this marks the original item as
    ``replayed`` but does NOT execute the replay — that's the
    ``papayya replay`` subprocess. Callers of this endpoint for the
    ``replayed`` disposition are therefore expected to have already
    created (or be about to create) the new item with
    ``replayed_from=<id>``.

    Idempotent by design: re-posting the same disposition on an already-
    resolved item is a no-op at the store layer.
    """
    row = conn.execute(
        f"""SELECT status, {_schema.COL_ITEM_DLQ_DISPOSITION} AS disp,
                   {_schema.COL_ITEM_RUN_ID} AS run_id
            FROM {_schema.TBL_ITEMS} WHERE {_schema.COL_ITEM_ID} = ?""",
        (record_id,),
    ).fetchone()
    if row is None:
        raise _ApiError(404, "item not found")
    if row["run_id"] != run_id:
        raise _ApiError(404, "item does not belong to this run")
    if row["status"] != "failed":
        raise _ApiError(409, "item is not failed; cannot mark DLQ disposition")
    if row["disp"] is not None:
        # Already resolved — return current state without re-setting.
        return {"noop": True, "disposition": row["disp"]}

    if disposition not in (
        _schema.DLQ_REPLAYED,
        _schema.DLQ_SKIPPED,
        _schema.DLQ_ACKNOWLEDGED,
    ):
        raise _ApiError(400, f"invalid disposition: {disposition!r}")

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        f"""UPDATE {_schema.TBL_ITEMS}
            SET {_schema.COL_ITEM_DLQ_DISPOSITION} = ?,
                {_schema.COL_ITEM_DLQ_RESOLVED_AT} = ?,
                updated_at = ?
            WHERE {_schema.COL_ITEM_ID} = ?""",
        (disposition, now, now, record_id),
    )
    _promote_partial_if_drained(conn, run_id, now)
    conn.commit()
    return {"noop": False, "disposition": disposition, "resolved_at": now}


def _h_dlq_replay(
    conn: sqlite3.Connection,
    db_path: str,
    run_id: str,
    record_id: str,
) -> dict[str, Any]:
    """Spawn ``papayya replay --item <record id>`` and wait for it.

    Runs the CLI in a subprocess from the dashboard server's cwd — which is
    the user's project dir, where their agent module lives. Times out at
    120s: LLM replays are almost always <30s, and a longer-running replay
    signals either a deep agent loop or a model that's unsuited to being
    blocked-on from a browser. In that case the operator can re-try from
    their terminal directly.

    Validates DLQ state before dispatching so the subprocess doesn't waste
    cycles discovering the item is already resolved.
    """
    import subprocess

    row = conn.execute(
        f"""SELECT status, {_schema.COL_ITEM_DLQ_DISPOSITION} AS disp,
                   {_schema.COL_ITEM_RUN_ID} AS run_id,
                   {_schema.COL_ITEM_INPUT_SNAPSHOT} AS input_snapshot
            FROM {_schema.TBL_ITEMS} WHERE {_schema.COL_ITEM_ID} = ?""",
        (record_id,),
    ).fetchone()
    if row is None:
        raise _ApiError(404, "item not found")
    if row["run_id"] != run_id:
        raise _ApiError(404, "item does not belong to this run")
    if row["status"] != "failed":
        raise _ApiError(409, "item is not failed; cannot replay")
    if row["disp"] is not None:
        return {"noop": True, "disposition": row["disp"]}
    if row["input_snapshot"] is None:
        raise _ApiError(
            409,
            "item has no captured input_snapshot; cannot replay",
        )

    try:
        proc = subprocess.run(
            ["papayya", "replay", "--item", record_id, "--db", db_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise _ApiError(
            504,
            "Replay still running after 120s. Check the DLQ list — if a new "
            "item appeared, replay is in-flight; otherwise retry from the terminal."
        )
    except FileNotFoundError:
        raise _ApiError(
            500,
            "papayya CLI not found on PATH. Install papayya into the active "
            "environment before using the Replay button.",
        )

    return {
        "exit_code": proc.returncode,
        "stdout": proc.stdout[-2000:] if proc.stdout else "",
        "stderr": proc.stderr[-2000:] if proc.stderr else "",
    }


def _h_run_cancel(conn: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    row = conn.execute(
        f"SELECT {_schema.COL_RUN_STATUS} FROM {_schema.TBL_RUNS} "
        f"WHERE {_schema.COL_RUN_ID} = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        raise _ApiError(404, "run not found")
    if row[_schema.COL_RUN_STATUS] in ("completed", "cancelled", "failed", "partial"):
        return {"noop": True, "status": row[_schema.COL_RUN_STATUS]}

    conn.execute(
        f"UPDATE {_schema.TBL_RUNS} SET {_schema.COL_RUN_STATUS} = 'cancelled' "
        f"WHERE {_schema.COL_RUN_ID} = ?",
        (run_id,),
    )
    conn.commit()
    return {"noop": False, "status": "cancelled"}


# --------------------------------------------------------------------------- #
#  Dispatcher                                                                  #
# --------------------------------------------------------------------------- #


# Static (no-path-parameter) GET routes
_GET_ROUTES: dict[str, Callable[[sqlite3.Connection, dict[str, str]], Any]] = {
    "/api/stats":          _h_stats,
    "/api/agents":         _h_agents_list,
    "/api/runs":           _h_runs_list,
    "/api/items":          _h_items_list,
    "/api/steps/search":   _h_steps_search,
    "/api/thrashing":      _h_thrashing,
    "/api/projection":     _h_projection,
    "/api/tier-recommendation": _h_tier_recommendation,
    # Legacy alias (one release): old UI/curl habits list invocations here.
    "/api/batches":        _h_runs_list,
}

# Parameterised GET routes. /api/batches/* aliases /api/runs/* — a batch
# id IS a run id post-v12. One deliberate exception, keyed on the prefix
# so the same path never carries two meanings: `/api/batches/:id/items`
# keeps its pre-0.3.0 semantics (per-CUSTOMER aggregates, the old Items
# tab), while `/api/runs/:id/items` lists the run's records. `/runs` and
# `/tasks` subroutes are the pre-0.3.0 spellings of the record list and a
# record's step trace.
_RUN_ID_RE = re.compile(
    r"^/api/(runs|batches)/([A-Za-z0-9_\-]+)"
    r"(/items/[A-Za-z0-9_\-]+|/items|/runs|/tenants|/clusters|/outliers|/dlq|/tasks)?$"
)
# item_id is user-supplied so we accept a wider charset than run ids.
# Still restricted to URL-safe characters to avoid path traversal issues.
_ITEM_ID_RE = re.compile(r"^/api/items/([A-Za-z0-9_\-\.:]+)$")

_POST_RE = re.compile(
    r"^/api/(?:runs|batches)/([A-Za-z0-9_\-]+)"
    r"(?:/cancel|/dlq/([A-Za-z0-9_\-]+)/(skip|acknowledge|replay))$"
)


class DevHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves the dashboard API and static files."""

    db_path: str = ".papayya/local.db"

    # Silence the default access-log spam; the dashboard is interactive.
    def log_message(self, format: str, *args: object) -> None:
        pass

    # ---- GET ----

    def do_GET(self) -> None:
        if self.path.startswith("/api/"):
            self._handle_api_get()
        else:
            self._serve_static()

    def _handle_api_get(self) -> None:
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

            conn = self._open_db()
            try:
                if path in _GET_ROUTES:
                    self._json(_GET_ROUTES[path](conn, params))
                    return

                m = _RUN_ID_RE.match(path)
                if m:
                    prefix, run_id, sub = m.group(1), m.group(2), m.group(3) or ""
                    if sub.startswith("/items/"):
                        record_id = sub.removeprefix("/items/")
                        self._json(_h_run_item_record(conn, run_id, record_id, params))
                    elif sub == "/items" and prefix == "batches":
                        # Legacy semantics: per-customer aggregates.
                        self._json(_h_run_customer_items(conn, run_id, params))
                    elif sub == "/items":
                        self._json(_h_run_items(conn, run_id, params))
                    elif sub == "/tenants":
                        self._json(_h_run_tenants(conn, run_id, params))
                    elif sub == "/clusters":
                        self._json(_h_run_clusters(conn, run_id, params))
                    elif sub == "/outliers":
                        self._json(_h_run_outliers(conn, run_id, params))
                    elif sub == "/dlq":
                        self._json(_h_run_dlq(conn, run_id, params))
                    elif sub == "/runs":
                        # Legacy: /api/batches/:id/runs listed the per-item rows.
                        self._json(_h_run_items(conn, run_id, params))
                    elif sub == "/tasks":
                        # Legacy: run_id here is a RECORD uuid (old sense).
                        self._json(_h_record_steps(conn, run_id, params))
                    else:
                        self._json(_h_run_detail(conn, run_id, params))
                    return

                m = _ITEM_ID_RE.match(path)
                if m:
                    self._json(_h_item_lineage(conn, m.group(1), params))
                    return

                raise _ApiError(404, "not found")
            finally:
                conn.close()
        except _ApiError as e:
            self._json({"error": e.message}, status=e.status)
        except sqlite3.Error as e:
            # Database-level problems (corrupt DB, locked, etc.) are worth
            # surfacing as 500 with the message — it's a dev tool.
            self._json({"error": f"db error: {e}"}, status=500)
        except Exception as e:  # noqa: BLE001  (dev tool — never propagate)
            self._json({"error": f"server error: {e}"}, status=500)

    # ---- POST ----

    def do_POST(self) -> None:
        if not self._is_localhost():
            self._json({"error": "forbidden"}, status=403)
            return

        try:
            parsed = urlparse(self.path)
            m = _POST_RE.match(parsed.path)
            if m:
                run_id, record_id, action = m.group(1), m.group(2), m.group(3)
                conn = self._open_db()
                try:
                    if record_id is None:
                        self._json(_h_run_cancel(conn, run_id))
                    elif action == "replay":
                        self._json(_h_dlq_replay(conn, self.db_path, run_id, record_id))
                    else:
                        disposition = {
                            "skip": _schema.DLQ_SKIPPED,
                            "acknowledge": _schema.DLQ_ACKNOWLEDGED,
                        }[action]
                        self._json(_h_dlq_disposition(conn, run_id, record_id, disposition))
                finally:
                    conn.close()
                return

            raise _ApiError(404, "not found")
        except _ApiError as e:
            self._json({"error": e.message}, status=e.status)
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"server error: {e}"}, status=500)

    # ---- Helpers ----

    def _is_localhost(self) -> bool:
        host = self.headers.get("Host", "")
        # Strip port
        host_no_port = host.split(":")[0]
        return host_no_port in ("127.0.0.1", "localhost", "::1")

    def _open_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _json(self, data: Any, *, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        # Map clean page routes to their HTML files, otherwise serve the
        # file as-is or fall back to the home page.
        if path in _PAGE_ROUTES:
            file_path = STATIC_DIR / _PAGE_ROUTES[path]
        else:
            file_path = STATIC_DIR / path.lstrip("/")
            if not file_path.is_file():
                file_path = STATIC_DIR / _PAGE_ROUTES["/"]

        if not file_path.is_file():
            # Legacy path — the pre-Slice-4 SPA lived at index.html. Keep
            # serving it if the new pages haven't landed yet.
            file_path = STATIC_DIR / "index.html"

        if not file_path.is_file():
            self.send_error(404)
            return

        content = file_path.read_bytes()
        content_type = _guess_type(file_path.suffix)
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def _guess_type(ext: str) -> str:
    return {
        ".html": "text/html; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".json": "application/json",
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".ico": "image/x-icon",
    }.get(ext, "application/octet-stream")


def serve(host: str = "127.0.0.1", port: int = 8585, db_path: str = ".papayya/local.db") -> None:
    """Start the local development dashboard server."""
    import sys

    db = Path(db_path)
    if not db.exists():
        sys.stderr.write(f"No local database found at {db.resolve()}\n")
        sys.stderr.write(
            "\nRun an agent with the Papayya SDK first to generate data.\n"
            "The SDK automatically writes to .papayya/local.db when no\n"
            "PAPAYYA_API_KEY is set.\n"
        )
        sys.exit(1)

    # A DB written by an older SDK may still be at a pre-v12 schema. The
    # dashboard opens raw sqlite3 connections and has no migration path of
    # its own, so upgrade once here before we start serving.
    from ..durable.sqlite_store import ensure_migrated
    ensure_migrated(db.resolve())

    DevHandler.db_path = str(db.resolve())
    try:
        server = ThreadingHTTPServer((host, port), DevHandler)
    except OSError as exc:
        import errno
        if exc.errno == errno.EADDRINUSE:
            sys.stderr.write(
                f"Port {port} on {host} is already in use.\n"
                f"Another `papayya dev` instance or another process is bound there.\n"
                f"Try `papayya dev --port <N>` with a different port.\n"
            )
            sys.exit(1)
        raise
    sys.stderr.write(f"Papayya Dev Dashboard: http://{host}:{port}\n")
    sys.stderr.write(f"Reading from: {db.resolve()}\n")
    sys.stderr.write("Press Ctrl+C to stop.\n\n")
    sys.stderr.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("\nShutting down.\n")
        server.shutdown()
