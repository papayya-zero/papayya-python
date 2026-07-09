"""Local development dashboard server.

Serves a static dashboard UI and a JSON API that reads from the local
SQLite database. Uses only Python stdlib — no frameworks.

Plan 34 noun consolidation, PR-1 scope: the queries below read the v12
schema (``runs`` = invocations, ``items`` = per-item records, ``steps`` =
trace nodes) while the ROUTES and the JSON field names keep the pre-v12
wire shape the shipped static UI reads (``/api/batches`` lists
invocations, ``/api/runs`` lists per-item records, item rows carry
``run_id``/``batch_id``). The Unit 3 pass renames the routes/nav and does
the keyspace redesign; endpoints here emit old field names (plus the new
``id`` on item rows) for one release.

The route table maps URL paths to handler functions. Handlers are short,
take a ``(conn, params)`` pair, and return a JSON-serialisable value.
Errors bubble up as ``_ApiError(status, message)`` and are translated to
clean 4xx/5xx responses — endpoints must never leak a 500 on malformed
input.

The server uses ``ThreadingHTTPServer`` so a slow query on one tab does
not block other requests. Writes remain single-writer via the SDK; the
dashboard's only state-mutating endpoint is run cancel, which is
localhost-gated and no-ops for terminal runs.
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
_PAGE_ROUTES: dict[str, str] = {
    "/": "batches.html",
    "/batches": "batches.html",
    "/batch": "batch.html",
    "/run": "run.html",
    "/item": "item.html",
    "/search": "search.html",
    "/upgrade": "upgrade.html",
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
#  Wire-compat row translation (one release; Unit 3 replaces this)             #
# --------------------------------------------------------------------------- #


def _item_json(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    """v12 items row → the pre-v12 per-item wire shape (+ new ``id``).

    The shipped static UI reads ``run_id`` as "the per-item record's id"
    and ``batch_id`` as "the invocation it belongs to". Those keys collide
    with the v12 column meanings, so the OLD meanings win on this wire for
    one release and the new surrogate is additionally exposed as ``id``.
    """
    d = dict(row)
    d["batch_id"] = d.get("run_id")
    d["run_id"] = d.get("id")
    return d


def _run_json(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    """v12 runs (invocation) row → the pre-v12 batch wire shape.

    Emits both the new ``run_id`` and the old ``batch_id`` alias (same
    value — no meaning collision on this row type).
    """
    d = dict(row)
    d["batch_id"] = d.get("run_id")
    return d


def _step_json(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    """v12 steps row → the pre-v12 task wire shape (+ new names).

    Old wire: ``run_id`` = the per-item record's id, ``item_id`` = the
    CUSTOMER identity. New columns keep flowing through under their v12
    names (``customer_item_id``), so both generations of readers work.
    """
    d = dict(row)
    d["run_id"] = d.get("item_id")
    d["item_id"] = d.get("customer_item_id")
    return d


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
    runs_total = conn.execute(f"SELECT COUNT(*) FROM {_schema.TBL_RUNS}").fetchone()[0]
    runs_completed = conn.execute(
        f"SELECT COUNT(*) FROM {_schema.TBL_RUNS} WHERE status='completed'"
    ).fetchone()[0]
    runs_running = conn.execute(
        f"SELECT COUNT(*) FROM {_schema.TBL_RUNS} WHERE status='running'"
    ).fetchone()[0]
    return {
        # New-noun keys.
        "items_total": items_total,
        "items_completed": items_completed,
        "items_failed": items_failed,
        "runs_total": runs_total,
        "runs_completed": runs_completed,
        "runs_in_progress": runs_running,
        # Pre-v12 keys the shipped UI reads (old "run" = item, old
        # "batch" = run). Kept for one release.
        "total_runs": items_total,
        "completed_runs": items_completed,
        "failed_runs": items_failed,
        "total_batches": runs_total,
        "batches_completed": runs_completed,
        "batches_in_progress": runs_running,
    }


def _h_runs_list(conn: sqlite3.Connection, params: dict[str, str]) -> list[dict[str, Any]]:
    limit = _require_int(params, "limit", 100, maximum=_MAX_LIMIT)
    rows = conn.execute(
        f"SELECT * FROM {_schema.TBL_ITEMS} ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [_item_json(r) for r in rows]


def _h_run_detail(conn: sqlite3.Connection, run_id: str, _: dict[str, str]) -> dict[str, Any]:
    row = conn.execute(
        f"SELECT * FROM {_schema.TBL_ITEMS} WHERE {_schema.COL_ITEM_ID} = ?", (run_id,)
    ).fetchone()
    if row is None:
        raise _ApiError(404, "run not found")
    return _item_json(row)


def _h_run_tasks(conn: sqlite3.Connection, run_id: str, _: dict[str, str]) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"SELECT * FROM {_schema.TBL_STEPS} WHERE {_schema.COL_STEP_ITEM_ID} = ? ORDER BY id",
        (run_id,),
    ).fetchall()
    return [_step_json(r) for r in rows]


def _h_run_steps(conn: sqlite3.Connection, run_id: str, _: dict[str, str]) -> list[dict[str, Any]]:
    """Pre-v12 the legacy `steps` table held raw LLM-call rows written by
    `record_step` — a surface with no production writer, dropped in v12.
    The endpoint stays (the shipped run page still fetches it) and always
    returns []; the item's real trace lives at ``/api/runs/:id/tasks``.
    """
    return []


def _h_item_detail(conn: sqlite3.Connection, item_id: str, _: dict[str, str]) -> dict[str, Any]:
    """Timeline of everything that happened to one record.

    Keyed by CUSTOMER item_id (the ``items.item_id`` /
    ``steps.customer_item_id`` value) across all runs — the lineage view.
    Per-record detail is ``/api/runs/:id``. Two endpoints, two keyspaces —
    that split is deliberate (Plan 34 Unit 3 keyspace decision).
    """
    items = conn.execute(
        f"""SELECT * FROM {_schema.TBL_ITEMS}
            WHERE {_schema.COL_ITEM_ITEM_ID} = ?
               OR {_schema.COL_ITEM_ID} IN (
                   SELECT DISTINCT {_schema.COL_STEP_ITEM_ID}
                   FROM {_schema.TBL_STEPS}
                   WHERE {_schema.COL_STEP_CUSTOMER_ITEM_ID} = ?
               )
            ORDER BY created_at""",
        (item_id, item_id),
    ).fetchall()

    steps = conn.execute(
        f"""SELECT s.*, i.agent AS run_agent,
                   i.{_schema.COL_ITEM_RUN_ID} AS batch_id
            FROM {_schema.TBL_STEPS} s
            JOIN {_schema.TBL_ITEMS} i ON s.{_schema.COL_STEP_ITEM_ID} = i.{_schema.COL_ITEM_ID}
            WHERE s.{_schema.COL_STEP_CUSTOMER_ITEM_ID} = ?
               OR (i.{_schema.COL_ITEM_ITEM_ID} = ?
                   AND s.{_schema.COL_STEP_CUSTOMER_ITEM_ID} IS NULL)
            ORDER BY s.completed_at, s.id""",
        (item_id, item_id),
    ).fetchall()

    if not items and not steps:
        raise _ApiError(404, "item not found")

    return {
        "item_id": item_id,
        "runs": [_item_json(r) for r in items],
        "tasks": [_step_json(t) for t in steps],
    }


def _h_batches_list(conn: sqlite3.Connection, params: dict[str, str]) -> list[dict[str, Any]]:
    limit = _require_int(params, "limit", 100, maximum=_MAX_LIMIT)
    rows = conn.execute(
        f"SELECT * FROM {_schema.TBL_RUNS} ORDER BY {_schema.COL_RUN_CREATED_AT} DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_run_json(r) for r in rows]


def _h_batch_detail(conn: sqlite3.Connection, batch_id: str, _: dict[str, str]) -> dict[str, Any]:
    row = conn.execute(
        f"SELECT * FROM {_schema.TBL_RUNS} WHERE {_schema.COL_RUN_ID} = ?",
        (batch_id,),
    ).fetchone()
    if row is None:
        raise _ApiError(404, "batch not found")
    return _run_json(row)


def _require_run_exists(conn: sqlite3.Connection, run_id: str) -> None:
    exists = conn.execute(
        f"SELECT 1 FROM {_schema.TBL_RUNS} WHERE {_schema.COL_RUN_ID} = ?",
        (run_id,),
    ).fetchone()
    if exists is None:
        raise _ApiError(404, "batch not found")


def _h_batch_runs(conn: sqlite3.Connection, batch_id: str, params: dict[str, str]) -> list[dict[str, Any]]:
    # Ensure the run exists so a typo returns 404, not an empty array
    _require_run_exists(conn, batch_id)

    limit = _require_int(params, "limit", 100, maximum=_MAX_LIMIT)
    offset = _require_int(params, "offset", 0)
    status = params.get("status")

    query = f"SELECT * FROM {_schema.TBL_ITEMS} WHERE {_schema.COL_ITEM_RUN_ID} = ?"
    args: list[Any] = [batch_id]
    if status:
        query += " AND status = ?"
        args.append(status)
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    args.extend([limit, offset])
    return [_item_json(r) for r in conn.execute(query, args).fetchall()]


def _h_batch_items(conn: sqlite3.Connection, batch_id: str, _: dict[str, str]) -> list[dict[str, Any]]:
    """Per-record aggregates for the detail page's 'Items' tab.

    Groups by the CUSTOMER item_id (denormalized onto item rows). Records
    without a customer item_id are omitted — this view is the lineage
    surface, not a general list.
    """
    _require_run_exists(conn, batch_id)

    rows = conn.execute(
        f"""SELECT i.{_schema.COL_ITEM_ITEM_ID} AS item_id,
                   COUNT(DISTINCT i.{_schema.COL_ITEM_ID}) AS run_count,
                   COUNT(s.id) AS step_count,
                   MIN(i.created_at) AS first_seen,
                   MAX(i.updated_at) AS last_seen,
                   SUM(CASE WHEN i.status = 'failed' THEN 1 ELSE 0 END) AS failed_runs
            FROM {_schema.TBL_ITEMS} i
            LEFT JOIN {_schema.TBL_STEPS} s
                ON s.{_schema.COL_STEP_ITEM_ID} = i.{_schema.COL_ITEM_ID}
            WHERE i.{_schema.COL_ITEM_RUN_ID} = ?
              AND i.{_schema.COL_ITEM_ITEM_ID} IS NOT NULL
            GROUP BY i.{_schema.COL_ITEM_ITEM_ID}
            ORDER BY last_seen DESC
            LIMIT ?""",
        (batch_id, _DEFAULT_LIMIT),
    ).fetchall()
    return [dict(r) for r in rows]


def _h_batch_dlq(conn: sqlite3.Connection, batch_id: str, _: dict[str, str]) -> list[dict[str, Any]]:
    """Dead letter queue: unresolved failed items for this run.

    A dead letter is an item with ``status='failed'`` whose
    ``dlq_disposition`` is still null — the operator hasn't decided what
    to do with it yet. The response carries the replay source
    (``input_snapshot``) and the recorded error so the UI can present a
    triage row without additional round-trips.
    """
    _require_run_exists(conn, batch_id)

    rows = conn.execute(
        f"""SELECT {_schema.COL_ITEM_ID} AS run_id, agent, created_at, updated_at,
                   output AS error,
                   {_schema.COL_ITEM_ITEM_ID} AS item_id,
                   {_schema.COL_ITEM_INPUT_SNAPSHOT} AS input_snapshot,
                   {_schema.COL_ITEM_ID} AS id
            FROM {_schema.TBL_ITEMS}
            WHERE {_schema.COL_ITEM_RUN_ID} = ?
              AND status = 'failed'
              AND {_schema.COL_ITEM_DLQ_DISPOSITION} IS NULL
            ORDER BY created_at DESC""",
        (batch_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _h_batch_clusters(conn: sqlite3.Connection, batch_id: str, _: dict[str, str]) -> list[dict[str, Any]]:
    """Failure clusters for a run.

    Pre-v12 this clustered the legacy LLM-call log on (error_code,
    input_hash); that table is gone (no production writer). Clusters now
    group the run's step rows by ``error_category`` — same response keys
    so the shipped UI renders unchanged (``input_hash`` is always null).
    """
    rows = conn.execute(
        f"""SELECT s.{_schema.COL_STEP_ERROR_CATEGORY} AS error_code,
                   NULL AS input_hash,
                   COUNT(*) AS count,
                   MIN(s.label) AS sample_label,
                   MIN(s.{_schema.COL_STEP_OUTCOME_REASON}) AS sample_response
            FROM {_schema.TBL_STEPS} s
            JOIN {_schema.TBL_ITEMS} i ON s.{_schema.COL_STEP_ITEM_ID} = i.{_schema.COL_ITEM_ID}
            WHERE i.{_schema.COL_ITEM_RUN_ID} = ?
              AND s.{_schema.COL_STEP_ERROR_CATEGORY} IS NOT NULL
            GROUP BY s.{_schema.COL_STEP_ERROR_CATEGORY}
            ORDER BY count DESC
            LIMIT ?""",
        (batch_id, _DEFAULT_LIMIT),
    ).fetchall()
    return [dict(r) for r in rows]


def _h_batch_outliers(conn: sqlite3.Connection, batch_id: str, _: dict[str, str]) -> list[dict[str, Any]]:
    """Top 10 longest-running items in a run — the local-dash outlier view.

    Ranked by wall-clock duration (``updated_at - created_at``) across all
    items in the run. In-flight items count toward the top of the list
    because their ``updated_at`` keeps advancing until terminal.
    """
    rows = conn.execute(
        f"""SELECT {_schema.COL_ITEM_ID} AS run_id, agent, status, created_at, updated_at,
                   {_schema.COL_ITEM_ID} AS id,
                   (julianday(updated_at) - julianday(created_at)) * 24 * 60 * 60 * 1000
                     AS duration_ms
            FROM {_schema.TBL_ITEMS}
            WHERE {_schema.COL_ITEM_RUN_ID} = ?
            ORDER BY duration_ms DESC
            LIMIT 10""",
        (batch_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _h_steps_search(conn: sqlite3.Connection, params: dict[str, str]) -> list[dict[str, Any]]:
    """Search step rows. Pre-v12 this searched the legacy LLM-call log by
    tool_name/error_code/input_hash; v12 searches the real trace table.
    ``error_code`` maps onto ``error_category``; ``tool_name`` has no v12
    column and matches nothing (the Unit 3 pass redesigns this page).
    """
    tool_name = params.get("tool_name")
    error_code = params.get("error_code")
    date_from = params.get("from")
    date_to = params.get("to")
    limit = _require_int(params, "limit", 200, maximum=_MAX_LIMIT)

    if tool_name:
        return []

    query = f"SELECT * FROM {_schema.TBL_STEPS} WHERE 1=1"
    args: list[Any] = []
    if error_code:
        query += f" AND {_schema.COL_STEP_ERROR_CATEGORY} = ?"
        args.append(error_code)
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
    """Repeated-identical-call detection.

    Fed by the legacy LLM-call log's (tool_name, input_hash) columns,
    which had no production writer and were dropped in v12. Returns []
    until the Unit 3 pass rebuilds it on step rows; the endpoint survives
    so the shipped UI doesn't 404/500.
    """
    run_id = params.get("run_id")
    batch_id = params.get("batch_id")
    if not run_id and not batch_id:
        raise _ApiError(400, "thrashing requires run_id or batch_id")
    return []


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
        # New-noun keys.
        "total_items": totals["total_items"],
        "runs_total": runs["total_runs"],
        "largest_run": runs["largest_run"],
        "compute_minutes": duration["compute_minutes"],
        # Pre-v12 keys (old "run" = item, old "batch" = run).
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
    batch_id: str,
    run_id: str,
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
                   {_schema.COL_ITEM_RUN_ID} AS batch_id
            FROM {_schema.TBL_ITEMS} WHERE {_schema.COL_ITEM_ID} = ?""",
        (run_id,),
    ).fetchone()
    if row is None:
        raise _ApiError(404, "run not found")
    if row["batch_id"] != batch_id:
        raise _ApiError(404, "run does not belong to this batch")
    if row["status"] != "failed":
        raise _ApiError(409, "run is not failed; cannot mark DLQ disposition")
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
        (disposition, now, now, run_id),
    )
    _promote_partial_if_drained(conn, batch_id, now)
    conn.commit()
    return {"noop": False, "disposition": disposition, "resolved_at": now}


def _h_dlq_replay(
    conn: sqlite3.Connection,
    db_path: str,
    batch_id: str,
    run_id: str,
) -> dict[str, Any]:
    """Spawn ``papayya replay --run <id>`` and wait for it.

    The id sent here is a per-ITEM id; the CLI's --run flag detects that
    and runs single-item replay (its slice mode applies to run ids).

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
                   {_schema.COL_ITEM_RUN_ID} AS batch_id,
                   {_schema.COL_ITEM_INPUT_SNAPSHOT} AS input_snapshot
            FROM {_schema.TBL_ITEMS} WHERE {_schema.COL_ITEM_ID} = ?""",
        (run_id,),
    ).fetchone()
    if row is None:
        raise _ApiError(404, "run not found")
    if row["batch_id"] != batch_id:
        raise _ApiError(404, "run does not belong to this batch")
    if row["status"] != "failed":
        raise _ApiError(409, "run is not failed; cannot replay")
    if row["disp"] is not None:
        return {"noop": True, "disposition": row["disp"]}
    if row["input_snapshot"] is None:
        raise _ApiError(
            409,
            "run has no captured input_snapshot; cannot replay",
        )

    try:
        proc = subprocess.run(
            ["papayya", "replay", "--run", run_id, "--db", db_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise _ApiError(
            504,
            "Replay still running after 120s. Check the DLQ list — if a new "
            "run appeared, replay is in-flight; otherwise retry from the terminal."
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


def _h_batch_cancel(conn: sqlite3.Connection, batch_id: str) -> dict[str, Any]:
    row = conn.execute(
        f"SELECT {_schema.COL_RUN_STATUS} FROM {_schema.TBL_RUNS} "
        f"WHERE {_schema.COL_RUN_ID} = ?",
        (batch_id,),
    ).fetchone()
    if row is None:
        raise _ApiError(404, "batch not found")
    if row[_schema.COL_RUN_STATUS] in ("completed", "cancelled", "failed", "partial"):
        return {"noop": True, "status": row[_schema.COL_RUN_STATUS]}

    conn.execute(
        f"UPDATE {_schema.TBL_RUNS} SET {_schema.COL_RUN_STATUS} = 'cancelled' "
        f"WHERE {_schema.COL_RUN_ID} = ?",
        (batch_id,),
    )
    conn.commit()
    return {"noop": False, "status": "cancelled"}


# --------------------------------------------------------------------------- #
#  Dispatcher                                                                  #
# --------------------------------------------------------------------------- #


# Static (no-path-parameter) GET routes
_GET_ROUTES: dict[str, Callable[[sqlite3.Connection, dict[str, str]], Any]] = {
    "/api/stats":          _h_stats,
    "/api/runs":           _h_runs_list,
    "/api/batches":        _h_batches_list,
    "/api/steps/search":   _h_steps_search,
    "/api/thrashing":      _h_thrashing,
    "/api/projection":     _h_projection,
    "/api/tier-recommendation": _h_tier_recommendation,
}

# Parameterised GET routes — pattern -> (regex, handler)
_RUN_ID_RE = re.compile(r"^/api/runs/([A-Za-z0-9_\-]+)(/tasks|/steps)?$")
_BATCH_ID_RE = re.compile(
    r"^/api/batches/([A-Za-z0-9_\-]+)(/runs|/clusters|/outliers|/items|/dlq)?$"
)
# item_id is user-supplied so we accept a wider charset than run_id / batch_id.
# Still restricted to URL-safe characters to avoid path traversal issues.
_ITEM_ID_RE = re.compile(r"^/api/items/([A-Za-z0-9_\-\.:]+)$")


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
                    run_id, sub = m.group(1), m.group(2) or ""
                    if sub == "/tasks":
                        self._json(_h_run_tasks(conn, run_id, params))
                    elif sub == "/steps":
                        self._json(_h_run_steps(conn, run_id, params))
                    else:
                        self._json(_h_run_detail(conn, run_id, params))
                    return

                m = _BATCH_ID_RE.match(path)
                if m:
                    batch_id, sub = m.group(1), m.group(2) or ""
                    if sub == "/runs":
                        self._json(_h_batch_runs(conn, batch_id, params))
                    elif sub == "/clusters":
                        self._json(_h_batch_clusters(conn, batch_id, params))
                    elif sub == "/outliers":
                        self._json(_h_batch_outliers(conn, batch_id, params))
                    elif sub == "/items":
                        self._json(_h_batch_items(conn, batch_id, params))
                    elif sub == "/dlq":
                        self._json(_h_batch_dlq(conn, batch_id, params))
                    else:
                        self._json(_h_batch_detail(conn, batch_id, params))
                    return

                m = _ITEM_ID_RE.match(path)
                if m:
                    self._json(_h_item_detail(conn, m.group(1), params))
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
            path = parsed.path
            m = re.match(
                r"^/api/batches/([A-Za-z0-9_\-]+)/cancel$", path
            )
            if m:
                conn = self._open_db()
                try:
                    self._json(_h_batch_cancel(conn, m.group(1)))
                finally:
                    conn.close()
                return

            m = re.match(
                r"^/api/batches/([A-Za-z0-9_\-]+)/dlq/([A-Za-z0-9_\-]+)/(skip|acknowledge|replay)$",
                path,
            )
            if m:
                batch_id, run_id, action = m.group(1), m.group(2), m.group(3)
                conn = self._open_db()
                try:
                    if action == "replay":
                        self._json(_h_dlq_replay(conn, self.db_path, batch_id, run_id))
                    else:
                        disposition = {
                            "skip": _schema.DLQ_SKIPPED,
                            "acknowledge": _schema.DLQ_ACKNOWLEDGED,
                        }[action]
                        self._json(_h_dlq_disposition(conn, batch_id, run_id, disposition))
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
