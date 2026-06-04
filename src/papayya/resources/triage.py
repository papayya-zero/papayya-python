"""Unified `Needs Attention` feed across the durable DLQ + quarantine lanes.

``GET /v1/triage`` (v1→v2 cutover: now backed by ``durable_runs``) aggregates
degraded/failed runs and quarantined runs into one tenant-scoped,
keyset-paginated stream. Each row is
``{kind, run_id, group_id?, agent, partition_key?, status, reason?,
available_actions, occurred_at}``. The dashboard's "Needs attention" lane and
the ``papayya triage`` CLI read from this endpoint.

Quarantine-lane actions (``runs.release`` / ``runs.discard``) are live on the
durable surface. The DLQ-lane disposition actions (skip/acknowledge/replay)
are a deferred follow-up — the durable model has no ``dlq_disposition`` column
yet — so this resource exposes only the read surface and the auto-paging
iterator the CLI uses.
"""
from __future__ import annotations

from typing import Any, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from papayya.api import APIClient


class Triage:
    def __init__(self, api: APIClient) -> None:
        self._api = api

    def list(
        self,
        *,
        partition_key: str | None = None,
        tenant: str | None = None,
        kind: str = "all",
        cursor: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """GET /v1/triage. Returns
        ``{"items": [...], "next_cursor": "..."|null, "total": N}``.

        ``kind``: ``"all"`` (default), ``"dlq"``, or ``"quarantine"``.

        ``partition_key`` filters on the durable run's per-tenant partition
        axis. ``tenant`` is a back-compat alias for the same filter (the
        server accepts either; ``partition_key`` wins when both are given).

        ``limit=0`` is the count-only mode used by the dashboard sidebar
        badge — the server skips the row fetch entirely and just returns
        ``{"items": [], "total": N}``.
        """
        params: dict[str, Any] = {"kind": kind, "limit": limit}
        if partition_key:
            params["partition_key"] = partition_key
        elif tenant:
            params["tenant"] = tenant
        if cursor:
            params["cursor"] = cursor
        return self._api._request("GET", "/v1/triage", params=params)

    def iter(
        self,
        *,
        partition_key: str | None = None,
        tenant: str | None = None,
        kind: str = "all",
        page_size: int = 50,
    ) -> Iterator[dict[str, Any]]:
        """Yield every triage row, transparently following ``next_cursor``.

        Backs ``papayya triage list`` so the CLI doesn't bake pagination
        into the command body. Exits when the server returns no cursor.
        """
        cursor: str | None = None
        while True:
            page = self.list(
                partition_key=partition_key,
                tenant=tenant,
                kind=kind,
                cursor=cursor,
                limit=page_size,
            )
            for row in page.get("items", []):
                yield row
            cursor = page.get("next_cursor")
            if not cursor:
                return
