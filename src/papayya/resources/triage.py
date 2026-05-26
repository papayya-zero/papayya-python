"""Unified `Needs Attention` feed across DLQ + quarantine.

Plan 09's `GET /v1/triage` aggregates failed/budget_exceeded DLQ rows and
quarantine rows into one tenant-scoped, keyset-paginated stream. The dashboard
fork (Plan 18) and the ``papayya triage`` CLI read from this endpoint.
Per-row state-machine actions stay on the existing endpoints
(``runs.release`` / ``runs.discard`` / ``runs.dlq_replay`` / ``runs.dlq_skip``
/ ``runs.dlq_acknowledge``); this resource only exposes the read surface and
the auto-paging iterator the CLI uses.
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
        workload: str | None = None,
        tenant: str | None = None,
        kind: str = "all",
        cursor: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """GET /v1/triage. Returns
        ``{"items": [...], "next_cursor": "..."|null, "total": N}``.

        ``kind``: ``"all"`` (default), ``"dlq"``, or ``"quarantine"``.

        ``workload`` is accepted but currently ignored by the server (a
        ``Warning: 299 - "workload filter not yet supported"`` response
        header is emitted). The column lands with Plans 10/11/12 — the
        parameter is wired today so the CLI and the dashboard need no
        change when it does.

        ``tenant`` filters on the user-supplied ``metadata.tenant`` value
        on the run's input payload. Placeholder until the first-class
        tenant column lands.

        ``limit=0`` is the count-only mode used by Plan 18's sidebar
        badge — the server skips the row fetch entirely and just returns
        ``{"items": [], "total": N}``.
        """
        params: dict[str, Any] = {"kind": kind, "limit": limit}
        if workload:
            params["workload"] = workload
        if tenant:
            params["tenant"] = tenant
        if cursor:
            params["cursor"] = cursor
        return self._api._request("GET", "/v1/triage", params=params)

    def iter(
        self,
        *,
        workload: str | None = None,
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
                workload=workload,
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
