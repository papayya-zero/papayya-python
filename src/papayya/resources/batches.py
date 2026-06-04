from __future__ import annotations

import json
from typing import Any, Iterable, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from papayya.api import APIClient


class Batches:
    """Client for the /v1/batches submission surface.

    v1→v2 cutover: a batch is no longer its own table row. Submitting
    (:meth:`create` / :meth:`create_stream`) mints N durable runs that share
    a ``group_id`` (a minted parent_run_id) and returns
    ``{group_id, agent_id, status, total_items, created_at}``.

    The v1 batch read + lifecycle endpoints (get/list/runs/cancel/
    retry_failed/dlq/results/…) retired with the v1 DROP; poll a group as
    durable runs filtered by ``group_id`` (``Runs.list`` / the durable runs
    surface) instead.
    """

    def __init__(self, api: APIClient) -> None:
        self._api = api

    def create(
        self,
        agent_id: str,
        items: list[dict[str, Any]],
        *,
        name: str | None = None,
        budget_cents_cap: int | None = None,
        concurrency_cap: int | None = None,
        callback_url: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Submit a batch via the JSON body path.

        Best for small batches — the backend caps this path at 1,000 items.
        For larger submissions, use :meth:`create_stream` which streams
        NDJSON and has no item ceiling (only a 1 GiB byte guard).

        Each item is ``{"input": <any>, "metadata"?: <any>}``. The run
        inherits the agent's configured budget/max_steps; per-item
        overrides are deliberately not supported.
        """
        body: dict[str, Any] = {"agent_id": agent_id, "items": items}
        if name is not None:
            body["name"] = name
        if budget_cents_cap is not None:
            body["budget_cents_cap"] = budget_cents_cap
        if concurrency_cap is not None:
            body["concurrency_cap"] = concurrency_cap
        if callback_url is not None:
            body["callback_url"] = callback_url
        if idempotency_key is not None:
            body["idempotency_key"] = idempotency_key
        return self._api._request("POST", "/v1/batches", json=body)

    def create_stream(
        self,
        agent_id: str,
        items: Iterable[dict[str, Any]],
        *,
        name: str | None = None,
        budget_cents_cap: int | None = None,
        concurrency_cap: int | None = None,
        callback_url: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Submit a batch via the NDJSON streaming path — no item ceiling.

        First NDJSON line is the header (batch-level config). Every
        subsequent line is one item. Backend materialises runs in 1k-row
        chunks while the stream is open, then flips status queued once
        EOF lands. Use this path whenever the item count is large or
        unknown ahead of time.
        """
        header: dict[str, Any] = {"agent_id": agent_id}
        if name is not None:
            header["name"] = name
        if budget_cents_cap is not None:
            header["budget_cents_cap"] = budget_cents_cap
        if concurrency_cap is not None:
            header["concurrency_cap"] = concurrency_cap
        if callback_url is not None:
            header["callback_url"] = callback_url
        if idempotency_key is not None:
            header["idempotency_key"] = idempotency_key

        def _lines() -> Iterator[bytes]:
            yield (json.dumps(header) + "\n").encode("utf-8")
            for item in items:
                yield (json.dumps(item) + "\n").encode("utf-8")

        resp = self._api._http.post(
            "/v1/batches",
            content=_lines(),
            headers={"Content-Type": "application/x-ndjson"},
        )
        if not resp.is_success:
            from papayya.api import PapayyaAPIError

            raise PapayyaAPIError(resp.status_code, resp.text)
        return resp.json()
