from __future__ import annotations

import time
from typing import Any, Iterable, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from papayya.api import APIClient


# Terminal statuses for a batch as far as the SDK's wait() is concerned.
# Paused and partial are included because both are stuck states that need
# caller intervention to leave: paused → bump the budget cap and resume;
# partial → triage the DLQ via Runs.dlq_skip/acknowledge/replay. Treating
# them as terminal forces the caller to notice rather than hang forever.
# True non-terminal: materializing, queued, running.
_TERMINAL_BATCH_STATUSES = frozenset({"completed", "failed", "cancelled", "paused", "partial"})


class Batches:
    """Client for the /v1/batches surface.

    Mirrors the Runs resource shape. A batch is a collection of runs
    submitted together under a single concurrency + budget cap; see
    memory/batch_primitive_design.md. The backend enforces both caps at
    dispatch — the SDK just hands over the submission and exposes the
    read + lifecycle endpoints.
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

        import json

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

    def get(self, batch_id: str) -> dict[str, Any]:
        return self._api._request("GET", f"/v1/batches/{batch_id}")

    def list(
        self,
        *,
        status: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if status is not None:
            params["status"] = status
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return self._api._request("GET", "/v1/batches", params=params)

    def runs(
        self,
        batch_id: str,
        *,
        status: str | None = None,
        page: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Paginated list of the batch's child runs."""
        params: dict[str, Any] = {}
        if status is not None:
            params["status"] = status
        if page is not None:
            params["page"] = page
        if limit is not None:
            params["limit"] = limit
        return self._api._request("GET", f"/v1/batches/{batch_id}/runs", params=params)

    def cancel(self, batch_id: str) -> dict[str, Any]:
        """Cancel a batch. Returns 202 with the current batch state — the
        backend fans cancellation out to child runs in the background."""
        return self._api._request("POST", f"/v1/batches/{batch_id}/cancel")

    def retry_failed(self, batch_id: str) -> dict[str, Any]:
        """Re-enqueue every failed child of the batch as a new run. The
        batch's total_items is bumped to match. Returns the updated batch.

        Distinct from :meth:`dlq` — retry-failed is a blanket re-run that
        doesn't link to the source or interact with DLQ disposition. For
        per-run, traceable replay use ``Runs.dlq_replay``."""
        return self._api._request("POST", f"/v1/batches/{batch_id}/retry-failed")

    def dlq(self, batch_id: str) -> list[dict[str, Any]]:
        """List failed/budget_exceeded child runs in this batch's Dead
        Letter Queue (i.e. without a dlq_disposition yet). Returns the
        same shape as :meth:`runs` plus the ``input_snapshot`` field on
        each row so the operator can replay from the original input.

        Pair with ``Runs.dlq_skip`` / ``dlq_acknowledge`` / ``dlq_replay``
        to drain the queue. Once empty, a batch sitting in 'partial'
        promotes to 'completed' automatically."""
        return self._api._request("GET", f"/v1/batches/{batch_id}/dlq")

    def wait(
        self,
        batch_id: str,
        *,
        timeout: float = 3600,
        poll_interval: float = 5,
    ) -> dict[str, Any]:
        """Block until the batch reaches a terminal status.

        Terminal here includes 'paused' — a paused batch isn't making
        progress without caller intervention (bumping the budget cap),
        so returning lets the caller decide what to do rather than
        hanging. Raises TimeoutError if no terminal transition within
        ``timeout`` seconds.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            batch = self.get(batch_id)
            if batch.get("status") in _TERMINAL_BATCH_STATUSES:
                return batch
            time.sleep(poll_interval)
        raise TimeoutError(f"Batch {batch_id} did not reach terminal status within {timeout}s")

    def stream_results(
        self,
        batch_id: str,
        *,
        poll_interval: float = 2,
        include_failed: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Yield child runs as they reach terminal status.

        Polls ``GET /v1/batches/{id}/runs?status=completed`` (and failed,
        if requested) and yields each newly-terminal run once. Generator
        exits when the parent batch itself reaches terminal status.

        Will be rewired to the SSE endpoint when Phase 4b ships — the
        public signature is designed to stay stable across that swap.
        """
        seen: set[str] = set()
        terminal_run_statuses = ["completed"]
        if include_failed:
            terminal_run_statuses.extend(["failed", "cancelled", "budget_exceeded"])

        while True:
            for run_status in terminal_run_statuses:
                page = 0
                while True:
                    children = self.runs(batch_id, status=run_status, page=page, limit=200)
                    if not children:
                        break
                    for run in children:
                        rid = run.get("id")
                        if rid and rid not in seen:
                            seen.add(rid)
                            yield run
                    if len(children) < 200:
                        break
                    page += 1

            batch = self.get(batch_id)
            if batch.get("status") in _TERMINAL_BATCH_STATUSES:
                # Drain once more after the batch goes terminal so any
                # run that completed between the last poll and the
                # terminal flip still gets yielded.
                for run_status in terminal_run_statuses:
                    for run in self.runs(batch_id, status=run_status, limit=200):
                        rid = run.get("id")
                        if rid and rid not in seen:
                            seen.add(rid)
                            yield run
                return

            time.sleep(poll_interval)
