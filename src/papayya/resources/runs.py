from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from papayya.api import APIClient


class Runs:
    def __init__(self, api: APIClient) -> None:
        self._api = api

    def create(
        self,
        agent_id: str,
        input: Any,
        *,
        model: str | None = None,
        max_steps: int | None = None,
        budget_cents: int | None = None,
        callback_url: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"input": input}
        if model:
            body["model"] = model
        if max_steps:
            body["max_steps"] = max_steps
        if budget_cents:
            body["budget_cents"] = budget_cents
        if callback_url:
            body["callback_url"] = callback_url
        return self._api._request("POST", f"/v1/agents/{agent_id}/runs", json=body)

    def get(self, run_id: str) -> dict[str, Any]:
        return self._api._request("GET", f"/v1/runs/{run_id}")

    def list(self) -> list[dict[str, Any]]:
        return self._api._request("GET", "/v1/runs")

    def cancel(self, run_id: str) -> dict[str, Any]:
        return self._api._request("POST", f"/v1/runs/{run_id}/cancel")

    def replay(self, run_id: str, *, from_step: int) -> dict[str, Any]:
        return self._api._request("POST", f"/v1/runs/{run_id}/replay", json={"from_step": from_step})

    def steps(self, run_id: str) -> list[dict[str, Any]]:
        return self._api._request("GET", f"/v1/runs/{run_id}/steps")

    # ── Dead Letter Queue ──────────────────────────────────────────────────
    # A failed/budget_exceeded run that belongs to a batch lands in the DLQ
    # until the operator triages it. Use one of:
    #   - dlq_skip       — accept the failure, don't replay
    #   - dlq_acknowledge — record review, don't replay
    #   - dlq_replay     — re-issue the run from input_snapshot
    # Once every failure in a batch has a disposition, the batch promotes
    # from 'partial' to 'completed'. See Batches.dlq() for the list endpoint.

    def dlq_skip(self, run_id: str) -> dict[str, Any]:
        """Mark a failed run as 'skipped' — accept the failure as terminal.
        Returns the updated run with dlq_disposition set."""
        return self._api._request("POST", f"/v1/runs/{run_id}/dlq/skip")

    def dlq_acknowledge(self, run_id: str) -> dict[str, Any]:
        """Mark a failed run as 'acknowledged' — record that the operator
        has reviewed the failure but is choosing to leave it. Functionally
        equivalent to skip; semantically distinct (skip ≈ "not worth
        looking at"; acknowledge ≈ "I've looked at this")."""
        return self._api._request("POST", f"/v1/runs/{run_id}/dlq/acknowledge")

    def dlq_replay(self, run_id: str) -> dict[str, Any]:
        """Re-issue the failed run from its input_snapshot as a new queued
        run. Marks the source as 'replayed' and links the new run via
        replayed_from. Returns the new run (HTTP 202)."""
        return self._api._request("POST", f"/v1/runs/{run_id}/dlq/replay")
