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
