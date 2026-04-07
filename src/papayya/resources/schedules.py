from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from papayya.api import APIClient


class Schedules:
    def __init__(self, api: APIClient) -> None:
        self._api = api

    def create(
        self,
        agent_id: str,
        cron: str,
        *,
        timezone: str | None = None,
        input: str | None = None,
        max_steps: int | None = None,
        budget_cents: int | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"cron_expression": cron}
        if timezone:
            body["timezone"] = timezone
        if input:
            body["input"] = input
        if max_steps:
            body["max_steps"] = max_steps
        if budget_cents:
            body["budget_cents"] = budget_cents
        return self._api._request("POST", f"/v1/agents/{agent_id}/schedules", json=body)

    def list(self, agent_id: str | None = None) -> list[dict[str, Any]]:
        if agent_id:
            return self._api._request("GET", f"/v1/agents/{agent_id}/schedules")
        return self._api._request("GET", "/v1/schedules")

    def get(self, schedule_id: str) -> dict[str, Any]:
        return self._api._request("GET", f"/v1/schedules/{schedule_id}")

    def update(self, schedule_id: str, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if "cron" in kwargs:
            body["cron_expression"] = kwargs["cron"]
        if "timezone" in kwargs:
            body["timezone"] = kwargs["timezone"]
        if "input" in kwargs:
            body["input"] = kwargs["input"]
        if "max_steps" in kwargs:
            body["max_steps"] = kwargs["max_steps"]
        if "budget_cents" in kwargs:
            body["budget_cents"] = kwargs["budget_cents"]
        if "enabled" in kwargs:
            body["enabled"] = kwargs["enabled"]
        return self._api._request("PATCH", f"/v1/schedules/{schedule_id}", json=body)

    def delete(self, schedule_id: str) -> None:
        self._api._http.request("DELETE", f"/v1/schedules/{schedule_id}")

    def enable(self, schedule_id: str) -> dict[str, Any]:
        return self.update(schedule_id, enabled=True)

    def disable(self, schedule_id: str) -> dict[str, Any]:
        return self.update(schedule_id, enabled=False)
