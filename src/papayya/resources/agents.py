from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from papayya.api import APIClient


class Agents:
    def __init__(self, api: APIClient) -> None:
        self._api = api

    def create(
        self,
        name: str,
        slug: str,
        project_id: str,
        *,
        config: dict[str, Any] | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"name": name, "slug": slug, "project_id": project_id}
        if config:
            body["config"] = config
        if description:
            body["description"] = description
        return self._api._request("POST", "/v1/agents", json=body)

    def list(self, project_id: str | None = None) -> list[dict[str, Any]]:
        return self._api.list_agents(project_id)

    def get(self, agent_id: str) -> dict[str, Any]:
        return self._api.get_agent(agent_id)

    def update(self, agent_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._api._request("PATCH", f"/v1/agents/{agent_id}", json=kwargs)

    def resume(self, agent_id: str) -> dict[str, Any]:
        """Resume a workload auto-paused by the degraded-rate fence (Plan 33).
        Clears the paused flag so the dispatcher resumes leasing this agent's
        runs. 409 if the agent is not auto-paused. A remediation agent calls
        this after fixing the root cause (e.g. ``update(agent_id, config=...)``
        to swap the model) — resume without a fix just re-trips the fence."""
        return self._api._request("POST", f"/v1/agents/{agent_id}/resume")

    def outcome_rate(self, agent: str, window: int = 30) -> dict[str, Any]:
        """Per-agent outcome-quality surface (Plan 35): ran-vs-worked counts +
        derived rates, a per-day trend, and a reason histogram over ``window``
        days. Keyed by the agent SLUG (not the UUID). The same data the
        dashboard's Outcome quality view renders — read it to decide what to
        fix before resuming."""
        return self._api._request(
            "GET",
            "/v1/durable/runs/outcome-rate",
            params={"agent": agent, "window": window},
        )
