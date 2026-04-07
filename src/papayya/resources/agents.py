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
