from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from papayya.api import APIClient


class Webhooks:
    def __init__(self, api: APIClient) -> None:
        self._api = api

    def create(self, agent_id: str, name: str, *, description: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"name": name}
        if description:
            body["description"] = description
        return self._api._request("POST", f"/v1/agents/{agent_id}/webhooks", json=body)

    def list(self, agent_id: str) -> list[dict[str, Any]]:
        return self._api._request("GET", f"/v1/agents/{agent_id}/webhooks")

    def delete(self, webhook_id: str) -> None:
        self._api._http.request("DELETE", f"/v1/webhooks/{webhook_id}")
