from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from papayya.api import APIClient


class ApiKeys:
    def __init__(self, api: APIClient) -> None:
        self._api = api

    def create(self, project_id: str, name: str) -> dict[str, Any]:
        return self._api.create_api_key(project_id, name)

    def list(self, project_id: str) -> list[dict[str, Any]]:
        return self._api._request("GET", f"/v1/projects/{project_id}/api-keys")

    def revoke(self, project_id: str, key_id: str) -> None:
        self._api._http.request("DELETE", f"/v1/projects/{project_id}/api-keys/{key_id}")
