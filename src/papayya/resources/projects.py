from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from papayya.api import APIClient


class Projects:
    def __init__(self, api: APIClient) -> None:
        self._api = api

    def create(self, name: str, slug: str) -> dict[str, Any]:
        return self._api.create_project(name, slug)

    def list(self) -> list[dict[str, Any]]:
        return self._api.list_projects()

    def get(self, project_id: str) -> dict[str, Any]:
        return self._api._request("GET", f"/v1/projects/{project_id}")

    def update(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._api._request("PATCH", f"/v1/projects/{project_id}", json=kwargs)

    def delete(self, project_id: str) -> None:
        self._api._http.request("DELETE", f"/v1/projects/{project_id}")
