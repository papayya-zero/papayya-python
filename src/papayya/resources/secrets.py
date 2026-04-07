from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from papayya.api import APIClient


class Secrets:
    def __init__(self, api: APIClient) -> None:
        self._api = api

    def set(self, project_id: str, name: str, value: str) -> dict[str, Any]:
        return self._api.set_secret(project_id, name, value)

    def list(self, project_id: str) -> list[dict[str, Any]]:
        return self._api.list_secrets(project_id)

    def delete(self, project_id: str, name: str) -> None:
        self._api.delete_secret(project_id, name)
