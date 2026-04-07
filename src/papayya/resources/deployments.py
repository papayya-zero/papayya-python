from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from papayya.api import APIClient


class Deployments:
    def __init__(self, api: APIClient) -> None:
        self._api = api

    def create(
        self,
        agent_id: str,
        tarball: bytes,
        *,
        runtime: str = "python",
        entrypoint: str = "agent.py",
    ) -> dict[str, Any]:
        return self._api.upload_deployment(agent_id, tarball, runtime, entrypoint)

    def get(self, deployment_id: str) -> dict[str, Any]:
        return self._api.get_deployment(deployment_id)

    def list(self, agent_id: str) -> list[dict[str, Any]]:
        return self._api.list_deployments(agent_id)
