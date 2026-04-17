"""Papayya — the primary SDK client for the Papayya platform.

Usage:
    from papayya import Papayya

    papayya = Papayya(api_key="cpk_...")
    run = papayya.runs.create(agent_id="...", input="hello")
"""

from __future__ import annotations

from papayya.api import APIClient, resolve_config
from papayya.resources.runs import Runs
from papayya.resources.batches import Batches
from papayya.resources.schedules import Schedules
from papayya.resources.webhooks import Webhooks
from papayya.resources.agents import Agents
from papayya.resources.deployments import Deployments
from papayya.resources.secrets import Secrets
from papayya.resources.projects import Projects
from papayya.resources.api_keys import ApiKeys
from papayya.resources.usage import Usage


class Papayya:
    """Resource-namespaced client for the Papayya platform.

    All platform operations are available through namespaced resources:
        papayya.runs.create(...)
        papayya.batches.create(...)
        papayya.schedules.create(...)
        papayya.webhooks.create(...)
        papayya.agents.list()
        papayya.secrets.set(...)
        papayya.usage.summary()
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        config = resolve_config(api_key, base_url)
        self._api = APIClient(config)

        self.runs = Runs(self._api)
        self.batches = Batches(self._api)
        self.schedules = Schedules(self._api)
        self.webhooks = Webhooks(self._api)
        self.agents = Agents(self._api)
        self.deployments = Deployments(self._api)
        self.secrets = Secrets(self._api)
        self.projects = Projects(self._api)
        self.api_keys = ApiKeys(self._api)
        self.usage = Usage(self._api)

    def close(self) -> None:
        """Close the underlying HTTP connection."""
        self._api.close()

    def __enter__(self) -> Papayya:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
