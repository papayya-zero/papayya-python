"""Papayya — the canonical SDK client.

One client class covers both surfaces:

* **Durable execution** — ``papayya.run(agent="...", metadata={...})`` returns
  a ``PapayyaRun`` you wrap your steps with. Works locally (SQLite) or
  against the hosted control plane (CloudStore) — same call, the right
  store gets selected automatically.
* **Platform resources** — ``papayya.runs``, ``papayya.batches``,
  ``papayya.agents``, ``papayya.schedules``, ``papayya.webhooks``,
  ``papayya.deployments``, ``papayya.secrets``, ``papayya.projects``,
  ``papayya.api_keys``, ``papayya.usage``. These talk to the hosted API
  and require an ``api_key``.

Resource namespaces lazy-resolve the API key, so a local-only script
that never touches a resource namespace runs without credentials. The
``papayya()`` lowercase factory is preserved as an ergonomic alias.

Usage::

    from papayya import Papayya

    client = Papayya(api_key="cpk_...")
    run = client.run(agent="my-agent", metadata={"organization_id": "org_42"})

    # Or use the factory for automatic env/config resolution:
    from papayya import papayya
    client = papayya()
"""

from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path
from typing import Any

from papayya._config import (
    PapayyaYaml,
    PapayyaYamlError,
    env_config,
    load_cli_config,
    load_yaml,
)
from papayya._defaults import DEFAULT_BASE_URL
from papayya.api import APIClient, resolve_config
from papayya.resources.agents import Agents
from papayya.resources.api_keys import ApiKeys
from papayya.resources.batches import Batches
from papayya.resources.deployments import Deployments
from papayya.resources.projects import Projects
from papayya.resources.runs import Runs
from papayya.resources.schedules import Schedules
from papayya.resources.secrets import Secrets
from papayya.resources.usage import Usage
from papayya.resources.webhooks import Webhooks


_PARTITION_KEY_SENTINEL = object()


def _resolve_durable_api_key(explicit: str | None) -> str | None:
    """Permissive API key resolution for the durable path.

    Unlike `resolve_config` (which raises when no key is found), this
    returns None silently — local-only durable runs are valid and the
    SDK falls back to SQLiteStore in that case.
    """
    if explicit:
        return explicit
    key = os.environ.get("PAPAYYA_API_KEY")
    if key:
        return key
    cfg = load_cli_config()
    return env_config(cfg).get("api_key")


def _resolve_durable_base_url(explicit: str | None) -> str:
    if explicit:
        return explicit
    return os.environ.get("PAPAYYA_BASE_URL") or DEFAULT_BASE_URL


def _resolve_yaml_partition_key_field(yaml_path: Path) -> str | None:
    """Read the partition_key declaration from papayya.yaml. None when absent.

    Missing yaml is silent — single-partition projects don't need to
    write one. Malformed yaml or version mismatches raise
    PapayyaYamlError so config bugs surface here rather than at first
    run().
    """
    if not yaml_path.exists():
        return None
    spec: PapayyaYaml = load_yaml(yaml_path)
    return spec.partition_key


def _extract_partition_key(
    metadata: dict[str, Any] | None,
    partition_key_field: str,
) -> str:
    """Pull the partition key value from run metadata. Strict by design.

    Raises ValueError when papayya.yaml declares a partition_key but
    the caller didn't include it in metadata, or included an
    empty/non-string value. The error names the missing key so the
    caller knows what contract they're violating.
    """
    if not metadata:
        raise ValueError(
            f"papayya.yaml declares partition_key={partition_key_field!r} but "
            f"run() was called with no metadata. Pass "
            f"metadata={{{partition_key_field!r}: ...}} to identify the partition."
        )
    if partition_key_field not in metadata:
        raise ValueError(
            f"papayya.yaml declares partition_key={partition_key_field!r} but "
            f"run() metadata is missing this key. "
            f"metadata.keys()={sorted(metadata.keys())}"
        )
    value = metadata[partition_key_field]
    if not isinstance(value, str) or value == "":
        raise ValueError(
            f"metadata[{partition_key_field!r}] must be a non-empty string; "
            f"got {value!r}"
        )
    return value


class Papayya:
    """Canonical Papayya SDK client.

    Combines the durable-execution runtime with platform resource
    namespaces. Use ``papayya.run(agent="...")`` for durable execution
    and ``papayya.runs.create(...)`` (and friends) for hosted-API
    resource operations.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        store: Any | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._store_override = store
        # Resource namespaces resolve the API client lazily so a
        # local-only durable script (no api_key) can construct Papayya
        # without ever needing credentials.
        self._api: APIClient | None = None
        # papayya.yaml partition_key declaration — resolved on first run().
        self._partition_key_field: Any = _PARTITION_KEY_SENTINEL

    # --- internal ----------------------------------------------------- #

    def _api_client(self) -> APIClient:
        """Lazily construct the platform API client.

        Resource namespaces call this on first use; durable runs that
        write to the local SQLite store never trigger it. Raises the
        usual `PapayyaAPIError(401, "No API key...")` when no key is
        resolvable.
        """
        if self._api is None:
            config = resolve_config(self._api_key, self._base_url)
            self._api = APIClient(config)
        return self._api

    def _project_partition_key_field(self) -> str | None:
        if self._partition_key_field is _PARTITION_KEY_SENTINEL:
            self._partition_key_field = _resolve_yaml_partition_key_field(
                Path("papayya.yaml")
            )
        return self._partition_key_field  # type: ignore[return-value]

    def _auto_store(self) -> Any:
        """Auto-select a CheckpointStore for durable runs.

        CloudStore when an API key is resolvable, SQLiteStore otherwise.
        Mirrors the legacy `papayya.durable.client._auto_store` shape.
        """
        from papayya.durable.cloud_store import CloudStore, CloudStoreConfig
        from papayya.durable.sqlite_store import SQLiteStore

        resolved_key = _resolve_durable_api_key(self._api_key)
        if resolved_key:
            resolved_url = _resolve_durable_base_url(self._base_url)
            return CloudStore(
                CloudStoreConfig(api_key=resolved_key, base_url=resolved_url)
            )
        db_path = os.environ.get("PAPAYYA_LOCAL_DB_PATH")
        if db_path:
            return SQLiteStore(db_path)
        return SQLiteStore()

    # --- durable runtime ---------------------------------------------- #

    def run(
        self,
        agent: str,
        *,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        item_id: str | None = None,
        store: Any | None = None,
    ) -> Any:
        """Create a new durable run.

        When ``papayya.yaml`` declares a ``partition_key:``, the
        supplied ``metadata`` MUST include that key —
        strict-when-declared. The extracted value is persisted in the
        indexed ``partition_key`` column on every row written under
        this run.
        """
        # Layer 3 #9: the documented pattern is now
        # ``def process_note(run, note): ...`` with ``run`` injected by
        # the @agent wrapper. Customers on the legacy pattern call this
        # method themselves from inside the fn body — that's the line
        # they need to delete, so we warn at the call site (the wrapper
        # sets a contextvar before invoking the legacy fn).
        from papayya.agent import legacy_agent_path_active
        if legacy_agent_path_active():
            import warnings
            warnings.warn(
                "Calling papayya().run() inside an @agent function is "
                "deprecated. Add `run` as the first positional parameter of "
                "your agent function (e.g. `def process_note(run, note):`) "
                "and it will be injected automatically. The legacy pattern "
                "will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )

        from papayya.durable._replay import consume_replay_hydration
        from papayya.durable.run import PapayyaRun
        from papayya.durable.types import DurableRunConfig

        partition_key_field = self._project_partition_key_field()
        partition_key_value: str | None = None
        if partition_key_field is not None:
            partition_key_value = _extract_partition_key(metadata, partition_key_field)

        resolved_store = store or self._store_override or self._auto_store()

        # Replay Phase 3: when papayya.durable._replay is driving us, the
        # one-shot _REPLAY_HYDRATION contextvar carries the new run's id
        # and the TaskEntry rows to seed the cache with. We force run_id
        # to the contextvar value (so caller-supplied run_id= is ignored
        # mid-replay — the replayer owns identity) and pass the rows
        # through as prepopulated_tasks. consume_* clears the contextvar
        # so only the first papayya.run() call inside the replayed
        # @agent body picks this up; subsequent intra-fn run() calls
        # construct normal fresh runs.
        hydration = consume_replay_hydration()
        if hydration is not None:
            forced_run_id, prepopulated = hydration
            return PapayyaRun(
                DurableRunConfig(
                    agent=agent,
                    run_id=forced_run_id,
                    metadata=metadata,
                    item_id=item_id,
                    store=resolved_store,
                    partition_key=partition_key_value,
                    prepopulated_tasks=prepopulated,
                )
            )

        return PapayyaRun(
            DurableRunConfig(
                agent=agent,
                run_id=run_id,
                metadata=metadata,
                item_id=item_id,
                store=resolved_store,
                partition_key=partition_key_value,
            )
        )

    # --- resource namespaces ------------------------------------------ #

    @cached_property
    def runs(self) -> Runs:
        return Runs(self._api_client())

    @cached_property
    def batches(self) -> Batches:
        return Batches(self._api_client())

    @cached_property
    def schedules(self) -> Schedules:
        return Schedules(self._api_client())

    @cached_property
    def webhooks(self) -> Webhooks:
        return Webhooks(self._api_client())

    @cached_property
    def agents(self) -> Agents:
        return Agents(self._api_client())

    @cached_property
    def deployments(self) -> Deployments:
        return Deployments(self._api_client())

    @cached_property
    def secrets(self) -> Secrets:
        return Secrets(self._api_client())

    @cached_property
    def projects(self) -> Projects:
        return Projects(self._api_client())

    @cached_property
    def api_keys(self) -> ApiKeys:
        return ApiKeys(self._api_client())

    @cached_property
    def usage(self) -> Usage:
        return Usage(self._api_client())

    # --- lifecycle ---------------------------------------------------- #

    def close(self) -> None:
        """Close the underlying HTTP connection if one was opened."""
        if self._api is not None:
            self._api.close()

    def __enter__(self) -> Papayya:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
