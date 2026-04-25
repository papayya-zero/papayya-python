"""Factory for creating durable runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from papayya._config import env_config, load_cli_config
from papayya._defaults import DEFAULT_BASE_URL

from .run import PapayyaRun
from .types import CheckpointStore, DurableRunConfig


@dataclass
class PapayyaClientConfig:
    """Configuration for the papayya() factory."""

    api_key: str | None = None
    base_url: str | None = None
    store: CheckpointStore | None = None


def _resolve_api_key(explicit: str | None = None) -> str | None:
    """Resolve API key from explicit param, env vars, or saved config."""
    if explicit:
        return explicit
    key = os.environ.get("PAPAYYA_API_KEY")
    if key:
        return key
    cfg = load_cli_config()
    return env_config(cfg).get("api_key")


def _resolve_base_url(explicit: str | None = None) -> str:
    """Resolve base URL from explicit param, env var, or default."""
    if explicit:
        return explicit
    return os.environ.get("PAPAYYA_BASE_URL") or DEFAULT_BASE_URL


def _auto_store(api_key: str | None, base_url: str | None) -> CheckpointStore:
    """Auto-select store: CloudStore if API key available, else SQLiteStore.

    SQLite path resolution: ``PAPAYYA_LOCAL_DB_PATH`` env var if set, else
    the default ``.papayya/local.db``. The env-var override is used by the
    runtime worker (`papayya.runtime`) to point customer SQLite writes at a
    shared file so multiple workers (and the dispatcher / dashboard) read
    consistent state.
    """
    resolved_key = _resolve_api_key(api_key)
    if resolved_key:
        resolved_url = _resolve_base_url(base_url)
        from .cloud_store import CloudStore, CloudStoreConfig
        return CloudStore(CloudStoreConfig(api_key=resolved_key, base_url=resolved_url))
    from .sqlite_store import SQLiteStore
    db_path = os.environ.get("PAPAYYA_LOCAL_DB_PATH")
    if db_path:
        return SQLiteStore(db_path)
    return SQLiteStore()


class PapayyaClient:
    """Client for creating durable runs.

    Usage::

        from papayya.durable import papayya

        # Auto-detects API key from env/config → uses CloudStore
        t = papayya()
        run = t.run(agent="my-agent")
    """

    def __init__(self, config: PapayyaClientConfig | None = None) -> None:
        self._config = config or PapayyaClientConfig()

    def run(
        self,
        agent: str,
        *,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        item_id: str | None = None,
        store: CheckpointStore | None = None,
    ) -> PapayyaRun:
        """Create a new durable run."""
        return PapayyaRun(
            DurableRunConfig(
                agent=agent,
                run_id=run_id,
                metadata=metadata,
                item_id=item_id,
                store=store or self._config.store,
            )
        )


def papayya(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    store: CheckpointStore | None = None,
) -> PapayyaClient:
    """Create a Papayya client for durable agent execution.

    If no store is provided, automatically uses CloudStore when an API key
    is found (from params, PAPAYYA_API_KEY env var, or ~/.papayya/config.json).
    Falls back to MemoryStore if no API key is available.

    Usage::

        from papayya.durable import papayya

        t = papayya()  # auto-detects API key, persists to cloud
        run = t.run(agent="my-agent")
    """
    resolved_store = store or _auto_store(api_key, base_url)
    return PapayyaClient(PapayyaClientConfig(api_key=api_key, base_url=base_url, store=resolved_store))
