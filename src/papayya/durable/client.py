"""Factory for creating durable runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
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


_TENANT_KEY_SENTINEL = object()


def _resolve_yaml_tenant_key_field(yaml_path: Path) -> str | None:
    """Return the metadata-key name declared in papayya.yaml, or None.

    Missing yaml is silent (treated as a single-tenant project). Malformed
    yaml or version mismatches raise — caller-visible config errors surface
    here rather than at first run().
    """
    if not yaml_path.exists():
        return None
    spec: PapayyaYaml = load_yaml(yaml_path)
    return spec.tenant_key


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
        # Lazy-resolved on first run() so construction stays cheap and
        # tests that monkeypatch cwd are unaffected by import order.
        self._tenant_key_field: Any = _TENANT_KEY_SENTINEL

    def _project_tenant_key_field(self) -> str | None:
        if self._tenant_key_field is _TENANT_KEY_SENTINEL:
            try:
                self._tenant_key_field = _resolve_yaml_tenant_key_field(
                    Path("papayya.yaml")
                )
            except PapayyaYamlError:
                raise
        return self._tenant_key_field  # type: ignore[return-value]

    def run(
        self,
        agent: str,
        *,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        item_id: str | None = None,
        store: CheckpointStore | None = None,
    ) -> PapayyaRun:
        """Create a new durable run.

        When ``papayya.yaml`` declares a ``tenant_key:``, the supplied
        ``metadata`` MUST include that key — strict-when-declared. The
        extracted value is persisted in the indexed ``tenant_key`` column
        on every row written under this run.
        """
        tenant_key_field = self._project_tenant_key_field()
        tenant_key_value: str | None = None
        if tenant_key_field is not None:
            tenant_key_value = _extract_tenant_key(metadata, tenant_key_field)

        return PapayyaRun(
            DurableRunConfig(
                agent=agent,
                run_id=run_id,
                metadata=metadata,
                item_id=item_id,
                store=store or self._config.store,
                tenant_key=tenant_key_value,
            )
        )


def _extract_tenant_key(
    metadata: dict[str, Any] | None,
    tenant_key_field: str,
) -> str:
    """Pull the tenant key value from run metadata. Strict by design.

    Raises ValueError when papayya.yaml declares a tenant_key but the
    caller didn't include it in metadata, or included an empty/non-string
    value. The error names the missing key so the caller knows what
    contract they're violating.
    """
    if not metadata:
        raise ValueError(
            f"papayya.yaml declares tenant_key={tenant_key_field!r} but "
            f"run() was called with no metadata. Pass "
            f"metadata={{{tenant_key_field!r}: ...}} to identify the tenant."
        )
    if tenant_key_field not in metadata:
        raise ValueError(
            f"papayya.yaml declares tenant_key={tenant_key_field!r} but "
            f"run() metadata is missing this key. "
            f"metadata.keys()={sorted(metadata.keys())}"
        )
    value = metadata[tenant_key_field]
    if not isinstance(value, str) or value == "":
        raise ValueError(
            f"metadata[{tenant_key_field!r}] must be a non-empty string; "
            f"got {value!r}"
        )
    return value


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
