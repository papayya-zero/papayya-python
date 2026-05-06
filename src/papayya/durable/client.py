"""Lowercase ``papayya()`` factory and back-compat shim.

The factory mirrors the unified ``Papayya`` class in
``papayya.papayya``; this module exists for the historical import path
``from papayya.durable import papayya`` and to keep ``PapayyaClient``
resolvable as an alias for one release.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from papayya.papayya import Papayya

from ._replay import ReplayError, replay
from .types import CheckpointStore


@dataclass
class PapayyaClientConfig:
    """Configuration for the papayya() factory.

    Kept as a dataclass for back-compat with code that imports it
    directly. The unified ``Papayya`` constructor takes the same fields
    as positional/keyword args.
    """

    api_key: str | None = None
    base_url: str | None = None
    store: CheckpointStore | None = None


class PapayyaClient(Papayya):
    """Back-compat shim for the previous internal class name.

    The unified ``Papayya`` constructor takes its config as discrete
    kwargs; the legacy ``PapayyaClient(config)`` form passed a single
    ``PapayyaClientConfig`` dataclass. This subclass translates the
    legacy shape so existing tests / library users keep working for one
    release. New code should use ``Papayya(...)`` directly.
    """

    def __init__(self, config: PapayyaClientConfig | None = None) -> None:
        cfg = config or PapayyaClientConfig()
        super().__init__(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            store=cfg.store,
        )


def papayya(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    store: CheckpointStore | None = None,
) -> Papayya:
    """Create a Papayya client for durable agent execution.

    Equivalent to ``Papayya(api_key=..., base_url=..., store=...)``.
    Resolves the api_key from env vars (``PAPAYYA_API_KEY``) and the CLI
    config (``~/.papayya/config.json``) when not passed explicitly.

    Usage::

        from papayya import papayya

        client = papayya()  # auto-detects api_key, persists to cloud
        run = client.run(agent="my-agent")
    """
    return Papayya(api_key=api_key, base_url=base_url, store=store)


__all__ = [
    "Papayya",
    "PapayyaClient",
    "PapayyaClientConfig",
    "papayya",
    "replay",
    "ReplayError",
]
