"""Papayya — the canonical SDK client.

One client class covers both surfaces:

* **Durable execution** — ``papayya.item(agent="...", metadata={...})``
  returns an ``Item`` you wrap your steps with. Works locally (SQLite) or
  against the hosted control plane (CloudStore) — same call, the right
  store gets selected automatically. (``papayya.run(...)`` is the
  pre-Plan-34 spelling, kept as a deprecated alias.)
* **Platform resources** — ``papayya.runs`` (invocations),
  ``papayya.items`` (per-item records), ``papayya.agents``,
  ``papayya.schedules``, ``papayya.webhooks``, ``papayya.deployments``,
  ``papayya.secrets``, ``papayya.projects``, ``papayya.api_keys``,
  ``papayya.usage``. These talk to the hosted API and require an
  ``api_key``.

BREAKING in 0.3.0 (Plan 34): ``papayya.runs`` used to be the per-item
resource; it now addresses invocations (the old ``batches`` surface).
Per-item access moved to ``papayya.items``. ``papayya.batches`` forwards
to ``papayya.runs`` as a deprecated alias.

Resource namespaces lazy-resolve the API key, so a local-only script
that never touches a resource namespace runs without credentials. The
``papayya()`` lowercase factory is preserved as an ergonomic alias.

Usage::

    from papayya import Papayya

    client = Papayya(api_key="cpk_...")
    item = client.item(agent="my-agent", metadata={"organization_id": "org_42"})

    # Or use the factory for automatic env/config resolution:
    from papayya import papayya
    client = papayya()
"""

from __future__ import annotations

import os
from functools import cached_property
from typing import Any

from papayya._config import (
    env_config,
    load_cli_config,
)
from papayya._defaults import DEFAULT_BASE_URL
from papayya.api import APIClient, PapayyaAPIError, resolve_config
from papayya.resources.agents import Agents
from papayya.resources.api_keys import ApiKeys
from papayya.resources.deployments import Deployments
from papayya.resources.items import Items
from papayya.resources.projects import Projects
from papayya.resources.runs import Runs
from papayya.resources.schedules import Schedules
from papayya.resources.secrets import Secrets
from papayya.resources.triage import Triage
from papayya.resources.usage import Usage
from papayya.resources.webhooks import Webhooks


# Distinguishes "item(partition_key=...) not passed" (unattributed) from an
# explicit value — including an explicit None, which records the run
# unattributed.
_PARTITION_KEY_UNSET = object()


def _resolve_durable_credentials(
    explicit_key: str | None, explicit_url: str | None
) -> tuple[str | None, str]:
    """Resolve the API key AND the control plane it is sent to, as a pair.

    They are only meaningful together: a key is minted by one control plane
    and means nothing to another, and `papayya login` writes both into the
    same env block for exactly that reason. Resolving them from independent
    ladders is what pointed a docker-compose key at api.getpapayya.com — the
    key was read out of the saved config while the URL fell through to the
    production default.

    So the SOURCE decides both. A key named explicitly or via
    PAPAYYA_API_KEY takes the explicit/env URL (or the default) — never a
    saved config's URL, which may describe a different control plane. Only
    when the key itself comes from the config does the config's URL apply.

    Permissive like the key resolution it replaces: returns ``(None, url)``
    rather than raising when no key is found, because the caller falls back
    to SQLiteStore in that case.
    """
    env_url = explicit_url or os.environ.get("PAPAYYA_BASE_URL")

    key = explicit_key or os.environ.get("PAPAYYA_API_KEY")
    if key:
        return key, env_url or DEFAULT_BASE_URL

    cfg = env_config(load_cli_config())
    return cfg.get("api_key"), env_url or cfg.get("base_url") or DEFAULT_BASE_URL


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

    def _auto_store(self) -> Any:
        """Auto-select a CheckpointStore for durable runs.

        Selection order:
          1. Runtime store — when the hosted worker set
             ``PAPAYYA_RUNTIME_STORE_BASE`` + ``PAPAYYA_PLATFORM_WORKER_KEY``.
             Customer @agent code running in-process on the worker pool writes
             its checkpoints through the platform-authed ``/v1/runtime`` lane
             (Plan 37 Unit 1), so hosted runs are visible in the dashboard —
             including the per-step token/cost trace — from the first run.
          2. CloudStore — when a real ``cpk_`` project key is resolvable
             (non-worker in-process clients).
          3. SQLiteStore — ONLY when an explicit ``PAPAYYA_LOCAL_DB_PATH`` is
             set. This is internal plumbing (the test harness + the retired
             local worker path), NOT a product surface: local dev is
             deactivated (Plan 37) — the local dashboard, the keyless demo,
             and `map`/`iter` are all removed — so nothing user-facing sets this.

        With none of the above resolvable the keyless local default is GONE
        (Plan 37): rather than silently writing to ``.papayya/local.db`` — the
        "second app" whose drift was the whole reason to collapse — this
        raises so work can only land on a control-plane (cloud or the
        docker-compose stack).
        """
        from papayya.durable.cloud_store import CloudStore, CloudStoreConfig
        from papayya.durable.sqlite_store import SQLiteStore

        runtime_base = os.environ.get("PAPAYYA_RUNTIME_STORE_BASE")
        runtime_key = os.environ.get("PAPAYYA_PLATFORM_WORKER_KEY")
        if runtime_base and runtime_key:
            from papayya.durable.cloud_store import make_runtime_store

            return make_runtime_store(runtime_base, runtime_key)

        resolved_key, resolved_url = _resolve_durable_credentials(
            self._api_key, self._base_url
        )
        if resolved_key:
            return CloudStore(
                CloudStoreConfig(api_key=resolved_key, base_url=resolved_url)
            )
        db_path = os.environ.get("PAPAYYA_LOCAL_DB_PATH")
        if db_path:
            return SQLiteStore(db_path)
        raise PapayyaAPIError(
            401,
            "No API key resolvable and local dev is deactivated. Set "
            "PAPAYYA_API_KEY (cloud) or run against the docker-compose stack.",
        )

    # --- durable runtime ---------------------------------------------- #

    def item(
        self,
        agent: str,
        *,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        item_id: str | None = None,
        parent_run_id: str | None = None,
        store: Any | None = None,
        partition_key: Any = _PARTITION_KEY_UNSET,
    ) -> Any:
        """Create a new durable per-item record, returned as an :class:`Item`.

        Plan 34 rename of ``papayya().run(...)`` — the old name is kept as
        a deprecated alias. A direct call like this is an implicit
        run-of-one: the local ledger wraps the item in its own run row.
        ``run_id=`` (the item's surrogate id) and ``parent_run_id=`` keep
        their pre-consolidation kwarg names for compatibility; ``item_id=``
        stays reserved for CUSTOMER identity (e.g. ``"co_007"``).

        Attribution is code-first: pass ``partition_key=`` to persist an
        indexed partition value (often a tenant / organization id) on every
        row written under this run. A non-empty string is used as-is; an
        explicit ``None`` (or omitting it) records the run unattributed.
        ``papayya.map(..., partition_key=…)`` is the per-item form for real
        per-partition attribution.

        When called from inside an ``@agent`` body, the outer run's id
        is picked up automatically (sub-runs lineage / Layer 3 #7). Pass
        ``parent_run_id=`` to override the auto-detected value or to
        link a run that was spawned out-of-band.
        """
        # Layer 3 #9: the documented pattern is now
        # ``def process_note(run, note): ...`` with ``run`` injected by
        # the @agent wrapper. Customers on the legacy pattern call this
        # method themselves from inside the fn body — that's the line
        # they need to delete, so we warn at the call site (the wrapper
        # sets a contextvar before invoking the legacy fn).
        # Tell the enclosing ambient isolate (if any) that this body opened a
        # run for itself. Without this the empty-agent warning cannot tell a
        # legacy body — which did record a run, just not via the isolate —
        # from one that recorded nothing at all.
        try:
            from papayya.iterators import note_manual_run

            note_manual_run()
        except Exception:  # pragma: no cover - never block a real run
            pass

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

        from papayya.agent import consume_bootstrap_run_id, get_active_run_id
        from papayya.durable._replay import consume_replay_hydration
        from papayya.durable.run import PapayyaRun
        from papayya.durable.types import DurableRunConfig

        partition_key_value: str | None
        if partition_key is not _PARTITION_KEY_UNSET:
            if partition_key is not None and (
                not isinstance(partition_key, str) or partition_key == ""
            ):
                raise ValueError(
                    f"partition_key must be a non-empty string or None; "
                    f"got {partition_key!r}"
                )
            partition_key_value = partition_key
        else:
            partition_key_value = None

        resolved_store = store or self._store_override or self._auto_store()

        # Sub-runs lineage (Layer 3 #7 Phase 2): explicit kwarg wins, else
        # auto-detect from the @agent wrapper's contextvar. Top-level
        # calls outside any @agent body leave it None.
        resolved_parent_run_id = (
            parent_run_id if parent_run_id is not None else get_active_run_id()
        )

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
            # Replay preserves lineage as it was — don't re-derive
            # parent_run_id from the current invocation context.
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

        # v1→v2 cutover: when a hosted worker injected the lease's run_id
        # (one-shot), adopt it so this run's checkpoints link to the
        # durable_run the submission pre-created. An explicit run_id=
        # still wins; outside a worker (local dev) this is None and the
        # run mints its own id as before.
        effective_run_id = run_id if run_id is not None else consume_bootstrap_run_id()

        return PapayyaRun(
            DurableRunConfig(
                agent=agent,
                run_id=effective_run_id,
                metadata=metadata,
                item_id=item_id,
                store=resolved_store,
                partition_key=partition_key_value,
                parent_run_id=resolved_parent_run_id,
            )
        )

    # Deprecated pre-Plan-34 alias: "run" now names the whole invocation,
    # not the per-item record this returns. Silent for one release —
    # internal callers and existing user code keep working unchanged.
    run = item

    # --- resource namespaces ------------------------------------------ #

    @cached_property
    def runs(self) -> Runs:
        """Invocation resource (BREAKING shift in 0.3.0 — was per-item;
        per-item access moved to :attr:`items`)."""
        return Runs(self._api_client())

    @cached_property
    def items(self) -> Items:
        """Per-item resource (the pre-0.3.0 ``runs`` surface, renamed)."""
        return Items(self._api_client())

    @property
    def batches(self) -> Runs:
        """Deprecated alias: forwards to :attr:`runs` (invocations)."""
        return self.runs

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

    @cached_property
    def triage(self) -> Triage:
        return Triage(self._api_client())

    # --- lifecycle ---------------------------------------------------- #

    def close(self) -> None:
        """Close the underlying HTTP connection if one was opened."""
        if self._api is not None:
            self._api.close()

    def __enter__(self) -> Papayya:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
