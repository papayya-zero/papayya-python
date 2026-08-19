"""Worker — long-lived process that pulls items from a dispatcher.

The worker process:

1. Imports the customer agent module *once* on boot (this triggers
   ``@agent`` decorator registration).
2. Loops: long-polls the dispatcher for the next leased item, looks
   up the registered ``@agent`` function by name, calls it with the
   item's submitted **input** (see :attr:`Lease.agent_argument` — it
   was the ``item_id`` until plan 43 B2a), reports completion (or
   failure).
3. Exits cleanly on SIGTERM / SIGINT.

The dispatcher protocol is intentionally minimal for Phase 1:

  GET  /lease?worker_id=X     -> 200 JSON {lease_id, agent, item_id} or 204
  POST /complete              -> 200 JSON {}, body {lease_id, status, error?}
  POST /release               -> 200 JSON {outcome}, body {lease_id, reason}

/complete and /release are the two ways a lease ends, and the difference is
the point: /complete is terminal, /release hands the item back to the queue
parked so an operator resume can re-drive it.

Future phases add: heartbeats, lease TTL, code-distribution version
negotiation, hot-reload signaling. None of that exists yet — Phase 1
prototype is the simplest thing that proves workers can serve a batch
with one module import.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request


log = logging.getLogger("papayya.runtime")

# Mirrors papayya.runtime.heartbeat.IDLE. Duplicated rather than imported so
# the worker never pulls the child module into its own process — the child
# must be startable even when this one is in a bad state.
_HEARTBEAT_IDLE = "-"


# Default heartbeat cadence. Must be well below the dispatcher's lease
# TTL (default 30s) so a single missed heartbeat doesn't expire the
# lease. 5s gives roughly 6× headroom.
_DEFAULT_HEARTBEAT_INTERVAL = 5.0


# Default SIGTERM drain budget. Aligns with Kubernetes' default
# `terminationGracePeriodSeconds` (30s) so a worker pod gets to finish
# the in-flight item before kubelet escalates to SIGKILL. ADR-0002 #12.
_DEFAULT_DRAIN_TIMEOUT_SECONDS = 30.0


# Per-request budget for every dispatcher call (lease, complete, release,
# heartbeat, run PATCH). Was a hardcoded 2.0 at each call site, which is
# not a response budget a busy control plane can be held to — plan 48 W1
# killed the pool by pausing the API for eight seconds, and a laptop
# sleeping does the same thing. 10s is comfortably under the 30s lease TTL
# so a single stalled call cannot outlive the lease it is about.
_DEFAULT_HTTP_TIMEOUT_SECONDS = 10.0


# Bundle downloads get their own, longer budget: this one transfers a
# tarball rather than a small JSON body, so it is sized for bytes on the
# wire, not for control-plane latency. Floor, not override — raising
# --http-timeout-seconds above it raises this too.
_DEFAULT_BUNDLE_TIMEOUT_SECONDS = 30.0


# ADR-0001 § 4 designed-but-unshipped recycling triggers. ADR-0002 #6
# closes the loop. Defaults are guesses — Phase 1 prototype must surface
# real memory growth and item-throughput numbers; Phase 2 tunes from
# data. 0 or negative on either disables that trigger.
_DEFAULT_MAX_ITEMS_BEFORE_RECYCLE = 100
_DEFAULT_MAX_RSS_PERCENT_BEFORE_RECYCLE = 80.0


def _default_rss_percent_provider() -> float:
    """Return this process's RSS as a percentage of system/container memory.

    Under containerized runtimes (Fargate, k8s) the cgroup memory
    limit normally surfaces as MemTotal, so this percentage tracks the
    container's allotment rather than the host. Swap to a cgroup-aware
    reader (`/sys/fs/cgroup/memory.max`) if production proves that wrong.

    Pulled into a module-level function (rather than a Worker method) so
    tests can swap it via the ``rss_percent_provider`` constructor kwarg.
    """
    import psutil

    return float(psutil.Process().memory_percent())


@dataclass
class Lease:
    """One unit of work assigned to this worker by the dispatcher."""
    lease_id: str
    agent: str
    item_id: str
    payload: dict[str, Any] | None = None
    # Set by the hosted dispatcher (control-pane RuntimeLease) when the
    # lease was enqueued against a specific deployed bundle. None when the
    # local LocalDispatcher served the lease — local dev loads the agent
    # module from --agent-module FILE and is version-unaware. ADR-0003 § 1.
    agent_version: str | None = None
    # account_id + project_id ride along on the lease wire so a v2
    # platform-actor worker (ADR-0004 § 2/3) can address the bundle
    # endpoint unambiguously: slug uniqueness is per-project, not per-
    # account, so the bundle handler needs both query params under
    # PrincipalPlatform auth. Project-scoped workers get them too —
    # harmless because the principal already carries scope. Both stay
    # ``None`` for LocalDispatcher leases.
    account_id: str | None = None
    project_id: str | None = None

    @property
    def agent_argument(self) -> Any:
        """What the customer's ``@agent`` function is called with.

        THE SUBMITTED INPUT, when the lease carries one — plan 43 B2a. The
        hosted dispatcher has published ``payload.input`` since the v2 cutover
        and this worker parsed it into :attr:`payload` and then ignored it,
        calling ``fn(item_id)`` instead. ``item_id`` on that path is
        ``dispatch.InputToItemID(input)``: a JSON *string* input arrives
        unquoted, and **any other shape arrives as its compact JSON text**. So
        a customer submitting ``{"order_id": …}`` had their function handed a
        string to re-parse, while the parsed object sat in the same lease.

        The docs never described that. quickstart.mdx has
        ``def triage(ticket: dict)`` with ``item_id=`` as *"your identity for
        each item"* — this restores the contract the docs already promise, and
        it is a change that can only be made once, before hosted deploys.

        KEY PRESENCE, NOT TRUTHINESS. ``input`` is legitimately ``null``,
        ``""``, ``0``, ``false``, ``{}`` or ``[]``, all of which are falsy in
        Python; ``payload.get("input") or item_id`` would send the *wrong*
        argument for every one of them, and the customer's function would
        receive a stringified id where it expected an empty list.

        The fallback exists for leases minted before the submitted input was
        persisted — LocalDispatcher leases, and any hosted path that sets no
        input. It is not a permanent dual contract.
        """
        if self.payload is not None and "input" in self.payload:
            return self.payload["input"]
        return self.item_id


class _InvocationOutcome(NamedTuple):
    """What one invocation resolved to, for the caller's run-status flip.

    Was a bare ``(run_status, output)`` tuple. The error and its category ride
    here because plan 48 R3 found the worker already computing the exception
    text — ``f"{type(exc).__name__}: {exc}"`` — reporting it to the LEASE table
    and then throwing it away, so ``durable_runs`` had nothing and a crashed
    record's entire diagnostic was the word "failed".

    ``status`` is None for a pause/credit signal, whose run status is
    authoritative server-side and must not be clobbered.
    """

    status: str | None
    output: Any = None
    error: str | None = None
    error_category: str | None = None


@dataclass
class _LoadedBundle:
    """Tracking entry for a bundle the worker has already imported.

    ADR-0003 § Worker #4 makes the multi-version registry keyed by
    ``(agent_name, agent_version)`` so slice 3 holds multiple versions
    resident. Storing the bundle path + sys.modules name lets a future
    eviction path (Slice E) clean up without rediscovering them.

    ``dep_hash`` carries forward from the bundle cache so
    ``_ensure_loaded`` can detect dep-graph changes without re-reading
    the on-disk sidecar on every miss (ADR-0003 § Worker #6).

    ``registration`` is the ``@agent`` registration this bundle produced
    at import time, snapshotted here rather than looked up at dispatch.
    The module-level ``papayya.agent._registry`` is keyed
    ``(name, version)`` with no account component — customer code cannot
    know its account — so on a shared worker two accounts deploying the
    same slug at the same version overwrite each other there. Holding
    the registration per residency makes that collision unreachable for
    hosted dispatch (plan 47 S1). ``None`` for the local /
    ``LocalDispatcher`` path, which falls back to ``get_agent``.
    """
    agent_name: str
    agent_version: str
    bundle_path: str
    module_name: str
    dep_hash: str | None = None
    registration: object | None = None


def _lookup_registration(agent: str, version: str | None):
    """``papayya.agent.get_agent`` behind a late import.

    ``papayya.agent`` is imported lazily throughout this module to keep
    worker boot free of the decorator machinery; this wrapper keeps that
    property while letting ``_ensure_loaded`` snapshot a registration.
    """
    from papayya.agent import get_agent

    return get_agent(agent, version)


def _residency_token(account_scope: str, agent_version: str) -> str:
    """Identifier-safe token naming one (account, version) residency.

    Used as the ``sys.modules`` name suffix and as the bundle loader's
    registration key, so both are partitioned per account rather than
    per version alone (plan 47 S1).

    Non-alphanumerics collapse to ``_`` because the token lands in a
    module name. Account ids are UUIDs, which differ in their hex digits
    and not only in punctuation, so the collapse cannot alias two
    accounts onto one token.
    """
    safe = "".join(c if c.isalnum() else "_" for c in account_scope)
    return f"{safe}__v{agent_version}"


class _VersionNotFound(Exception):
    """Bundle endpoint returned 404 for the lease's agent_version.

    Worker maps this to ``_report_complete(status="failed",
    error_category="version_not_found")`` so the dispatcher's
    idempotent-complete + lease TTL can clean up. Distinct exception
    type so generic ``Exception`` handlers in ``_handle_lease`` route
    it through the categorised path rather than a stringified
    ``RuntimeError`` message.
    """


class _RecyclePending(Exception):
    """A new bundle's dep-graph differs from the resident version's.

    ``importlib.reload()`` is unreliable for transitively-imported
    modules + new C extensions, so when a deploy ships a new
    ``requirements.txt`` (or ``pyproject.toml``) the worker can't
    safely import the new version in this process — it must recycle.

    The triggering lease is failed with
    ``error_category="recycle_pending"``; the dispatcher's lease TTL
    re-dispatches it to a fresh worker that has no resident versions.
    The current worker drains its main loop cleanly (``_running =
    False``) and exits; the orchestrator's restart policy brings up a
    new worker.

    ADR-0003 § Worker #6, extends ADR-0002 #6.
    """


class _AgentTimeout(BaseException):
    """Raised by the SIGALRM handler when an agent fn exceeds its
    ``max_duration_seconds`` budget.

    Subclasses BaseException (not Exception) so customer ``except
    Exception`` blocks inside the agent fn don't accidentally swallow
    the timeout. The worker handles it explicitly.
    """


def _on_agent_alarm(_signum: int, _frame: Any) -> None:
    raise _AgentTimeout()


class _PollOutcome:
    """String constants for the three states `_poll_lease` can return.

    A small string-based discriminant rather than an Enum keeps the
    main loop's branching trivially readable in tracebacks.
    """
    LEASED = "leased"
    IDLE = "idle"
    UNREACHABLE = "unreachable"


class _ReconnectBackoff:
    """Exponential backoff for dispatcher unreachability.

    Stateful by design — the worker holds one instance across the life
    of the run loop. Each ``on_failure`` advances the wait (doubles up
    to ``max_seconds``), each ``on_success`` snaps back to zero so the
    *next* poll after recovery has zero added latency. ADR-0002 #15.
    """

    def __init__(
        self,
        *,
        initial_seconds: float = 0.1,
        max_seconds: float = 2.0,
    ) -> None:
        self._initial = initial_seconds
        self._max = max_seconds
        self._current = 0.0

    def on_failure(self) -> float:
        if self._current == 0.0:
            self._current = self._initial
        else:
            self._current = min(self._current * 2.0, self._max)
        return self._current

    def on_success(self) -> None:
        self._current = 0.0

    @property
    def current(self) -> float:
        return self._current


class Worker:
    """Long-running worker. Polls a dispatcher, runs ``@agent`` functions.

    Args:
        dispatcher_url: Base URL of the dispatcher (e.g. ``http://127.0.0.1:8765``).
        store_path: Path to the SQLite file the customer's ``papayya()``
            client should write through. Set as ``PAPAYYA_LOCAL_DB_PATH``
            so customer code transparently picks it up.
        agent_module_path: Path to the customer's ``.py`` file containing
            ``@agent``-decorated function(s). Imported once on construction.
        worker_id: Stable id for this worker (defaults to a random short id).
        poll_idle_seconds: Sleep between empty-lease polls. Keep small for
            responsive iteration loop; tune in Phase 2 from real load data.
    """

    _idle_log_interval = 30.0

    def __init__(
        self,
        *,
        dispatcher_url: str,
        store_path: str,
        agent_module_path: Optional[str] = None,
        worker_id: Optional[str] = None,
        poll_idle_seconds: float = 0.05,
        heartbeat_interval_seconds: float = _DEFAULT_HEARTBEAT_INTERVAL,
        drain_timeout_seconds: float = _DEFAULT_DRAIN_TIMEOUT_SECONDS,
        http_timeout_seconds: float = _DEFAULT_HTTP_TIMEOUT_SECONDS,
        api_key: Optional[str] = None,
        bundle_url_base: Optional[str] = None,
        max_items_before_recycle: int = _DEFAULT_MAX_ITEMS_BEFORE_RECYCLE,
        max_rss_percent_before_recycle: float = _DEFAULT_MAX_RSS_PERCENT_BEFORE_RECYCLE,
        rss_percent_provider: Optional[Callable[[], float]] = None,
    ) -> None:
        self.dispatcher_url = dispatcher_url.rstrip("/")
        self.store_path = store_path
        self.worker_id = worker_id or f"w-{uuid.uuid4().hex[:8]}"
        self.poll_idle_seconds = poll_idle_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.drain_timeout_seconds = drain_timeout_seconds
        self.http_timeout_seconds = http_timeout_seconds
        self._bundle_timeout_seconds = max(
            _DEFAULT_BUNDLE_TIMEOUT_SECONDS, http_timeout_seconds
        )
        # Bootstrap mode: hosted workers boot without --agent-module and
        # load every bundle on demand from lease.agent_version. A lease
        # with agent_version=None (LocalDispatcher) in this mode is a
        # misconfiguration — the lease handler emits error_category=
        # "no_agent_module" so the failure is loud. ADR-0003 § Worker #5.
        self._bootstrap_mode = agent_module_path is None
        # v1→v2 cutover: durable-run base for the queued→running flip.
        # The hosted dispatcher_url carries the ``/v1/runtime`` prefix, so
        # the sibling durable surface is ``/v1/durable``. Local-dev
        # (LocalDispatcher, no ``/runtime`` suffix) leaves this None — and
        # local leases carry no run_id, so the flip is skipped entirely.
        if self.dispatcher_url.endswith("/runtime"):
            self._durable_api_base = self.dispatcher_url[: -len("/runtime")] + "/durable"
        else:
            self._durable_api_base = None
        # Sent as X-Api-Key on lease/complete/heartbeat. Matches the
        # dispatcher's API-key middleware (control-pane auth.go) which
        # requires a project-scoped key — JWT Bearer tokens are rejected
        # for runtime endpoints. None = no header (LocalDispatcher accepts).
        self._api_key = api_key
        # Base URL of the bundle download endpoint (ADR-0003 § 7).
        # Hosted: ``https://api.papayya.com/v1/runtime/bundles``. Local
        # dev: ``LocalDispatcher`` doesn't host this route, so the field
        # is unused — only consulted when a lease arrives carrying
        # ``agent_version`` (which LocalDispatcher never sets). Tests
        # that exercise the fetch path point this at their own fake
        # bundle server.
        #
        # When derived from ``dispatcher_url`` we append ``/bundles``
        # only — the dispatcher_url already carries the ``/v1/runtime``
        # path prefix (see ``/lease``, ``/complete``, ``/heartbeat``
        # builders below at :727, :767, :1218 which all use the same
        # base). Appending ``/v1/runtime/bundles`` here would double the
        # path and 404 in production.
        self._bundle_url_base = (bundle_url_base or f"{self.dispatcher_url}/bundles").rstrip("/")
        # Per-(agent_name, agent_version) cache of bundle entries the
        # worker has already loaded into the registry. Slice 2 guarantees
        # at most one resident entry per (name, version) tuple — multi-
        # version dispatch is slice 3. Hot-path lookup avoids re-reading
        # the on-disk cache and re-importing the module on every lease.
        # Keyed (account_scope, agent_slug, agent_version) — the account
        # leads because a shared platform worker serves many accounts and
        # slug+version alone collides across them (plan 47 S1).
        self._loaded_versions: dict[tuple[str, str, str], "_LoadedBundle"] = {}
        self._running = True
        now = time.monotonic()
        self._last_activity_at = now
        self._last_idle_log_at = now

        # In-flight lease tracking for heartbeats. Set to the current
        # Lease just before the agent fn runs and cleared in the finally
        # block. Heartbeat thread reads it under _hb_lock and POSTs
        # to /heartbeat while it's set.
        self._in_flight_lease: Optional[Lease] = None
        self._hb_lock = threading.Lock()
        self._hb_stop = threading.Event()
        # Started at the end of __init__ via _start_heartbeat() below.

        # Backoff state for dispatcher unreachability. Without this the
        # poll loop hammers a dead/recovering dispatcher at the
        # poll_idle_seconds rate (~20 retries/sec by default).
        self._reconnect_backoff = _ReconnectBackoff()

        # Drain coordination (ADR-0002 #12). Watchdog thread is started
        # lazily on first SIGTERM. Pre-spawning + Event.wait() would be
        # the cleaner pattern, but a long-blocked daemon thread inside
        # the worker subprocess interferes with cross-process SQLite WAL
        # visibility under load (commits land but other processes read
        # stale state). Lazy-start sidesteps that completely.
        self._drain_started: bool = False
        self._drain_lock = threading.Lock()
        self._drain_thread: Optional[threading.Thread] = None

        # Recycle-pending flag (ADR-0003 § Worker #6, ADR-0002 #6).
        # Set when:
        #   1. ``_ensure_loaded`` detects a different ``requirements.txt``
        #      hash between the resident version and a newly-fetched
        #      version of the same agent slug (the triggering lease is
        #      failed with ``error_category="recycle_pending"``).
        #   2. ``_check_recycle_thresholds`` observes
        #      ``items_processed`` or RSS% past their configured
        #      ceilings (between items, after the current item finished
        #      normally — no lease failure needed).
        # Either path also sets ``self._running = False`` so the main
        # loop exits cleanly and the orchestrator brings up a fresh
        # process.
        self._recycle_pending: bool = False

        # ADR-0002 #6 / ADR-0001 § 4 recycling counters.
        self._items_processed: int = 0
        self._max_items_before_recycle = max_items_before_recycle
        self._max_rss_percent_before_recycle = max_rss_percent_before_recycle
        # Module-level default lazy-imports psutil so tests that inject
        # their own provider don't pay for the dep — and so a busted
        # psutil install doesn't break worker boot when the operator
        # has the trigger disabled (max=0).
        self._rss_percent_provider: Callable[[], float] = (
            rss_percent_provider or _default_rss_percent_provider
        )

        # Point the customer's in-process papayya() client at the right
        # CheckpointStore BEFORE importing the agent module (customer code may
        # call papayya() at module top-level). Two modes:
        #
        #   Hosted worker pool (dispatcher_url carries the /v1/runtime prefix,
        #   so _durable_api_base is set) → the platform-authed runtime lane.
        #   Every checkpoint — including the per-step token/cost trace — is
        #   POSTed to /v1/runtime/runs/{id}/checkpoints with the shared
        #   platform worker key, so hosted runs are visible in the dashboard
        #   from the first run (Plan 37 Unit 1). The control-plane resolves
        #   the tenant off the pre-created run row.
        #
        #   Local prototype (LocalDispatcher, no /runtime prefix) → worker-
        #   local SQLite, kept until Plan 37 Unit 4 removes SQLite.
        #
        # A stray PAPAYYA_API_KEY from the parent shell is popped either way so
        # it can't shadow the selected store.
        os.environ.pop("PAPAYYA_API_KEY", None)
        if self._durable_api_base is not None:
            # make_runtime_store() wants the control-plane ROOT (it appends
            # the /v1/runtime/runs path itself), so strip the WHOLE
            # /v1/runtime prefix off dispatcher_url — NOT just /runtime.
            # Leaving a stray /v1 makes httpx concatenate a doubled
            # /v1/v1/runtime/... path that 404s every checkpoint write
            # (Plan 37 Unit 1 regression). Fall back to stripping a bare
            # /runtime for a non-versioned dispatcher prefix.
            runtime_store_base = self.dispatcher_url
            for suffix in ("/v1/runtime", "/runtime"):
                if runtime_store_base.endswith(suffix):
                    runtime_store_base = runtime_store_base[: -len(suffix)]
                    break
            os.environ["PAPAYYA_RUNTIME_STORE_BASE"] = runtime_store_base
            if self._api_key:
                os.environ["PAPAYYA_PLATFORM_WORKER_KEY"] = self._api_key
            os.environ.pop("PAPAYYA_LOCAL_DB_PATH", None)
        else:
            os.environ["PAPAYYA_LOCAL_DB_PATH"] = store_path

        if agent_module_path is not None:
            self._import_agent_module(agent_module_path)
        else:
            log.info(
                "starting in bootstrap mode (no agent module pre-loaded; "
                "first lease's agent_version triggers first import)"
            )

        # Heartbeat starts after module import so any import error fails
        # fast without leaving a heartbeat behind.
        self._start_heartbeat()

    # --- agent module loading ------------------------------------------ #

    def _import_agent_module(self, path: str) -> None:
        """Import the customer's agent file by absolute path.

        This is the *one* import that should happen for the lifetime of
        the worker. The acceptance test verifies this via an external
        counter — see tests/integration/test_worker_acceptance.py.
        """
        p = Path(path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"agent module not found: {p}")

        spec = importlib.util.spec_from_file_location(f"_papayya_user_{p.stem}", p)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot build module spec for: {p}")

        module = importlib.util.module_from_spec(spec)
        # Insert into sys.modules so the @agent decorator's module-level
        # registry write side effect persists across this loader's lifetime.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        log.info("imported agent module: %s", p)

    # --- versioned bundle loading (ADR-0003 § Worker #2/#3) ----------- #

    def _ensure_loaded(self, lease: "Lease") -> None:
        """Make sure the lease's ``(agent, agent_version)`` is importable.

        Called from ``_handle_lease`` *before* ``get_agent(lease.agent)``.
        Three cases:

        1. ``lease.agent_version is None`` — local-dev / legacy. The
           file-loaded module from ``--agent-module`` already populated
           the registry; no-op.
        2. ``(account, agent, agent_version)`` already in
           ``self._loaded_versions`` — hot path. The earlier import
           already registered the ``@agent``; no-op.
        3. Cache miss. Fetch the tarball from the bundle endpoint,
           extract under the on-disk cache, build a ``ModuleSpec`` from
           the entrypoint, ``exec_module`` it (the ``@agent`` decorator
           re-registers under the agent's slug), record the bundle in
           ``self._loaded_versions``.

        Every residency key leads with the account. A shared platform
        worker leases across accounts, so ``(slug, version)`` alone
        aliases two customers' bundles onto one entry — the collision
        plan 47 S1 demonstrated, where one account's code ran against
        another's items. The same scope flows into the on-disk cache
        path, the ``sys.modules`` name, and the bundle loader's root
        registration, so no layer can re-introduce it.

        Raises ``_VersionNotFound`` on 404 from the bundle endpoint;
        ``_handle_lease`` maps that to a categorised failure. Network errors
        bubble as ``OSError`` and everything else (a bad import in the
        customer's entrypoint, a corrupt tarball) bubbles as ``Exception``:
        ``_handle_lease`` catches both. The lease TTL is only a safety net if
        the worker is alive to be caught by it, which is what plan 48 W1 was
        about — before that these escaped ``run()`` and exited the process.
        """
        if lease.agent_version is None:
            return

        key = self._residency_key(lease)
        if key in self._loaded_versions:
            return

        # Late import to keep the hot path (no agent_version) free of the
        # bundle-cache module's tarfile/fcntl pull-in cost. Worker boot
        # is unaffected; only the first hosted lease pays it.
        from papayya.runtime import _bundle_cache

        version_int = self._parse_version(lease.agent_version)
        bundle = _bundle_cache.ensure_bundle(
            account_id=self._account_scope(lease),
            agent_slug=lease.agent,
            version=version_int,
            fetch=lambda: self._fetch_bundle(
                lease.agent,
                version_int,
                account_id=lease.account_id,
                project_id=lease.project_id,
            ),
        )

        # ADR-0003 § Worker #6 — if a *different* version of this
        # agent slug is already resident with a *different* dep-hash,
        # the new version's pip deps can't be loaded safely into this
        # process. Mark recycle pending and bail; the lease will be
        # failed with ``error_category="recycle_pending"`` and the
        # main loop will exit cleanly so the orchestrator brings up
        # a fresh worker. ``None`` on either side (no manifest in the
        # bundle) means we can't tell deps apart, so we proceed —
        # explicit absence is treated as "no dep change."
        # Same account, same slug, different version. Scoping the search
        # to the account matters in both directions: another account's
        # resident v1 is not a *version* conflict with this one, so an
        # unscoped search would fire a spurious recycle every time two
        # customers happened to share a slug.
        scope = self._account_scope(lease)
        prior = next(
            (
                lb
                for (acct, slug, _v), lb in self._loaded_versions.items()
                if acct == scope and slug == lease.agent and _v != lease.agent_version
            ),
            None,
        )
        if (
            prior is not None
            and prior.dep_hash is not None
            and bundle.dep_hash is not None
            and prior.dep_hash != bundle.dep_hash
        ):
            self._recycle_pending = True
            self._running = False
            log.warning(
                "scheduling recycle: agent=%s prior=v%s new=v%s dep-hash differs (%s != %s)",
                lease.agent,
                prior.agent_version,
                lease.agent_version,
                prior.dep_hash[:12],
                bundle.dep_hash[:12],
            )
            raise _RecyclePending(
                f"agent {lease.agent} dep-hash differs between v{prior.agent_version} "
                f"and v{lease.agent_version}; recycling worker for fresh pip env"
            )

        module_name = self._import_bundle_module(
            bundle_path=Path(bundle.path),
            entrypoint=bundle.entrypoint or "agent.py",
            agent_name=lease.agent,
            agent_version=lease.agent_version,
            account_scope=scope,
        )
        # Snapshot the registration this import just produced. The global
        # registry is keyed (name, version) with no account, so a later
        # import by another account would overwrite it — dispatching from
        # the snapshot keeps each residency pointed at its own code.
        self._loaded_versions[key] = _LoadedBundle(
            agent_name=lease.agent,
            agent_version=lease.agent_version,
            bundle_path=str(bundle.path),
            module_name=module_name,
            dep_hash=bundle.dep_hash,
            registration=_lookup_registration(lease.agent, lease.agent_version),
        )

    @staticmethod
    def _account_scope(lease: "Lease") -> str:
        """Account partition for every residency key derived from ``lease``.

        ``Lease.account_id`` is populated on the hosted wire and absent
        for ``LocalDispatcher``. Falling back to the reserved
        ``LOCAL_SCOPE`` keeps single-tenant local dev on its own
        partition, which cannot alias a real (UUID) account id.
        """
        from papayya.runtime._bundle_cache import LOCAL_SCOPE

        return lease.account_id or LOCAL_SCOPE

    def _residency_key(self, lease: "Lease") -> tuple[str, str, str]:
        """``(account_scope, agent_slug, agent_version)`` for ``_loaded_versions``."""
        return (
            self._account_scope(lease),
            lease.agent,
            lease.agent_version or "",
        )

    def _activation_token(self, lease: "Lease") -> str | None:
        """Bundle-loader key for this lease, or ``None`` for local dev.

        Mirrors the token ``_import_bundle_module`` registered under, so
        function-body imports during dispatch resolve against the bundle
        root that was actually imported for this account and version.
        """
        if lease.agent_version is None:
            return None
        return _residency_token(self._account_scope(lease), lease.agent_version)

    @staticmethod
    def _parse_version(version: str) -> int:
        """Parse the wire ``agent_version`` into the int the bundle endpoint expects.

        Accepts ``"3"`` and ``"v3"`` symmetrically with the control-pane
        handler, which strips a leading ``v`` before atoi.
        """
        cleaned = version.lstrip("v") if version.startswith("v") else version
        try:
            n = int(cleaned)
        except ValueError as exc:
            raise _VersionNotFound(
                f"agent_version {version!r} is not parseable as an integer"
            ) from exc
        if n < 1:
            raise _VersionNotFound(
                f"agent_version {version!r} must be a positive integer"
            )
        return n

    def _fetch_bundle(
        self,
        agent: str,
        version: int,
        account_id: str | None = None,
        project_id: str | None = None,
    ) -> Any:
        """HTTP GET the bundle endpoint and adapt to ``FetchedBundle``.

        Stored as a closure so ``ensure_bundle``'s lazy-fetch contract
        works: zero-arg callable, only invoked on cache miss. Response
        headers — entrypoint, account_id, agent_id, deployment_id, ETag
        — ride along on the ``FetchedBundle`` so ``ensure_bundle`` can
        annotate the resulting cache entry without a second round-trip.

        ``account_id`` + ``project_id`` are passed through as query
        params under platform-actor auth (ADR-0004 § 2): the bundle
        handler at control-pane needs both to disambiguate slug
        uniqueness, since slug is unique per-project not per-account.
        Project-scoped (PrincipalAPIKey) callers can omit them — the
        principal already carries scope and the handler ignores extra
        query params.
        """
        from papayya.runtime._bundle_cache import FetchedBundle

        url = f"{self._bundle_url_base}?agent={agent}&version={version}"
        if account_id:
            url += f"&account={account_id}"
        if project_id:
            url += f"&project={project_id}"
        req = urllib_request.Request(url, headers=self._auth_headers())
        try:
            resp = urllib_request.urlopen(req, timeout=self._bundle_timeout_seconds)
        except urllib_error.HTTPError as exc:
            if exc.code == 404:
                raise _VersionNotFound(
                    f"bundle endpoint returned 404 for agent={agent} version={version}"
                ) from exc
            raise

        with resp:
            body = resp.read()
            account_id = resp.headers.get("X-Papayya-Account-Id")
            agent_id = resp.headers.get("X-Papayya-Agent-Id")
            entrypoint = resp.headers.get("X-Papayya-Entrypoint") or "agent.py"
            deployment_id = resp.headers.get("X-Papayya-Deployment-Id")
            etag = resp.headers.get("ETag")
            artifact_hash = etag.strip('"') if etag else None

        return FetchedBundle(
            tarball_bytes=body,
            entrypoint=entrypoint,
            artifact_hash=artifact_hash,
            account_id=account_id,
            agent_id=agent_id,
            deployment_id=deployment_id,
        )

    def _import_bundle_module(
        self,
        *,
        bundle_path: Path,
        entrypoint: str,
        agent_name: str,
        agent_version: str,
        account_scope: str,
    ) -> str:
        """exec_module the bundle's entrypoint; return the sys.modules key.

        The entrypoint is interpreted relative to ``bundle_path`` (the
        extracted tarball root). We use ``importlib.util`` to keep the
        loader path-aware, and we register sys.modules under a name
        suffixed with the agent_version so a future multi-version
        registry (slice 3) can keep both modules resident.

        Module identity collision is the slice-2 risk the hand-off
        flagged: two bundles sharing entrypoint stems will produce
        identical ``_papayya_user_<stem>`` keys without the version
        suffix. Slice 2 namespaces the suffix so the warning fires only
        when an actual collision happens.

        ``account_scope`` extends that namespace across tenants. The
        version suffix alone is not enough on a shared worker: two
        accounts deploying ``agent.py`` at v1 both produce
        ``_papayya_user_agent__v1`` and the second silently overwrites
        the first in ``sys.modules`` — the "already in sys.modules —
        overwriting" warning observed in plan 47 S1. The same scope keys
        the bundle loader's root registration, so a sibling
        ``from helpers import x`` resolves within the right account's
        bundle rather than whichever landed first.
        """
        entry_path = (bundle_path / entrypoint).resolve()
        if not entry_path.exists():
            raise _VersionNotFound(
                f"bundle for {agent_name}@{agent_version} missing entrypoint {entrypoint!r}"
            )

        # ADR-0003 § Worker #4 — register the bundle root with the
        # per-version MetaPathFinder instead of mutating ``sys.path``.
        # The finder, scoped via ``activate(version)`` below, intercepts
        # top-level imports made *during* the bundle's execution so two
        # versions' ``helpers.py`` siblings don't collide in
        # ``sys.modules``.
        from papayya.runtime import _bundle_loader

        residency = _residency_token(account_scope, agent_version)
        _bundle_loader.register_bundle(residency, bundle_path)

        module_name = f"_papayya_user_{entry_path.stem}__{residency}"
        if module_name in sys.modules:
            log.warning(
                "module name %s already in sys.modules — overwriting",
                module_name,
            )

        spec = importlib.util.spec_from_file_location(module_name, entry_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot build module spec for: {entry_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        # Pin ``PAPAYYA_AGENT_VERSION`` for the duration of the bundle's
        # exec so the customer's ``@agent`` decorator (which resolves
        # version via decorator-arg → env → git → "unknown") stamps the
        # registration with the lease's version. The env-cache is
        # cleared before AND after so the resolution actually re-runs
        # against this scoped value, and we don't poison subsequent
        # imports with a cached "v1" after we've moved on.
        # ADR-0003 § Worker #4.
        from papayya.agent import _clear_agent_version_cache

        prior_env = os.environ.get("PAPAYYA_AGENT_VERSION")
        os.environ["PAPAYYA_AGENT_VERSION"] = agent_version
        _clear_agent_version_cache()
        try:
            # ``activate`` wires top-level imports made during
            # exec_module (e.g., the entrypoint's ``from helpers
            # import ...``) to this version's bundle root, so two
            # bundles' sibling files don't collide in sys.modules.
            with _bundle_loader.activate(residency):
                spec.loader.exec_module(module)
        finally:
            if prior_env is None:
                os.environ.pop("PAPAYYA_AGENT_VERSION", None)
            else:
                os.environ["PAPAYYA_AGENT_VERSION"] = prior_env
            _clear_agent_version_cache()
        log.info(
            "loaded bundle %s@%s from %s (module=%s)",
            agent_name, agent_version, entry_path, module_name,
        )
        return module_name

    # --- main loop ----------------------------------------------------- #

    def run(self) -> None:
        """Block, pulling items from the dispatcher, until stopped."""
        # ``signal.signal`` raises ValueError when called off the main
        # thread (CPython implementation constraint). Production workers
        # always boot ``run()`` from the main thread of a subprocess so
        # this is the normal path. In-process tests that drive ``run()``
        # from a worker thread skip the registration — ``stop()`` and
        # ``_running=False`` are still the orderly exit path.
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, self._on_signal)
            signal.signal(signal.SIGINT, self._on_signal)

        try:
            while self._running:
                outcome, lease = self._poll_lease()
                if outcome == _PollOutcome.LEASED:
                    if self._reconnect_backoff.current > 0.0:
                        log.info("dispatcher reachable again; resuming normal poll cadence")
                    self._reconnect_backoff.on_success()
                    assert lease is not None
                    try:
                        self._handle_lease(lease)
                    except Exception:  # noqa: BLE001 — one item, not the pool
                        # Backstop for plan 48 W1's shape rather than its
                        # specific cause: every KNOWN failure is categorised
                        # and reported inside _handle_lease, so reaching here
                        # is a papayya bug. Log it whole (exc_info) and keep
                        # polling — the lease TTL re-dispatches this item, and
                        # one unanticipated exception taking down a pool that
                        # serves every account is a strictly worse outcome
                        # than one item going around again.
                        log.error(
                            "worker %s: unhandled error on lease %s item=%s; "
                            "item returns to the queue at lease expiry",
                            self.worker_id, lease.lease_id[:8], lease.item_id,
                            exc_info=True,
                        )
                    continue
                if outcome == _PollOutcome.IDLE:
                    if self._reconnect_backoff.current > 0.0:
                        log.info("dispatcher reachable again; resuming normal poll cadence")
                    self._reconnect_backoff.on_success()
                    self._maybe_log_idle()
                    time.sleep(self.poll_idle_seconds)
                    continue
                # UNREACHABLE — connection refused or timeout.
                was_healthy = self._reconnect_backoff.current == 0.0
                wait = self._reconnect_backoff.on_failure()
                if was_healthy:
                    # Surface the first failure at INFO so operators see it
                    # without DEBUG. Sustained outages stay quiet (the
                    # individual urlopen exception still logs at DEBUG).
                    log.warning(
                        "dispatcher unreachable; backing off (next poll in %.1fs)",
                        wait,
                    )
                time.sleep(wait)
        finally:
            # Stop the heartbeat thread cleanly so in-process callers
            # don't leak it across runs. The drain watchdog (if it was
            # spawned) checks _hb_stop and exits silently when the main
            # thread reaches this point — clean shutdown short-circuits
            # the deadline.
            self._hb_stop.set()
            self._stop_heartbeat()

    def _maybe_log_idle(self) -> None:
        now = time.monotonic()
        if (
            now - self._last_activity_at >= self._idle_log_interval
            and now - self._last_idle_log_at >= self._idle_log_interval
        ):
            log.info(
                "worker %s idle, no work for %ds",
                self.worker_id,
                int(now - self._last_activity_at),
            )
            self._last_idle_log_at = now

    def stop(self) -> None:
        self._running = False

    def _on_signal(self, signum: int, _frame: Any) -> None:
        # Idempotent: a second SIGTERM during drain is a no-op so the
        # operator's only escape is SIGKILL.
        with self._drain_lock:
            if self._drain_started:
                return
            self._drain_started = True
            self._running = False
            if self.drain_timeout_seconds > 0:
                # Lazy-spawn the watchdog. Pre-spawning + Event.wait()
                # would be cleaner, but a long-blocked daemon thread in
                # the worker subprocess interferes with cross-process
                # SQLite WAL visibility under load. Spawning from a
                # signal handler is safe here: the only other thread
                # that calls Thread.start() is __init__ (already done)
                # and the heartbeat thread (never spawns).
                self._drain_thread = threading.Thread(
                    target=self._drain_watchdog,
                    args=(time.monotonic(),),
                    daemon=True,
                    name=f"papayya-worker-drain-{self.worker_id}",
                )
                self._drain_thread.start()
        # Log outside the lock — signal handler interrupting another
        # log call could deadlock the logging lock if it ran inside it.
        log.info(
            "worker %s received signal %s; draining (deadline %.0fs, "
            "SIGKILL to force-exit)",
            self.worker_id, signum, self.drain_timeout_seconds,
        )

    # --- dispatcher I/O ------------------------------------------------ #

    def _auth_headers(self) -> dict[str, str]:
        if self._api_key is None:
            return {}
        return {"X-Api-Key": self._api_key}

    def _poll_lease(self) -> tuple[str, Lease | None]:
        """Poll the dispatcher for one lease.

        Returns a (outcome, lease) tuple. The outcome distinguishes
        "no work right now" (IDLE) from "couldn't reach the dispatcher"
        (UNREACHABLE) so the caller can apply different sleep policies —
        the latter triggers exponential backoff.

        Catches ``OSError``, not ``URLError``. urllib only converts to
        ``URLError`` around the *request* phase; a timeout waiting for the
        **response** comes out of ``h.getresponse()`` as a bare
        ``TimeoutError``, which is outside that conversion. Under the old
        handler it escaped here, escaped ``run()``, and exited the process —
        plan 48 W1 killed the whole pool by pausing the API for eight
        seconds. ``URLError`` and ``TimeoutError`` are both ``OSError``, so
        one clause covers the request phase and the response phase alike.
        """
        url = f"{self.dispatcher_url}/lease?worker_id={self.worker_id}"
        req = urllib_request.Request(url, headers=self._auth_headers())
        try:
            with urllib_request.urlopen(req, timeout=self.http_timeout_seconds) as resp:
                if resp.status == 204:
                    return (_PollOutcome.IDLE, None)
                if resp.status != 200:
                    log.warning("unexpected lease status: %s", resp.status)
                    return (_PollOutcome.IDLE, None)
                body = json.loads(resp.read().decode("utf-8"))
        except OSError as exc:
            log.debug("lease poll failed: %s", exc)
            return (_PollOutcome.UNREACHABLE, None)

        return (_PollOutcome.LEASED, Lease(
            lease_id=body["lease_id"],
            agent=body["agent"],
            item_id=body["item_id"],
            payload=body.get("payload"),
            agent_version=body.get("agent_version"),
            account_id=body.get("account_id"),
            project_id=body.get("project_id"),
        ))

    def _report_complete(
        self,
        lease_id: str,
        status: str,
        error: str | None = None,
        error_category: str | None = None,
    ) -> None:
        body = {
            "lease_id": lease_id,
            "status": status,
            "worker_id": self.worker_id,
        }
        if error is not None:
            body["error"] = error
        if error_category is not None:
            body["error_category"] = error_category
        data = json.dumps(body).encode("utf-8")
        req = urllib_request.Request(
            f"{self.dispatcher_url}/complete",
            data=data,
            headers={"Content-Type": "application/json", **self._auth_headers()},
            method="POST",
        )

        # Bounded retry. The dispatcher's /complete handler is idempotent
        # on lease_id (a duplicate POST emits stale_complete and is a
        # no-op), so retrying a transient failure is always safe. ADR-0002
        # #4. On exhaustion the dispatcher's lease TTL is the safety net:
        # the lease eventually re-dispatches and at-least-once semantics
        # are preserved.
        #
        # ``OSError`` for the reason given on ``_poll_lease``. It matters more
        # here: this call lands *after* the model call has been paid for, so
        # the old ``URLError`` handler let a response-phase timeout kill the
        # worker at the one moment the item's result was still unreported.
        attempts = 5
        wait = 0.1
        for attempt in range(1, attempts + 1):
            try:
                with urllib_request.urlopen(req, timeout=self.http_timeout_seconds):
                    return
            except OSError as exc:
                if attempt == attempts:
                    log.error(
                        "failed to report completion for %s after %d attempts: %s",
                        lease_id, attempts, exc,
                    )
                    return
                log.debug(
                    "complete report attempt %d/%d failed: %s; retrying in %.2fs",
                    attempt, attempts, exc, wait,
                )
                time.sleep(wait)
                wait = min(wait * 2.0, 2.0)

    def _report_release(self, lease_id: str, reason: str) -> None:
        """Hand a lease BACK to the queue instead of completing it.

        POST /release {lease_id, worker_id, reason}. The dispatcher parks the
        item at ``available_at='infinity'`` so an operator resume can unpark
        it and a worker replays the run from its saved checkpoints.

        This is the fix for the pause path's oldest lie. Until Plan 41 R6 a
        paused run reported ``/complete`` with ``error_category="paused"``:
        the item landed in ``runtime_completed``, which is terminal by
        construction, so nothing could ever re-queue it. ``POST /resume``
        flipped the run to ``running`` and left it with no lease and no
        worker — "a human is needed here" rewritten as "work is in progress",
        in the product whose whole wedge is silent failure.

        ``reason`` is ``"paused"`` (a fence stopped the run; the server
        already knows and wrote the status) or ``"credit"`` (the provider
        reported credit exhaustion, which is classified entirely client-side,
        so this call is the only way the server learns of it at all).

        Never raises. Same bounded retry as :meth:`_report_complete`, and
        safe for the same reason: the dispatcher's release is idempotent on
        lease_id, returning 200 for duplicate and stale alike.
        """
        body = {"lease_id": lease_id, "worker_id": self.worker_id, "reason": reason}
        data = json.dumps(body).encode("utf-8")
        req = urllib_request.Request(
            f"{self.dispatcher_url}/release",
            data=data,
            headers={"Content-Type": "application/json", **self._auth_headers()},
            method="POST",
        )

        attempts = 5
        wait = 0.1
        for attempt in range(1, attempts + 1):
            try:
                with urllib_request.urlopen(req, timeout=self.http_timeout_seconds):
                    return
            except urllib_error.HTTPError as exc:
                if exc.code in (404, 405):
                    # The dispatcher predates the release verb. Don't retry a
                    # route that doesn't exist, and don't leave the lease
                    # dangling either: a dangling lease expires in ~30s, the
                    # reaper re-queues it, and a worker picks the PAUSED run
                    # straight back up — worse than the bug we're fixing,
                    # because a re-leased run doesn't re-read its own paused
                    # status. So resolve the lease the old way and say
                    # plainly what the operator has lost.
                    log.warning(
                        "dispatcher has no /release (HTTP %d) — completing lease %s "
                        "instead. The run's item leaves the queue permanently and "
                        "resume will NOT re-drive it; replay the run instead.",
                        exc.code, lease_id,
                    )
                    self._report_complete(
                        lease_id,
                        status="failed",
                        error=f"paused ({reason}); dispatcher has no /release",
                        error_category="paused",
                    )
                    return
                if attempt == attempts:
                    log.error(
                        "failed to release lease %s after %d attempts: %s",
                        lease_id, attempts, exc,
                    )
                    return
                time.sleep(wait)
                wait = min(wait * 2.0, 2.0)
            # After the HTTPError clause, never before: HTTPError is itself an
            # OSError, and the 404/405 branch above is what keeps a dispatcher
            # with no /release from stranding the lease.
            except OSError as exc:
                if attempt == attempts:
                    log.error(
                        "failed to release lease %s after %d attempts: %s",
                        lease_id, attempts, exc,
                    )
                    return
                log.debug(
                    "release attempt %d/%d failed: %s; retrying in %.2fs",
                    attempt, attempts, exc, wait,
                )
                time.sleep(wait)
                wait = min(wait * 2.0, 2.0)

    def _mark_run_running(self, run_id: str) -> None:
        """Best-effort flip the durable run queued→running on lease pickup
        (v1→v2 cutover). PATCH /v1/runtime/runs/{run_id} {status:running} on
        the platform-authed runtime lane — the worker holds the shared
        platform key (no tenant scope), so the tenant-scoped /v1/durable PATCH
        silently no-ops for it (Plan 37 Unit 1). The server guards the
        transition (only 'queued' rows move, output untouched) and resolves
        the tenant off the run row. Never raises: a transient failure just
        leaves the run 'queued' until the SDK writes its terminal status, so
        it must not block execution. No-op when there's no runtime base
        (local dev)."""
        if not self._durable_api_base:
            return
        data = json.dumps({"status": "running"}).encode("utf-8")
        req = urllib_request.Request(
            f"{self.dispatcher_url}/runs/{run_id}",
            data=data,
            headers={"Content-Type": "application/json", **self._auth_headers()},
            method="PATCH",
        )
        try:
            with urllib_request.urlopen(req, timeout=self.http_timeout_seconds):
                return
        except OSError as exc:
            # "Never raises" was the declared contract and OSError is what
            # makes it true — a response-phase timeout here used to take the
            # worker down between lease and execution (plan 48 W1).
            log.warning("failed to mark run %s running: %s", run_id, exc)

    def _mark_run_terminal(
        self,
        run_id: str,
        status: str,
        output: Any = None,
        *,
        error: str | None = None,
        error_category: str | None = None,
    ) -> None:
        """Flip the durable run to its terminal state (running→completed/failed)
        on the platform-authed runtime lane (Plan 37 Unit 1). The sibling of
        :meth:`_mark_run_running`: PATCH /v1/runtime/runs/{run_id}
        {status, output?, error?, error_category?}. The server guards the
        transition, resolves the
        tenant off the run row, and runs the same terminal-status side effects
        (workload fence + run-failed notification) as the tenant path.

        ``output`` (the agent's return value) is attached on completion when it
        is JSON-serialisable, so the dashboard shows the run's result; a
        non-serialisable return just leaves the run output empty rather than
        failing the flip. Never raises — a transient failure leaves the run
        'running' until the reaper/next write reconciles it, so it must not
        block the worker. No-op on the local-dev path (no runtime base)."""
        if not self._durable_api_base:
            return
        body: dict[str, Any] = {"status": status}
        # Why it failed, on the SAME request that records that it failed
        # (plan 48 R3). Not a second call: a follow-up write could land on a
        # row whose status the server's guard refused to move, putting one
        # attempt's error on another attempt's record.
        if status == "failed":
            if error:
                body["error"] = error
            if error_category:
                body["error_category"] = error_category
        if status == "completed" and output is not None:
            try:
                json.dumps(output)  # probe serialisability
                body["output"] = output
            except (TypeError, ValueError):
                pass  # non-serialisable return — leave run output empty
        data = json.dumps(body).encode("utf-8")
        req = urllib_request.Request(
            f"{self.dispatcher_url}/runs/{run_id}",
            data=data,
            headers={"Content-Type": "application/json", **self._auth_headers()},
            method="PATCH",
        )
        try:
            with urllib_request.urlopen(req, timeout=self.http_timeout_seconds):
                return
        except OSError as exc:
            log.warning("failed to mark run %s %s: %s", run_id, status, exc)

    # --- lease handling ------------------------------------------------ #

    def _handle_lease(self, lease: Lease) -> None:
        """Run the @agent function for a single leased item."""
        # Late import: the customer module's @agent decorations registered
        # into this same module-level dict, so a top-level import here
        # would create a cycle / shadow.
        from papayya.agent import (
            get_agent,
            reset_bootstrap_item_id,
            reset_bootstrap_lease_id,
            reset_bootstrap_run_id,
            set_bootstrap_item_id,
            set_bootstrap_lease_id,
            set_bootstrap_run_id,
        )

        short = lease.lease_id[:8]
        log.info(
            "started  %s agent=%s item=%s",
            short, lease.agent, lease.item_id,
        )
        started_at = time.monotonic()
        self._last_activity_at = started_at

        # v1→v2 cutover: the hosted submission pre-created a durable_run
        # and put its id on the lease payload. Adopt it so the @agent's
        # first Papayya.run() links its checkpoints to that exact run, and
        # flip the run queued→running now that a worker owns it. Both are
        # no-ops on the local-dev path (LocalDispatcher leases carry no
        # run_id), keeping the SQLite guardrail green.
        run_id = None
        if isinstance(lease.payload, dict):
            run_id = lease.payload.get("run_id")
        bootstrap_token = set_bootstrap_run_id(run_id)
        # Plan 56 F3: stamp every checkpoint / status write this invocation
        # makes with the lease it is running under, so the control-plane can
        # reject writes from a worker whose lease was revoked. The 410 this
        # worker already handles clears LOCAL tracking only — it cannot stop
        # the invocation (the customer's function is on the main thread,
        # possibly inside C code), which is why the fence has to be at the
        # write door and not here.
        lease_bootstrap_token = set_bootstrap_lease_id(lease.lease_id)
        # The DECLARED item_id, handed down rather than guessed from the
        # function's arguments (plan 43 B2b C11). Since B2a the argument is the
        # customer's INPUT, so `args[0]` is an object — and an object reached
        # `papayya().run(item_id=...)` and then every checkpoint payload, where
        # the wire wants a string: measured, every hosted run.step() 400'd and no
        # hosted run could save a checkpoint at all.
        #
        # `or None` because the dispatcher normalizes an omitted id to "" rather
        # than to NULL, and an empty string is not an id — it would blank the
        # local fallback instead of deferring to it.
        item_bootstrap_token = set_bootstrap_item_id(lease.item_id or None)
        if run_id:
            self._mark_run_running(run_id)

        # Publish the lease so the heartbeat starts pinging /heartbeat for
        # it. Cleared in the finally block. The in-process copy is still
        # tracked (the drain watchdog and the 410 reader both read it); the
        # process gets told separately because it cannot see this object.
        with self._hb_lock:
            self._in_flight_lease = lease
        self._publish_lease_to_heartbeat(lease.lease_id)
        try:
            # ADR-0003 § Worker #3 — make sure the lease's agent_version
            # is loaded before resolving the registration. No-op when
            # agent_version is None (local-dev parity).
            try:
                self._ensure_loaded(lease)
            except _VersionNotFound as exc:
                duration_ms = int((time.monotonic() - started_at) * 1000)
                log.warning(
                    "failed   %s item=%s duration=%dms category=version_not_found %s",
                    short, lease.item_id, duration_ms, exc,
                )
                self._report_complete(
                    lease.lease_id,
                    status="failed",
                    error=str(exc),
                    error_category="version_not_found",
                )
                return
            except _RecyclePending as exc:
                # ADR-0003 § Worker #6 — fail this lease with a
                # categorised error so the dispatcher's lease-TTL
                # path can re-dispatch it; main loop exits via the
                # ``_running = False`` set inside ``_ensure_loaded``.
                duration_ms = int((time.monotonic() - started_at) * 1000)
                log.warning(
                    "failed   %s item=%s duration=%dms category=recycle_pending %s",
                    short, lease.item_id, duration_ms, exc,
                )
                self._report_complete(
                    lease.lease_id,
                    status="failed",
                    error=str(exc),
                    error_category="recycle_pending",
                )
                return
            except OSError as exc:
                # Couldn't reach the bundle endpoint. Transient by
                # assumption, so this does NOT complete the lease: burning a
                # customer's item because the control plane blinked is the
                # wrong trade when the lease TTL will re-dispatch it in 30s.
                # Deliberately no /complete — dropping the lease is the retry.
                log.warning(
                    "deferred %s item=%s bundle fetch unreachable (%s); "
                    "dropping the lease for TTL re-dispatch",
                    short, lease.item_id, exc,
                )
                return
            except Exception as exc:  # noqa: BLE001 — one lease, not the pool
                # A bad import in the customer's entrypoint, a corrupt
                # tarball, a verification failure. Deterministic: TTL
                # re-dispatch would just fail the same way forever, so this
                # one is reported. Before plan 48 W1 it exited the process
                # instead — one customer's broken bundle stopped every other
                # account's work.
                from papayya.runtime.error_category import classify_exception

                duration_ms = int((time.monotonic() - started_at) * 1000)
                category = classify_exception(exc)
                log.warning(
                    "failed   %s item=%s duration=%dms category=%s "
                    "bundle load: %s: %s",
                    short, lease.item_id, duration_ms, category,
                    type(exc).__name__, exc,
                    exc_info=True,
                )
                self._report_complete(
                    lease.lease_id,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    error_category=category,
                )
                # /complete records the LEASE. The run is a separate row and
                # nothing else will move it, so without this the item reads
                # "in progress" forever — plan 48 W3's exact shape, on a path
                # that has already decided the work is over.
                if run_id:
                    self._mark_run_terminal(
                        run_id,
                        "failed",
                        error=f"{type(exc).__name__}: {exc}",
                        error_category=category,
                    )
                return

            # ADR-0003 § Worker #4 — dispatch to the registration that
            # matches the lease's version. ``lease.agent_version is
            # None`` (LocalDispatcher) preserves single-resident
            # behaviour: ``get_agent`` returns the latest-registered
            # entry for the slug.
            #
            # Hosted leases dispatch from the residency's own snapshot
            # rather than the global registry. The registry is keyed
            # (name, version) with no account — customer code cannot
            # know its account — so on a shared worker the second
            # account to import a shared slug overwrites the first's
            # entry, and every later lease for *either* account would
            # get the survivor's function (plan 47 S1).
            resident = self._loaded_versions.get(self._residency_key(lease))
            registration = (
                resident.registration
                if resident is not None and resident.registration is not None
                else get_agent(lease.agent, lease.agent_version)
            )
            if registration is None:
                duration_ms = int((time.monotonic() - started_at) * 1000)
                # ADR-0003 § Worker #5 — bootstrap workers have no
                # pre-loaded module, so a lease without agent_version
                # is a misconfiguration (LocalDispatcher pointed at a
                # hosted worker, or env var leaked into a container).
                # Emit a distinct category so it doesn't get lumped
                # in with "agent name typo" / version_not_found.
                if self._bootstrap_mode and lease.agent_version is None:
                    log.warning(
                        "failed   %s item=%s duration=%dms category=no_agent_module",
                        short, lease.item_id, duration_ms,
                    )
                    self._report_complete(
                        lease.lease_id,
                        status="failed",
                        error=(
                            "bootstrap worker received lease without "
                            "agent_version (LocalDispatcher misconfigured "
                            "against hosted worker?)"
                        ),
                        error_category="no_agent_module",
                    )
                    return
                log.warning(
                    "failed   %s item=%s duration=%dms unknown-agent=%s version=%s",
                    short, lease.item_id, duration_ms, lease.agent, lease.agent_version,
                )
                self._report_complete(
                    lease.lease_id,
                    status="failed",
                    error=f"unknown agent: {lease.agent} (version={lease.agent_version})",
                )
                return

            # Resolve the timeout for this invocation. Per-call payload
            # override (ADR-0002 #2 user choice) wins over the per-agent
            # default. None at both levels disables the watchdog.
            max_duration = None
            if isinstance(lease.payload, dict):
                payload_override = lease.payload.get("max_duration_seconds")
                if payload_override is not None:
                    max_duration = payload_override
            if max_duration is None:
                max_duration = registration.max_duration_seconds

            outcome = self._invoke_with_timeout(
                fn=registration.fn,
                lease=lease,
                started_at=started_at,
                max_duration=max_duration,
                short=short,
            )
            # v1→v2 cutover: the dispatcher's /complete records only the
            # *lease* (runtime_completed). The durable *run* status is owned
            # on the platform runtime lane, so flip it here now that the
            # invocation resolved. run_status is None for a pause/credit
            # signal — the run's status is authoritative server-side and must
            # not be clobbered. No-op on the local-dev path (no run_id).
            if run_id and outcome.status is not None:
                self._mark_run_terminal(
                    run_id,
                    outcome.status,
                    outcome.output,
                    error=outcome.error,
                    error_category=outcome.error_category,
                )
        finally:
            reset_bootstrap_run_id(bootstrap_token)
            reset_bootstrap_item_id(item_bootstrap_token)
            reset_bootstrap_lease_id(lease_bootstrap_token)
            with self._hb_lock:
                self._in_flight_lease = None
            self._publish_lease_to_heartbeat(None)
            self._last_activity_at = time.monotonic()
            self._items_processed += 1
            self._check_recycle_thresholds()

    def _check_recycle_thresholds(self) -> None:
        """Trip the recycle flags if item-count or RSS% exceed their caps.

        Called from ``_handle_lease``'s finally block — between items,
        after the current lease has fully released the worker. Unlike
        the dep-hash branch this never fails a lease: the item that
        triggered the threshold already completed normally, so we just
        flip ``_recycle_pending`` and ``_running=False`` and let the
        main loop exit on its next iteration. ADR-0002 #6 / ADR-0001 § 4.
        """
        if (
            self._max_items_before_recycle > 0
            and self._items_processed >= self._max_items_before_recycle
        ):
            log.warning(
                "scheduling recycle: reason=item_count items_processed=%d max=%d",
                self._items_processed,
                self._max_items_before_recycle,
            )
            self._recycle_pending = True
            self._running = False
            return

        if self._max_rss_percent_before_recycle > 0:
            try:
                rss_pct = self._rss_percent_provider()
            except Exception as exc:  # noqa: BLE001 — observability over crash
                # Reading RSS must never crash the worker. The dep-hash
                # and SIGTERM triggers stay armed; we just skip the RSS
                # check this iteration.
                log.debug("rss percent provider failed: %s", exc)
                return
            if rss_pct >= self._max_rss_percent_before_recycle:
                log.warning(
                    "scheduling recycle: reason=rss_percent rss_percent=%.1f max=%.1f",
                    rss_pct,
                    self._max_rss_percent_before_recycle,
                )
                self._recycle_pending = True
                self._running = False

    def _invoke_with_timeout(
        self,
        *,
        fn: Any,
        lease: Lease,
        started_at: float,
        max_duration: float | None,
        short: str,
    ) -> tuple[str | None, Any]:
        """Run ``fn(lease.agent_argument)``; arm SIGALRM if max_duration is set.

        Returns ``(run_status, output)`` for the caller to flip the durable
        run's terminal state on the hosted path (Plan 37 Unit 1 — the
        dispatcher's ``/complete`` only records the *lease*; the *run* status
        is owned here). ``run_status`` is ``"completed"`` / ``"failed"``, or
        ``None`` for a pass-through signal (pause/credit) whose run status is
        authoritative server-side and must not be clobbered. ``output`` is the
        agent's return value on success, else ``None``.

        Terminal paths:
          - Success: report completed lease → ``("completed", result)``.
          - _AgentTimeout: report failed (category=timeout) → ``("failed", None)``.
          - WorkloadPaused / CreditExhausted: RELEASE the lease back to the
            queue (parked) and leave the run status alone → ``(None, None)``.
            Not a completion: a completed lease is terminal and resume would
            have nothing to re-drive (Plan 41 R6).
          - Any other exception: report failed → ``("failed", None)``.

        The signal arming is local to this call. ``setitimer(0)`` and
        the handler restore in the finally block guarantee no SIGALRM
        leaks across leases.

        Async registrations branch off to ``_invoke_async`` — the SIGALRM
        watchdog is unsafe inside a running event loop (raising into
        ``epoll_wait`` from a signal handler can leave the loop in an
        inconsistent state). The async path uses ``asyncio.wait_for``
        for the same wall-clock guarantee.
        """
        if inspect.iscoroutinefunction(fn):
            return self._invoke_async(
                fn=fn,
                lease=lease,
                started_at=started_at,
                max_duration=max_duration,
                short=short,
            )

        # Late import: keep the worker boot path free of the bundle
        # loader's importlib pull-in cost when no bundles are involved.
        from papayya.runtime import _bundle_loader
        from papayya.runtime.error_category import classify_exception
        from papayya.errors import CreditExhausted, WorkloadPaused

        prior_handler = None
        watchdog_armed = max_duration is not None and max_duration > 0
        if watchdog_armed:
            prior_handler = signal.signal(signal.SIGALRM, _on_agent_alarm)
            signal.setitimer(signal.ITIMER_REAL, max_duration)
        try:
            # Activate this residency's bundle finder so function-body
            # imports (``def fn(): from helpers import x``) resolve
            # against the right account+version's siblings. ``None`` is
            # a no-op so local-dev / LocalDispatcher leases pay nothing.
            with _bundle_loader.activate(self._activation_token(lease)):
                result = fn(lease.agent_argument)
        except _AgentTimeout:
            duration_ms = int((time.monotonic() - started_at) * 1000)
            log.warning(
                "failed   %s item=%s duration=%dms category=timeout limit=%.2fs",
                short, lease.item_id, duration_ms, max_duration,
            )
            timeout_error = f"timeout: agent ran for >{max_duration}s"
            self._report_complete(
                lease.lease_id,
                status="failed",
                error=timeout_error,
                error_category="timeout",
            )
            return _InvocationOutcome("failed", None, timeout_error, "timeout")
        except (WorkloadPaused, CreditExhausted) as exc:
            # Not a body failure: the run stopped, its checkpoints are all
            # saved, and it must be resumable. RELEASE the lease — do not
            # complete it. Completing it puts the item in runtime_completed,
            # which is terminal by construction, so resume would have
            # nothing to re-drive (Plan 41 R6). The two exceptions differ
            # only in the reason they carry: a fence pause the server
            # already recorded, versus a provider-credit stop it can learn
            # about nowhere else.
            duration_ms = int((time.monotonic() - started_at) * 1000)
            reason = "credit" if isinstance(exc, CreditExhausted) else "paused"
            log.warning(
                "paused   %s item=%s duration=%dms reason=%s %s: %s",
                short, lease.item_id, duration_ms, reason, type(exc).__name__, exc,
            )
            self._report_release(lease.lease_id, reason=reason)
            return _InvocationOutcome(None)
        except Exception as exc:  # noqa: BLE001 — customer code; isolate
            duration_ms = int((time.monotonic() - started_at) * 1000)
            log.exception(
                "failed   %s item=%s duration=%dms",
                short, lease.item_id, duration_ms,
            )
            # The category the lease table never got: this branch is the
            # common case (the customer's own function raised) and it was the
            # one path that reported no category at all (plan 48 R3).
            message = f"{type(exc).__name__}: {exc}"
            category = classify_exception(exc)
            self._report_complete(
                lease.lease_id,
                status="failed",
                error=message,
                error_category=category,
            )
            return _InvocationOutcome("failed", None, message, category)
        finally:
            if watchdog_armed:
                signal.setitimer(signal.ITIMER_REAL, 0)
                # Restore whatever was on SIGALRM before us — could be
                # the default handler (None on the C side) or a customer
                # handler installed before we hooked. signal.signal
                # returns the prior callable / SIG_DFL marker.
                if prior_handler is not None:
                    signal.signal(signal.SIGALRM, prior_handler)

        # Success path (no exception, no early return).
        duration_ms = int((time.monotonic() - started_at) * 1000)
        log.info(
            "finished %s item=%s duration=%dms",
            short, lease.item_id, duration_ms,
        )
        self._report_complete(lease.lease_id, status="completed")
        return _InvocationOutcome("completed", result)

    def _invoke_async(
        self,
        *,
        fn: Any,
        lease: Lease,
        started_at: float,
        max_duration: float | None,
        short: str,
    ) -> tuple[str | None, Any]:
        """Run a coroutine ``fn(lease.agent_argument)`` to completion.

        Returns ``(run_status, output)`` on the same contract as
        :meth:`_invoke_with_timeout` — the caller flips the durable run's
        terminal state from it (``None`` = pause/credit signal, leave the run
        status alone).

        Uses ``asyncio.wait_for`` for timeout enforcement instead of the
        sync path's SIGALRM watchdog. Signal handlers raising into a
        running event loop can leave the loop in inconsistent state;
        ``wait_for`` cancels the inner coroutine cleanly so any
        ``finally`` / cleanup blocks the agent installed run before we
        report failure.

        Terminal paths:
          - Success: report completed → ``("completed", result)``.
          - ``asyncio.TimeoutError`` from ``wait_for``: report failed
            with ``error_category="timeout"`` (parity with sync path).
          - ``asyncio.CancelledError``: report failed with
            ``error_category="cancelled"``. Distinct from ``timeout``
            because the operator response differs — ``timeout`` says
            "max_duration_seconds is too tight", ``cancelled`` says
            "look for who issued the cancel". CancelledError extends
            ``BaseException`` so the generic ``except Exception`` below
            doesn't catch it; without an explicit branch this would
            propagate out of ``_handle_lease`` and the lease would only
            recover via TTL.
          - WorkloadPaused / CreditExhausted: RELEASE the lease back to the
            queue (parked), leaving the run status alone → ``(None, None)``.
          - Any other ``Exception``: existing stringified-error path.
        """
        from papayya.runtime import _bundle_loader
        from papayya.runtime.error_category import classify_exception
        from papayya.errors import CreditExhausted, WorkloadPaused

        coro = fn(lease.agent_argument)
        try:
            # Same activate-scope rationale as the sync path; mirrored
            # here because ``asyncio.run`` runs the coroutine on a new
            # loop and we want imports inside it to see the right
            # account+version's siblings.
            with _bundle_loader.activate(self._activation_token(lease)):
                if max_duration is not None and max_duration > 0:
                    result = asyncio.run(asyncio.wait_for(coro, timeout=max_duration))
                else:
                    result = asyncio.run(coro)
        except asyncio.TimeoutError:
            duration_ms = int((time.monotonic() - started_at) * 1000)
            log.warning(
                "failed   %s item=%s duration=%dms category=timeout limit=%.2fs",
                short, lease.item_id, duration_ms, max_duration,
            )
            timeout_error = f"timeout: agent ran for >{max_duration}s"
            self._report_complete(
                lease.lease_id,
                status="failed",
                error=timeout_error,
                error_category="timeout",
            )
            return _InvocationOutcome("failed", None, timeout_error, "timeout")
        except asyncio.CancelledError:
            duration_ms = int((time.monotonic() - started_at) * 1000)
            log.warning(
                "failed   %s item=%s duration=%dms category=cancelled",
                short, lease.item_id, duration_ms,
            )
            cancel_error = "cancelled: asyncio.CancelledError"
            self._report_complete(
                lease.lease_id,
                status="failed",
                error=cancel_error,
                error_category="cancelled",
            )
            return _InvocationOutcome("failed", None, cancel_error, "cancelled")
        except (WorkloadPaused, CreditExhausted) as exc:
            # Mirrors the sync path exactly: release the lease so the item
            # goes back to the queue parked, rather than completing it into
            # a terminal table resume can't reach (Plan 41 R6).
            duration_ms = int((time.monotonic() - started_at) * 1000)
            reason = "credit" if isinstance(exc, CreditExhausted) else "paused"
            log.warning(
                "paused   %s item=%s duration=%dms reason=%s %s: %s",
                short, lease.item_id, duration_ms, reason, type(exc).__name__, exc,
            )
            self._report_release(lease.lease_id, reason=reason)
            return _InvocationOutcome(None)
        except Exception as exc:  # noqa: BLE001 — customer code; isolate
            duration_ms = int((time.monotonic() - started_at) * 1000)
            log.exception(
                "failed   %s item=%s duration=%dms",
                short, lease.item_id, duration_ms,
            )
            message = f"{type(exc).__name__}: {exc}"
            category = classify_exception(exc)
            self._report_complete(
                lease.lease_id,
                status="failed",
                error=message,
                error_category=category,
            )
            return _InvocationOutcome("failed", None, message, category)

        duration_ms = int((time.monotonic() - started_at) * 1000)
        log.info(
            "finished %s item=%s duration=%dms",
            short, lease.item_id, duration_ms,
        )
        self._report_complete(lease.lease_id, status="completed")
        return _InvocationOutcome("completed", result)

    # --- drain watchdog ----------------------------------------------- #

    def _drain_watchdog(self, started_at: float) -> None:
        """Bound the SIGTERM drain phase; force-exit on deadline.

        Spawned lazily from ``_on_signal`` so an idle worker doesn't
        hold a blocked daemon thread (which interferes with
        cross-process SQLite WAL visibility on macOS). Gives the
        in-flight item ``drain_timeout_seconds`` to finish naturally;
        if the main thread reaches ``run()``'s finally before that
        deadline, ``_hb_stop`` is set and the watchdog exits silently.

        On deadline expiry the watchdog flushes log handlers and calls
        ``os._exit(1)``. The recovery path is the dispatcher's lease
        TTL: the orphaned lease is released and the item re-dispatched,
        with the idempotent ``/complete`` (#4) preventing
        double-accounting if a late completion lands.
        """
        deadline = started_at + self.drain_timeout_seconds
        while time.monotonic() < deadline:
            if self._hb_stop.is_set():
                return  # run() returned cleanly; nothing to escalate
            time.sleep(0.2)
        with self._hb_lock:
            in_flight = self._in_flight_lease
        lease_short = in_flight.lease_id[:8] if in_flight else "?"
        log.error(
            "worker %s drain deadline exceeded (%.0fs); forcing exit. "
            "Lease %s will be released by dispatcher TTL.",
            self.worker_id, self.drain_timeout_seconds, lease_short,
        )
        # Flush handlers so the error line above reaches the operator
        # before os._exit skips Python finalization.
        for h in logging.getLogger().handlers:
            try:
                h.flush()
            except Exception:  # noqa: BLE001
                pass
        os._exit(1)

    # --- heartbeat ----------------------------------------------------- #

    def _start_heartbeat(self) -> None:
        """Start the out-of-process heartbeat, falling back to the thread.

        THE PROCESS IS THE POINT (plan 56 F2). A background THREAD cannot run
        while the customer's step holds the GIL inside a C loop, and measured
        across three kinds of work it gets 1-6% of its due ticks against 97%
        for I/O-bound work. That is not a monitoring gap: the lease expires,
        the reaper re-queues the item, and a second worker starts the same
        document again. Plan 55 D1 measured three concurrent 135-second
        executions of one customer function, every checkpoint recorded
        attempt=1, and the run finished `ok`.

        A separate interpreter has its own GIL, so customer compute cannot
        starve it.

        THE FALLBACK IS THE THREAD, not an exception. If the child cannot be
        spawned — a locked-down container, no /proc, an exotic sys.executable —
        a degraded heartbeat is strictly better than no worker. The thread is
        correct for every I/O-bound step, which is most of them, and the
        dispatcher's reaper remains the backstop for the rest. Refusing to
        start would turn a partial defence into an outage.
        """
        self._hb_proc: Optional[subprocess.Popen] = None
        self._hb_thread: Optional[threading.Thread] = None
        try:
            self._hb_proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "papayya.runtime.heartbeat",
                    "--dispatcher-url", self.dispatcher_url,
                    "--worker-id", self.worker_id,
                    "--interval-seconds", str(self.heartbeat_interval_seconds),
                    "--timeout-seconds", str(self.http_timeout_seconds),
                ]
                + (["--api-key", self._api_key] if self._api_key else []),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,  # line-buffered: the protocol is one line per message
            )
        except Exception as exc:  # noqa: BLE001 — see docstring
            log.warning(
                "could not start the heartbeat process (%s); falling back to the "
                "in-process thread, which cannot heartbeat through a step that "
                "holds the GIL", exc,
            )
            self._hb_proc = None

        if self._hb_proc is not None:
            # Drain the child's stdout so a 410 clears in-flight tracking the
            # same way the thread's own 410 branch does. Daemon: this thread
            # only ever blocks on a pipe, never on customer code.
            self._hb_reader = threading.Thread(
                target=self._heartbeat_process_reader,
                daemon=True,
                name=f"papayya-worker-hb-reader-{self.worker_id}",
            )
            self._hb_reader.start()
            log.info(
                "heartbeat process started (pid=%s, interval=%.1fs)",
                self._hb_proc.pid, self.heartbeat_interval_seconds,
            )
            return

        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name=f"papayya-worker-hb-{self.worker_id}",
        )
        self._hb_thread.start()

    def _stop_heartbeat(self) -> None:
        """Shut the heartbeat down, whichever form it took.

        Closing stdin is the ordinary stop: the child sees EOF and exits. The
        kill is for a child wedged in a socket timeout — it holds no state
        worth draining, and a heartbeat that outlives its worker actively lies
        to the dispatcher about a dead item.
        """
        proc = getattr(self, "_hb_proc", None)
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
                proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            self._hb_proc = None
        thread = getattr(self, "_hb_thread", None)
        if thread is not None:
            thread.join(timeout=2)

    def _publish_lease_to_heartbeat(self, lease_id: Optional[str]) -> None:
        """Tell the heartbeat process which lease is in flight (None = idle).

        A write failure means the child is gone. Log once and carry on: the
        item in flight is still being processed, and killing the worker over a
        dead heartbeat would abandon work the dispatcher would then have to
        re-deliver — the very loop F2 exists to stop.
        """
        proc = getattr(self, "_hb_proc", None)
        if proc is None or proc.stdin is None:
            return
        try:
            proc.stdin.write(f"{lease_id or _HEARTBEAT_IDLE}\n")
            proc.stdin.flush()
        except Exception as exc:  # noqa: BLE001 — see docstring
            log.warning("heartbeat process is not accepting leases (%s)", exc)
            self._hb_proc = None

    def _heartbeat_process_reader(self) -> None:
        """Apply the child's `gone <lease_id>` lines to in-flight tracking.

        Same effect as the thread's 409/410 branch and for the same reason:
        drop local tracking so a late /complete for a stolen item is not
        reported. It does NOT stop the invocation — nothing can, the customer's
        function is on the main thread inside C code — which is exactly why
        plan 56 F3 fences the WRITES server-side rather than trusting this.
        """
        proc = self._hb_proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                parts = line.split()
                if len(parts) != 2 or parts[0] != "gone":
                    continue
                lease_id = parts[1]
                with self._hb_lock:
                    if (
                        self._in_flight_lease is not None
                        and self._in_flight_lease.lease_id == lease_id
                    ):
                        log.warning(
                            "lease %s rejected by dispatcher; worker dropping "
                            "in-flight tracking", lease_id[:8],
                        )
                        self._in_flight_lease = None
        except Exception:  # noqa: BLE001 — pipe closed on shutdown
            return

    def _heartbeat_loop(self) -> None:
        """Background loop: ping /heartbeat for the in-flight lease.

        Runs for the worker's lifetime. A missing in-flight lease is
        legal (worker is between items) and just skips the iteration.
        Network failures are soft — the dispatcher's reaper handles
        actual death; heartbeat-loop errors are surface-only.
        """
        while not self._hb_stop.is_set():
            if self._hb_stop.wait(timeout=self.heartbeat_interval_seconds):
                return
            with self._hb_lock:
                lease = self._in_flight_lease
            if lease is None:
                continue
            try:
                self._send_heartbeat(lease.lease_id)
            except Exception as exc:  # noqa: BLE001
                log.debug("heartbeat for %s failed: %s", lease.lease_id[:8], exc)

    def _send_heartbeat(self, lease_id: str) -> None:
        body = json.dumps({
            "lease_id": lease_id,
            "worker_id": self.worker_id,
        }).encode("utf-8")
        req = urllib_request.Request(
            f"{self.dispatcher_url}/heartbeat",
            data=body,
            headers={"Content-Type": "application/json", **self._auth_headers()},
            method="POST",
        )
        # The SHORTER of one interval and the request budget. A heartbeat that
        # hasn't landed by the time the next one is due has been superseded,
        # and the loop is serial — a long budget here would stretch the cadence
        # toward the lease TTL it exists to stay under. Timeouts surface as
        # OSError and _heartbeat_loop's except Exception already absorbs them.
        try:
            with urllib_request.urlopen(
                req, timeout=min(self.heartbeat_interval_seconds, self.http_timeout_seconds)
            ):
                pass
        except urllib_error.HTTPError as exc:
            # 410 Gone: dispatcher released this lease (TTL expired or
            # never existed). Drop our local tracking so a late /complete
            # for this stolen item doesn't get reported.
            # 409 Conflict: another worker holds it (zombie scenario).
            if exc.code in (409, 410):
                with self._hb_lock:
                    if self._in_flight_lease is not None and self._in_flight_lease.lease_id == lease_id:
                        log.warning(
                            "lease %s rejected by dispatcher (HTTP %d); worker dropping in-flight tracking",
                            lease_id[:8], exc.code,
                        )
                        self._in_flight_lease = None
                return
            raise
