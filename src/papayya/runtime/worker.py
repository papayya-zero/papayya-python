"""Worker — long-lived process that pulls items from a dispatcher.

The worker process:

1. Imports the customer agent module *once* on boot (this triggers
   ``@agent`` decorator registration).
2. Loops: long-polls the dispatcher for the next leased item, looks
   up the registered ``@agent`` function by name, calls it with the
   ``item_id``, reports completion (or failure).
3. Exits cleanly on SIGTERM / SIGINT.

The dispatcher protocol is intentionally minimal for Phase 1:

  GET  /lease?worker_id=X     -> 200 JSON {lease_id, agent, item_id} or 204
  POST /complete              -> 200 JSON {}, body {lease_id, status, error?}

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
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request


log = logging.getLogger("papayya.runtime")


# Default heartbeat cadence. Must be well below the dispatcher's lease
# TTL (default 30s) so a single missed heartbeat doesn't expire the
# lease. 5s gives roughly 6× headroom.
_DEFAULT_HEARTBEAT_INTERVAL = 5.0


# Default SIGTERM drain budget. Aligns with Kubernetes' default
# `terminationGracePeriodSeconds` (30s) so a worker pod gets to finish
# the in-flight item before kubelet escalates to SIGKILL. ADR-0002 #12.
_DEFAULT_DRAIN_TIMEOUT_SECONDS = 30.0


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
    """
    agent_name: str
    agent_version: str
    bundle_path: str
    module_name: str
    dep_hash: str | None = None


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
        # Bootstrap mode: hosted workers boot without --agent-module and
        # load every bundle on demand from lease.agent_version. A lease
        # with agent_version=None (LocalDispatcher) in this mode is a
        # misconfiguration — the lease handler emits error_category=
        # "no_agent_module" so the failure is loud. ADR-0003 § Worker #5.
        self._bootstrap_mode = agent_module_path is None
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
        self._bundle_url_base = (bundle_url_base or f"{self.dispatcher_url}/v1/runtime/bundles").rstrip("/")
        # Per-(agent_name, agent_version) cache of bundle entries the
        # worker has already loaded into the registry. Slice 2 guarantees
        # at most one resident entry per (name, version) tuple — multi-
        # version dispatch is slice 3. Hot-path lookup avoids re-reading
        # the on-disk cache and re-importing the module on every lease.
        self._loaded_versions: dict[tuple[str, str], "_LoadedBundle"] = {}
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

        # Point the customer's papayya() client at our shared SQLite. Must be
        # set BEFORE importing the agent module — the customer code may
        # call `papayya()` at module top-level (rare but legal).
        os.environ["PAPAYYA_LOCAL_DB_PATH"] = store_path
        # Ensure CloudStore isn't picked up if a stray PAPAYYA_API_KEY is in
        # env from the parent shell.
        os.environ.pop("PAPAYYA_API_KEY", None)

        if agent_module_path is not None:
            self._import_agent_module(agent_module_path)
        else:
            log.info(
                "starting in bootstrap mode (no agent module pre-loaded; "
                "first lease's agent_version triggers first import)"
            )

        # Heartbeat thread starts after module import so any import
        # error fails fast without leaving a daemon thread behind.
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name=f"papayya-worker-hb-{self.worker_id}",
        )
        self._hb_thread.start()

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
        2. ``(agent, agent_version)`` already in ``self._loaded_versions``
           — hot path. The earlier import already registered the
           ``@agent``; no-op.
        3. Cache miss. Fetch the tarball from the bundle endpoint,
           extract under the on-disk cache, build a ``ModuleSpec`` from
           the entrypoint, ``exec_module`` it (the ``@agent`` decorator
           re-registers under the agent's slug), record the bundle in
           ``self._loaded_versions``.

        Raises ``_VersionNotFound`` on 404 from the bundle endpoint;
        ``_handle_lease`` maps that to a categorised failure. Network /
        verification errors bubble — the lease TTL is the safety net.
        """
        if lease.agent_version is None:
            return

        key = (lease.agent, lease.agent_version)
        if key in self._loaded_versions:
            return

        # Late import to keep the hot path (no agent_version) free of the
        # bundle-cache module's tarfile/fcntl pull-in cost. Worker boot
        # is unaffected; only the first hosted lease pays it.
        from papayya.runtime import _bundle_cache

        version_int = self._parse_version(lease.agent_version)
        bundle = _bundle_cache.ensure_bundle(
            agent_slug=lease.agent,
            version=version_int,
            fetch=lambda: self._fetch_bundle(lease.agent, version_int),
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
        prior = next(
            (
                lb
                for (slug, _v), lb in self._loaded_versions.items()
                if slug == lease.agent and _v != lease.agent_version
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
        )
        self._loaded_versions[key] = _LoadedBundle(
            agent_name=lease.agent,
            agent_version=lease.agent_version,
            bundle_path=str(bundle.path),
            module_name=module_name,
            dep_hash=bundle.dep_hash,
        )

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

    def _fetch_bundle(self, agent: str, version: int) -> Any:
        """HTTP GET the bundle endpoint and adapt to ``FetchedBundle``.

        Stored as a closure so ``ensure_bundle``'s lazy-fetch contract
        works: zero-arg callable, only invoked on cache miss. Response
        headers — entrypoint, account_id, agent_id, deployment_id, ETag
        — ride along on the ``FetchedBundle`` so ``ensure_bundle`` can
        annotate the resulting cache entry without a second round-trip.
        """
        from papayya.runtime._bundle_cache import FetchedBundle

        url = f"{self._bundle_url_base}?agent={agent}&version={version}"
        req = urllib_request.Request(url, headers=self._auth_headers())
        try:
            resp = urllib_request.urlopen(req, timeout=30.0)
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

        _bundle_loader.register_bundle(agent_version, bundle_path)

        module_name = f"_papayya_user_{entry_path.stem}__v{agent_version}"
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
            with _bundle_loader.activate(agent_version):
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
                    self._handle_lease(lease)
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
            self._hb_thread.join(timeout=2)

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
        """
        url = f"{self.dispatcher_url}/lease?worker_id={self.worker_id}"
        req = urllib_request.Request(url, headers=self._auth_headers())
        try:
            with urllib_request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 204:
                    return (_PollOutcome.IDLE, None)
                if resp.status != 200:
                    log.warning("unexpected lease status: %s", resp.status)
                    return (_PollOutcome.IDLE, None)
                body = json.loads(resp.read().decode("utf-8"))
        except urllib_error.URLError as exc:
            log.debug("lease poll failed: %s", exc)
            return (_PollOutcome.UNREACHABLE, None)

        return (_PollOutcome.LEASED, Lease(
            lease_id=body["lease_id"],
            agent=body["agent"],
            item_id=body["item_id"],
            payload=body.get("payload"),
            agent_version=body.get("agent_version"),
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
        attempts = 5
        wait = 0.1
        for attempt in range(1, attempts + 1):
            try:
                with urllib_request.urlopen(req, timeout=2.0):
                    return
            except urllib_error.URLError as exc:
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

    # --- lease handling ------------------------------------------------ #

    def _handle_lease(self, lease: Lease) -> None:
        """Run the @agent function for a single leased item."""
        # Late import: the customer module's @agent decorations registered
        # into this same module-level dict, so a top-level import here
        # would create a cycle / shadow.
        from papayya.agent import get_agent

        short = lease.lease_id[:8]
        log.info(
            "started  %s agent=%s item=%s",
            short, lease.agent, lease.item_id,
        )
        started_at = time.monotonic()
        self._last_activity_at = started_at
        # Publish the lease so the heartbeat thread starts pinging
        # /heartbeat for it. Cleared in the finally block.
        with self._hb_lock:
            self._in_flight_lease = lease
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

            # ADR-0003 § Worker #4 — dispatch to the registration that
            # matches the lease's version. ``lease.agent_version is
            # None`` (LocalDispatcher) preserves single-resident
            # behaviour: ``get_agent`` returns the latest-registered
            # entry for the slug.
            registration = get_agent(lease.agent, lease.agent_version)
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

            self._invoke_with_timeout(
                fn=registration.fn,
                lease=lease,
                started_at=started_at,
                max_duration=max_duration,
                short=short,
            )
        finally:
            with self._hb_lock:
                self._in_flight_lease = None
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
    ) -> None:
        """Run ``fn(lease.item_id)``; arm SIGALRM if max_duration is set.

        Three terminal paths:
          - Success: report completed.
          - _AgentTimeout: report failed with error_category=timeout.
          - Any other exception: report failed with stringified error.

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
            self._invoke_async(
                fn=fn,
                lease=lease,
                started_at=started_at,
                max_duration=max_duration,
                short=short,
            )
            return

        # Late import: keep the worker boot path free of the bundle
        # loader's importlib pull-in cost when no bundles are involved.
        from papayya.runtime import _bundle_loader

        prior_handler = None
        watchdog_armed = max_duration is not None and max_duration > 0
        if watchdog_armed:
            prior_handler = signal.signal(signal.SIGALRM, _on_agent_alarm)
            signal.setitimer(signal.ITIMER_REAL, max_duration)
        try:
            # Activate the version's bundle finder so function-body
            # imports (``def fn(): from helpers import x``) resolve
            # against the right version's siblings. ``None`` is a
            # no-op so local-dev / LocalDispatcher leases pay nothing.
            with _bundle_loader.activate(lease.agent_version):
                fn(lease.item_id)
        except _AgentTimeout:
            duration_ms = int((time.monotonic() - started_at) * 1000)
            log.warning(
                "failed   %s item=%s duration=%dms category=timeout limit=%.2fs",
                short, lease.item_id, duration_ms, max_duration,
            )
            self._report_complete(
                lease.lease_id,
                status="failed",
                error=f"timeout: agent ran for >{max_duration}s",
                error_category="timeout",
            )
            return
        except Exception as exc:  # noqa: BLE001 — customer code; isolate
            duration_ms = int((time.monotonic() - started_at) * 1000)
            log.exception(
                "failed   %s item=%s duration=%dms",
                short, lease.item_id, duration_ms,
            )
            self._report_complete(
                lease.lease_id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            return
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

    def _invoke_async(
        self,
        *,
        fn: Any,
        lease: Lease,
        started_at: float,
        max_duration: float | None,
        short: str,
    ) -> None:
        """Run a coroutine ``fn(lease.item_id)`` to completion.

        Uses ``asyncio.wait_for`` for timeout enforcement instead of the
        sync path's SIGALRM watchdog. Signal handlers raising into a
        running event loop can leave the loop in inconsistent state;
        ``wait_for`` cancels the inner coroutine cleanly so any
        ``finally`` / cleanup blocks the agent installed run before we
        report failure.

        Four terminal paths:
          - Success: report completed.
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
          - Any other ``Exception``: existing stringified-error path.
        """
        from papayya.runtime import _bundle_loader

        coro = fn(lease.item_id)
        try:
            # Same activate-scope rationale as the sync path; mirrored
            # here because ``asyncio.run`` runs the coroutine on a new
            # loop and we want imports inside it to see the right
            # version's siblings.
            with _bundle_loader.activate(lease.agent_version):
                if max_duration is not None and max_duration > 0:
                    asyncio.run(asyncio.wait_for(coro, timeout=max_duration))
                else:
                    asyncio.run(coro)
        except asyncio.TimeoutError:
            duration_ms = int((time.monotonic() - started_at) * 1000)
            log.warning(
                "failed   %s item=%s duration=%dms category=timeout limit=%.2fs",
                short, lease.item_id, duration_ms, max_duration,
            )
            self._report_complete(
                lease.lease_id,
                status="failed",
                error=f"timeout: agent ran for >{max_duration}s",
                error_category="timeout",
            )
            return
        except asyncio.CancelledError:
            duration_ms = int((time.monotonic() - started_at) * 1000)
            log.warning(
                "failed   %s item=%s duration=%dms category=cancelled",
                short, lease.item_id, duration_ms,
            )
            self._report_complete(
                lease.lease_id,
                status="failed",
                error="cancelled: asyncio.CancelledError",
                error_category="cancelled",
            )
            return
        except Exception as exc:  # noqa: BLE001 — customer code; isolate
            duration_ms = int((time.monotonic() - started_at) * 1000)
            log.exception(
                "failed   %s item=%s duration=%dms",
                short, lease.item_id, duration_ms,
            )
            self._report_complete(
                lease.lease_id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            return

        duration_ms = int((time.monotonic() - started_at) * 1000)
        log.info(
            "finished %s item=%s duration=%dms",
            short, lease.item_id, duration_ms,
        )
        self._report_complete(lease.lease_id, status="completed")

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
        try:
            with urllib_request.urlopen(req, timeout=2.0):
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
