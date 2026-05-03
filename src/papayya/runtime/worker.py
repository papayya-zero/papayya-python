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
from typing import Any, Optional
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
        agent_module_path: str,
        worker_id: Optional[str] = None,
        poll_idle_seconds: float = 0.05,
        heartbeat_interval_seconds: float = _DEFAULT_HEARTBEAT_INTERVAL,
        drain_timeout_seconds: float = _DEFAULT_DRAIN_TIMEOUT_SECONDS,
        api_key: Optional[str] = None,
    ) -> None:
        self.dispatcher_url = dispatcher_url.rstrip("/")
        self.store_path = store_path
        self.worker_id = worker_id or f"w-{uuid.uuid4().hex[:8]}"
        self.poll_idle_seconds = poll_idle_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.drain_timeout_seconds = drain_timeout_seconds
        # Sent as X-Api-Key on lease/complete/heartbeat. Matches the
        # dispatcher's API-key middleware (control-pane auth.go) which
        # requires a project-scoped key — JWT Bearer tokens are rejected
        # for runtime endpoints. None = no header (LocalDispatcher accepts).
        self._api_key = api_key
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

        # Point the customer's papayya() client at our shared SQLite. Must be
        # set BEFORE importing the agent module — the customer code may
        # call `papayya()` at module top-level (rare but legal).
        os.environ["PAPAYYA_LOCAL_DB_PATH"] = store_path
        # Ensure CloudStore isn't picked up if a stray PAPAYYA_API_KEY is in
        # env from the parent shell.
        os.environ.pop("PAPAYYA_API_KEY", None)

        self._import_agent_module(agent_module_path)

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

    # --- main loop ----------------------------------------------------- #

    def run(self) -> None:
        """Block, pulling items from the dispatcher, until stopped."""
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
            registration = get_agent(lease.agent)
            if registration is None:
                duration_ms = int((time.monotonic() - started_at) * 1000)
                log.warning(
                    "failed   %s item=%s duration=%dms unknown-agent=%s",
                    short, lease.item_id, duration_ms, lease.agent,
                )
                self._report_complete(
                    lease.lease_id,
                    status="failed",
                    error=f"unknown agent: {lease.agent}",
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

        prior_handler = None
        watchdog_armed = max_duration is not None and max_duration > 0
        if watchdog_armed:
            prior_handler = signal.signal(signal.SIGALRM, _on_agent_alarm)
            signal.setitimer(signal.ITIMER_REAL, max_duration)
        try:
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
        coro = fn(lease.item_id)
        try:
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
