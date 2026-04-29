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

import importlib.util
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


@dataclass
class Lease:
    """One unit of work assigned to this worker by the dispatcher."""
    lease_id: str
    agent: str
    item_id: str
    payload: dict[str, Any] | None = None


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
    ) -> None:
        self.dispatcher_url = dispatcher_url.rstrip("/")
        self.store_path = store_path
        self.worker_id = worker_id or f"w-{uuid.uuid4().hex[:8]}"
        self.poll_idle_seconds = poll_idle_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
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
        log.info("worker %s received signal %s; shutting down", self.worker_id, signum)
        self._running = False

    # --- dispatcher I/O ------------------------------------------------ #

    def _poll_lease(self) -> tuple[str, Lease | None]:
        """Poll the dispatcher for one lease.

        Returns a (outcome, lease) tuple. The outcome distinguishes
        "no work right now" (IDLE) from "couldn't reach the dispatcher"
        (UNREACHABLE) so the caller can apply different sleep policies —
        the latter triggers exponential backoff.
        """
        url = f"{self.dispatcher_url}/lease?worker_id={self.worker_id}"
        try:
            with urllib_request.urlopen(url, timeout=2.0) as resp:
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
            headers={"Content-Type": "application/json"},
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
        """
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
            headers={"Content-Type": "application/json"},
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
