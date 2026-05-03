"""LocalDispatcher — local-development dispatcher for the worker runtime.

This is the *local-only* dispatcher. It runs an HTTP server on
localhost that:

  • accepts items via POST /enqueue (or .enqueue() in-process)
  • leases items to workers via GET /lease?worker_id=X
  • records completion via POST /complete
  • exposes a snapshot via GET /stats

Workers connect to it the same way they will eventually connect to the
hosted control-pane dispatcher — same wire protocol, same lease/complete
semantics. Phase 2 builds the production dispatcher in control-pane;
Phase 3 swaps workers' --dispatcher URL to point at it. The protocol is
the contract between the two; this class is one implementation.

Run as a CLI:

    python -m papayya.runtime.dispatcher --port 8765 --enqueue enrich:a,b,c

In-process (tests, scripts):

    d = LocalDispatcher(port=0)            # 0 picks a random port
    d.enqueue(agent="enrich", item_id="co_42")
    d.wait_until_drained(timeout=10)
    d.shutdown()
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse


log = logging.getLogger("papayya.runtime.dispatcher")


@dataclass
class _PendingItem:
    lease_id: str
    agent: str
    item_id: str
    payload: dict | None
    # Optional bundle version tag. Local dev typically leaves this None
    # (the worker boots from --agent-module FILE and is version-unaware);
    # set when a test or external caller exercises the hosted code-
    # distribution wire shape. ADR-0003 § 1.
    agent_version: str | None = None


@dataclass
class _LeasedRecord:
    """A leased item plus the bookkeeping needed for TTL-based recovery.

    ``leased_at`` is the lease grant time. ``last_heartbeat`` advances
    on POST /heartbeat from the lease's owning worker; the reaper
    releases the lease when ``now - last_heartbeat`` exceeds the TTL.
    """
    item: _PendingItem
    worker_id: str
    leased_at: float       # time.monotonic()
    last_heartbeat: float  # time.monotonic()


# Default lease TTL: long enough to absorb a slow step + a few missed
# heartbeats; short enough that worker death recovers in under a minute.
# Phase 2 ADR-0002 #1 calls for "30–60 seconds"; defaulting to 30 here so
# the reaper releases dead leases within roughly the polling window the
# operator already observes in the dispatcher event log.
_DEFAULT_LEASE_TTL_SECONDS = 30.0


class LocalDispatcher:
    """In-memory HTTP dispatcher for local development.

    Thread-safe: all state mutations go through ``self._lock``. The HTTP
    handler runs in dispatcher-server threads; in-process callers
    (tests, the CLI) acquire the same lock.

    Lease TTL + heartbeats (Phase 2 ADR-0002 #1):
      • Workers heartbeat to ``POST /heartbeat`` every few seconds while a
        lease is in flight. Dispatcher updates the lease's
        ``last_heartbeat`` timestamp.
      • A reaper thread scans leased items every ``heartbeat_check_interval``
        seconds and releases any lease whose heartbeat is older than
        ``lease_ttl_seconds``.
      • Released items are re-queued at the *front* of pending with a
        *fresh* lease_id. The fresh ID means a zombie worker's late
        ``/complete`` becomes an unknown lease and gets dropped — it
        cannot collide with the lease the next worker takes.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        on_event: Callable[[str, dict], None] | None = None,
        lease_ttl_seconds: float = _DEFAULT_LEASE_TTL_SECONDS,
        heartbeat_check_interval: float | None = None,
        expected_api_key: str | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._pending: deque[_PendingItem] = deque()
        self._leased: dict[str, _LeasedRecord] = {}
        self._completed: dict[str, dict[str, Any]] = {}
        self._enqueued_total = 0
        self._on_event = on_event or (lambda _kind, _data: None)
        # When set, lease/complete/heartbeat require X-Api-Key to match.
        # Mirrors the hosted dispatcher's API-key middleware. Default
        # None preserves the unauthenticated local-dev surface.
        self._expected_api_key = expected_api_key

        self._lease_ttl = lease_ttl_seconds
        self._heartbeat_check_interval = (
            heartbeat_check_interval
            if heartbeat_check_interval is not None
            else max(0.1, lease_ttl_seconds / 4)
        )

        self._server = ThreadingHTTPServer((host, port), self._handler_factory())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

        # Reaper thread runs after the server is listening so its events
        # never refer to an HTTP path that doesn't exist yet.
        self._reaper_stop = threading.Event()
        self._reaper_thread = threading.Thread(
            target=self._reaper_loop,
            daemon=True,
            name="papayya-dispatcher-reaper",
        )
        self._reaper_thread.start()

    # --- lifecycle ----------------------------------------------------- #

    @property
    def url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    @property
    def host(self) -> str:
        return self._server.server_address[0]

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def shutdown(self) -> None:
        self._reaper_stop.set()
        self._reaper_thread.join(timeout=2)
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)

    # --- in-process API ------------------------------------------------ #

    def enqueue(
        self,
        *,
        agent: str,
        item_id: str,
        payload: dict | None = None,
        agent_version: str | None = None,
    ) -> str:
        """Enqueue one item. Returns the lease_id (used as completion handle)."""
        lease_id = uuid.uuid4().hex
        item = _PendingItem(
            lease_id=lease_id,
            agent=agent,
            item_id=item_id,
            payload=payload,
            agent_version=agent_version,
        )
        with self._lock:
            self._pending.append(item)
            self._enqueued_total += 1
        self._on_event("enqueued", {"lease_id": lease_id, "agent": agent, "item_id": item_id})
        return lease_id

    def wait_until_drained(self, timeout: float = 5.0) -> None:
        """Block until every enqueued item has reached terminal status.

        Raises TimeoutError if not drained within ``timeout`` seconds.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                drained = (
                    not self._pending
                    and not self._leased
                    and len(self._completed) >= self._enqueued_total
                )
            if drained:
                return
            time.sleep(0.02)
        with self._lock:
            raise TimeoutError(
                f"dispatcher did not drain within {timeout}s: "
                f"pending={len(self._pending)} leased={len(self._leased)} "
                f"completed={len(self._completed)} enqueued={self._enqueued_total}"
            )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enqueued_total": self._enqueued_total,
                "pending": len(self._pending),
                "leased": len(self._leased),
                "completed": sum(1 for v in self._completed.values() if v["status"] == "completed"),
                "failed": sum(1 for v in self._completed.values() if v["status"] != "completed"),
            }

    def completed_count(self) -> int:
        return self.stats()["completed"]

    def failed(self) -> list[tuple[str, str]]:
        with self._lock:
            return [
                (k, v.get("error", "") or "")
                for k, v in self._completed.items()
                if v["status"] != "completed"
            ]

    # --- HTTP handler -------------------------------------------------- #

    def _handler_factory(self):
        dispatcher = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                # Quiet by default; the dispatcher emits its own structured
                # events via on_event. The CLI subscribes to those.
                return

            def _send_json(self, status: int, body: dict | list | None) -> None:
                if body is None:
                    self.send_response(status)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                payload = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _read_json(self) -> dict | None:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    return json.loads(raw.decode("utf-8"))
                except ValueError:
                    return None

            def _check_auth(self) -> bool:
                """Return True iff the request passes the API-key check.

                When the dispatcher has no expected key, every request
                passes (default local-dev posture). When set, the
                request's X-Api-Key must match exactly. Mismatch writes
                a 401 and returns False so the caller can early-exit.
                """
                expected = dispatcher._expected_api_key
                if expected is None:
                    return True
                provided = self.headers.get("X-Api-Key", "")
                if provided == expected:
                    return True
                self._send_json(401, {"error": "invalid api key"})
                return False

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/lease":
                    if not self._check_auth():
                        return
                    qs = parse_qs(parsed.query)
                    worker_id = (qs.get("worker_id") or ["unknown"])[0]
                    lease = dispatcher._take_lease(worker_id)
                    if lease is None:
                        self.send_response(204)
                        self.end_headers()
                        return
                    body = {
                        "lease_id": lease.lease_id,
                        "agent": lease.agent,
                        "item_id": lease.item_id,
                        "payload": lease.payload,
                    }
                    # Mirror the control-pane wire shape: omit the key
                    # entirely when unset rather than emitting a null.
                    # Keeps legacy callers' parsed-body shape unchanged.
                    if lease.agent_version is not None:
                        body["agent_version"] = lease.agent_version
                    self._send_json(200, body)
                    return

                if parsed.path == "/stats":
                    self._send_json(200, dispatcher.stats())
                    return

                self._send_json(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)

                if parsed.path == "/enqueue":
                    body = self._read_json()
                    if body is None or "agent" not in body or "item_id" not in body:
                        self._send_json(400, {"error": "agent and item_id required"})
                        return
                    lease_id = dispatcher.enqueue(
                        agent=body["agent"],
                        item_id=body["item_id"],
                        payload=body.get("payload"),
                        agent_version=body.get("agent_version"),
                    )
                    self._send_json(200, {"lease_id": lease_id})
                    return

                if parsed.path == "/complete":
                    if not self._check_auth():
                        return
                    body = self._read_json()
                    if body is None or "lease_id" not in body:
                        self._send_json(400, {"error": "lease_id required"})
                        return
                    dispatcher._mark_complete(
                        lease_id=body["lease_id"],
                        status=body.get("status", "completed"),
                        error=body.get("error"),
                        worker_id=body.get("worker_id"),
                        error_category=body.get("error_category"),
                    )
                    self._send_json(200, None)
                    return

                if parsed.path == "/heartbeat":
                    if not self._check_auth():
                        return
                    body = self._read_json()
                    if body is None or "lease_id" not in body or "worker_id" not in body:
                        self._send_json(400, {"error": "lease_id and worker_id required"})
                        return
                    outcome = dispatcher._record_heartbeat(
                        lease_id=body["lease_id"],
                        worker_id=body["worker_id"],
                    )
                    if outcome == "ok":
                        self._send_json(200, None)
                    elif outcome == "wrong_worker":
                        # 409: another worker holds this lease (zombie
                        # double-claim). The heartbeating worker should
                        # drop its in-flight tracking.
                        self._send_json(409, {"error": "lease held by another worker"})
                    else:  # "unknown"
                        # 410: lease released (reaper) or never existed.
                        # The worker should clear its in-flight state.
                        self._send_json(410, {"error": "lease no longer recognized"})
                    return

                self._send_json(404, {"error": "not found"})

        return _Handler

    # --- internal state transitions ----------------------------------- #

    def _take_lease(self, worker_id: str) -> _PendingItem | None:
        with self._lock:
            if not self._pending:
                return None
            item = self._pending.popleft()
            now = time.monotonic()
            self._leased[item.lease_id] = _LeasedRecord(
                item=item,
                worker_id=worker_id,
                leased_at=now,
                last_heartbeat=now,
            )
        self._on_event("leased", {
            "lease_id": item.lease_id,
            "agent": item.agent,
            "item_id": item.item_id,
            "worker_id": worker_id,
        })
        return item

    def _mark_complete(
        self,
        *,
        lease_id: str,
        status: str,
        error: str | None,
        worker_id: str | None,
        error_category: str | None = None,
    ) -> bool:
        """Record completion. Returns True if the lease was recognized.

        A False return means the lease was already released (typically by
        the reaper after a heartbeat timeout); the caller's /complete is
        a stale write from a zombie worker and gets dropped. The
        ``stale_complete`` event surfaces it so operators see the drop.
        """
        with self._lock:
            record = self._leased.pop(lease_id, None)
            if record is None:
                self._on_event("stale_complete", {
                    "lease_id": lease_id,
                    "status": status,
                    "worker_id": worker_id,
                })
                return False
            self._completed[lease_id] = {
                "status": status,
                "error": error,
                "error_category": error_category,
            }
            # Computed dispatcher-side from leased_at to keep the
            # measurement single-clock — worker clock skew can't
            # mislead operators reading the event log.
            duration_ms = int((time.monotonic() - record.leased_at) * 1000)
        self._on_event("completed", {
            "lease_id": lease_id,
            "status": status,
            "error": error,
            "error_category": error_category,
            "worker_id": worker_id,
            "duration_ms": duration_ms,
        })
        return True

    def _record_heartbeat(self, *, lease_id: str, worker_id: str) -> str:
        """Update last_heartbeat for a lease.

        Returns a status string the HTTP layer maps to a code:
          "ok"            → 200, lease recognized and refreshed
          "unknown"       → 410 Gone, lease already released or never existed
          "wrong_worker"  → 409 Conflict, lease held by a different worker
        """
        with self._lock:
            record = self._leased.get(lease_id)
            if record is None:
                return "unknown"
            if record.worker_id != worker_id:
                return "wrong_worker"
            record.last_heartbeat = time.monotonic()
            return "ok"

    def _reaper_loop(self) -> None:
        """Background loop: release leases past TTL, re-issue with fresh ID."""
        while not self._reaper_stop.is_set():
            if self._reaper_stop.wait(timeout=self._heartbeat_check_interval):
                return
            try:
                self._reap_expired()
            except Exception as exc:  # noqa: BLE001
                # Reaper must never die — that would silently disable
                # the recovery mechanism. Log and continue.
                log.warning("reaper iteration failed: %s", exc)

    def _reap_expired(self) -> None:
        now = time.monotonic()
        # Collect-then-emit pattern: release under the lock, fire events
        # outside it. ``on_event`` is user-supplied and may block.
        expired: list[tuple[str, str, _LeasedRecord, float]] = []
        with self._lock:
            for old_lease_id in list(self._leased.keys()):
                record = self._leased[old_lease_id]
                age = now - record.last_heartbeat
                if age <= self._lease_ttl:
                    continue
                self._leased.pop(old_lease_id)
                # Fresh lease_id so a zombie worker's late /complete
                # cannot collide with the lease the next worker takes.
                new_lease_id = uuid.uuid4().hex
                new_item = _PendingItem(
                    lease_id=new_lease_id,
                    agent=record.item.agent,
                    item_id=record.item.item_id,
                    payload=record.item.payload,
                )
                # Front of the deque — re-process ASAP. The item already
                # waited for the reaper; don't make it wait behind newer
                # arrivals too.
                self._pending.appendleft(new_item)
                expired.append((old_lease_id, new_lease_id, record, age))
        for old_lease_id, new_lease_id, record, age in expired:
            self._on_event("lease_expired", {
                "old_lease_id": old_lease_id,
                "new_lease_id": new_lease_id,
                "worker_id": record.worker_id,
                "agent": record.item.agent,
                "item_id": record.item.item_id,
                "age_s": age,
            })


# --------------------------------------------------------------------------- #
#  CLI: python -m papayya.runtime.dispatcher                                   #
# --------------------------------------------------------------------------- #

def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog="python -m papayya.runtime.dispatcher",
        description="Run a local Papayya dispatcher for development workers.",
    )
    p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    p.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765; use 0 for random).")
    p.add_argument(
        "--enqueue",
        action="append",
        default=[],
        metavar="AGENT:id1,id2,...",
        help="Initial batch to enqueue at startup. Repeatable. Example: --enqueue enrich:co_42,co_43",
    )
    p.add_argument("--log-level", default="INFO")
    p.add_argument(
        "--lease-ttl-seconds",
        type=float,
        default=_DEFAULT_LEASE_TTL_SECONDS,
        help=(
            "Maximum seconds without a worker heartbeat before the dispatcher "
            f"releases a lease and re-queues the item (default: {_DEFAULT_LEASE_TTL_SECONDS})."
        ),
    )
    return p


def _format_event(kind: str, data: dict) -> str:
    ts = time.strftime("%H:%M:%S")
    if kind == "enqueued":
        return f"[{ts}] enqueued  {data['lease_id'][:8]}  agent={data['agent']}  item={data['item_id']}"
    if kind == "leased":
        return (
            f"[{ts}] leased    {data['lease_id'][:8]}  agent={data['agent']}  "
            f"item={data['item_id']}  worker={data['worker_id']}"
        )
    if kind == "completed":
        marker = "✓" if data["status"] == "completed" else "✗"
        suffix = f" error={data['error']}" if data.get("error") else ""
        duration = data.get("duration_ms")
        duration_field = f"  duration={duration}ms" if duration is not None else ""
        category = data.get("error_category")
        category_field = f"  category={category}" if category else ""
        return (
            f"[{ts}] {marker} completed {data['lease_id'][:8]}  "
            f"status={data['status']}  worker={data.get('worker_id')}"
            f"{category_field}{duration_field}{suffix}"
        )
    if kind == "lease_expired":
        return (
            f"[{ts}] ⏰ expired   {data['old_lease_id'][:8]}  agent={data['agent']}  "
            f"item={data['item_id']}  worker={data['worker_id']}  "
            f"age={data['age_s']:.1f}s  re-leased={data['new_lease_id'][:8]}"
        )
    if kind == "stale_complete":
        return (
            f"[{ts}] ⚠️  stale     {data['lease_id'][:8]}  status={data['status']}  "
            f"worker={data.get('worker_id')}  (lease already released; drop)"
        )
    return f"[{ts}] {kind} {data}"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    def emit(kind: str, data: dict) -> None:
        print(_format_event(kind, data), flush=True)

    d = LocalDispatcher(
        host=args.host,
        port=args.port,
        on_event=emit,
        lease_ttl_seconds=args.lease_ttl_seconds,
    )

    print(f"Papayya local dispatcher listening on {d.url}", flush=True)
    print(f"  worker:  python -m papayya.runtime --agent-module FILE --dispatcher {d.url} --store /tmp/papayya.db", flush=True)
    print(f"  enqueue: curl -X POST {d.url}/enqueue -H 'Content-Type: application/json' \\", flush=True)
    print(f"             -d '{{\"agent\":\"NAME\",\"item_id\":\"co_42\"}}'", flush=True)
    print(f"  stats:   curl {d.url}/stats", flush=True)
    print("  Ctrl+C to stop.", flush=True)
    print("", flush=True)

    for spec in args.enqueue:
        if ":" not in spec:
            print(f"  WARN: ignoring --enqueue '{spec}' (expected AGENT:id1,id2)", flush=True)
            continue
        agent, ids_str = spec.split(":", 1)
        for item_id in [s.strip() for s in ids_str.split(",") if s.strip()]:
            d.enqueue(agent=agent.strip(), item_id=item_id)

    try:
        # Block forever; the server runs in a daemon thread.
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nshutting down...", flush=True)
        d.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
