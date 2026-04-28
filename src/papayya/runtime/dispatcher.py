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


class LocalDispatcher:
    """In-memory HTTP dispatcher for local development.

    Thread-safe: all state mutations go through ``self._lock``. The HTTP
    handler runs in dispatcher-server threads; in-process callers
    (tests, the CLI) acquire the same lock.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        on_event: Callable[[str, dict], None] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._pending: deque[_PendingItem] = deque()
        self._leased: dict[str, _PendingItem] = {}
        self._completed: dict[str, dict[str, Any]] = {}
        self._enqueued_total = 0
        self._on_event = on_event or (lambda _kind, _data: None)

        self._server = ThreadingHTTPServer((host, port), self._handler_factory())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

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
    ) -> str:
        """Enqueue one item. Returns the lease_id (used as completion handle)."""
        lease_id = uuid.uuid4().hex
        item = _PendingItem(lease_id=lease_id, agent=agent, item_id=item_id, payload=payload)
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

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/lease":
                    qs = parse_qs(parsed.query)
                    worker_id = (qs.get("worker_id") or ["unknown"])[0]
                    lease = dispatcher._take_lease(worker_id)
                    if lease is None:
                        self.send_response(204)
                        self.end_headers()
                        return
                    self._send_json(200, {
                        "lease_id": lease.lease_id,
                        "agent": lease.agent,
                        "item_id": lease.item_id,
                        "payload": lease.payload,
                    })
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
                    )
                    self._send_json(200, {"lease_id": lease_id})
                    return

                if parsed.path == "/complete":
                    body = self._read_json()
                    if body is None or "lease_id" not in body:
                        self._send_json(400, {"error": "lease_id required"})
                        return
                    dispatcher._mark_complete(
                        lease_id=body["lease_id"],
                        status=body.get("status", "completed"),
                        error=body.get("error"),
                        worker_id=body.get("worker_id"),
                    )
                    self._send_json(200, None)
                    return

                self._send_json(404, {"error": "not found"})

        return _Handler

    # --- internal state transitions ----------------------------------- #

    def _take_lease(self, worker_id: str) -> _PendingItem | None:
        with self._lock:
            if not self._pending:
                return None
            item = self._pending.popleft()
            self._leased[item.lease_id] = item
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
    ) -> None:
        with self._lock:
            self._leased.pop(lease_id, None)
            self._completed[lease_id] = {"status": status, "error": error}
        self._on_event("completed", {
            "lease_id": lease_id,
            "status": status,
            "error": error,
            "worker_id": worker_id,
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
        return (
            f"[{ts}] {marker} completed {data['lease_id'][:8]}  "
            f"status={data['status']}  worker={data.get('worker_id')}{suffix}"
        )
    return f"[{ts}] {kind} {data}"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    def emit(kind: str, data: dict) -> None:
        print(_format_event(kind, data), flush=True)

    d = LocalDispatcher(host=args.host, port=args.port, on_event=emit)

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
