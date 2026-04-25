"""FakeDispatcher — in-test HTTP dispatcher for the worker subprocess.

Implements the minimal protocol the worker speaks:

  GET  /lease?worker_id=X   -> 200 {lease_id, agent, item_id} | 204
  POST /complete            -> 200 {}, body {lease_id, status, error?}

Plus a Python API the test process uses to drive scenarios:

  d.enqueue(agent=..., item_id=...) -> lease_id
  d.wait_until_drained(timeout=...) -> blocks until empty + completed
  d.completed_count() -> int
  d.failed() -> list[(lease_id, error)]

Lifecycle: bind to localhost:0 (random port), serve in a daemon thread,
shut down cleanly when the fixture tears down.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse


class _PendingItem:
    __slots__ = ("lease_id", "agent", "item_id", "payload")

    def __init__(self, lease_id: str, agent: str, item_id: str, payload: dict | None) -> None:
        self.lease_id = lease_id
        self.agent = agent
        self.item_id = item_id
        self.payload = payload


class FakeDispatcher:
    """In-memory dispatcher with an HTTP front-end on localhost.

    Thread-safe: all state mutations go through ``self._lock``. The HTTP
    handler runs in dispatcher-server threads; the test process and worker
    subprocess both read/write through the same lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: deque[_PendingItem] = deque()
        self._leased: dict[str, _PendingItem] = {}
        self._completed: dict[str, dict[str, Any]] = {}
        self._enqueued_total = 0

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler_factory())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    # --- lifecycle ----------------------------------------------------- #

    @property
    def url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)

    # --- test API ------------------------------------------------------ #

    def enqueue(self, *, agent: str, item_id: str, payload: dict | None = None) -> str:
        """Enqueue one item. Returns the lease_id (also used as run id)."""
        lease_id = uuid.uuid4().hex
        item = _PendingItem(lease_id=lease_id, agent=agent, item_id=item_id, payload=payload)
        with self._lock:
            self._pending.append(item)
            self._enqueued_total += 1
        return lease_id

    def wait_until_drained(self, timeout: float = 5.0) -> None:
        """Block until all enqueued items have reached terminal status.

        Raises TimeoutError if not drained within ``timeout`` seconds.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if not self._pending and not self._leased and len(self._completed) >= self._enqueued_total:
                    return
            time.sleep(0.02)
        with self._lock:
            raise TimeoutError(
                f"dispatcher did not drain within {timeout}s: "
                f"pending={len(self._pending)} leased={len(self._leased)} "
                f"completed={len(self._completed)} enqueued={self._enqueued_total}"
            )

    def completed_count(self) -> int:
        with self._lock:
            return sum(1 for v in self._completed.values() if v["status"] == "completed")

    def failed(self) -> list[tuple[str, str]]:
        with self._lock:
            return [(k, v.get("error", "")) for k, v in self._completed.items() if v["status"] != "completed"]

    # --- HTTP handler -------------------------------------------------- #

    def _handler_factory(self):
        dispatcher = self

        class _Handler(BaseHTTPRequestHandler):
            # Suppress noisy access logs in test output.
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

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
                    body = json.dumps({
                        "lease_id": lease.lease_id,
                        "agent": lease.agent,
                        "item_id": lease.item_id,
                        "payload": lease.payload,
                    }).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                self.send_response(404)
                self.end_headers()

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/complete":
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length) if length else b"{}"
                    try:
                        body = json.loads(raw.decode("utf-8"))
                    except ValueError:
                        self.send_response(400)
                        self.end_headers()
                        return
                    dispatcher._mark_complete(
                        lease_id=body["lease_id"],
                        status=body.get("status", "completed"),
                        error=body.get("error"),
                    )
                    self.send_response(200)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return

                self.send_response(404)
                self.end_headers()

        return _Handler

    # --- internal state transitions ----------------------------------- #

    def _take_lease(self, worker_id: str) -> _PendingItem | None:
        with self._lock:
            if not self._pending:
                return None
            item = self._pending.popleft()
            self._leased[item.lease_id] = item
            return item

    def _mark_complete(self, *, lease_id: str, status: str, error: str | None) -> None:
        with self._lock:
            self._leased.pop(lease_id, None)
            self._completed[lease_id] = {"status": status, "error": error}
