"""Plan 56 F2 — the heartbeat survives a step that holds the GIL.

This is the defect plan 55 D1 measured, reproduced at the level that caused
it. A background THREAD gets ~1% of its due ticks while the main thread is
inside a C loop; the lease then expires, the reaper re-queues the item, and a
second worker starts the same document again. Three concurrent 135-second
executions of one customer function, all reported `ok`.

TWO THINGS ABOUT THIS HARNESS, both learned by getting them wrong first:

  * The mock dispatcher runs in its OWN PROCESS. The first version served it
    from a thread of the test process, which is the process holding the GIL —
    it recorded 0 heartbeats and blamed the child. A mock that shares a GIL
    with the blocked code cannot measure whether the child escaped it.

  * The blocking work is a real catastrophic-backtracking regex, not a stub.
    `time.sleep` releases the GIL, which is precisely why plan 55's DOC-1005
    (45s of sleep, over the 30s TTL) survived and DOC-1008 did not.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

import pytest

# Runs ~15-20s of GIL-held compute. Kept at module scope so the value is
# visible next to the interval it is measured against.
_BACKTRACK = (r"(a+)+$", "a" * 29 + "b")
_INTERVAL = 0.25

_SERVER = r"""
import http.server, json, socketserver, sys, time
OUT = open(sys.argv[1], "a", buffering=1)
STATUS = int(sys.argv[2])
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        OUT.write(json.dumps({"t": time.time(), "lease_id": body.get("lease_id")}) + "\n")
        self.send_response(STATUS); self.end_headers()
    def log_message(self, *a): pass
srv = socketserver.TCPServer(("127.0.0.1", 0), H)
print(srv.server_address[1], flush=True)
srv.serve_forever()
"""


@pytest.fixture
def dispatcher(tmp_path):
    """A mock /heartbeat endpoint in a separate process. Yields a factory."""
    procs = []

    def start(status: int = 200):
        hits = tmp_path / f"hits-{status}-{len(procs)}.jsonl"
        p = subprocess.Popen(
            [sys.executable, "-c", _SERVER, str(hits), str(status)],
            stdout=subprocess.PIPE, text=True,
        )
        procs.append(p)
        port = int(p.stdout.readline().strip())
        return port, hits

    yield start
    for p in procs:
        p.kill()


def _start_child(port: int, interval: float = _INTERVAL) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "papayya.runtime.heartbeat",
         "--dispatcher-url", f"http://127.0.0.1:{port}",
         "--worker-id", "w-test",
         "--interval-seconds", str(interval)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
    )


def _hits(path, since: float = 0.0) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return [r for r in rows if r["t"] > since]


@pytest.mark.slow
def test_heartbeat_survives_a_step_that_holds_the_gil(dispatcher):
    port, hits = dispatcher()
    child = _start_child(port)
    try:
        child.stdin.write("lease-abc\n")
        child.stdin.flush()
        time.sleep(_INTERVAL * 2)  # let it pick the lease up

        started = time.time()
        re.match(*_BACKTRACK)  # the main thread is now wedged in C
        elapsed = time.time() - started

        during = _hits(hits, since=started)
        due = int(elapsed / _INTERVAL)
        pct = 100 * len(during) / max(due, 1)

        # The thread form scores ~1% here. Anything near that means the
        # heartbeat is back inside the GIL and the D1 loop is live again.
        assert pct >= 80, (
            f"heartbeat starved during {elapsed:.1f}s of GIL-held compute: "
            f"{len(during)} of {due} due ticks ({pct:.0f}%)"
        )
        assert all(h["lease_id"] == "lease-abc" for h in during)
    finally:
        child.kill()


def test_idle_sentinel_stops_heartbeating_a_released_lease(dispatcher):
    # A heartbeat that outlives its lease tells the dispatcher a dead item is
    # healthy — worse than no heartbeat, because it defeats the reaper.
    port, hits = dispatcher()
    child = _start_child(port)
    try:
        child.stdin.write("lease-abc\n")
        child.stdin.flush()
        time.sleep(_INTERVAL * 3)
        assert _hits(hits), "no heartbeats at all — the harness is broken"

        child.stdin.write("-\n")
        child.stdin.flush()
        time.sleep(_INTERVAL)
        mark = time.time()
        time.sleep(_INTERVAL * 4)

        assert not _hits(hits, since=mark), "kept heartbeating a released lease"
    finally:
        child.kill()


def test_stdin_eof_is_the_death_contract(dispatcher):
    # EOF, not a pidfile or a parent-pid poll, because the kernel delivers it
    # whether the parent exited cleanly, called os._exit (the drain watchdog
    # does), or was SIGKILLed.
    port, _ = dispatcher()
    child = _start_child(port)
    child.stdin.write("lease-abc\n")
    child.stdin.flush()
    time.sleep(_INTERVAL * 2)

    child.stdin.close()
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        pytest.fail("heartbeat process survived stdin EOF")
    assert child.returncode == 0


def test_reports_a_revoked_lease_to_the_parent_exactly_once(dispatcher):
    # The parent uses this to drop in-flight tracking so a late /complete for
    # a stolen item is not reported. Repeat suppression matters because the
    # child keeps ticking: without it every interval would emit another line
    # and the parent's reader would log the same warning forever.
    port, _ = dispatcher(status=410)
    child = _start_child(port)
    try:
        child.stdin.write("lease-gone\n")
        child.stdin.flush()
        assert child.stdout.readline().strip() == "gone lease-gone"

        time.sleep(_INTERVAL * 4)
        child.stdin.close()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
        assert child.stdout.read().strip() == "", "re-reported the same revoked lease"
    finally:
        child.kill()
