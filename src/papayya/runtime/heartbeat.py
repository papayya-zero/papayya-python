"""Out-of-process lease heartbeat (plan 56 F2).

WHY THIS IS NOT A THREAD ANY MORE, which is the only thing worth knowing
about this file.

The worker used to heartbeat from a background thread. That works for
I/O-bound steps and fails completely for CPU-bound ones, because a Python
thread cannot run while another thread holds the GIL inside a C loop.
Measured across three kinds of work (plans/56-gil-matrix.py in papayya-zero):

    sleep (I/O-like)       31 of  32 due ticks ( 97%)
    re backtracking         1 of  71 due ticks (  1%)
    big-int arithmetic      1 of  17 due ticks (  6%)

One percent. Catastrophic backtracking over extracted text is an ordinary
production hang in a document pipeline, and the consequence was not a missing
graph line: the lease expired, the dispatcher's reaper re-queued the item, and
ANOTHER worker started the same document again. Plan 55 D1 measured three
concurrent 135-second executions of one customer function, all reported `ok`.

A separate process has its own interpreter and its own GIL, so no amount of
customer compute can starve it. That is the entire design.

WHY NOT SIGALRM, which is immune to the same starvation and is already used
for the invocation watchdog. A signal handler runs on the main thread, and it
can fire while that thread is inside httpx or urllib — reentrant socket use on
the connection the handler would need. The one case where an in-process
heartbeat is safe (the main thread is blocked in I/O, so the GIL is free) is
exactly the case where the old thread already worked.

PROTOCOL, deliberately the dumbest thing that survives a kill -9:

    parent → child (stdin, one line each)   "<lease_id>"   now heartbeating this
                                            "-"            idle, nothing in flight
    child → parent (stdout, one line each)  "gone <lease_id>"   dispatcher said 409/410

    parent dies → stdin EOF → child exits.

EOF is the death contract because it is delivered by the kernel whether the
parent exited cleanly, called os._exit (which the drain watchdog does), or was
SIGKILLed. A pidfile or a parent-pid poll misses the last case for a full
interval, and a heartbeat that outlives its worker is worse than none: it
tells the dispatcher a dead item is healthy.

The child imports nothing from the rest of the SDK. It must start even when
the customer's bundle is unimportable, and it must not pull the bundle loader
into a second process.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error as urllib_error
import urllib.request as urllib_request

# The sentinel the parent writes when no lease is in flight. A literal
# rather than an empty line so a stray newline cannot be mistaken for it.
IDLE = "-"


def _post_heartbeat(
    *,
    dispatcher_url: str,
    lease_id: str,
    worker_id: str,
    api_key: str | None,
    timeout: float,
) -> int | None:
    """POST one heartbeat. Returns the HTTP error code, or None on success.

    Never raises. A transient network failure is not this process's problem to
    solve — the dispatcher's reaper is the backstop, and an exception here
    would kill the only thing keeping every subsequent lease alive.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-Api-Key"] = api_key
    body = json.dumps({"lease_id": lease_id, "worker_id": worker_id}).encode("utf-8")
    req = urllib_request.Request(
        f"{dispatcher_url}/heartbeat", data=body, headers=headers, method="POST"
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout):
            return None
    except urllib_error.HTTPError as exc:
        return exc.code
    except Exception:  # noqa: BLE001 — see docstring
        return None


def _read_stdin(state: dict, stop: threading.Event) -> None:
    """Track the parent's current lease until stdin closes.

    Reading blocks, so it gets its own thread — safe here because this process
    runs no customer code and therefore never holds the GIL for long. The
    starvation this whole module exists to escape is a property of the WORKER's
    process, not of threads as such.
    """
    try:
        for line in sys.stdin:
            value = line.strip()
            state["lease_id"] = None if value == IDLE or not value else value
    finally:
        # EOF: the parent is gone.
        stop.set()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="papayya-heartbeat")
    ap.add_argument("--dispatcher-url", required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--interval-seconds", type=float, default=5.0)
    ap.add_argument("--timeout-seconds", type=float, default=5.0)
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args(argv)

    state: dict = {"lease_id": None}
    stop = threading.Event()
    reader = threading.Thread(target=_read_stdin, args=(state, stop), daemon=True)
    reader.start()

    # The lease ids already reported gone, so a parent that has not yet
    # processed our stdout line doesn't make us re-report every interval.
    reported: set[str] = set()

    while not stop.wait(timeout=args.interval_seconds):
        lease_id = state["lease_id"]
        if not lease_id:
            continue
        code = _post_heartbeat(
            dispatcher_url=args.dispatcher_url,
            lease_id=lease_id,
            worker_id=args.worker_id,
            api_key=args.api_key,
            # Never longer than one interval: a heartbeat that hasn't landed
            # by the time the next is due has been superseded, and a long
            # budget here stretches the cadence toward the TTL it exists to
            # stay under. Same reasoning as the thread it replaces.
            timeout=min(args.interval_seconds, args.timeout_seconds),
        )
        if code in (409, 410) and lease_id not in reported:
            reported.add(lease_id)
            try:
                sys.stdout.write(f"gone {lease_id}\n")
                sys.stdout.flush()
            except Exception:  # noqa: BLE001 — parent's pipe closed; we exit next tick
                pass

    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(main())
