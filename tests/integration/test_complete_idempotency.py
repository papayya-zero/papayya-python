"""Idempotent /complete + worker-side retry — Phase 2 ADR-0002 #4.

Two scenarios cover the partition-recovery story:

  1. The worker's POST /complete fails transiently (network blip).
     Worker retries with bounded backoff. Eventual delivery records
     exactly one completion. Without this, a single transient failure
     in the response window means the dispatcher reaps the lease
     (TTL) and the item gets re-dispatched even though the agent fn
     completed.

  2. The dispatcher receives /complete twice with the same lease_id
     (e.g. response was lost and the worker retried, or two zombie
     workers both racing to report). The second POST emits a
     stale_complete event but has no other side effect — no double
     accounting, no item leaking back into pending.

Together these prove "the worker can safely retry" + "the dispatcher
safely accepts the retry" — the two sides of #4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import json
import textwrap
import threading
import time

import pytest


@pytest.fixture
def noop_agent_module(tmp_path: Path) -> Path:
    """Trivial agent module the Worker can import on construction.

    The /complete tests don't actually run the agent fn — they exercise
    the report/dispatcher path directly. The module just needs to load
    cleanly so Worker.__init__ succeeds.
    """
    src = textwrap.dedent("""\
        from papayya import agent

        @agent(name="noop")
        def noop(item_id: str) -> dict:
            return {"id": item_id}
    """)
    path = tmp_path / "noop_agent.py"
    path.write_text(src)
    return path


def test_worker_retries_complete_on_transient_failure(
    tmp_path: Path,
    noop_agent_module: Path,
) -> None:
    """If the first /complete POST fails with URLError, the worker
    retries and the dispatcher records exactly one completion.

    Asserts:
      • Worker calls urlopen on /complete twice (1 failure + 1 success).
      • Dispatcher's `_completed` contains the lease_id.
      • Dispatcher's `_leased` no longer contains it.
      • `_pending` is empty (the item didn't re-queue — retry beat any
        TTL fire, which is the whole point of the bounded backoff).
      • Exactly one `completed` event fired; zero `stale_complete`
        (the failed first attempt never reached the dispatcher).
    """
    from papayya.runtime import worker as worker_module
    from papayya.runtime.dispatcher import LocalDispatcher
    from papayya.runtime.worker import Worker

    events: list[tuple[str, dict]] = []
    dispatcher = LocalDispatcher(
        host="127.0.0.1",
        port=0,
        on_event=lambda kind, data: events.append((kind, data)),
        lease_ttl_seconds=30.0,
    )
    worker: Worker | None = None
    original_urlopen = worker_module.urllib_request.urlopen
    try:
        store_path = str(tmp_path / "complete.db")
        worker = Worker(
            dispatcher_url=dispatcher.url,
            store_path=store_path,
            agent_module_path=str(noop_agent_module),
            worker_id="retry-w1",
        )

        dispatcher.enqueue(agent="noop", item_id="item-A")
        item = dispatcher._take_lease(worker_id="retry-w1")
        assert item is not None

        # Inject a one-shot failure on the FIRST POST to /complete.
        # urllib_request is a module-level alias inside worker.py, so
        # patching the symbol on that module is sufficient — the
        # Request constructed in _report_complete will call this stub.
        urlopen_calls: list[str] = []

        def flaky_urlopen(req: Any, *args: Any, **kwargs: Any) -> Any:
            url = getattr(req, "full_url", str(req))
            urlopen_calls.append(url)
            if "/complete" in url and len(
                [u for u in urlopen_calls if "/complete" in u]
            ) == 1:
                raise urllib_error.URLError("simulated transient")
            return original_urlopen(req, *args, **kwargs)

        worker_module.urllib_request.urlopen = flaky_urlopen
        try:
            worker._report_complete(item.lease_id, status="completed")
        finally:
            worker_module.urllib_request.urlopen = original_urlopen

        complete_calls = [u for u in urlopen_calls if "/complete" in u]
        assert len(complete_calls) == 2, (
            f"expected 2 /complete attempts (1 fail + 1 retry-success), "
            f"got {len(complete_calls)}: {complete_calls}"
        )

        with dispatcher._lock:
            assert item.lease_id in dispatcher._completed, (
                "retry succeeded but dispatcher state missing completion record"
            )
            assert item.lease_id not in dispatcher._leased
            assert len(dispatcher._pending) == 0, (
                "item should not have re-queued — retry beats TTL fire"
            )

        completed_events = [d for kind, d in events if kind == "completed"]
        stale_events = [d for kind, d in events if kind == "stale_complete"]
        assert len(completed_events) == 1, (
            f"expected exactly one completed event, got {completed_events}"
        )
        assert len(stale_events) == 0, (
            "first attempt failed at the network layer — no POST reached "
            f"the dispatcher, so no stale_complete should fire. Got: {stale_events}"
        )
        assert completed_events[0]["lease_id"] == item.lease_id
        assert completed_events[0]["status"] == "completed"
    finally:
        if worker is not None:
            worker._hb_stop.set()
        dispatcher.shutdown()


def test_dispatcher_double_complete_is_idempotent_and_emits_stale_event(
    tmp_path: Path,
) -> None:
    """A duplicate /complete with the same lease_id is safe.

    Models the response-lost scenario: dispatcher accepted the first
    POST, response never reached the worker, worker retried. The
    second POST must:
      • not re-add the item to pending
      • not double-count it in completed
      • emit a stale_complete event so operators see the duplicate

    This works because dispatcher._mark_complete pops from _leased; the
    second call finds nothing and returns without side effects beyond
    the event.
    """
    from papayya.runtime.dispatcher import LocalDispatcher

    events: list[tuple[str, dict]] = []
    dispatcher = LocalDispatcher(
        host="127.0.0.1",
        port=0,
        on_event=lambda kind, data: events.append((kind, data)),
        lease_ttl_seconds=30.0,
    )
    try:
        dispatcher.enqueue(agent="noop", item_id="item-B")
        item = dispatcher._take_lease(worker_id="dup-w1")
        assert item is not None

        body = json.dumps({
            "lease_id": item.lease_id,
            "status": "completed",
            "worker_id": "dup-w1",
        }).encode("utf-8")

        def post_complete() -> int:
            req = urllib_request.Request(
                f"{dispatcher.url}/complete",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib_request.urlopen(req, timeout=2.0) as resp:
                return resp.status

        # First POST: dispatcher records completion.
        assert post_complete() == 200
        # Second POST: lease already gone, dispatcher returns 200 + stale_complete event.
        assert post_complete() == 200

        with dispatcher._lock:
            assert item.lease_id in dispatcher._completed
            assert item.lease_id not in dispatcher._leased
            assert len(dispatcher._pending) == 0, (
                "item must not re-queue on duplicate /complete"
            )
            # Completion record stored exactly once.
            assert len(dispatcher._completed) == 1

        completed_events = [d for kind, d in events if kind == "completed"]
        stale_events = [d for kind, d in events if kind == "stale_complete"]
        assert len(completed_events) == 1, (
            f"expected exactly one completed event, got {len(completed_events)}"
        )
        assert len(stale_events) == 1, (
            f"expected one stale_complete event for the duplicate POST, "
            f"got {len(stale_events)}"
        )
        assert stale_events[0]["lease_id"] == item.lease_id
    finally:
        dispatcher.shutdown()


def test_complete_retry_exhausts_after_max_attempts_and_lease_ttl_recovers(
    tmp_path: Path,
    noop_agent_module: Path,
) -> None:
    """If every retry fails, the worker logs and gives up — but the
    item is not lost: the dispatcher's lease TTL will release it for
    re-dispatch. This is the documented safety-net behavior in
    ADR-0002 #4 ("at-least-once semantics are preserved either way").

    Asserts:
      • _report_complete returns without raising even when every
        urlopen call fails (the worker must keep running).
      • The dispatcher state is unchanged — lease still in _leased,
        _completed empty — because no /complete ever landed.
      • Subsequent reaper firing (TTL expiry) would re-queue the item,
        which test_lease_ttl.py already covers; we just confirm here
        that the lease is still alive and reapable.
    """
    from papayya.runtime import worker as worker_module
    from papayya.runtime.dispatcher import LocalDispatcher
    from papayya.runtime.worker import Worker

    events: list[tuple[str, dict]] = []
    dispatcher = LocalDispatcher(
        host="127.0.0.1",
        port=0,
        on_event=lambda kind, data: events.append((kind, data)),
        lease_ttl_seconds=60.0,  # long enough we never reap during the test
    )
    worker: Worker | None = None
    original_urlopen = worker_module.urllib_request.urlopen
    try:
        store_path = str(tmp_path / "exhaust.db")
        worker = Worker(
            dispatcher_url=dispatcher.url,
            store_path=store_path,
            agent_module_path=str(noop_agent_module),
            worker_id="exhaust-w1",
        )

        dispatcher.enqueue(agent="noop", item_id="item-C")
        item = dispatcher._take_lease(worker_id="exhaust-w1")
        assert item is not None

        # Every /complete attempt fails. urllib calls for /heartbeat
        # (from the worker's heartbeat thread) still pass through.
        complete_attempts = 0

        def always_fail_complete(req: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal complete_attempts
            url = getattr(req, "full_url", str(req))
            if "/complete" in url:
                complete_attempts += 1
                raise urllib_error.URLError("simulated permanent")
            return original_urlopen(req, *args, **kwargs)

        worker_module.urllib_request.urlopen = always_fail_complete
        try:
            t0 = time.monotonic()
            # Must not raise; worker survival is non-negotiable.
            worker._report_complete(item.lease_id, status="completed")
            elapsed = time.monotonic() - t0
        finally:
            worker_module.urllib_request.urlopen = original_urlopen

        # 5 attempts is the documented limit (ADR-0002 #4 spec).
        assert complete_attempts == 5, (
            f"expected exactly 5 /complete attempts before giving up, got {complete_attempts}"
        )
        # Backoff sum: 0.1 + 0.2 + 0.4 + 0.8 = 1.5s of waits between
        # attempts (the 5th attempt has no wait after it). Cap the
        # upper bound generously for CI noise.
        assert elapsed < 8.0, (
            f"retry loop took too long: {elapsed:.2f}s — backoff math wrong"
        )

        # Dispatcher state untouched: no /complete reached it.
        with dispatcher._lock:
            assert item.lease_id in dispatcher._leased, (
                "lease must remain leased so the TTL reaper can recover it"
            )
            assert item.lease_id not in dispatcher._completed
            assert len(dispatcher._pending) == 0
        completed_events = [d for kind, d in events if kind == "completed"]
        assert len(completed_events) == 0
    finally:
        if worker is not None:
            worker._hb_stop.set()
        dispatcher.shutdown()
