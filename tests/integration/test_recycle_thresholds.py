"""Worker recycling triggers — ADR-0002 #6 / ADR-0001 § 4.

Covers the two recycling triggers that ADR-0001 § 4 designed but the
slice 3 dep-hash work didn't ship:

  • item-count: worker recycles after N items have flowed through it
  • RSS%:       worker recycles when resident memory crosses a ceiling

The dep-hash and SIGTERM-graceful-drain triggers are covered elsewhere
(`test_multi_version_dispatch.py` and `test_sigterm_drain.py`). All four
share the same ``_recycle_pending`` plumbing so an observability surface
that consumes "recycle reason" can treat them uniformly.

Both tests construct the Worker in-process and let ``worker.run()``
drive a real ``LocalDispatcher`` over loopback. That mirrors the pattern
in ``test_complete_idempotency.py`` and is the surgical proof that the
threshold check fires *between* items (not mid-item, not on the very
first lease).
"""

from __future__ import annotations

import textwrap
import threading
from pathlib import Path

import pytest


@pytest.fixture
def noop_agent_module(tmp_path: Path) -> Path:
    """Trivial agent module — returns a dict, no run lineage.

    The recycle triggers don't care what the agent does; they fire in
    ``_handle_lease``'s finally block after each item finishes (success
    or failure). A bare-return agent keeps the test focused on the
    threshold logic.
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


def _run_worker_until_exit(worker, timeout: float = 10.0) -> threading.Thread:
    """Drive ``worker.run()`` in a daemon thread; return the thread.

    Caller asserts ``thread.is_alive()`` is False after join — that's
    the signal that ``_running`` flipped (recycle triggered) and the
    main loop exited cleanly.
    """
    t = threading.Thread(
        target=worker.run, daemon=True, name="recycle-test-worker"
    )
    t.start()
    t.join(timeout=timeout)
    return t


def test_worker_recycles_after_item_count_threshold(
    tmp_path: Path,
    noop_agent_module: Path,
) -> None:
    """Worker exits after ``max_items_before_recycle`` items.

    Asserts:
      • ``_recycle_pending`` flips to True
      • ``_items_processed`` equals the cap exactly (not over, not under)
      • Dispatcher records exactly ``cap`` completed; remaining items
        stay in ``pending``/``leased`` for the *next* worker
      • The thread driving ``worker.run()`` exits cleanly within the
        timeout (no busy-loop after recycle)
    """
    from papayya.runtime.dispatcher import LocalDispatcher
    from papayya.runtime.worker import Worker

    dispatcher = LocalDispatcher(
        host="127.0.0.1", port=0, lease_ttl_seconds=30.0
    )
    worker: Worker | None = None
    try:
        worker = Worker(
            dispatcher_url=dispatcher.url,
            store_path=str(tmp_path / "store.db"),
            agent_module_path=str(noop_agent_module),
            worker_id="recycle-items-w1",
            poll_idle_seconds=0.01,
            max_items_before_recycle=2,
            # Disable RSS trigger so this test isolates the item-count path.
            max_rss_percent_before_recycle=0,
        )

        for i in range(4):
            dispatcher.enqueue(agent="noop", item_id=f"item-{i}")

        thread = _run_worker_until_exit(worker, timeout=10.0)
        assert not thread.is_alive(), (
            "worker.run() did not exit after the item-count threshold "
            "fired; main loop is still running"
        )

        assert worker._recycle_pending is True, (
            "expected _recycle_pending=True after threshold breach"
        )
        assert worker._items_processed == 2, (
            f"expected exactly 2 items processed, got {worker._items_processed}"
        )

        stats = dispatcher.stats()
        assert stats["completed"] == 2, (
            f"expected 2 completed in dispatcher, got {stats}"
        )
        assert stats["pending"] + stats["leased"] == 2, (
            "expected 2 items still queued for the next worker, got "
            f"{stats}"
        )
    finally:
        if worker is not None:
            worker._hb_stop.set()
        dispatcher.shutdown()


def test_worker_recycles_when_rss_percent_exceeds_threshold(
    tmp_path: Path,
    noop_agent_module: Path,
) -> None:
    """Worker exits when injected RSS provider returns above the cap.

    Uses an injected provider so the test is deterministic — no dep on
    actual psutil readings or process memory state. The provider is the
    documented test seam (constructor kwarg ``rss_percent_provider``).

    Asserts:
      • ``_recycle_pending`` flips after the first item finishes
      • Dispatcher records exactly 1 completed even though 3 are enqueued
      • ``_items_processed == 1`` — the trigger fires *between* items,
        not mid-item
    """
    from papayya.runtime.dispatcher import LocalDispatcher
    from papayya.runtime.worker import Worker

    dispatcher = LocalDispatcher(
        host="127.0.0.1", port=0, lease_ttl_seconds=30.0
    )
    worker: Worker | None = None
    try:
        worker = Worker(
            dispatcher_url=dispatcher.url,
            store_path=str(tmp_path / "store.db"),
            agent_module_path=str(noop_agent_module),
            worker_id="recycle-rss-w1",
            poll_idle_seconds=0.01,
            # Disable item-count trigger so this test isolates the RSS path.
            max_items_before_recycle=0,
            max_rss_percent_before_recycle=50.0,
            rss_percent_provider=lambda: 75.0,
        )

        for i in range(3):
            dispatcher.enqueue(agent="noop", item_id=f"item-{i}")

        thread = _run_worker_until_exit(worker, timeout=10.0)
        assert not thread.is_alive(), (
            "worker.run() did not exit after the RSS threshold fired"
        )

        assert worker._recycle_pending is True, (
            "expected _recycle_pending=True after RSS threshold breach"
        )
        assert worker._items_processed == 1, (
            f"expected exactly 1 item processed before recycle, got "
            f"{worker._items_processed}"
        )

        stats = dispatcher.stats()
        assert stats["completed"] == 1, (
            f"expected 1 completed in dispatcher, got {stats}"
        )
        assert stats["pending"] + stats["leased"] == 2, (
            "expected 2 items still queued for the next worker, got "
            f"{stats}"
        )
    finally:
        if worker is not None:
            worker._hb_stop.set()
        dispatcher.shutdown()


def test_rss_provider_failure_does_not_crash_worker(
    tmp_path: Path,
    noop_agent_module: Path,
) -> None:
    """A raising RSS provider is logged at DEBUG and the check is skipped.

    Reading RSS must never crash the worker. The dep-hash and SIGTERM
    triggers stay armed, but a transiently broken provider (e.g. psutil
    misconfigured in a stripped container) shouldn't take the worker
    down. Item-count trigger still fires normally.
    """
    from papayya.runtime.dispatcher import LocalDispatcher
    from papayya.runtime.worker import Worker

    def broken_provider() -> float:
        raise RuntimeError("simulated psutil failure")

    dispatcher = LocalDispatcher(
        host="127.0.0.1", port=0, lease_ttl_seconds=30.0
    )
    worker: Worker | None = None
    try:
        worker = Worker(
            dispatcher_url=dispatcher.url,
            store_path=str(tmp_path / "store.db"),
            agent_module_path=str(noop_agent_module),
            worker_id="recycle-rss-fail-w1",
            poll_idle_seconds=0.01,
            # Item-count fallback at 1 — worker should still recycle on
            # the item-count path even though RSS provider raises.
            max_items_before_recycle=1,
            max_rss_percent_before_recycle=50.0,
            rss_percent_provider=broken_provider,
        )

        dispatcher.enqueue(agent="noop", item_id="item-0")
        dispatcher.enqueue(agent="noop", item_id="item-1")

        thread = _run_worker_until_exit(worker, timeout=10.0)
        assert not thread.is_alive()

        # Item-count trigger fired (broke at 1); RSS provider raised but
        # was swallowed.
        assert worker._recycle_pending is True
        assert worker._items_processed == 1
    finally:
        if worker is not None:
            worker._hb_stop.set()
        dispatcher.shutdown()
