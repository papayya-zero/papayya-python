"""Lease TTL + heartbeat recovery — Phase 2 ADR-0002 #1.

Two scenarios:

  1. Worker dies mid-item (SIGKILL): heartbeats stop, dispatcher reaper
     releases the lease after TTL, re-queues the item with a fresh
     lease_id, and emits a ``lease_expired`` event.

  2. Worker is healthy but the item is slow: heartbeats arrive faster
     than the TTL, so the lease stays alive across multiple TTL
     intervals and the reaper does not falsely release it.

These cover the recovery shape and the most obvious false-positive.
Networking-partition recovery (zombie /complete after re-lease) is
covered separately when item #4 lands.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


@pytest.fixture
def slow_agent_module(tmp_path: Path) -> Path:
    """Agent that blocks for 60s. Long enough that any reasonable test
    SIGKILL'ing it always catches it mid-flight; short enough that the
    OS reaps cleanly when pytest tears down."""
    src = textwrap.dedent("""\
        import time
        from papayya import agent

        @agent(name="slow")
        def slow(item_id: str) -> dict:
            time.sleep(60)
            return {"id": item_id}
    """)
    path = tmp_path / "slow_agent.py"
    path.write_text(src)
    return path


def _spawn_worker(
    *,
    agent_module: Path,
    dispatcher_url: str,
    store_path: str,
    worker_id: str,
    heartbeat_interval: float,
    log_path: Path,
) -> subprocess.Popen:
    log = open(log_path, "w")
    return subprocess.Popen(
        [
            sys.executable, "-m", "papayya.runtime",
            "--agent-module", str(agent_module),
            "--dispatcher", dispatcher_url,
            "--store", store_path,
            "--worker-id", worker_id,
            "--log-level", "INFO",
            "--heartbeat-interval-seconds", str(heartbeat_interval),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def _wait_for_lease(dispatcher, timeout: float = 5.0) -> str | None:
    """Return the lease_id of whichever item is currently leased, or None."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with dispatcher._lock:
            if dispatcher._leased:
                return next(iter(dispatcher._leased.keys()))
        time.sleep(0.02)
    return None


def test_lease_expires_when_worker_dies_and_reissues_with_fresh_id(
    tmp_path: Path,
    slow_agent_module: Path,
) -> None:
    """The recovery the whole feature exists to provide.

    Without this: a SIGKILL'd worker holds an item permanently — the
    dispatcher never re-dispatches, the dashboard never surfaces it as
    failed, and the customer's batch silently stalls forever. The test
    asserts the three things that close that gap:
      • the stale lease is removed from `_leased` after TTL
      • the item is re-queued in `_pending` with a different lease_id
      • a `lease_expired` event fires for operators
    """
    from papayya.runtime.dispatcher import LocalDispatcher

    events: list[tuple[str, dict]] = []
    dispatcher = LocalDispatcher(
        port=0,
        on_event=lambda kind, data: events.append((kind, data)),
        lease_ttl_seconds=1.5,
        heartbeat_check_interval=0.2,
    )
    try:
        store_path = str(tmp_path / "ttl.db")
        dispatcher.enqueue(agent="slow", item_id="co_dies")

        proc = _spawn_worker(
            agent_module=slow_agent_module,
            dispatcher_url=dispatcher.url,
            store_path=store_path,
            worker_id="ttl_w1",
            heartbeat_interval=0.4,
            log_path=tmp_path / "worker_ttl.log",
        )
        try:
            initial_lease_id = _wait_for_lease(dispatcher, timeout=5.0)
            assert initial_lease_id is not None, "worker never picked up the lease"

            # SIGKILL — heartbeats stop instantly. SIGTERM would trigger
            # graceful drain (Phase 2 #12, not yet shipped), which we'd
            # want to test separately.
            proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=3)

            # Reaper window: TTL=1.5s, reaper tick=0.2s → release should
            # happen within ~1.7-2.0s of last heartbeat. Pad to 5s for
            # CI noise tolerance.
            deadline = time.time() + 5.0
            while time.time() < deadline:
                with dispatcher._lock:
                    if initial_lease_id not in dispatcher._leased:
                        break
                time.sleep(0.05)

            with dispatcher._lock:
                assert initial_lease_id not in dispatcher._leased, (
                    f"reaper did not release lease {initial_lease_id} "
                    "within TTL+padding window — recovery mechanism broken"
                )
                pending = list(dispatcher._pending)
                assert len(pending) == 1, (
                    f"expected the item to be re-queued, got pending={len(pending)}"
                )
                new_item = pending[0]
                assert new_item.lease_id != initial_lease_id, (
                    "re-issued lease_id must be fresh — a colliding ID would "
                    "let a zombie worker's late /complete claim the new lease"
                )
                assert new_item.item_id == "co_dies"
                assert new_item.agent == "slow"

            expired = [d for kind, d in events if kind == "lease_expired"]
            assert len(expired) == 1, (
                f"expected exactly one lease_expired event, got {len(expired)}"
            )
            ev = expired[0]
            assert ev["old_lease_id"] == initial_lease_id
            assert ev["worker_id"] == "ttl_w1"
            assert ev["item_id"] == "co_dies"
            assert ev["age_s"] >= 1.5  # at least the TTL elapsed
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGKILL)
                proc.wait(timeout=3)
    finally:
        dispatcher.shutdown()


def test_healthy_worker_heartbeats_keep_lease_alive_past_ttl(
    tmp_path: Path,
    slow_agent_module: Path,
) -> None:
    """Sanity: a slow-but-alive worker must NOT have its lease released.

    Without this guarantee, the recovery mechanism is a false-positive
    machine — long-running steps would get re-dispatched while the
    original worker is still successfully running them, and the same
    item would execute concurrently on multiple workers.
    """
    from papayya.runtime.dispatcher import LocalDispatcher

    events: list[tuple[str, dict]] = []
    dispatcher = LocalDispatcher(
        port=0,
        on_event=lambda kind, data: events.append((kind, data)),
        lease_ttl_seconds=1.0,
        heartbeat_check_interval=0.2,
    )
    try:
        store_path = str(tmp_path / "ttl_alive.db")
        dispatcher.enqueue(agent="slow", item_id="co_alive")

        proc = _spawn_worker(
            agent_module=slow_agent_module,
            dispatcher_url=dispatcher.url,
            store_path=store_path,
            worker_id="ttl_alive",
            heartbeat_interval=0.3,  # well below TTL
            log_path=tmp_path / "worker_alive.log",
        )
        try:
            initial_lease_id = _wait_for_lease(dispatcher, timeout=5.0)
            assert initial_lease_id is not None

            # Wait several TTL windows. Heartbeats every 0.3s should keep
            # last_heartbeat fresh; the reaper should never release.
            time.sleep(3.0)  # = 3× TTL

            with dispatcher._lock:
                assert initial_lease_id in dispatcher._leased, (
                    "reaper falsely released a healthy worker's lease — "
                    "heartbeats are not preventing TTL expiry"
                )
                assert len(dispatcher._pending) == 0, (
                    "no item should be re-queued for a healthy worker"
                )

            expired = [d for kind, d in events if kind == "lease_expired"]
            assert len(expired) == 0, (
                f"expected no lease_expired events for healthy worker, got {expired}"
            )
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGKILL)
                proc.wait(timeout=3)
    finally:
        dispatcher.shutdown()
