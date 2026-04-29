"""Graceful SIGTERM drain — Phase 2 ADR-0002 #12.

Three scenarios cover the full surface of the drain contract:

  1. **Clean drain.** SIGTERM during a normal-duration in-flight item:
     poll loop stops, current item finishes, completion is reported,
     worker exits cleanly. Without this, deploy-driven worker recycling
     drops in-flight work.

  2. **Idle exit.** SIGTERM with nothing in flight: worker exits
     immediately. No watchdog escalation, no spurious wait.

  3. **Hung-item escalation + recovery.** SIGTERM during an item with
     no max_duration, drain budget too tight to finish: watchdog
     force-exits; lease TTL releases the orphaned lease and the
     dispatcher re-dispatches with a fresh lease_id. This is the
     production safety story for the "customer code hangs" failure
     mode that #2 doesn't catch (no max_duration set).

These tests run a real subprocess so the SIGTERM path is exercised
end-to-end (handler install → drain event → watchdog → exit codes),
not just the in-process logic.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


def _spawn_worker(
    *,
    agent_module: Path,
    dispatcher_url: str,
    store_path: str,
    worker_id: str,
    drain_timeout_seconds: float,
    log_path: Path,
    heartbeat_interval: float = 0.5,
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
            "--drain-timeout-seconds", str(drain_timeout_seconds),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def _wait_for_lease(dispatcher, timeout: float = 5.0) -> str | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with dispatcher._lock:
            if dispatcher._leased:
                return next(iter(dispatcher._leased.keys()))
        time.sleep(0.02)
    return None


def _wait_for_event(events: list, kind: str, timeout: float) -> dict | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for k, d in events:
            if k == kind:
                return d
        time.sleep(0.02)
    return None


def test_sigterm_lets_in_flight_item_finish(tmp_path: Path) -> None:
    """SIGTERM during a normal-duration item: item completes; worker
    exits cleanly with zero status.

    Without this, every deploy / pod recycle silently drops whatever
    items happened to be in flight at the moment the orchestrator
    sent SIGTERM. With it, the lease completes and the next deploy
    starts from a clean queue.
    """
    from papayya.runtime.dispatcher import LocalDispatcher

    src = textwrap.dedent("""\
        import time
        from papayya import agent

        @agent(name="medium", max_duration_seconds=10.0)
        def medium(item_id: str) -> dict:
            time.sleep(1.5)
            return {"id": item_id}
    """)
    agent_path = tmp_path / "medium_agent.py"
    agent_path.write_text(src)

    events: list[tuple[str, dict]] = []
    dispatcher = LocalDispatcher(
        port=0,
        on_event=lambda kind, data: events.append((kind, data)),
        lease_ttl_seconds=60.0,
    )
    try:
        store_path = str(tmp_path / "drain_clean.db")
        dispatcher.enqueue(agent="medium", item_id="co_clean")

        proc = _spawn_worker(
            agent_module=agent_path,
            dispatcher_url=dispatcher.url,
            store_path=store_path,
            worker_id="drain_clean_w1",
            drain_timeout_seconds=10.0,
            log_path=tmp_path / "worker_clean.log",
        )
        try:
            initial_lease = _wait_for_lease(dispatcher, timeout=5.0)
            assert initial_lease is not None, "worker never picked up the lease"

            # SIGTERM mid-flight. The agent fn sleeps 1.5s; drain budget
            # is 10s, so the item must finish naturally.
            proc.send_signal(signal.SIGTERM)

            # Worker should exit within ~2.5s (sleep 1.5 + completion
            # round-trip + thread shutdown). Give 5s for CI noise.
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.send_signal(signal.SIGKILL)
                proc.wait(timeout=2)
                pytest.fail(
                    "worker did not exit within drain window — graceful "
                    "drain hung instead of letting the in-flight item finish"
                )

            assert proc.returncode == 0, (
                f"worker should exit cleanly after drain, got {proc.returncode} "
                "— watchdog likely escalated when it shouldn't have"
            )

            completed = _wait_for_event(events, "completed", timeout=2.0)
            assert completed is not None, (
                "no completed event — in-flight item was dropped on SIGTERM"
            )
            assert completed["lease_id"] == initial_lease
            assert completed["status"] == "completed"

            expired = [d for kind, d in events if kind == "lease_expired"]
            assert expired == [], (
                f"lease was expired by reaper instead of completed by worker: "
                f"{expired}"
            )
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGKILL)
                proc.wait(timeout=3)
    finally:
        dispatcher.shutdown()


def test_sigterm_idle_worker_exits_immediately(tmp_path: Path) -> None:
    """SIGTERM with no in-flight item: worker exits within ~1s.

    Sanity check that the drain machinery doesn't introduce a
    spurious wait when there's nothing to drain. Without this, every
    pod-stop would pay a 200ms-poll-cycle latency tax for no reason.
    """
    from papayya.runtime.dispatcher import LocalDispatcher

    src = textwrap.dedent("""\
        from papayya import agent

        @agent(name="noop")
        def noop(item_id: str) -> dict:
            return {"id": item_id}
    """)
    agent_path = tmp_path / "noop_agent.py"
    agent_path.write_text(src)

    events: list[tuple[str, dict]] = []
    dispatcher = LocalDispatcher(
        port=0,
        on_event=lambda kind, data: events.append((kind, data)),
        lease_ttl_seconds=60.0,
    )
    try:
        store_path = str(tmp_path / "drain_idle.db")

        proc = _spawn_worker(
            agent_module=agent_path,
            dispatcher_url=dispatcher.url,
            store_path=store_path,
            worker_id="drain_idle_w1",
            drain_timeout_seconds=10.0,
            log_path=tmp_path / "worker_idle.log",
        )
        try:
            # Wait long enough for the worker to be polling idle.
            time.sleep(0.5)

            t0 = time.monotonic()
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.send_signal(signal.SIGKILL)
                proc.wait(timeout=2)
                pytest.fail(
                    "idle worker did not exit within 2s of SIGTERM — "
                    "drain machinery is blocking on something"
                )
            elapsed = time.monotonic() - t0

            assert proc.returncode == 0, (
                f"idle worker should exit clean, got {proc.returncode}"
            )
            assert elapsed < 1.5, (
                f"idle drain took {elapsed:.2f}s — should exit ~immediately"
            )
            # No items were ever enqueued; no completed events expected.
            assert [d for kind, d in events if kind == "completed"] == []
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGKILL)
                proc.wait(timeout=3)
    finally:
        dispatcher.shutdown()


def test_sigterm_hung_item_force_exits_and_lease_recovers(tmp_path: Path) -> None:
    """Hung-item escalation: drain deadline → os._exit(1) → TTL recovery.

    The agent fn sleeps 60s with NO max_duration, so the per-item
    watchdog (#2) won't fire. Drain budget is 0.5s — well under the
    item's runtime — to force the watchdog to escalate. The dispatcher's
    lease TTL (1s) then releases the orphaned lease and re-dispatches
    the item with a fresh lease_id. This proves the recovery path
    that ADR-0002 #12 documents:

      "drain deadline exceeded ... lease ... will be released by
       dispatcher TTL and re-dispatched."

    Without this, a stuck step at SIGTERM time would block the worker
    until kubelet's terminationGracePeriod expired and SIGKILL'd it,
    with no observable signal in the dispatcher event log explaining
    why the item disappeared.
    """
    from papayya.runtime.dispatcher import LocalDispatcher

    src = textwrap.dedent("""\
        import time
        from papayya import agent

        # No max_duration: the per-item watchdog stays disarmed so the
        # drain watchdog is the only mechanism that can stop this fn.
        @agent(name="hangy_no_timeout")
        def hangy(item_id: str) -> dict:
            time.sleep(60)
            return {"id": item_id}
    """)
    agent_path = tmp_path / "hangy_no_timeout_agent.py"
    agent_path.write_text(src)

    events: list[tuple[str, dict]] = []
    dispatcher = LocalDispatcher(
        port=0,
        on_event=lambda kind, data: events.append((kind, data)),
        lease_ttl_seconds=1.0,
        heartbeat_check_interval=0.2,
    )
    log_path = tmp_path / "worker_hung.log"
    try:
        store_path = str(tmp_path / "drain_hung.db")
        dispatcher.enqueue(agent="hangy_no_timeout", item_id="co_hung")

        proc = _spawn_worker(
            agent_module=agent_path,
            dispatcher_url=dispatcher.url,
            store_path=store_path,
            worker_id="drain_hung_w1",
            drain_timeout_seconds=0.5,
            heartbeat_interval=0.3,
            log_path=log_path,
        )
        try:
            initial_lease = _wait_for_lease(dispatcher, timeout=5.0)
            assert initial_lease is not None, "worker never picked up the lease"

            t0 = time.monotonic()
            proc.send_signal(signal.SIGTERM)

            # Watchdog deadline = 0.5s; expect force-exit shortly after.
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.send_signal(signal.SIGKILL)
                proc.wait(timeout=2)
                pytest.fail(
                    "drain watchdog did not force-exit within deadline + "
                    "padding — escalation path is broken"
                )
            elapsed = time.monotonic() - t0

            # Force-exit code is 1 per the watchdog implementation.
            assert proc.returncode == 1, (
                f"expected exit code 1 from os._exit(1) on drain deadline, "
                f"got {proc.returncode}. Log: {log_path.read_text()[-1500:]}"
            )
            # Should have exited well inside drain_timeout + watchdog
            # poll grain (0.5 + 0.2) + process teardown.
            assert elapsed < 2.0, (
                f"force-exit took {elapsed:.2f}s — watchdog deadline math wrong"
            )

            # Lease TTL recovery — wait for the reaper to fire.
            # TTL=1s, reaper tick=0.2s → recovery within ~1.4s.
            expired = _wait_for_event(events, "lease_expired", timeout=3.0)
            assert expired is not None, (
                "TTL reaper never released the orphaned lease — "
                "recovery story broken"
            )
            assert expired["old_lease_id"] == initial_lease
            assert expired["worker_id"] == "drain_hung_w1"
            assert expired["item_id"] == "co_hung"

            # Item must be re-queued for re-dispatch with a fresh lease_id.
            with dispatcher._lock:
                pending = list(dispatcher._pending)
                assert len(pending) == 1, (
                    f"expected the item to be re-queued, got pending={len(pending)}"
                )
                assert pending[0].lease_id != initial_lease, (
                    "fresh lease_id required so a stale completion can't "
                    "claim the new lease"
                )
                assert initial_lease not in dispatcher._completed, (
                    "no /complete should have landed for the orphaned lease"
                )

            # The error log line is the operator's diagnostic — verify it
            # actually reached the log file before os._exit ran.
            log_text = log_path.read_text()
            assert "drain deadline exceeded" in log_text, (
                f"missing drain-deadline diagnostic in worker log: {log_text[-2000:]}"
            )
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGKILL)
                proc.wait(timeout=3)
    finally:
        dispatcher.shutdown()
