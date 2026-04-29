"""Per-item soft timeout — Phase 2 ADR-0002 #2.

Three cases the watchdog has to handle correctly:

  1. Hanging item → fails within max_duration; worker recovers and
     processes the next item normally.
  2. Per-call payload override beats the per-agent decorator default.
  3. No max_duration anywhere → watchdog stays disarmed; fast items
     don't get penalized.

The watchdog is signal-based (Unix only); these tests run in
subprocesses so the SIGALRM only ever fires inside the worker process,
never inside the test process.
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
            "--heartbeat-interval-seconds", "0.5",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def _wait_for_completed(events: list, target_count: int, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if sum(1 for kind, _ in events if kind == "completed") >= target_count:
            return
        time.sleep(0.05)


@pytest.fixture
def hanging_agent_module(tmp_path: Path) -> Path:
    """Hangs forever unless the watchdog interrupts."""
    src = textwrap.dedent("""\
        import time
        from papayya import agent

        @agent(name="hangy", max_duration_seconds=0.5)
        def hangy(item_id: str) -> dict:
            time.sleep(60)
            return {"id": item_id}
    """)
    path = tmp_path / "hangy_agent.py"
    path.write_text(src)
    return path


def test_hanging_item_times_out_and_worker_continues(
    tmp_path: Path,
    hanging_agent_module: Path,
) -> None:
    """The recovery path: a stuck step must surface as a timeout-failed
    completion, and the worker must process the next item normally.
    Without this, one bad item silently halts the customer's batch.
    """
    from papayya.runtime.dispatcher import LocalDispatcher

    events: list[tuple[str, dict]] = []
    dispatcher = LocalDispatcher(
        port=0,
        on_event=lambda kind, data: events.append((kind, data)),
        # Long TTL so the reaper isn't a confound.
        lease_ttl_seconds=60.0,
    )
    try:
        store_path = str(tmp_path / "timeout.db")

        # The hanging agent will time out; the next call to the same
        # registered name will also hang+timeout. To prove the worker
        # processes the *next* item normally, write a second module
        # at the same path that finishes fast — but a worker only
        # imports once. Instead, just enqueue two items of the hanging
        # agent and assert both surface as failed-timeout in finite time.
        # That's the recovery test: "worker did not get permanently
        # stuck on the first one."
        dispatcher.enqueue(agent="hangy", item_id="co_first")
        dispatcher.enqueue(agent="hangy", item_id="co_second")

        proc = _spawn_worker(
            agent_module=hanging_agent_module,
            dispatcher_url=dispatcher.url,
            store_path=store_path,
            worker_id="timeout_w1",
            log_path=tmp_path / "worker_timeout.log",
        )
        try:
            # Each item should fail in ~0.5s + some overhead. 5s for
            # both is generous.
            _wait_for_completed(events, target_count=2, timeout=5.0)

            completed = [d for kind, d in events if kind == "completed"]
            assert len(completed) == 2, (
                f"expected 2 completions, got {len(completed)}; "
                "worker likely stalled on the first hang"
            )
            for ev in completed:
                assert ev["status"] == "failed", ev
                assert ev["error_category"] == "timeout", ev
                assert "timeout" in (ev["error"] or "").lower()
                # Bound the duration: was the watchdog actually firing,
                # or did the lease TTL release it as a side effect?
                assert ev["duration_ms"] < 2000, (
                    f"item ran {ev['duration_ms']}ms — watchdog should "
                    "have fired around 500ms"
                )
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGKILL)
                proc.wait(timeout=3)
    finally:
        dispatcher.shutdown()


def test_per_call_payload_override_wins_over_decorator(tmp_path: Path) -> None:
    """Decorator default = 60s (won't fire); payload override = 0.5s.
    Proves the priority order in `_handle_lease`: payload first, then
    registration, then None."""
    from papayya.runtime.dispatcher import LocalDispatcher

    src = textwrap.dedent("""\
        import time
        from papayya import agent

        @agent(name="generous", max_duration_seconds=60.0)
        def generous(item_id: str) -> dict:
            time.sleep(60)
            return {"id": item_id}
    """)
    agent_path = tmp_path / "generous_agent.py"
    agent_path.write_text(src)

    events: list[tuple[str, dict]] = []
    dispatcher = LocalDispatcher(
        port=0,
        on_event=lambda kind, data: events.append((kind, data)),
        lease_ttl_seconds=60.0,
    )
    try:
        store_path = str(tmp_path / "override.db")
        dispatcher.enqueue(
            agent="generous",
            item_id="co_override",
            payload={"max_duration_seconds": 0.5},
        )

        proc = _spawn_worker(
            agent_module=agent_path,
            dispatcher_url=dispatcher.url,
            store_path=store_path,
            worker_id="override_w1",
            log_path=tmp_path / "worker_override.log",
        )
        try:
            _wait_for_completed(events, target_count=1, timeout=5.0)
            completed = [d for kind, d in events if kind == "completed"]
            assert len(completed) == 1
            ev = completed[0]
            assert ev["status"] == "failed"
            assert ev["error_category"] == "timeout"
            # 60s decorator default would have run nearly forever;
            # 2s ceiling proves the payload override was honored.
            assert ev["duration_ms"] < 2000, (
                f"item ran {ev['duration_ms']}ms — payload override "
                "of 0.5s did not take effect over decorator default of 60s"
            )
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGKILL)
                proc.wait(timeout=3)
    finally:
        dispatcher.shutdown()


def test_no_timeout_when_neither_decorator_nor_payload(tmp_path: Path) -> None:
    """Watchdog must stay disarmed when nothing requested a timeout —
    fast items shouldn't pay any cost and there must be no leftover
    SIGALRM handler that could fire later."""
    from papayya.runtime.dispatcher import LocalDispatcher

    src = textwrap.dedent("""\
        from papayya import agent

        @agent(name="quick")
        def quick(item_id: str) -> dict:
            return {"id": item_id}
    """)
    agent_path = tmp_path / "quick_agent.py"
    agent_path.write_text(src)

    events: list[tuple[str, dict]] = []
    dispatcher = LocalDispatcher(
        port=0,
        on_event=lambda kind, data: events.append((kind, data)),
        lease_ttl_seconds=60.0,
    )
    try:
        store_path = str(tmp_path / "no_timeout.db")
        dispatcher.enqueue(agent="quick", item_id="co_fast")

        proc = _spawn_worker(
            agent_module=agent_path,
            dispatcher_url=dispatcher.url,
            store_path=store_path,
            worker_id="quick_w1",
            log_path=tmp_path / "worker_quick.log",
        )
        try:
            _wait_for_completed(events, target_count=1, timeout=5.0)
            completed = [d for kind, d in events if kind == "completed"]
            assert len(completed) == 1
            ev = completed[0]
            assert ev["status"] == "completed"
            # No category set when watchdog stays disarmed.
            assert ev.get("error_category") is None
            assert ev.get("error") is None
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGKILL)
                proc.wait(timeout=3)
    finally:
        dispatcher.shutdown()
