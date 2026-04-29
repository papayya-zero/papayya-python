"""Worker dispatcher-reconnect backoff — Phase 2 ADR-0002 #15.

The Phase 1.5 mechanical feel-test (2026-04-28) showed that without
backoff the worker hammers a dead/recovering dispatcher at the
poll_idle_seconds rate (~75 connection-refused failures during a 4s
outage). The backoff state machine here replaces that tight retry loop
with exponential intervals capped at 2s, snapping back to zero on the
first successful poll so recovery latency stays unchanged.

The state machine is small enough to test in isolation; the worker's
run loop wiring is exercised by the integration tests.
"""

from __future__ import annotations

from papayya.runtime.worker import _ReconnectBackoff


def test_initial_state_is_zero():
    b = _ReconnectBackoff()
    assert b.current == 0.0


def test_first_failure_uses_initial_value():
    b = _ReconnectBackoff(initial_seconds=0.1, max_seconds=2.0)
    assert b.on_failure() == 0.1
    assert b.current == 0.1


def test_subsequent_failures_double_until_cap():
    b = _ReconnectBackoff(initial_seconds=0.1, max_seconds=2.0)
    waits = [b.on_failure() for _ in range(8)]
    # 0.1, 0.2, 0.4, 0.8, 1.6, 2.0, 2.0, 2.0 — caps at 2.0 thereafter.
    assert waits[0] == 0.1
    assert waits[1] == 0.2
    assert waits[2] == 0.4
    assert waits[3] == 0.8
    assert waits[4] == 1.6
    assert all(w == 2.0 for w in waits[5:]), waits


def test_success_resets_to_zero():
    """The whole point: the first poll after recovery has no added wait,
    so the worker doesn't sit out a backoff window unnecessarily."""
    b = _ReconnectBackoff(initial_seconds=0.1, max_seconds=2.0)
    for _ in range(5):
        b.on_failure()
    assert b.current == 1.6
    b.on_success()
    assert b.current == 0.0
    # Next failure starts the ramp over, not from where it left off.
    assert b.on_failure() == 0.1


def test_repeated_success_is_idempotent():
    b = _ReconnectBackoff()
    b.on_success()
    b.on_success()
    assert b.current == 0.0


def test_cap_is_respected_with_aggressive_initial():
    """If the initial value is already near the cap, the next failure
    should clamp to the cap instead of overshooting."""
    b = _ReconnectBackoff(initial_seconds=1.5, max_seconds=2.0)
    assert b.on_failure() == 1.5
    assert b.on_failure() == 2.0  # 1.5 * 2 = 3.0 → clamped to 2.0
    assert b.on_failure() == 2.0
