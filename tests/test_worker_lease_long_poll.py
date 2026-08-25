"""The worker half of the lease long-poll (plan 65 L1).

An idle worker used to cost the control plane ~18 lease requests a second —
0.05s of sleep between polls that each ran a write-path transaction against
runtime_pending. Measured on an idle local stack: `polls=1065 granted=0` in a
60-second window, from one worker.

The worker's side of the fix is three things, and the third is the one worth
testing hardest:

  1. send ?wait= so the server knows it may hold the poll,
  2. budget the socket timeout for a request that is SUPPOSED to be slow,
  3. NOT sleep again on top of a wait the server already served.

(3) is where the dangerous failure lives. If the worker assumed the server
honoured `wait` and skipped its sleep, then an older control plane — or a proxy
that drops the query string, or a deployment that clamped the value to zero —
would turn this worker into a 100%-CPU spin loop. So the decision is made from
MEASURED elapsed time, never from a promise, and that is what these pin.
"""

from __future__ import annotations

import urllib.request

import pytest

from papayya.runtime.worker import (
    _DEFAULT_LEASE_WAIT_SECONDS,
    _MAX_LEASE_WAIT_SECONDS,
    _PollOutcome,
    Worker,
)


class _Resp:
    """Minimal stand-in for the urlopen context manager."""

    def __init__(self, status: int, body: bytes = b""):
        self.status = status
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _worker(**kw) -> Worker:
    return Worker(
        dispatcher_url="http://cp:8090/v1/runtime",
        store_path=":memory:",
        agent_module_path=None,
        worker_id="w-test",
        **kw,
    )


def test_lease_url_carries_the_wait(monkeypatch):
    w = _worker(lease_wait_seconds=20.0)
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        return _Resp(204)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    outcome, lease = w._poll_lease()

    assert outcome == _PollOutcome.IDLE
    assert lease is None
    assert "wait=20" in seen["url"]
    # The socket budget must cover the wait plus the normal round trip, or the
    # worker times out on exactly the behaviour it asked for.
    assert seen["timeout"] == pytest.approx(w.http_timeout_seconds + 20.0)


def test_zero_wait_sends_no_parameter_and_keeps_the_old_timeout(monkeypatch):
    """wait=0 must be wire-identical to the pre-L1 worker, so the opt-out is a
    real opt-out and not a differently-shaped request."""
    w = _worker(lease_wait_seconds=0)
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        return _Resp(204)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    w._poll_lease()

    assert "wait=" not in seen["url"]
    assert seen["timeout"] == w.http_timeout_seconds


def test_wait_is_clamped_to_the_server_ceiling():
    """Clamped, not rejected. A worker configured past the ceiling should get
    the largest wait the server will honour, not fail every poll."""
    assert _worker(lease_wait_seconds=600).lease_wait_seconds == _MAX_LEASE_WAIT_SECONDS
    assert _worker(lease_wait_seconds=-5).lease_wait_seconds == 0.0
    assert _worker().lease_wait_seconds == _DEFAULT_LEASE_WAIT_SECONDS


def test_no_extra_sleep_after_a_poll_the_server_actually_held():
    """The wait and the sleep are one budget, paid once."""
    w = _worker(lease_wait_seconds=20.0)
    w._last_poll_seconds = 20.0
    assert w._idle_sleep_seconds() == 0.0


def test_falls_back_to_pacing_itself_against_a_server_that_ignores_wait():
    """THE ONE THAT MATTERS. A control plane that does not honour `wait`
    answers instantly; if the worker skipped its sleep on the strength of
    having ASKED, it would spin at 100% CPU forever. The elapsed time is what
    it believes, so it paces itself instead."""
    w = _worker(lease_wait_seconds=20.0)
    w._last_poll_seconds = 0.002
    assert w._idle_sleep_seconds() == w.poll_idle_seconds


def test_a_partially_honoured_wait_still_counts_as_held():
    """The server clamps at 25s and may answer early; anything past half the
    request could only have blocked, so it must not be re-slept."""
    w = _worker(lease_wait_seconds=20.0)
    w._last_poll_seconds = 11.0
    assert w._idle_sleep_seconds() == 0.0
    w._last_poll_seconds = 9.0
    assert w._idle_sleep_seconds() == w.poll_idle_seconds


def test_wait_disabled_always_paces_itself():
    w = _worker(lease_wait_seconds=0)
    w._last_poll_seconds = 0.0
    assert w._idle_sleep_seconds() == w.poll_idle_seconds


def test_unreachable_dispatcher_still_records_elapsed(monkeypatch):
    """The backoff path reads the same field. Leaving it stale from the last
    successful hold would make a dead dispatcher look like a held poll and
    suppress the reconnect sleep."""
    w = _worker(lease_wait_seconds=20.0)
    w._last_poll_seconds = 20.0

    def boom(_req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    outcome, _ = w._poll_lease()

    assert outcome == _PollOutcome.UNREACHABLE
    assert w._last_poll_seconds < 1.0
