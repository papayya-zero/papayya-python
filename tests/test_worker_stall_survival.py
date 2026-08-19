"""Plan 48 W1 — an API stall must not kill the worker pool.

The defect: every dispatcher call handled ``except urllib_error.URLError``,
and urllib only converts to ``URLError`` around the *request* phase. A timeout
waiting for the **response** — CPython's ``do_open`` calls ``h.getresponse()``
outside that conversion — comes out as a bare ``TimeoutError``. It escaped the
handler, escaped ``run()``, and exited the process 1. Reproduced in the walk by
pausing the API container for eight seconds; a laptop sleeping does it too.

So every test here raises ``TimeoutError``, not ``URLError``. Asserting on
``URLError`` would have passed against the broken code — that is exactly why
the bug survived a test suite.

The second half covers the same shape one layer up: an exception from ONE
lease (a bundle fetch that can't reach the endpoint, a customer entrypoint
that won't import) took down a pool serving every account.
"""

from __future__ import annotations

import urllib.error as urllib_error

import pytest

from papayya.runtime import worker as worker_mod
from papayya.runtime.worker import (
    _DEFAULT_HTTP_TIMEOUT_SECONDS,
    Lease,
    Worker,
    _PollOutcome,
)


def _make_worker(**kwargs) -> Worker:
    w = Worker(
        dispatcher_url="http://control-pane-api:8090/v1/runtime",
        store_path="",
        agent_module_path=None,  # bootstrap mode: no import side effects
        api_key="plat-secret",
        heartbeat_interval_seconds=3600,
        **kwargs,
    )
    # The constructor starts a daemon heartbeat thread; stop it immediately.
    w.stop()
    w._hb_stop.set()
    # Plan 56 F2: the constructor now starts a heartbeat PROCESS (a
    # thread cannot heartbeat through a step that holds the GIL). These
    # tests never run the poll loop, whose finally block would stop it,
    # so they have to tear it down themselves or leak one child per test.
    w._stop_heartbeat()
    return w


class _Stall:
    """A urlopen replacement that times out waiting for the response.

    ``TimeoutError`` with no URLError wrapper is precisely what CPython
    surfaces from ``h.getresponse()``.
    """

    def __init__(self, exc: BaseException | None = None):
        self.calls: list[float | None] = []
        self._exc = exc or TimeoutError("timed out")

    def __call__(self, req, timeout=None):
        self.calls.append(timeout)
        raise self._exc


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Retry loops back off with time.sleep; don't pay for it in tests."""
    monkeypatch.setattr(worker_mod.time, "sleep", lambda _s: None)


# --- the five dispatcher calls ---------------------------------------- #


def test_poll_lease_reports_unreachable_on_response_timeout(monkeypatch):
    w = _make_worker()
    monkeypatch.setattr(worker_mod.urllib_request, "urlopen", _Stall())

    outcome, lease = w._poll_lease()

    assert outcome == _PollOutcome.UNREACHABLE
    assert lease is None


def test_poll_lease_still_handles_the_request_phase_error(monkeypatch):
    """URLError is a subclass of OSError, so widening the clause kept it."""
    w = _make_worker()
    monkeypatch.setattr(
        worker_mod.urllib_request,
        "urlopen",
        _Stall(urllib_error.URLError("connection refused")),
    )

    assert w._poll_lease()[0] == _PollOutcome.UNREACHABLE


def test_report_complete_survives_a_stall_and_exhausts_its_retries(monkeypatch):
    """The worst place to die: the model call is already paid for."""
    w = _make_worker()
    stall = _Stall()
    monkeypatch.setattr(worker_mod.urllib_request, "urlopen", stall)

    w._report_complete("lease-1", status="completed")  # must not raise

    assert len(stall.calls) == 5, "bounded retry should still run all attempts"


def test_report_release_survives_a_stall(monkeypatch):
    w = _make_worker()
    stall = _Stall()
    monkeypatch.setattr(worker_mod.urllib_request, "urlopen", stall)

    w._report_release("lease-1", reason="paused")

    assert len(stall.calls) == 5


def test_release_still_falls_back_when_the_route_is_missing(monkeypatch):
    """The HTTPError clause must stay ahead of OSError — HTTPError IS one."""
    w = _make_worker()
    monkeypatch.setattr(
        worker_mod.urllib_request,
        "urlopen",
        _Stall(urllib_error.HTTPError("u", 404, "nope", {}, None)),  # type: ignore[arg-type]
    )
    completed: list[dict] = []
    monkeypatch.setattr(
        w, "_report_complete",
        lambda lease_id, **kw: completed.append({"lease_id": lease_id, **kw}),
    )

    w._report_release("lease-1", reason="paused")

    assert completed and completed[0]["error_category"] == "paused"


def test_mark_run_running_survives_a_stall(monkeypatch):
    w = _make_worker()
    monkeypatch.setattr(worker_mod.urllib_request, "urlopen", _Stall())

    w._mark_run_running("run-1")  # declared "never raises"; now true


def test_mark_run_terminal_survives_a_stall(monkeypatch):
    w = _make_worker()
    monkeypatch.setattr(worker_mod.urllib_request, "urlopen", _Stall())

    w._mark_run_terminal("run-1", "failed", error="boom", error_category="runtime")


# --- the loop ---------------------------------------------------------- #


def test_run_loop_backs_off_through_a_stall_instead_of_exiting(monkeypatch):
    """The end-to-end claim: an 8s API stall costs poll attempts, not the pool."""
    w = _make_worker()
    stall = _Stall()
    monkeypatch.setattr(worker_mod.urllib_request, "urlopen", stall)

    polls = {"n": 0}
    real_poll = w._poll_lease

    def counting_poll():
        polls["n"] += 1
        if polls["n"] >= 4:
            w.stop()
        return real_poll()

    monkeypatch.setattr(w, "_poll_lease", counting_poll)
    w._running = True

    w.run()  # returned rather than raised == the worker is still alive

    assert polls["n"] == 4
    assert w._reconnect_backoff.current > 0.0, "stall should engage backoff"


# --- timeouts ---------------------------------------------------------- #


def test_default_request_budget_is_ten_seconds(monkeypatch):
    """2s was not a budget a busy control plane could be held to."""
    assert _DEFAULT_HTTP_TIMEOUT_SECONDS == 10.0
    w = _make_worker()
    stall = _Stall()
    monkeypatch.setattr(worker_mod.urllib_request, "urlopen", stall)

    w._poll_lease()

    assert stall.calls == [10.0]


def test_request_budget_is_configurable(monkeypatch):
    w = _make_worker(http_timeout_seconds=25.0)
    stall = _Stall()
    monkeypatch.setattr(worker_mod.urllib_request, "urlopen", stall)

    w._poll_lease()
    w._mark_run_running("run-1")

    assert stall.calls == [25.0, 25.0]


def test_bundle_budget_is_a_floor_not_an_override():
    """Bundle downloads are sized for bytes on the wire, not for latency."""
    assert _make_worker()._bundle_timeout_seconds == 30.0
    assert _make_worker(http_timeout_seconds=60.0)._bundle_timeout_seconds == 60.0


# --- one lease must not take the pool --------------------------------- #


def _lease(**kw) -> Lease:
    base = dict(
        lease_id="lease-abcdef12",
        agent="triage",
        item_id="item-1",
        payload={"run_id": "run-1"},
        agent_version="3",
        account_id="acct-1",
        project_id="proj-1",
    )
    base.update(kw)
    return Lease(**base)  # type: ignore[arg-type]


def test_unreachable_bundle_endpoint_defers_the_item_rather_than_burning_it(
    monkeypatch,
):
    w = _make_worker()
    monkeypatch.setattr(
        w, "_ensure_loaded",
        lambda lease: (_ for _ in ()).throw(TimeoutError("timed out")),
    )
    completed: list[tuple] = []
    monkeypatch.setattr(w, "_report_complete", lambda *a, **k: completed.append((a, k)))
    monkeypatch.setattr(w, "_mark_run_running", lambda _r: None)

    w._handle_lease(_lease())  # must not raise

    assert completed == [], (
        "a transient control-plane blip must not complete the customer's item; "
        "the lease TTL re-dispatches it"
    )


def test_a_bundle_that_will_not_import_fails_its_item_and_spares_the_pool(
    monkeypatch,
):
    w = _make_worker()
    monkeypatch.setattr(
        w, "_ensure_loaded",
        lambda lease: (_ for _ in ()).throw(
            ModuleNotFoundError("No module named 'pandas'")
        ),
    )
    completed: list[dict] = []
    terminal: list[dict] = []
    monkeypatch.setattr(
        w, "_report_complete",
        lambda lease_id, **kw: completed.append({"lease_id": lease_id, **kw}),
    )
    monkeypatch.setattr(
        w, "_mark_run_terminal",
        lambda run_id, status, output=None, **kw: terminal.append(
            {"run_id": run_id, "status": status, **kw}
        ),
    )
    monkeypatch.setattr(w, "_mark_run_running", lambda _r: None)

    w._handle_lease(_lease())  # must not raise

    assert len(completed) == 1
    assert completed[0]["status"] == "failed"
    assert completed[0]["error_category"] == "deps"
    assert "No module named 'pandas'" in completed[0]["error"]

    # The lease is not the run. Without this the item reads "in progress"
    # forever on a path that has already decided the work is over.
    assert terminal == [{
        "run_id": "run-1",
        "status": "failed",
        "error": completed[0]["error"],
        "error_category": "deps",
    }]
