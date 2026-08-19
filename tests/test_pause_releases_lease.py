"""Plan 41 R6 step 5 — a paused worker RELEASES its lease, it does not complete it.

The bug this pins (parent plan 41 A0): the worker's pause branch called
``_report_complete(status="failed", error_category="paused")``. The
dispatcher moved the item to ``runtime_completed``, which is terminal by
construction, so nothing could ever re-queue it. ``POST /resume`` flipped the
run's status back to ``running`` and left it with no lease and no worker —
"a human is needed here" rewritten as "work is in progress".

Both invocation paths carry the branch (sync at ``_invoke_with_timeout``,
async at ``_invoke_async``) and both are tested here, because they are
separate copies of the same seven lines and a fix applied to one of them is
the exact shape of a bug that ships.
"""

from __future__ import annotations

import pytest

from papayya.errors import CreditExhausted, WorkloadPaused
from papayya.runtime.worker import Lease, Worker


class _RecordingWorker(Worker):
    """A Worker with both report verbs replaced by recorders.

    Subclassing rather than monkeypatching so the branch under test calls
    the real method name on the real object — if someone renames
    ``_report_release`` the test fails to record rather than passing
    vacuously.
    """

    def __init__(self):
        super().__init__(
            dispatcher_url="http://localhost:1/v1/runtime",
            store_path="/tmp/papayya-r6-pause-test.db",
            agent_module_path=None,
            api_key="k",
            heartbeat_interval_seconds=3600,
        )
        self.stop()
        self._hb_stop.set()
        # Plan 56 F2: the constructor now starts a heartbeat PROCESS (a
        # thread cannot heartbeat through a step that holds the GIL). These
        # tests never run the poll loop, whose finally block would stop it,
        # so they have to tear it down themselves or leak one child per test.
        self._stop_heartbeat()
        self.releases: list[tuple[str, str]] = []
        self.completions: list[dict] = []

    def _report_release(self, lease_id, reason):  # type: ignore[override]
        self.releases.append((lease_id, reason))

    def _report_complete(self, lease_id, status, error=None, error_category=None):  # type: ignore[override]
        self.completions.append({
            "lease_id": lease_id,
            "status": status,
            "error": error,
            "error_category": error_category,
        })


def _lease() -> Lease:
    return Lease(lease_id="L1", agent="enrich", item_id="co_42", payload={})


@pytest.fixture
def worker():
    return _RecordingWorker()


def _raiser(exc):
    def fn(_item_id):
        raise exc
    return fn


def _async_raiser(exc):
    async def fn(_item_id):
        raise exc
    return fn


@pytest.mark.parametrize(
    "exc, want_reason",
    [
        (WorkloadPaused("3 consecutive degraded steps", "run_1"), "paused"),
        (CreditExhausted("provider credits exhausted"), "credit"),
    ],
    ids=["fence-pause", "credit-exhausted"],
)
@pytest.mark.parametrize("path", ["sync", "async"])
def test_pause_releases_the_lease(worker, exc, want_reason, path):
    fn = _raiser(exc) if path == "sync" else _async_raiser(exc)

    outcome = worker._invoke_with_timeout(
        fn=fn, lease=_lease(), started_at=0.0, max_duration=None, short="enrich",
    )

    assert worker.releases == [("L1", want_reason)], (
        f"{path} pause path did not release the lease "
        f"(releases={worker.releases}, completions={worker.completions})"
    )
    assert worker.completions == [], (
        "the pause path completed the lease — the item lands in "
        "runtime_completed, which is terminal, and resume can never re-drive it"
    )
    # The run's status is authoritative server-side; the worker must not
    # clobber it with a terminal value on the way out.
    assert outcome.status is None
    assert outcome.output is None
    # A pause is not a failure. Recording an error here would put a message on
    # a record that is going to be resumed and succeed (plan 50).
    assert outcome.error is None
    assert outcome.error_category is None


@pytest.mark.parametrize("path", ["sync", "async"])
def test_ordinary_failures_still_complete(worker, path):
    """The split must be surgical. An ordinary exception is a real failure
    and still completes the lease terminally — releasing it would re-queue
    genuinely broken work forever."""
    exc = ValueError("boom")
    fn = _raiser(exc) if path == "sync" else _async_raiser(exc)

    outcome = worker._invoke_with_timeout(
        fn=fn, lease=_lease(), started_at=0.0, max_duration=None, short="enrich",
    )

    assert worker.releases == []
    assert len(worker.completions) == 1
    assert worker.completions[0]["status"] == "failed"
    assert outcome.status == "failed"
    # Plan 50: the message the worker computes is now CARRIED OUT, not just
    # reported to the lease table and dropped. This is what durable_runs.error
    # is written from, and it is the whole of R3.
    assert outcome.error == "ValueError: boom"
    assert outcome.error_category == "customer_code"
    # The generic exception path used to report NO category at all, which is
    # why every crashed record in the walk had error_category empty.
    assert worker.completions[0]["error_category"] == "customer_code"
