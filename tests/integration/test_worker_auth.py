"""Worker → dispatcher API-key auth wiring.

The hosted dispatcher (control-pane/internal/runtime/dispatcher) sits
behind an X-Api-Key middleware that requires a project-scoped API key
on every /lease, /complete, /heartbeat call. This test pins the worker
side: when ``--api-key`` (or ``PAPAYYA_API_KEY``) is configured, the
worker sends ``X-Api-Key: <key>`` on all three endpoints, and the
dispatcher's matching ``expected_api_key`` parameter accepts/rejects
accordingly.

Two paths covered:
  • happy path: matching key → item completes end-to-end
  • rejection: wrong key → no leases granted, item stays pending
"""

from __future__ import annotations

import time as _t

from ._dispatcher import FakeDispatcher
from .conftest import write_test_agent


_AGENT_SOURCE = '''\
"""Auth-test agent: minimal one-step run."""

from papayya import agent
from papayya.durable import papayya


@agent(name="enrich")
def enrich(item_id: str) -> dict:
    run = papayya().run("enrich", item_id=item_id)
    step = run.step("extract", lambda: {"id": item_id, "ok": True})
    result = step()
    run.complete(result)
    return result
'''


def test_worker_with_matching_api_key_completes_run(
    tmp_path,
    in_memory_store,
    worker_subprocess,
):
    """X-Api-Key matches → lease/complete/heartbeat all succeed."""
    api_key = "pk_test_match"
    dispatcher = FakeDispatcher(expected_api_key=api_key)
    try:
        agent_path = write_test_agent(tmp_path, _AGENT_SOURCE)
        dispatcher.enqueue(agent="enrich", item_id="co_42")

        worker = worker_subprocess(
            agent_module=agent_path,
            dispatcher=dispatcher,
            store=in_memory_store,
            api_key=api_key,
        )

        dispatcher.wait_until_drained(timeout=10.0)

        deadline = _t.monotonic() + 2.0
        run = in_memory_store.run_for_item("co_42")
        while run is None and _t.monotonic() < deadline:
            _t.sleep(0.02)
            run = in_memory_store.run_for_item("co_42")
        assert run is not None, "run never landed in the store"
        assert [t.label for t in run.tasks] == ["extract"]

        worker.stop(timeout=5.0)
        assert worker.exit_code == 0
    finally:
        dispatcher.shutdown()


def test_worker_with_missing_api_key_cannot_lease(
    tmp_path,
    in_memory_store,
    worker_subprocess,
):
    """No X-Api-Key against an auth-required dispatcher → 401, no lease.

    The worker's _poll_lease maps the 401 HTTPError to UNREACHABLE and
    backs off — the item stays pending. We assert pending count > 0
    after a window long enough for many poll attempts to have happened.
    """
    dispatcher = FakeDispatcher(expected_api_key="pk_test_required")
    try:
        agent_path = write_test_agent(tmp_path, _AGENT_SOURCE)
        dispatcher.enqueue(agent="enrich", item_id="co_42")

        # Spawn a worker with NO api_key. Should get rejected by every
        # /lease attempt; backoff keeps it from spinning.
        worker = worker_subprocess(
            agent_module=agent_path,
            dispatcher=dispatcher,
            store=in_memory_store,
            api_key=None,
        )

        # Give it a window to attempt several leases. With backoff the
        # worker may sit idle most of this window — that's fine; we
        # only need to confirm zero leases were granted.
        _t.sleep(1.0)

        stats = dispatcher.stats()
        assert stats["pending"] == 1, (
            f"expected item to stay pending after auth rejection; "
            f"stats={stats}"
        )
        assert stats["leased"] == 0
        assert stats["completed"] == 0
        assert stats["failed"] == 0

        # No lineage row either — the agent never ran.
        assert in_memory_store.run_for_item("co_42") is None

        worker.stop(timeout=5.0)
    finally:
        dispatcher.shutdown()
