"""Worker reads `agent_version` off the lease wire (ADR-0003 § Worker #1).

Smallest cutover step toward hosted code distribution: the dispatcher's
JSON body for /lease may include an `agent_version` tag identifying the
deployed bundle the lease should run. The worker's Lease dataclass needs
to carry that field; everything later in the code-distribution build
(`_ensure_loaded`, multi-version registry, hot reload) reads it.

These tests pin the deserialization contract directly. Pattern is
in-process (Worker class + FakeDispatcher) rather than the subprocess
pattern used by `test_worker_acceptance.py` — we only need to verify
`Lease.agent_version` round-trips, not full agent execution. A single
call to `Worker._poll_lease()` is the surgical proof; spinning a
subprocess for it would be overkill.

Coverage:
  • happy path: enqueue with agent_version="v2" → Lease.agent_version == "v2"
  • legacy: enqueue without agent_version → Lease.agent_version is None
  • absence in JSON body: dispatcher's HTTP /lease omits the key when None
"""

from __future__ import annotations

import json
from urllib import request as urllib_request

from papayya.runtime.worker import Worker, _PollOutcome

from ._dispatcher import FakeDispatcher
from .conftest import write_test_agent


# Minimal agent module — registered so the worker's `_import_agent_module`
# completes cleanly. We never actually invoke the agent fn in these tests:
# `_poll_lease` is called directly without entering `run()`.
_MINIMAL_AGENT = '''\
"""Minimal agent: registered so the worker's import succeeds."""

from papayya import agent


@agent(name="enrich")
def enrich(item_id: str) -> dict:
    return {"id": item_id}
'''


def _build_worker(tmp_path, dispatcher_url: str) -> Worker:
    """Construct a Worker without entering its run loop.

    The Worker constructor starts a heartbeat daemon thread
    (worker.py:207-212); the test is responsible for stopping it before
    the test function returns or the daemon will leak past the test.
    Use ``_teardown_worker`` for that.
    """
    agent_path = write_test_agent(tmp_path, _MINIMAL_AGENT)
    return Worker(
        dispatcher_url=dispatcher_url,
        store_path=str(tmp_path / "store.db"),
        agent_module_path=str(agent_path),
    )


def _teardown_worker(w: Worker) -> None:
    """Stop the heartbeat thread the constructor spawned.

    `Worker.run()` would do this in its finally block, but tests in this
    module bypass `run()` entirely.
    """
    w._hb_stop.set()
    w._hb_thread.join(timeout=2)


def test_worker_reads_agent_version_from_lease_json(tmp_path, monkeypatch):
    """Happy path: enqueue with agent_version → Lease carries it."""
    # Worker constructor pops PAPAYYA_API_KEY from env (worker.py:201)
    # to keep CloudStore from being chosen. Use monkeypatch so a stray
    # value in the parent shell doesn't bleed into the test.
    monkeypatch.delenv("PAPAYYA_API_KEY", raising=False)

    dispatcher = FakeDispatcher()
    try:
        dispatcher.enqueue(agent="enrich", item_id="co_42", agent_version="v2")

        w = _build_worker(tmp_path, dispatcher.url)
        try:
            outcome, lease = w._poll_lease()
            assert outcome == _PollOutcome.LEASED
            assert lease is not None
            assert lease.agent == "enrich"
            assert lease.item_id == "co_42"
            assert lease.agent_version == "v2"
        finally:
            _teardown_worker(w)
    finally:
        dispatcher.shutdown()


def test_worker_lease_has_none_agent_version_when_unset(tmp_path, monkeypatch):
    """Legacy path: enqueue without agent_version → Lease.agent_version is None.

    Existing local-dev usage (no agent_version on the wire) must keep
    working; the new field defaults to None on both Lease and the
    dispatcher's _PendingItem.
    """
    monkeypatch.delenv("PAPAYYA_API_KEY", raising=False)

    dispatcher = FakeDispatcher()
    try:
        dispatcher.enqueue(agent="enrich", item_id="co_42")

        w = _build_worker(tmp_path, dispatcher.url)
        try:
            outcome, lease = w._poll_lease()
            assert outcome == _PollOutcome.LEASED
            assert lease is not None
            assert lease.agent_version is None
        finally:
            _teardown_worker(w)
    finally:
        dispatcher.shutdown()


def test_lease_http_body_omits_agent_version_when_none(tmp_path):
    """Wire shape: the /lease JSON body omits agent_version when unset.

    Mirrors the control-pane's `agent_version,omitempty` json tag on
    `model.RuntimeLease` so the two implementations produce byte-equal
    response bodies for the legacy case.
    """
    dispatcher = FakeDispatcher()
    try:
        dispatcher.enqueue(agent="enrich", item_id="co_42")

        with urllib_request.urlopen(
            f"{dispatcher.url}/lease?worker_id=t", timeout=2.0
        ) as resp:
            assert resp.status == 200
            body = json.loads(resp.read().decode("utf-8"))

        assert body["item_id"] == "co_42"
        assert "agent_version" not in body
    finally:
        dispatcher.shutdown()


def test_lease_http_body_includes_agent_version_when_set(tmp_path):
    """Wire shape: the /lease JSON body includes agent_version when set."""
    dispatcher = FakeDispatcher()
    try:
        dispatcher.enqueue(agent="enrich", item_id="co_42", agent_version="v2")

        with urllib_request.urlopen(
            f"{dispatcher.url}/lease?worker_id=t", timeout=2.0
        ) as resp:
            assert resp.status == 200
            body = json.loads(resp.read().decode("utf-8"))

        assert body["agent_version"] == "v2"
    finally:
        dispatcher.shutdown()
