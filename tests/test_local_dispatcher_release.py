"""Plan 41 R6 step 6 — LocalDispatcher serves /release.

``dispatcher.py``'s module docstring declares this the wire-protocol parity
reference for the hosted dispatcher. A worker pointed at it would have 404'd
on the release verb, and there is nothing sensible to fall back to inside an
exception handler — so either the route exists or the parity claim stops
being true. It exists.

Local dev has no ``available_at`` and no resume verb, so the park is a
separate dict rather than a timestamp. What matters is the observable
property, and it is the same one: a parked item is NOT leasable.
"""

from __future__ import annotations

import json
import urllib.request

from papayya.runtime.dispatcher import LocalDispatcher


def _post(dispatcher, path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{dispatcher.port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        raw = resp.read()
        return resp.status, (json.loads(raw) if raw else None)


def test_release_parks_the_item_and_it_is_not_leasable():
    d = LocalDispatcher(port=0, lease_ttl_seconds=60.0)
    try:
        lease_id = d.enqueue(agent="enrich", item_id="co_42")
        item = d._take_lease(worker_id="w1")
        assert item is not None and item.lease_id == lease_id

        status, body = _post(d, "/release", {
            "lease_id": lease_id, "worker_id": "w1", "reason": "paused",
        })
        assert status == 200
        assert body == {"outcome": "released"}

        # THE POINT: parked means unleasable. If the release simply
        # re-queued the item, a worker would pick the paused run straight
        # back up and the pause would mean nothing.
        assert d._take_lease(worker_id="w2") is None
        assert d.stats()["parked"] == 1
        assert d.stats()["leased"] == 0
        # And it did NOT land in the terminal bucket.
        assert d.stats()["completed"] == 0
        assert d.stats()["failed"] == 0
    finally:
        d.shutdown()


def test_release_outcomes_match_the_hosted_vocabulary():
    d = LocalDispatcher(port=0, lease_ttl_seconds=60.0)
    try:

        # stale — never leased here.
        _, body = _post(d, "/release", {"lease_id": "nope", "worker_id": "w1"})
        assert body == {"outcome": "stale"}

        # duplicate — the lease already completed. Releasing it must not
        # resurrect finished work.
        lease_id = d.enqueue(agent="enrich", item_id="co_43")
        d._take_lease(worker_id="w1")
        d._mark_complete(lease_id=lease_id, status="completed", error=None, worker_id="w1")
        _, body = _post(d, "/release", {"lease_id": lease_id, "worker_id": "w1"})
        assert body == {"outcome": "duplicate"}
        assert d.stats()["parked"] == 0

        # A worker's own retry of a successful release: the lease_id was
        # re-minted, so the retry is stale, not an error. Both are 200 —
        # anything else spins the worker's bounded retry.
        lease2 = d.enqueue(agent="enrich", item_id="co_44")
        d._take_lease(worker_id="w1")
        _post(d, "/release", {"lease_id": lease2, "worker_id": "w1"})
        status, body = _post(d, "/release", {"lease_id": lease2, "worker_id": "w1"})
        assert status == 200
        assert body == {"outcome": "stale"}
    finally:
        d.shutdown()


def test_release_rejects_an_unknown_reason():
    d = LocalDispatcher(port=0, lease_ttl_seconds=60.0)
    try:
        lease_id = d.enqueue(agent="enrich", item_id="co_45")
        d._take_lease(worker_id="w1")
        try:
            _post(d, "/release", {"lease_id": lease_id, "reason": "whatever"})
            raise AssertionError("expected 400")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
        # The lease is untouched — a rejected request must not move it.
        assert d.stats()["leased"] == 1
        assert d.stats()["parked"] == 0
    finally:
        d.shutdown()
