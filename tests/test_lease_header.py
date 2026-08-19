"""Plan 56 F3 — every hosted write carries the lease it is running under.

The server-side fence is only as good as the header, and the header has to be
PER REQUEST: the CloudStore is constructed once per worker process, before the
customer's module is imported, and the lease changes with every item. A
constructor header would pin the whole process to whichever lease was first —
which is worse than none, because it would authorise a zombie's writes under
the successor's id.
"""

from __future__ import annotations

import httpx
import pytest

from papayya.agent import reset_bootstrap_lease_id, set_bootstrap_lease_id
from papayya.durable.cloud_store import CloudStore, CloudStoreConfig, make_runtime_store

HEADER = "X-Papayya-Lease"


def _store_with_capture(seen: list[dict]) -> CloudStore:
    store = make_runtime_store("http://cp.invalid", "plat-secret")

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.headers))
        return httpx.Response(200, json={})

    store._client = httpx.Client(
        base_url="http://cp.invalid",
        headers=dict(store._client.headers),
        event_hooks=store._client.event_hooks,
        transport=httpx.MockTransport(handler),
    )
    return store


def test_write_carries_the_current_lease(monkeypatch):
    seen: list[dict] = []
    store = _store_with_capture(seen)

    token = set_bootstrap_lease_id("lease-one")
    try:
        store._client.post("/v1/runtime/runs/r1/checkpoints", json={})
    finally:
        reset_bootstrap_lease_id(token)

    assert seen[0].get(HEADER.lower()) == "lease-one"


def test_the_header_follows_the_lease_not_the_client(monkeypatch):
    # The whole reason this is an event hook. One client, two leases.
    seen: list[dict] = []
    store = _store_with_capture(seen)

    for lease in ("lease-one", "lease-two"):
        token = set_bootstrap_lease_id(lease)
        try:
            store._client.post("/v1/runtime/runs/r1/checkpoints", json={})
        finally:
            reset_bootstrap_lease_id(token)

    assert [h.get(HEADER.lower()) for h in seen] == ["lease-one", "lease-two"]


def test_absent_outside_a_hosted_invocation():
    # The local path and the tenant-scoped API have no lease. Sending a stale
    # or invented one would be worse than sending nothing: the server fences
    # only /v1/runtime/*, and a bogus value there reads as "revoked".
    seen: list[dict] = []
    store = _store_with_capture(seen)

    store._client.post("/v1/runtime/runs/r1/checkpoints", json={})

    assert HEADER.lower() not in seen[0]


def test_matches_the_name_the_server_fences_on():
    # Two spellings of this constant would disarm the fence silently: the
    # server would see no header and, in observe mode, log nothing alarming.
    from pathlib import Path

    go = Path(__file__).resolve().parents[2] / "control-pane" / "internal" / "server" / "handler" / "checkpoints.go"
    if not go.exists():
        pytest.skip("control-pane checkout not present")
    assert f'LeaseHeader = "{HEADER}"' in go.read_text()
