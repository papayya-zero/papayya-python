"""Unit tests for the Triage SDK resource + quarantine actions on Runs.

httpx.MockTransport stubs the control-pane so these tests stay
network-free. We assert the HTTP contract: path, method, query string,
request body — and that ``Triage.iter`` follows ``next_cursor`` until
the server returns null.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import httpx

from papayya.api import APIClient, APIConfig
from papayya.resources.items import Items
from papayya.resources.triage import Triage


def _make_clients(handler: Callable[[httpx.Request], httpx.Response]):
    transport = httpx.MockTransport(handler)
    config = APIConfig(api_key="cpk_test", base_url="http://mock")
    api = APIClient(config)
    api._http = httpx.Client(
        base_url=config.base_url,
        timeout=config.timeout,
        headers=api._http.headers,
        transport=transport,
    )
    return Items(api), Triage(api), api


# ── Runs.quarantine / release / discard ──

def test_runs_quarantine_sends_reason() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json={"id": "r1", "status": "quarantine"})

    runs, _, _ = _make_clients(handler)
    out = runs.quarantine("r1", "schema drift")

    assert captured["method"] == "POST"
    assert captured["path"] == "/v1/durable/runs/r1/quarantine"
    assert captured["body"] == {"reason": "schema drift"}
    assert out == {"id": "r1", "status": "quarantine"}


def test_runs_release_no_body() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["body"] = request.read().decode()
        return httpx.Response(200, json={"id": "r2", "status": "running"})

    runs, _, _ = _make_clients(handler)
    runs.release("r2")

    assert captured["method"] == "POST"
    assert captured["path"] == "/v1/durable/runs/r2/release"
    # No body is sent by the SDK on release (server treats as empty/{}).
    assert captured["body"] == ""


def test_runs_discard_no_body() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = request.read().decode()
        return httpx.Response(200, json={"id": "r3", "status": "cancelled"})

    runs, _, _ = _make_clients(handler)
    runs.discard("r3")

    assert captured["path"] == "/v1/durable/runs/r3/discard"
    assert captured["body"] == ""


# ── Triage.list ──

def test_triage_list_request_shape() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["query"] = dict(request.url.params)
        return httpx.Response(200, json={"items": [], "total": 0})

    _, triage, _ = _make_clients(handler)
    triage.list()

    assert captured["method"] == "GET"
    assert captured["path"] == "/v1/triage"
    # `kind=all` and `limit=50` are always sent; optional filters are omitted.
    assert captured["query"]["kind"] == "all"
    assert captured["query"]["limit"] == "50"
    assert "partition_key" not in captured["query"]
    assert "tenant" not in captured["query"]
    assert "cursor" not in captured["query"]


def test_triage_list_forwards_all_filters() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["query"] = dict(request.url.params)
        return httpx.Response(200, json={"items": [], "total": 0})

    _, triage, _ = _make_clients(handler)
    triage.list(
        partition_key="shard-1",
        kind="quarantine",
        cursor="c0",
        limit=25,
    )

    assert captured["query"]["partition_key"] == "shard-1"
    assert captured["query"]["kind"] == "quarantine"
    assert captured["query"]["cursor"] == "c0"
    assert captured["query"]["limit"] == "25"
    assert "workload" not in captured["query"]


# ── Triage.iter ──

def test_triage_iter_follows_next_cursor() -> None:
    pages = [
        {
            "items": [{"run_id": "a", "kind": "dlq"}, {"run_id": "b", "kind": "quarantine"}],
            "next_cursor": "c1",
            "total": 3,
        },
        {
            "items": [{"run_id": "c", "kind": "dlq"}],
            # null/missing next_cursor — iteration exits.
            "total": 3,
        },
    ]
    cursors_seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        cursors_seen.append(request.url.params.get("cursor"))
        # Return next page each time.
        return httpx.Response(200, json=pages.pop(0))

    _, triage, _ = _make_clients(handler)
    rows = list(triage.iter(page_size=2))

    # Two server hits — first without cursor, second carrying "c1".
    assert cursors_seen == [None, "c1"]
    assert [r["run_id"] for r in rows] == ["a", "b", "c"]


def test_triage_iter_stops_on_empty_page() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": [], "total": 0})

    _, triage, _ = _make_clients(handler)
    rows = list(triage.iter())
    assert rows == []
