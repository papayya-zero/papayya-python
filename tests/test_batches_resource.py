"""Unit tests for the Batches SDK resource.

Uses httpx.MockTransport to stub the control-pane, so these don't need a
running backend. v1→v2 cutover: only the submission surface
(create / create_stream → POST /v1/batches) survives; the v1 batch read +
lifecycle endpoints retired with the v1 DROP.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import httpx
import pytest

from papayya.api import APIClient, APIConfig, PapayyaAPIError
from papayya.resources.batches import Batches


def _make_batches(handler: Callable[[httpx.Request], httpx.Response]) -> tuple[Batches, APIClient]:
    """Wire a Batches resource backed by a mocked httpx transport."""
    transport = httpx.MockTransport(handler)
    config = APIConfig(api_key="cpk_test_key", base_url="http://mock")
    api = APIClient(config)
    api._http = httpx.Client(
        base_url=config.base_url,
        timeout=config.timeout,
        headers=api._http.headers,
        transport=transport,
    )
    return Batches(api), api


def test_create_sends_items_and_optional_caps() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["body"] = request.read().decode()
        captured["api_key"] = request.headers.get("X-Api-Key")
        return httpx.Response(200, json={"group_id": "grp-1", "status": "queued"})

    batches, _ = _make_batches(handler)
    result = batches.create(
        agent_id="agent-1",
        items=[{"input": {"prompt": "a"}}, {"input": {"prompt": "b"}}],
        name="lead enrichment",
        budget_cents_cap=2000,
        concurrency_cap=5,
    )

    assert result == {"group_id": "grp-1", "status": "queued"}
    assert captured["method"] == "POST"
    assert captured["path"] == "/v1/batches"
    assert captured["api_key"] == "cpk_test_key"

    body = json.loads(captured["body"])
    assert body["agent_id"] == "agent-1"
    assert body["items"] == [{"input": {"prompt": "a"}}, {"input": {"prompt": "b"}}]
    assert body["name"] == "lead enrichment"
    assert body["budget_cents_cap"] == 2000
    assert body["concurrency_cap"] == 5
    # Optional fields not passed should not appear in the body.
    assert "callback_url" not in body
    assert "idempotency_key" not in body


def test_create_minimal_body_omits_optionals() -> None:
    """Caller passes only required fields → no optional keys in the JSON."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json={"group_id": "grp-2"})

    batches, _ = _make_batches(handler)
    batches.create(agent_id="agent-1", items=[{"input": "x"}])

    assert set(captured["body"].keys()) == {"agent_id", "items"}


def test_create_stream_sends_ndjson_header_then_items() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["content_type"] = request.headers.get("Content-Type")
        captured["body"] = request.read().decode()
        return httpx.Response(200, json={"group_id": "grp-3", "status": "queued"})

    batches, _ = _make_batches(handler)
    result = batches.create_stream(
        agent_id="agent-9",
        items=[{"input": "a"}, {"input": "b"}, {"input": "c"}],
        budget_cents_cap=500,
    )

    assert result == {"group_id": "grp-3", "status": "queued"}
    assert captured["content_type"] == "application/x-ndjson"
    lines = captured["body"].rstrip("\n").split("\n")
    assert len(lines) == 4  # 1 header + 3 items

    header = json.loads(lines[0])
    assert header["agent_id"] == "agent-9"
    assert header["budget_cents_cap"] == 500
    assert "items" not in header  # items go in subsequent lines, not the header
    for i, expected in enumerate(["a", "b", "c"], start=1):
        item = json.loads(lines[i])
        assert item == {"input": expected}


def test_create_stream_raises_papayya_api_error_on_non_2xx() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    batches, _ = _make_batches(handler)
    with pytest.raises(PapayyaAPIError) as exc:
        batches.create_stream(agent_id="agent-1", items=[{"input": "x"}])
    assert exc.value.status == 403
