"""Unit tests for the Batches SDK resource.

Uses httpx.MockTransport to stub the control-pane, so these don't need a
running backend. Covers the HTTP contract of each method — body shape,
path, query params, auth header. Wait/stream_results happy paths are
exercised with small, deterministic polling so the tests stay under
100ms each.
"""

from __future__ import annotations

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
        return httpx.Response(200, json={"id": "batch-1", "status": "queued"})

    batches, _ = _make_batches(handler)
    result = batches.create(
        agent_id="agent-1",
        items=[{"input": {"prompt": "a"}}, {"input": {"prompt": "b"}}],
        name="lead enrichment",
        budget_cents_cap=2000,
        concurrency_cap=5,
    )

    assert result == {"id": "batch-1", "status": "queued"}
    assert captured["method"] == "POST"
    assert captured["path"] == "/v1/batches"
    assert captured["api_key"] == "cpk_test_key"
    import json

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
        import json

        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json={"id": "batch-2"})

    batches, _ = _make_batches(handler)
    batches.create(agent_id="agent-1", items=[{"input": "x"}])

    assert set(captured["body"].keys()) == {"agent_id", "items"}


def test_create_stream_sends_ndjson_header_then_items() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["content_type"] = request.headers.get("Content-Type")
        captured["body"] = request.read().decode()
        return httpx.Response(200, json={"id": "batch-3", "status": "queued"})

    batches, _ = _make_batches(handler)
    result = batches.create_stream(
        agent_id="agent-9",
        items=[{"input": "a"}, {"input": "b"}, {"input": "c"}],
        budget_cents_cap=500,
    )

    assert result == {"id": "batch-3", "status": "queued"}
    assert captured["content_type"] == "application/x-ndjson"
    lines = captured["body"].rstrip("\n").split("\n")
    assert len(lines) == 4  # 1 header + 3 items
    import json

    header = json.loads(lines[0])
    assert header["agent_id"] == "agent-9"
    assert header["budget_cents_cap"] == 500
    assert "items" not in header  # items go in subsequent lines, not the header
    for i, expected in enumerate(["a", "b", "c"], start=1):
        item = json.loads(lines[i])
        assert item == {"input": expected}


def test_get_hits_batch_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/batches/batch-42"
        return httpx.Response(200, json={"id": "batch-42", "status": "running"})

    batches, _ = _make_batches(handler)
    assert batches.get("batch-42")["status"] == "running"


def test_list_passes_filters_as_query_params() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["query"] = dict(request.url.params)
        return httpx.Response(200, json=[])

    batches, _ = _make_batches(handler)
    batches.list(status="completed", limit=50, offset=100)

    assert captured["query"] == {"status": "completed", "limit": "50", "offset": "100"}


def test_runs_paginates_children() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["query"] = dict(request.url.params)
        return httpx.Response(200, json=[{"id": "run-1"}])

    batches, _ = _make_batches(handler)
    batches.runs("batch-1", status="failed", page=2, limit=10)

    assert captured["path"] == "/v1/batches/batch-1/runs"
    assert captured["query"] == {"status": "failed", "page": "2", "limit": "10"}


def test_cancel_and_retry_failed_are_posts() -> None:
    captured_paths: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_paths.append((request.method, request.url.path))
        return httpx.Response(202, json={"id": "batch-1", "status": "cancelled"})

    batches, _ = _make_batches(handler)
    batches.cancel("batch-1")
    batches.retry_failed("batch-1")

    assert captured_paths == [
        ("POST", "/v1/batches/batch-1/cancel"),
        ("POST", "/v1/batches/batch-1/retry-failed"),
    ]


def test_wait_returns_on_terminal_status() -> None:
    # Sequence: running → running → completed. wait() must poll through
    # the non-terminal states and return the completed payload.
    states = iter([
        {"id": "b", "status": "running"},
        {"id": "b", "status": "running"},
        {"id": "b", "status": "completed"},
    ])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(states))

    batches, _ = _make_batches(handler)
    result = batches.wait("b", timeout=5, poll_interval=0.001)
    assert result["status"] == "completed"


def test_wait_treats_paused_as_terminal() -> None:
    # Paused batches are stuck until the caller bumps the cap — wait()
    # must return rather than loop indefinitely.
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "b", "status": "paused"})

    batches, _ = _make_batches(handler)
    result = batches.wait("b", timeout=5, poll_interval=0.001)
    assert result["status"] == "paused"


def test_wait_times_out() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "b", "status": "running"})

    batches, _ = _make_batches(handler)
    with pytest.raises(TimeoutError):
        batches.wait("b", timeout=0.05, poll_interval=0.01)


def test_stream_results_yields_completed_runs_then_exits() -> None:
    """Backend: first poll returns 2 completed runs + batch running, next poll
    returns 2 more + batch completed. stream_results should yield all 4 once
    each and then stop."""
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/runs"):
            page = int(request.url.params.get("page", "0"))
            status = request.url.params.get("status")
            if status != "completed" or page > 0:
                return httpx.Response(200, json=[])
            call_count["n"] += 1
            if call_count["n"] == 1:
                return httpx.Response(200, json=[{"id": "r1"}, {"id": "r2"}])
            return httpx.Response(200, json=[{"id": "r1"}, {"id": "r2"}, {"id": "r3"}, {"id": "r4"}])
        # batch get
        if call_count["n"] <= 1:
            return httpx.Response(200, json={"id": "b", "status": "running"})
        return httpx.Response(200, json={"id": "b", "status": "completed"})

    batches, _ = _make_batches(handler)
    out = list(batches.stream_results("b", poll_interval=0.001))
    ids = [r["id"] for r in out]
    assert ids == ["r1", "r2", "r3", "r4"]


def test_error_response_raises_papayya_api_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    batches, _ = _make_batches(handler)
    with pytest.raises(PapayyaAPIError) as exc:
        batches.get("b")
    assert exc.value.status == 403


# ── results() — server-streamed NDJSON export ─────────────────────────────


def test_results_streams_ndjson_lines_in_order() -> None:
    """Server emits one JSON object per line; results() yields them as
    parsed dicts in arrival order."""
    captured: dict[str, Any] = {}
    payload_lines = [
        {"id": "r-completed", "status": "completed", "output": "ok"},
        {"id": "r-failed", "status": "failed", "error_code": "rate_limited"},
        {"id": "r-cancelled", "status": "cancelled"},
    ]
    body = "\n".join(__import__("json").dumps(row) for row in payload_lines) + "\n"

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        return httpx.Response(
            200,
            headers={
                "Content-Type": "application/x-ndjson",
                "X-Batch-Status": "terminal",
            },
            content=body.encode("utf-8"),
        )

    batches, _ = _make_batches(handler)
    rows = list(batches.results("batch-x"))

    assert captured["method"] == "GET"
    assert captured["path"] == "/v1/batches/batch-x/results"
    assert rows == payload_lines


def test_results_raises_before_yielding_on_non_200() -> None:
    """A 404 from the server should raise PapayyaAPIError on the first
    call to ``next()``, before any rows are yielded — caller should never
    see a half-streamed response disguised as an empty iterator."""
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="batch not found")

    batches, _ = _make_batches(handler)
    iterator = batches.results("missing")
    with pytest.raises(PapayyaAPIError) as exc:
        next(iterator)
    assert exc.value.status == 404


def test_results_empty_body_yields_nothing() -> None:
    """A batch with no terminal children yet still returns 200 + NDJSON;
    the body is just empty. results() should yield zero rows and exit."""
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "Content-Type": "application/x-ndjson",
                "X-Batch-Status": "running",
            },
            content=b"",
        )

    batches, _ = _make_batches(handler)
    rows = list(batches.results("running-batch"))
    assert rows == []
