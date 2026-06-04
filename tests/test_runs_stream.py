"""Unit tests for client.runs.stream() — the SSE step-event iterator.

Uses httpx.MockTransport to serve a canned event-stream body so the
tests exercise both the wire parser (_parse_sse) and the httpx
integration without a live backend.
"""

from __future__ import annotations

from typing import Callable

import httpx
import pytest

from papayya.api import APIClient, APIConfig, PapayyaAPIError
from papayya.resources.runs import Runs, _parse_sse


def _make_runs(handler: Callable[[httpx.Request], httpx.Response]) -> tuple[Runs, APIClient]:
    transport = httpx.MockTransport(handler)
    config = APIConfig(api_key="cpk_test_key", base_url="http://mock")
    api = APIClient(config)
    api._http = httpx.Client(
        base_url=config.base_url,
        timeout=config.timeout,
        headers=api._http.headers,
        transport=transport,
    )
    return Runs(api), api


# ── _parse_sse: wire-format parser ────────────────────────────────────────

class TestParseSSE:
    def test_single_step_event(self) -> None:
        wire = [
            "id: 1",
            "event: step",
            'data: {"step_number": 1, "step_type": "llm_call"}',
            "",
        ]
        events = list(_parse_sse(iter(wire)))
        assert events == [
            {"event": "step", "data": {"step_number": 1, "step_type": "llm_call"}, "id": "1"}
        ]

    def test_comment_keepalive_is_ignored(self) -> None:
        wire = [
            ": keepalive",
            "",
            "id: 2",
            "event: step",
            'data: {"step_number": 2}',
            "",
        ]
        events = list(_parse_sse(iter(wire)))
        assert len(events) == 1
        assert events[0]["id"] == "2"

    def test_terminal_event_no_id(self) -> None:
        wire = [
            "event: terminal",
            'data: {"status": "completed"}',
            "",
        ]
        events = list(_parse_sse(iter(wire)))
        assert events == [{"event": "terminal", "data": {"status": "completed"}}]

    def test_backfill_plus_terminal(self) -> None:
        wire = [
            "id: 1",
            "event: step",
            'data: {"n": 1}',
            "",
            "id: 2",
            "event: step",
            'data: {"n": 2}',
            "",
            "event: terminal",
            'data: {"status": "completed"}',
            "",
        ]
        events = list(_parse_sse(iter(wire)))
        assert [e["event"] for e in events] == ["step", "step", "terminal"]
        assert events[1]["id"] == "2"

    def test_non_json_data_falls_back_to_string(self) -> None:
        wire = ["event: raw", "data: hello world", ""]
        events = list(_parse_sse(iter(wire)))
        assert events == [{"event": "raw", "data": "hello world"}]


# ── Runs.stream: httpx integration ────────────────────────────────────────

def _sse_bytes(*frames: str) -> bytes:
    """Stitch SSE frames into a newline-terminated bytes body."""
    return ("\n".join(frames) + "\n").encode("utf-8")


class TestRunsStream:
    def test_stream_yields_backfilled_step_events(self) -> None:
        body = _sse_bytes(
            "id: 1",
            "event: step",
            'data: {"step_number": 1, "step_type": "llm_call"}',
            "",
            "id: 2",
            "event: step",
            'data: {"step_number": 2, "step_type": "llm_call"}',
            "",
            "event: terminal",
            'data: {"status": "completed"}',
            "",
        )

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/durable/runs/run-123/events"
            assert request.headers["Accept"] == "text/event-stream"
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=body,
            )

        runs, _ = _make_runs(handler)
        events = list(runs.stream("run-123"))
        assert [e["event"] for e in events] == ["step", "step", "terminal"]
        assert events[0]["id"] == "1"
        assert events[1]["data"]["step_number"] == 2
        assert events[2]["data"]["status"] == "completed"

    def test_from_step_sends_last_event_id_header(self) -> None:
        captured: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["last_event_id"] = request.headers.get("Last-Event-ID", "")
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=_sse_bytes("event: terminal", 'data: {"status": "completed"}', ""),
            )

        runs, _ = _make_runs(handler)
        list(runs.stream("run-123", from_step=7))
        assert captured["last_event_id"] == "7"

    def test_omitting_from_step_sends_no_last_event_id(self) -> None:
        captured: dict[str, str | None] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["last_event_id"] = request.headers.get("Last-Event-ID")
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=_sse_bytes("event: terminal", 'data: {"status": "completed"}', ""),
            )

        runs, _ = _make_runs(handler)
        list(runs.stream("run-123"))
        assert captured["last_event_id"] is None

    def test_non_200_response_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "run not found"})

        runs, _ = _make_runs(handler)
        with pytest.raises(PapayyaAPIError) as excinfo:
            list(runs.stream("missing"))
        assert excinfo.value.status == 404

    def test_keepalive_comments_do_not_produce_events(self) -> None:
        body = _sse_bytes(
            ": keepalive",
            "",
            ": keepalive",
            "",
            "id: 1",
            "event: step",
            'data: {"step_number": 1}',
            "",
            "event: terminal",
            'data: {"status": "completed"}',
            "",
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=body,
            )

        runs, _ = _make_runs(handler)
        events = list(runs.stream("run-123"))
        assert [e["event"] for e in events] == ["step", "terminal"]
