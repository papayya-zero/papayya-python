"""Tests for the ``papayya runs`` CLI group (hosted run ops).

The group mirrors ``client.runs`` methods one-for-one. Tests swap the
Papayya client with a recording fake so we can assert the CLI's
translation layer in isolation, without a running backend.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from click.testing import CliRunner

from papayya import cli as cli_module
from papayya.api import PapayyaAPIError


class _FakeRuns:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.list_return: list[dict[str, Any]] = []
        self.cancel_return: dict[str, Any] = {"id": "r1", "status": "cancelled"}
        self.replay_return: dict[str, Any] = {"id": "r1-new", "status": "queued"}
        self.stream_events: list[dict[str, Any]] = []
        self.raise_on: str | None = None

    def _maybe_raise(self, method: str) -> None:
        if self.raise_on == method:
            raise PapayyaAPIError(500, "boom")

    def list(self) -> list[dict[str, Any]]:
        self.calls.append(("list", {}))
        self._maybe_raise("list")
        return self.list_return

    def cancel(self, run_id: str) -> dict[str, Any]:
        self.calls.append(("cancel", {"run_id": run_id}))
        self._maybe_raise("cancel")
        return self.cancel_return

    def replay(self, run_id: str, *, from_step: int, latest: bool = False) -> dict[str, Any]:
        self.calls.append(
            ("replay", {"run_id": run_id, "from_step": from_step, "latest": latest})
        )
        self._maybe_raise("replay")
        return self.replay_return

    def stream(self, run_id: str, *, from_step: int | None = None):
        self.calls.append(("stream", {"run_id": run_id, "from_step": from_step}))
        self._maybe_raise("stream")
        yield from self.stream_events


class _FakeClient:
    def __init__(self) -> None:
        self.runs = _FakeRuns()
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> _FakeClient:
    client = _FakeClient()
    monkeypatch.setattr(cli_module, "_make_papayya_client", lambda ctx: client)
    return client


def _run(args: list[str]) -> Any:
    return CliRunner().invoke(cli_module.main, args, catch_exceptions=False)


def test_runs_list_outputs_ndjson(fake_client: _FakeClient) -> None:
    fake_client.runs.list_return = [{"id": "r1"}, {"id": "r2"}]
    result = _run(["runs", "list"])
    assert result.exit_code == 0, result.output
    assert ("list", {}) in fake_client.runs.calls
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert [json.loads(ln)["id"] for ln in lines] == ["r1", "r2"]


def test_runs_cancel_calls_sdk_and_prints_run(fake_client: _FakeClient) -> None:
    result = _run(["runs", "cancel", "r1"])
    assert result.exit_code == 0, result.output
    assert ("cancel", {"run_id": "r1"}) in fake_client.runs.calls
    payload = json.loads(result.output)
    assert payload["status"] == "cancelled"


def test_runs_stream_emits_one_event_per_line(fake_client: _FakeClient) -> None:
    fake_client.runs.stream_events = [
        {"event": "step", "data": {"step_type": "llm"}, "id": 1},
        {"event": "terminal", "data": {"status": "completed"}},
    ]
    result = _run(["runs", "stream", "r1", "--from-step", "5"])
    assert result.exit_code == 0, result.output
    assert ("stream", {"run_id": "r1", "from_step": 5}) in fake_client.runs.calls
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert [json.loads(ln)["event"] for ln in lines] == ["step", "terminal"]


def test_runs_replay_requires_from_step_and_forwards_latest(
    fake_client: _FakeClient,
) -> None:
    result = _run(["runs", "replay", "r1", "--from-step", "3", "--latest"])
    assert result.exit_code == 0, result.output
    assert (
        "replay",
        {"run_id": "r1", "from_step": 3, "latest": True},
    ) in fake_client.runs.calls
    payload = json.loads(result.output)
    assert payload["status"] == "queued"


def test_runs_replay_errors_without_from_step(fake_client: _FakeClient) -> None:
    result = CliRunner().invoke(cli_module.main, ["runs", "replay", "r1"])
    assert result.exit_code != 0
    assert "--from-step" in result.output
