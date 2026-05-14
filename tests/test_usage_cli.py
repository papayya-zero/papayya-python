"""Tests for the ``papayya usage`` CLI group."""

from __future__ import annotations

import json
from typing import Any

import pytest
from click.testing import CliRunner

from papayya import cli as cli_module
from papayya.api import PapayyaAPIError


class _FakeUsage:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.summary_return: dict[str, Any] = {
            "total_cost_cents": 10000,
            "total_runs": 1234,
        }
        self.breakdown_return: list[dict[str, Any]] = []
        self.raise_on: str | None = None

    def _maybe_raise(self, method: str) -> None:
        if self.raise_on == method:
            raise PapayyaAPIError(500, "boom")

    def summary(self, *, from_date: str | None = None, to_date: str | None = None) -> dict[str, Any]:
        self.calls.append(("summary", {"from_date": from_date, "to_date": to_date}))
        self._maybe_raise("summary")
        return self.summary_return

    def breakdown(self, *, from_date: str | None = None, to_date: str | None = None) -> list[dict[str, Any]]:
        self.calls.append(("breakdown", {"from_date": from_date, "to_date": to_date}))
        self._maybe_raise("breakdown")
        return self.breakdown_return


class _FakeClient:
    def __init__(self) -> None:
        self.usage = _FakeUsage()
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


def test_summary_forwards_dates(fake_client: _FakeClient) -> None:
    result = _run([
        "usage", "summary",
        "--from", "2026-05-01", "--to", "2026-05-14",
    ])
    assert result.exit_code == 0, result.output
    assert (
        "summary",
        {"from_date": "2026-05-01", "to_date": "2026-05-14"},
    ) in fake_client.usage.calls
    payload = json.loads(result.output)
    assert payload["total_runs"] == 1234


def test_summary_with_no_dates(fake_client: _FakeClient) -> None:
    result = _run(["usage", "summary"])
    assert result.exit_code == 0, result.output
    assert ("summary", {"from_date": None, "to_date": None}) in fake_client.usage.calls


def test_breakdown_outputs_ndjson(fake_client: _FakeClient) -> None:
    fake_client.usage.breakdown_return = [
        {"agent_id": "a1", "cost_cents": 1234},
        {"agent_id": "a2", "cost_cents": 567},
    ]
    result = _run(["usage", "breakdown", "--from", "2026-05-01"])
    assert result.exit_code == 0, result.output
    assert (
        "breakdown",
        {"from_date": "2026-05-01", "to_date": None},
    ) in fake_client.usage.calls
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert [json.loads(ln)["agent_id"] for ln in lines] == ["a1", "a2"]
