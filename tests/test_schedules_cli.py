"""Tests for the ``papayya schedules`` CLI group."""

from __future__ import annotations

import json
from typing import Any

import pytest
from click.testing import CliRunner

from papayya import cli as cli_module
from papayya.api import PapayyaAPIError


class _FakeSchedules:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.create_return: dict[str, Any] = {"id": "s1", "cron_expression": "0 * * * *"}
        self.list_return: list[dict[str, Any]] = []
        self.get_return: dict[str, Any] = {"id": "s1"}
        self.update_return: dict[str, Any] = {"id": "s1", "cron_expression": "*/15 * * * *"}
        self.enable_return: dict[str, Any] = {"id": "s1", "enabled": True}
        self.disable_return: dict[str, Any] = {"id": "s1", "enabled": False}
        self.raise_on: str | None = None

    def _maybe_raise(self, method: str) -> None:
        if self.raise_on == method:
            raise PapayyaAPIError(500, "boom")

    def create(self, agent_id: str, cron: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("create", {"agent_id": agent_id, "cron": cron, **kwargs}))
        self._maybe_raise("create")
        return self.create_return

    def list(self, agent_id: str | None = None) -> list[dict[str, Any]]:
        self.calls.append(("list", {"agent_id": agent_id}))
        self._maybe_raise("list")
        return self.list_return

    def get(self, schedule_id: str) -> dict[str, Any]:
        self.calls.append(("get", {"schedule_id": schedule_id}))
        self._maybe_raise("get")
        return self.get_return

    def update(self, schedule_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("update", {"schedule_id": schedule_id, **kwargs}))
        self._maybe_raise("update")
        return self.update_return

    def delete(self, schedule_id: str) -> None:
        self.calls.append(("delete", {"schedule_id": schedule_id}))
        self._maybe_raise("delete")

    def enable(self, schedule_id: str) -> dict[str, Any]:
        self.calls.append(("enable", {"schedule_id": schedule_id}))
        self._maybe_raise("enable")
        return self.enable_return

    def disable(self, schedule_id: str) -> dict[str, Any]:
        self.calls.append(("disable", {"schedule_id": schedule_id}))
        self._maybe_raise("disable")
        return self.disable_return


class _FakeClient:
    def __init__(self) -> None:
        self.schedules = _FakeSchedules()
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


def test_create_converts_budget_dollars_to_cents(fake_client: _FakeClient) -> None:
    result = _run([
        "schedules", "create",
        "--agent", "agent-1",
        "--cron", "0 */6 * * *",
        "--timezone", "America/Toronto",
        "--input", "hello",
        "--max-steps", "20",
        "--budget", "5",
    ])
    assert result.exit_code == 0, result.output
    method, kwargs = fake_client.schedules.calls[-1]
    assert method == "create"
    assert kwargs["agent_id"] == "agent-1"
    assert kwargs["cron"] == "0 */6 * * *"
    assert kwargs["timezone"] == "America/Toronto"
    assert kwargs["input"] == "hello"
    assert kwargs["max_steps"] == 20
    assert kwargs["budget_cents"] == 500  # $5 → 500¢


def test_list_forwards_agent_filter(fake_client: _FakeClient) -> None:
    fake_client.schedules.list_return = [{"id": "s1"}, {"id": "s2"}]
    result = _run(["schedules", "list", "--agent", "agent-1"])
    assert result.exit_code == 0, result.output
    assert ("list", {"agent_id": "agent-1"}) in fake_client.schedules.calls
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert [json.loads(ln)["id"] for ln in lines] == ["s1", "s2"]


def test_get_prints_schedule(fake_client: _FakeClient) -> None:
    result = _run(["schedules", "get", "s1"])
    assert result.exit_code == 0, result.output
    assert ("get", {"schedule_id": "s1"}) in fake_client.schedules.calls


def test_update_sends_only_provided_fields(fake_client: _FakeClient) -> None:
    result = _run([
        "schedules", "update", "s1",
        "--cron", "*/15 * * * *",
        "--budget", "10",
    ])
    assert result.exit_code == 0, result.output
    method, kwargs = fake_client.schedules.calls[-1]
    assert method == "update"
    assert kwargs["schedule_id"] == "s1"
    assert kwargs["cron"] == "*/15 * * * *"
    assert kwargs["budget_cents"] == 1000
    assert "timezone" not in kwargs
    assert "input" not in kwargs


def test_update_requires_at_least_one_field(fake_client: _FakeClient) -> None:
    result = CliRunner().invoke(cli_module.main, ["schedules", "update", "s1"])
    assert result.exit_code != 0
    assert "at least one of" in result.output


def test_delete_calls_sdk(fake_client: _FakeClient) -> None:
    result = _run(["schedules", "delete", "s1"])
    assert result.exit_code == 0, result.output
    assert ("delete", {"schedule_id": "s1"}) in fake_client.schedules.calls
    assert "deleted" in result.output


def test_enable_and_disable_call_sdk(fake_client: _FakeClient) -> None:
    result = _run(["schedules", "enable", "s1"])
    assert result.exit_code == 0
    assert ("enable", {"schedule_id": "s1"}) in fake_client.schedules.calls

    result = _run(["schedules", "disable", "s1"])
    assert result.exit_code == 0
    assert ("disable", {"schedule_id": "s1"}) in fake_client.schedules.calls
