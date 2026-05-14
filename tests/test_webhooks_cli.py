"""Tests for the ``papayya webhooks`` CLI group."""

from __future__ import annotations

import json
from typing import Any

import pytest
from click.testing import CliRunner

from papayya import cli as cli_module
from papayya.api import PapayyaAPIError


class _FakeWebhooks:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.create_return: dict[str, Any] = {"id": "wh-1", "secret": "whsec_..."}
        self.list_return: list[dict[str, Any]] = []
        self.raise_on: str | None = None

    def _maybe_raise(self, method: str) -> None:
        if self.raise_on == method:
            raise PapayyaAPIError(500, "boom")

    def create(self, agent_id: str, name: str, *, description: str | None = None) -> dict[str, Any]:
        self.calls.append(
            ("create", {"agent_id": agent_id, "name": name, "description": description})
        )
        self._maybe_raise("create")
        return self.create_return

    def list(self, agent_id: str) -> list[dict[str, Any]]:
        self.calls.append(("list", {"agent_id": agent_id}))
        self._maybe_raise("list")
        return self.list_return

    def delete(self, webhook_id: str) -> None:
        self.calls.append(("delete", {"webhook_id": webhook_id}))
        self._maybe_raise("delete")


class _FakeClient:
    def __init__(self) -> None:
        self.webhooks = _FakeWebhooks()
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


def test_create_forwards_args(fake_client: _FakeClient) -> None:
    result = _run([
        "webhooks", "create",
        "--agent", "agent-1",
        "--name", "Slack delivery",
        "--description", "fires on terminal status",
    ])
    assert result.exit_code == 0, result.output
    assert (
        "create",
        {"agent_id": "agent-1", "name": "Slack delivery", "description": "fires on terminal status"},
    ) in fake_client.webhooks.calls
    payload = json.loads(result.output)
    assert payload["id"] == "wh-1"


def test_list_outputs_ndjson(fake_client: _FakeClient) -> None:
    fake_client.webhooks.list_return = [{"id": "wh-1"}, {"id": "wh-2"}]
    result = _run(["webhooks", "list", "agent-1"])
    assert result.exit_code == 0, result.output
    assert ("list", {"agent_id": "agent-1"}) in fake_client.webhooks.calls
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert [json.loads(ln)["id"] for ln in lines] == ["wh-1", "wh-2"]


def test_delete_calls_sdk(fake_client: _FakeClient) -> None:
    result = _run(["webhooks", "delete", "wh-1"])
    assert result.exit_code == 0, result.output
    assert ("delete", {"webhook_id": "wh-1"}) in fake_client.webhooks.calls
    assert "deleted" in result.output
