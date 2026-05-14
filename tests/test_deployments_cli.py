"""Tests for the ``papayya deployments`` CLI group."""

from __future__ import annotations

import json
from typing import Any

import pytest
from click.testing import CliRunner

from papayya import cli as cli_module
from papayya.api import PapayyaAPIError


class _FakeDeployments:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.list_return: list[dict[str, Any]] = []
        self.get_return: dict[str, Any] = {"id": "dep_1", "status": "active"}
        self.raise_on: str | None = None

    def _maybe_raise(self, method: str) -> None:
        if self.raise_on == method:
            raise PapayyaAPIError(500, "boom")

    def list(self, agent_id: str) -> list[dict[str, Any]]:
        self.calls.append(("list", {"agent_id": agent_id}))
        self._maybe_raise("list")
        return self.list_return

    def get(self, deployment_id: str) -> dict[str, Any]:
        self.calls.append(("get", {"deployment_id": deployment_id}))
        self._maybe_raise("get")
        return self.get_return


class _FakeClient:
    def __init__(self) -> None:
        self.deployments = _FakeDeployments()
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


def test_list_outputs_ndjson(fake_client: _FakeClient) -> None:
    fake_client.deployments.list_return = [{"id": "dep_1"}, {"id": "dep_2"}]
    result = _run(["deployments", "list", "agent-1"])
    assert result.exit_code == 0, result.output
    assert ("list", {"agent_id": "agent-1"}) in fake_client.deployments.calls
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert [json.loads(ln)["id"] for ln in lines] == ["dep_1", "dep_2"]


def test_get_prints_deployment(fake_client: _FakeClient) -> None:
    result = _run(["deployments", "get", "dep_1"])
    assert result.exit_code == 0, result.output
    assert ("get", {"deployment_id": "dep_1"}) in fake_client.deployments.calls
    payload = json.loads(result.output)
    assert payload["status"] == "active"
