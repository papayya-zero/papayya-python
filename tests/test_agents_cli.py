"""Tests for the ``papayya agents`` CLI group."""

from __future__ import annotations

import json
from typing import Any

import pytest
from click.testing import CliRunner

from papayya import cli as cli_module
from papayya.api import PapayyaAPIError


class _FakeAgents:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.create_return: dict[str, Any] = {"id": "agent-1", "name": "Enricher"}
        self.list_return: list[dict[str, Any]] = []
        self.get_return: dict[str, Any] = {"id": "agent-1"}
        self.update_return: dict[str, Any] = {"id": "agent-1", "name": "Renamed"}
        self.raise_on: str | None = None

    def _maybe_raise(self, method: str) -> None:
        if self.raise_on == method:
            raise PapayyaAPIError(500, "boom")

    def create(
        self,
        name: str,
        slug: str,
        project_id: str,
        *,
        config: dict[str, Any] | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "create",
                {
                    "name": name,
                    "slug": slug,
                    "project_id": project_id,
                    "config": config,
                    "description": description,
                },
            )
        )
        self._maybe_raise("create")
        return self.create_return

    def list(self, project_id: str | None = None) -> list[dict[str, Any]]:
        self.calls.append(("list", {"project_id": project_id}))
        self._maybe_raise("list")
        return self.list_return

    def get(self, agent_id: str) -> dict[str, Any]:
        self.calls.append(("get", {"agent_id": agent_id}))
        self._maybe_raise("get")
        return self.get_return

    def update(self, agent_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("update", {"agent_id": agent_id, **kwargs}))
        self._maybe_raise("update")
        return self.update_return


class _FakeClient:
    def __init__(self) -> None:
        self.agents = _FakeAgents()
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


def test_create_forwards_args_and_parses_config(fake_client: _FakeClient) -> None:
    result = _run([
        "agents", "create",
        "--name", "Enricher",
        "--slug", "enricher",
        "--project-id", "proj_1",
        "--description", "lead enricher",
        "--config", '{"model": "gpt-4o"}',
    ])
    assert result.exit_code == 0, result.output
    method, kwargs = fake_client.agents.calls[-1]
    assert method == "create"
    assert kwargs["name"] == "Enricher"
    assert kwargs["slug"] == "enricher"
    assert kwargs["project_id"] == "proj_1"
    assert kwargs["description"] == "lead enricher"
    assert kwargs["config"] == {"model": "gpt-4o"}


def test_create_rejects_invalid_config_json(fake_client: _FakeClient) -> None:
    result = CliRunner().invoke(cli_module.main, [
        "agents", "create",
        "--name", "X", "--slug", "x", "--project-id", "p",
        "--config", "{not-json",
    ])
    assert result.exit_code != 0
    assert "must be valid JSON" in result.output


def test_list_forwards_project_filter(fake_client: _FakeClient) -> None:
    fake_client.agents.list_return = [{"id": "a1"}, {"id": "a2"}]
    result = _run(["agents", "list", "--project-id", "proj_1"])
    assert result.exit_code == 0, result.output
    assert ("list", {"project_id": "proj_1"}) in fake_client.agents.calls
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert [json.loads(ln)["id"] for ln in lines] == ["a1", "a2"]


def test_get_prints_agent(fake_client: _FakeClient) -> None:
    fake_client.agents.get_return = {"id": "agent-1", "name": "Enricher"}
    result = _run(["agents", "get", "agent-1"])
    assert result.exit_code == 0, result.output
    assert ("get", {"agent_id": "agent-1"}) in fake_client.agents.calls
    payload = json.loads(result.output)
    assert payload["id"] == "agent-1"


def test_update_sends_only_provided_fields(fake_client: _FakeClient) -> None:
    result = _run([
        "agents", "update", "agent-1",
        "--name", "Renamed",
        "--config", '{"model": "gpt-4o-mini"}',
    ])
    assert result.exit_code == 0, result.output
    method, kwargs = fake_client.agents.calls[-1]
    assert method == "update"
    assert kwargs["agent_id"] == "agent-1"
    assert kwargs["name"] == "Renamed"
    assert kwargs["config"] == {"model": "gpt-4o-mini"}
    assert "description" not in kwargs  # omitted on the CLI → omitted in patch


def test_update_requires_at_least_one_field(fake_client: _FakeClient) -> None:
    result = CliRunner().invoke(cli_module.main, ["agents", "update", "agent-1"])
    assert result.exit_code != 0
    assert "at least one of" in result.output
