"""Tests for the ``papayya projects`` (plural) CLI group."""

from __future__ import annotations

import json
from typing import Any

import pytest
from click.testing import CliRunner

from papayya import cli as cli_module
from papayya.api import PapayyaAPIError


class _FakeProjects:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.list_return: list[dict[str, Any]] = []
        self.get_return: dict[str, Any] = {"id": "proj_1", "name": "Acme"}
        self.update_return: dict[str, Any] = {"id": "proj_1", "name": "Acme Renamed"}
        self.raise_on: str | None = None

    def _maybe_raise(self, method: str) -> None:
        if self.raise_on == method:
            raise PapayyaAPIError(500, "boom")

    def list(self) -> list[dict[str, Any]]:
        self.calls.append(("list", {}))
        self._maybe_raise("list")
        return self.list_return

    def get(self, project_id: str) -> dict[str, Any]:
        self.calls.append(("get", {"project_id": project_id}))
        self._maybe_raise("get")
        return self.get_return

    def update(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("update", {"project_id": project_id, **kwargs}))
        self._maybe_raise("update")
        return self.update_return

    def delete(self, project_id: str) -> None:
        self.calls.append(("delete", {"project_id": project_id}))
        self._maybe_raise("delete")


class _FakeClient:
    def __init__(self) -> None:
        self.projects = _FakeProjects()
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> _FakeClient:
    client = _FakeClient()
    monkeypatch.setattr(cli_module, "_make_papayya_client", lambda ctx: client)
    return client


def _run(args: list[str], **kwargs: Any) -> Any:
    return CliRunner().invoke(cli_module.main, args, catch_exceptions=False, **kwargs)


def test_list_outputs_ndjson(fake_client: _FakeClient) -> None:
    fake_client.projects.list_return = [{"id": "p1"}, {"id": "p2"}]
    result = _run(["projects", "list"])
    assert result.exit_code == 0, result.output
    assert ("list", {}) in fake_client.projects.calls
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert [json.loads(ln)["id"] for ln in lines] == ["p1", "p2"]


def test_get_prints_project(fake_client: _FakeClient) -> None:
    result = _run(["projects", "get", "proj_1"])
    assert result.exit_code == 0, result.output
    assert ("get", {"project_id": "proj_1"}) in fake_client.projects.calls


def test_update_sends_only_provided_fields(fake_client: _FakeClient) -> None:
    result = _run(["projects", "update", "proj_1", "--name", "Renamed"])
    assert result.exit_code == 0, result.output
    method, kwargs = fake_client.projects.calls[-1]
    assert method == "update"
    assert kwargs["project_id"] == "proj_1"
    assert kwargs["name"] == "Renamed"
    assert "slug" not in kwargs


def test_update_requires_at_least_one_field(fake_client: _FakeClient) -> None:
    result = CliRunner().invoke(cli_module.main, ["projects", "update", "proj_1"])
    assert result.exit_code != 0
    assert "at least one of" in result.output


def test_delete_requires_confirmation_and_calls_sdk(fake_client: _FakeClient) -> None:
    # The --yes flag is auto-added by @click.confirmation_option; pass via input.
    result = CliRunner().invoke(
        cli_module.main, ["projects", "delete", "proj_1"], input="y\n", catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert ("delete", {"project_id": "proj_1"}) in fake_client.projects.calls
    assert "deleted" in result.output


def test_delete_aborts_on_decline(fake_client: _FakeClient) -> None:
    result = CliRunner().invoke(
        cli_module.main, ["projects", "delete", "proj_1"], input="n\n"
    )
    assert result.exit_code != 0
    assert not any(call[0] == "delete" for call in fake_client.projects.calls)
