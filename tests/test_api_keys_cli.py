"""Tests for the ``papayya api-keys`` CLI group."""

from __future__ import annotations

import json
from typing import Any

import pytest
from click.testing import CliRunner

from papayya import cli as cli_module
from papayya.api import PapayyaAPIError


class _FakeApiKeys:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.list_return: list[dict[str, Any]] = []
        self.raise_on: str | None = None

    def _maybe_raise(self, method: str) -> None:
        if self.raise_on == method:
            raise PapayyaAPIError(500, "boom")

    def list(self, project_id: str) -> list[dict[str, Any]]:
        self.calls.append(("list", {"project_id": project_id}))
        self._maybe_raise("list")
        return self.list_return

    def revoke(self, project_id: str, key_id: str) -> None:
        self.calls.append(("revoke", {"project_id": project_id, "key_id": key_id}))
        self._maybe_raise("revoke")


class _FakeClient:
    def __init__(self) -> None:
        self.api_keys = _FakeApiKeys()
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
    fake_client.api_keys.list_return = [
        {"id": "key_1", "prefix": "cpk_abc"},
        {"id": "key_2", "prefix": "cpk_def"},
    ]
    result = _run(["api-keys", "list", "--project-id", "proj_1"])
    assert result.exit_code == 0, result.output
    assert ("list", {"project_id": "proj_1"}) in fake_client.api_keys.calls
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert [json.loads(ln)["id"] for ln in lines] == ["key_1", "key_2"]


def test_revoke_requires_confirmation_and_calls_sdk(fake_client: _FakeClient) -> None:
    result = CliRunner().invoke(
        cli_module.main,
        ["api-keys", "revoke", "key_1", "--project-id", "proj_1"],
        input="y\n",
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert (
        "revoke",
        {"project_id": "proj_1", "key_id": "key_1"},
    ) in fake_client.api_keys.calls
    assert "revoked" in result.output


def test_revoke_aborts_on_decline(fake_client: _FakeClient) -> None:
    result = CliRunner().invoke(
        cli_module.main,
        ["api-keys", "revoke", "key_1", "--project-id", "proj_1"],
        input="n\n",
    )
    assert result.exit_code != 0
    assert not any(call[0] == "revoke" for call in fake_client.api_keys.calls)
