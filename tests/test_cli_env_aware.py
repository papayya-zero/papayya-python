"""Parametric tests: every env-aware CLI command picks up --env and uses the
right env's api_key + project_id + base_url when calling the server."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from papayya import _config as cfg_module
from papayya import cli as cli_module
from papayya._config import save_cli_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config_dir = tmp_path / ".papayya"
    config_file = config_dir / "config.json"
    monkeypatch.setattr(cfg_module, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cfg_module, "CONFIG_FILE", config_file)
    monkeypatch.setattr(cli_module, "_CONFIG_FILE", config_file)
    return config_file


@pytest.fixture(autouse=True)
def _clear_papayya_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("PAPAYYA_API_KEY", "PAPAYYA_PROJECT_ID", "PAPAYYA_BASE_URL", "PAPAYYA_ENV"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def two_env_config(tmp_config: Path) -> None:
    save_cli_config({
        "version": 2,
        "current_env": "dev",
        "envs": {
            "dev": {
                "api_key": "cpk_dev",
                "project_id": "p-dev",
                "base_url": "https://dev.papayya.test",
            },
            "staging": {
                "api_key": "cpk_staging",
                "project_id": "p-staging",
                "base_url": "https://staging.papayya.test",
            },
        },
    })


class _FakeAPIClient:
    """Records the APIConfig it was built with and stubs out the methods the
    commands under test call."""

    instances: list["_FakeAPIClient"] = []

    def __init__(self, config: Any) -> None:
        self.config = config
        self.closed = False
        self.calls: list[tuple[str, tuple, dict]] = []
        _FakeAPIClient.instances.append(self)

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))

    # Run / steps
    def get_run(self, run_id: str) -> dict[str, Any]:
        self._record("get_run", run_id)
        return {"id": run_id, "status": "completed", "current_step": 0, "total_cost_cents": 0}

    def get_steps(self, run_id: str) -> list[dict[str, Any]]:
        self._record("get_steps", run_id)
        return []

    # Secrets
    def set_secret(self, project_id: str, name: str, value: str) -> dict[str, Any]:
        self._record("set_secret", project_id, name, value)
        return {}

    def list_secrets(self, project_id: str) -> list[dict[str, Any]]:
        self._record("list_secrets", project_id)
        return []

    def delete_secret(self, project_id: str, name: str) -> dict[str, Any]:
        self._record("delete_secret", project_id, name)
        return {}

    # Rate card
    def get_rate_card(self, project_id: str) -> dict[str, Any]:
        self._record("get_rate_card", project_id)
        return {}

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_instances() -> None:
    _FakeAPIClient.instances.clear()


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> type[_FakeAPIClient]:
    monkeypatch.setattr(cli_module, "APIClient", _FakeAPIClient)
    return _FakeAPIClient


def _invoke(*args: str) -> tuple[int, str]:
    result = CliRunner().invoke(cli_module.main, list(args))
    return result.exit_code, result.output + (result.stderr if result.stderr_bytes else "")


def _last_instance() -> _FakeAPIClient:
    assert _FakeAPIClient.instances, "no APIClient was instantiated"
    return _FakeAPIClient.instances[-1]


# ---------------------------------------------------------------------------
# Direct APIClient consumers
# ---------------------------------------------------------------------------


def test_status_uses_env_credentials(two_env_config: None, fake_api: type[_FakeAPIClient]) -> None:
    code, _ = _invoke("--env", "staging", "status", "run-42")
    assert code == 0
    inst = _last_instance()
    assert inst.config.api_key == "cpk_staging"
    assert inst.config.base_url == "https://staging.papayya.test"


def test_status_default_env_uses_current_env(two_env_config: None, fake_api: type[_FakeAPIClient]) -> None:
    code, _ = _invoke("status", "run-42")
    assert code == 0
    inst = _last_instance()
    assert inst.config.api_key == "cpk_dev"
    assert inst.config.base_url == "https://dev.papayya.test"


def test_logs_uses_env_credentials(two_env_config: None, fake_api: type[_FakeAPIClient]) -> None:
    code, _ = _invoke("--env", "staging", "logs", "run-42")
    assert code == 0
    inst = _last_instance()
    assert inst.config.api_key == "cpk_staging"
    assert inst.config.base_url == "https://staging.papayya.test"


# ---------------------------------------------------------------------------
# secrets — also the legacy project_id bug fix verification
# ---------------------------------------------------------------------------


def test_secrets_set_uses_env_project_id(two_env_config: None, fake_api: type[_FakeAPIClient]) -> None:
    code, out = _invoke("--env", "staging", "secrets", "set", "FOO", "bar")
    assert code == 0, out
    inst = _last_instance()
    assert inst.config.api_key == "cpk_staging"
    assert inst.calls == [("set_secret", ("p-staging", "FOO", "bar"), {})]


def test_secrets_list_uses_env_project_id(two_env_config: None, fake_api: type[_FakeAPIClient]) -> None:
    code, out = _invoke("--env", "staging", "secrets", "list")
    assert code == 0, out
    inst = _last_instance()
    assert inst.calls == [("list_secrets", ("p-staging",), {})]


def test_secrets_delete_uses_env_project_id(two_env_config: None, fake_api: type[_FakeAPIClient]) -> None:
    code, out = _invoke("--env", "staging", "secrets", "delete", "FOO")
    assert code == 0, out
    inst = _last_instance()
    assert inst.calls == [("delete_secret", ("p-staging", "FOO"), {})]


def test_secrets_migrated_account_no_legacy_project_id_still_works(
    tmp_config: Path, fake_api: type[_FakeAPIClient]
) -> None:
    """Regression guard: pre-Phase-1 flat project_id was dropped during
    migration; secrets must read from envs[current_env].project_id instead."""
    save_cli_config({
        "version": 2,
        "current_env": "dev",
        "envs": {"dev": {"api_key": "cpk_dev", "project_id": "p-dev"}},
    })
    code, out = _invoke("secrets", "set", "TOKEN", "val")
    assert code == 0, out
    inst = _last_instance()
    assert inst.calls == [("set_secret", ("p-dev", "TOKEN", "val"), {})]


def test_secrets_flag_overrides_env_project(two_env_config: None, fake_api: type[_FakeAPIClient]) -> None:
    code, out = _invoke("secrets", "set", "--project-id", "p-custom", "X", "y")
    assert code == 0, out
    inst = _last_instance()
    assert inst.calls == [("set_secret", ("p-custom", "X", "y"), {})]


# ---------------------------------------------------------------------------
# rate-card — fixes the subtle api_key/env mismatch bug
# ---------------------------------------------------------------------------


def test_rate_card_show_uses_env_credentials(
    two_env_config: None, fake_api: type[_FakeAPIClient]
) -> None:
    code, out = _invoke("--env", "staging", "rate-card", "show")
    assert code == 0, out
    inst = _last_instance()
    assert inst.config.api_key == "cpk_staging"
    assert inst.config.base_url == "https://staging.papayya.test"
    assert inst.calls == [("get_rate_card", ("p-staging",), {})]


# ---------------------------------------------------------------------------
# runs submit — via _make_papayya_client / Papayya SDK
# ---------------------------------------------------------------------------


class _SpyPapayya:
    instances: list["_SpyPapayya"] = []

    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.closed = False
        _SpyPapayya.instances.append(self)

        class _Runs:
            def __init__(self) -> None:
                self.calls: list[tuple[str, tuple, dict]] = []

            def create_stream(self, **kwargs: Any) -> dict[str, Any]:
                list(kwargs.pop("items", []))  # drain the lazy iterator
                self.calls.append(("create_stream", (), kwargs))
                return {"id": "b-1", "status": "queued", "total_items": 1}

        self.runs = _Runs()

    def close(self) -> None:
        self.closed = True


def test_runs_submit_uses_env_credentials(
    two_env_config: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _SpyPapayya.instances.clear()
    # _make_papayya_client imports Papayya locally — patch the module the CLI imports from.
    import papayya

    monkeypatch.setattr(papayya, "Papayya", _SpyPapayya)

    items = tmp_path / "items.jsonl"
    items.write_text('{"input": "x"}\n', encoding="utf-8")

    code, out = _invoke(
        "--env", "staging", "runs", "submit", "--agent", "a", "--file", str(items)
    )
    assert code == 0, out
    assert _SpyPapayya.instances, "Papayya client was not instantiated"
    spy = _SpyPapayya.instances[-1]
    assert spy.api_key == "cpk_staging"
    assert spy.base_url == "https://staging.papayya.test"


# ---------------------------------------------------------------------------
# explicit --base-url wins over env-stored
# ---------------------------------------------------------------------------


def test_explicit_base_url_wins_over_env_stored(
    two_env_config: None, fake_api: type[_FakeAPIClient]
) -> None:
    code, out = _invoke(
        "--env", "staging",
        "--base-url", "https://override.example",
        "status", "run-1",
    )
    assert code == 0, out
    inst = _last_instance()
    assert inst.config.base_url == "https://override.example"


# ---------------------------------------------------------------------------
# Missing credentials surface the expected message
# ---------------------------------------------------------------------------


def test_secrets_missing_project_id_errors_with_env_hint(
    tmp_config: Path, fake_api: type[_FakeAPIClient]
) -> None:
    save_cli_config({
        "version": 2,
        "current_env": "dev",
        "envs": {"dev": {"api_key": "cpk_dev"}},  # no project_id
    })
    result = CliRunner().invoke(cli_module.main, ["secrets", "set", "FOO", "bar"])
    assert result.exit_code != 0
    # Error goes to stderr; click.testing merges by default
    assert "No project ID for env 'dev'" in result.output or "No project ID for env 'dev'" in (result.stderr or "")
