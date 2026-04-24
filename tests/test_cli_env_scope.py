"""Unit tests for `_env_scope` + `_require_*` helpers in cli.py."""

from __future__ import annotations

from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from papayya import _config as cfg_module
from papayya import cli as cli_module
from papayya._config import save_cli_config
from papayya._defaults import DEFAULT_BASE_URL


@pytest.fixture
def tmp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config_dir = tmp_path / ".papayya"
    config_file = config_dir / "config.json"
    monkeypatch.setattr(cfg_module, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cfg_module, "CONFIG_FILE", config_file)
    monkeypatch.setattr(cli_module, "_CONFIG_FILE", config_file)
    return config_file


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("PAPAYYA_API_KEY", "PAPAYYA_PROJECT_ID", "PAPAYYA_BASE_URL", "PAPAYYA_ENV"):
        monkeypatch.delenv(name, raising=False)


def _two_env_config(tmp_config: Path) -> None:
    save_cli_config({
        "version": 2,
        "current_env": "dev",
        "envs": {
            "dev": {"api_key": "cpk_dev", "project_id": "p-dev"},
            "staging": {
                "api_key": "cpk_staging",
                "project_id": "p-staging",
                "base_url": "https://staging.papayya.dev",
            },
        },
    })


# ---------------------------------------------------------------------------
# _env_scope
# ---------------------------------------------------------------------------


def test_env_flag_overrides_current_env(tmp_config: Path) -> None:
    _two_env_config(tmp_config)
    ctx_obj = {"api_key": None, "base_url": DEFAULT_BASE_URL, "env": "staging", "base_url_source": "DEFAULT"}
    scope = cli_module._env_scope(ctx_obj)
    assert scope.env == "staging"
    assert scope.api_key == "cpk_staging"
    assert scope.project_id == "p-staging"
    assert scope.base_url == "https://staging.papayya.dev"


def test_env_omitted_uses_current_env(tmp_config: Path) -> None:
    _two_env_config(tmp_config)
    ctx_obj = {"api_key": None, "base_url": DEFAULT_BASE_URL, "env": None, "base_url_source": "DEFAULT"}
    scope = cli_module._env_scope(ctx_obj)
    assert scope.env == "dev"
    assert scope.api_key == "cpk_dev"
    assert scope.project_id == "p-dev"
    # dev has no base_url → falls back to DEFAULT_BASE_URL
    assert scope.base_url == DEFAULT_BASE_URL


def test_papayya_api_key_env_wins_over_env_config(tmp_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _two_env_config(tmp_config)
    monkeypatch.setenv("PAPAYYA_API_KEY", "cpk_override")
    ctx_obj = {"api_key": None, "base_url": DEFAULT_BASE_URL, "env": "staging", "base_url_source": "DEFAULT"}
    scope = cli_module._env_scope(ctx_obj)
    assert scope.api_key == "cpk_override"
    # project still comes from env config (not overridden)
    assert scope.project_id == "p-staging"


def test_papayya_project_id_env_wins_over_env_config(tmp_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _two_env_config(tmp_config)
    monkeypatch.setenv("PAPAYYA_PROJECT_ID", "p-override")
    ctx_obj = {"api_key": None, "base_url": DEFAULT_BASE_URL, "env": "staging", "base_url_source": "DEFAULT"}
    scope = cli_module._env_scope(ctx_obj)
    assert scope.project_id == "p-override"


def test_ctx_api_key_flag_wins(tmp_config: Path) -> None:
    _two_env_config(tmp_config)
    ctx_obj = {
        "api_key": "cpk_flag",
        "base_url": DEFAULT_BASE_URL,
        "env": "dev",
        "base_url_source": "DEFAULT",
    }
    scope = cli_module._env_scope(ctx_obj)
    assert scope.api_key == "cpk_flag"


def test_explicit_base_url_wins_over_env_stored(tmp_config: Path) -> None:
    _two_env_config(tmp_config)
    ctx_obj = {
        "api_key": None,
        "base_url": "https://override.example",
        "env": "staging",
        "base_url_source": "COMMANDLINE",
    }
    scope = cli_module._env_scope(ctx_obj)
    assert scope.base_url == "https://override.example"


def test_env_stored_base_url_wins_over_default(tmp_config: Path) -> None:
    _two_env_config(tmp_config)
    ctx_obj = {
        "api_key": None,
        "base_url": DEFAULT_BASE_URL,
        "env": "staging",
        "base_url_source": "DEFAULT",
    }
    scope = cli_module._env_scope(ctx_obj)
    assert scope.base_url == "https://staging.papayya.dev"


def test_env_var_base_url_counts_as_explicit(tmp_config: Path) -> None:
    _two_env_config(tmp_config)
    ctx_obj = {
        "api_key": None,
        "base_url": "https://from-envvar.example",
        "env": "staging",
        "base_url_source": "ENVIRONMENT",
    }
    scope = cli_module._env_scope(ctx_obj)
    assert scope.base_url == "https://from-envvar.example"


def test_empty_config_returns_none_credentials(tmp_config: Path) -> None:
    ctx_obj = {"api_key": None, "base_url": DEFAULT_BASE_URL, "env": None, "base_url_source": "DEFAULT"}
    scope = cli_module._env_scope(ctx_obj)
    assert scope.env == "dev"  # default
    assert scope.api_key is None
    assert scope.project_id is None
    assert scope.base_url == DEFAULT_BASE_URL


# ---------------------------------------------------------------------------
# _require_api_key / _require_project_id
# ---------------------------------------------------------------------------


def test_require_api_key_returns_key() -> None:
    scope = cli_module._EnvScope(env="dev", api_key="cpk_x", project_id="p-1", base_url=DEFAULT_BASE_URL)
    assert cli_module._require_api_key(scope) == "cpk_x"


def test_require_api_key_exits_when_missing(capsys: pytest.CaptureFixture[str]) -> None:
    scope = cli_module._EnvScope(env="dev", api_key=None, project_id="p-1", base_url=DEFAULT_BASE_URL)
    with pytest.raises(SystemExit) as exc:
        cli_module._require_api_key(scope)
    assert exc.value.code == 1
    assert "No API key" in capsys.readouterr().err


def test_require_project_id_returns_id() -> None:
    scope = cli_module._EnvScope(env="dev", api_key="cpk_x", project_id="p-1", base_url=DEFAULT_BASE_URL)
    assert cli_module._require_project_id(scope) == "p-1"


def test_require_project_id_exits_with_env_hint(capsys: pytest.CaptureFixture[str]) -> None:
    scope = cli_module._EnvScope(env="staging", api_key="cpk_x", project_id=None, base_url=DEFAULT_BASE_URL)
    with pytest.raises(SystemExit) as exc:
        cli_module._require_project_id(scope)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "staging" in err
    assert "envs link" in err


# ---------------------------------------------------------------------------
# Root callback stashes base_url_source so helpers can read it
# ---------------------------------------------------------------------------


@pytest.fixture
def probe_command():
    """Register a throwaway subcommand that captures ctx.obj, then unregister."""
    captured: dict = {}

    @click.pass_context
    def _probe(ctx: click.Context) -> None:
        captured.update(ctx.obj)

    cmd = click.Command(name="__probe_src", callback=_probe, params=[])
    cli_module.main.add_command(cmd)
    try:
        yield captured
    finally:
        cli_module.main.commands.pop("__probe_src", None)


def test_root_callback_stashes_base_url_source_default(tmp_config: Path, probe_command: dict) -> None:
    """Invoking the group without --base-url should record source=DEFAULT."""
    result = CliRunner().invoke(cli_module.main, ["__probe_src"])
    assert result.exit_code == 0, result.output
    assert probe_command["base_url_source"] == "DEFAULT"
    assert probe_command["base_url"] == DEFAULT_BASE_URL


def test_root_callback_records_commandline_source(tmp_config: Path, probe_command: dict) -> None:
    result = CliRunner().invoke(
        cli_module.main, ["--base-url", "https://x.example", "__probe_src"]
    )
    assert result.exit_code == 0, result.output
    assert probe_command["base_url_source"] == "COMMANDLINE"
    assert probe_command["base_url"] == "https://x.example"


def test_root_callback_records_environment_source(
    tmp_config: Path, probe_command: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PAPAYYA_BASE_URL", "https://from-envvar.example")
    result = CliRunner().invoke(cli_module.main, ["__probe_src"])
    assert result.exit_code == 0, result.output
    assert probe_command["base_url_source"] == "ENVIRONMENT"
    assert probe_command["base_url"] == "https://from-envvar.example"
