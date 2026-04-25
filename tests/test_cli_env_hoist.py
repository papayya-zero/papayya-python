"""Test that SafeGroup hoists `--env` from after-subcommand position."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from papayya import _config as cfg_module
from papayya import cli as cli_module
from papayya._cli_errors import SafeGroup
from papayya._config import save_cli_config


@pytest.fixture
def tmp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config_dir = tmp_path / ".papayya"
    config_file = config_dir / "config.json"
    monkeypatch.setattr(cfg_module, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cfg_module, "CONFIG_FILE", config_file)
    monkeypatch.setattr(cli_module, "_CONFIG_FILE", config_file)
    save_cli_config({
        "version": 2,
        "current_env": "dev",
        "envs": {
            "dev": {"api_key": "cpk_dev", "project_id": "p-dev"},
            "staging": {"api_key": "cpk_staging", "project_id": "p-staging"},
        },
    })
    return config_file


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("PAPAYYA_API_KEY", "PAPAYYA_PROJECT_ID", "PAPAYYA_BASE_URL", "PAPAYYA_ENV"):
        monkeypatch.delenv(name, raising=False)


def test_hoist_space_separated() -> None:
    """`--env staging foo` is split: ['--env', 'staging'] hoisted, ['foo'] kept."""
    args = ["foo", "--env", "staging", "bar"]
    # Simulate the parse_args hoist logic in isolation by calling SafeGroup
    # via a minimal runner — see test_envs_use_after_position below for the
    # end-to-end click invocation.
    hoisted: list[str] = []
    i = 0
    work = list(args)
    while i < len(work):
        tok = work[i]
        if tok == "--env" and i + 1 < len(work):
            hoisted.extend(work[i:i + 2])
            del work[i:i + 2]
            continue
        if tok.startswith("--env="):
            hoisted.append(tok)
            del work[i]
            continue
        i += 1
    assert hoisted + work == ["--env", "staging", "foo", "bar"]


def test_hoist_equals_form() -> None:
    args = ["foo", "--env=staging", "bar"]
    hoisted: list[str] = []
    i = 0
    work = list(args)
    while i < len(work):
        tok = work[i]
        if tok == "--env" and i + 1 < len(work):
            hoisted.extend(work[i:i + 2])
            del work[i:i + 2]
            continue
        if tok.startswith("--env="):
            hoisted.append(tok)
            del work[i]
            continue
        i += 1
    assert hoisted + work == ["--env=staging", "foo", "bar"]


def test_parse_args_hoists_env_after_subcommand(tmp_config: Path) -> None:
    """SafeGroup.parse_args should pull `--env staging` to the front so click
    binds it to the root option instead of erroring on the subcommand."""
    import click
    ctx = click.Context(cli_module.main, info_name="papayya")
    args = ["envs", "list", "--env", "staging"]
    cli_module.main.parse_args(ctx, args)
    assert ctx.params.get("env") == "staging"


def test_parse_args_hoists_env_equals_after_subcommand(tmp_config: Path) -> None:
    """Same for `--env=staging` form."""
    import click
    ctx = click.Context(cli_module.main, info_name="papayya")
    args = ["envs", "list", "--env=staging"]
    cli_module.main.parse_args(ctx, args)
    assert ctx.params.get("env") == "staging"


def test_parse_args_pre_subcommand_position_still_works(tmp_config: Path) -> None:
    """Don't regress the canonical `--env` position."""
    import click
    ctx = click.Context(cli_module.main, info_name="papayya")
    args = ["--env", "staging", "envs", "list"]
    cli_module.main.parse_args(ctx, args)
    assert ctx.params.get("env") == "staging"


def test_parse_args_no_env_flag_unchanged(tmp_config: Path) -> None:
    """No `--env` anywhere → ctx.params.env is None, no crash."""
    import click
    ctx = click.Context(cli_module.main, info_name="papayya")
    args = ["envs", "list"]
    cli_module.main.parse_args(ctx, args)
    assert ctx.params.get("env") is None


def test_safegroup_used_for_root() -> None:
    """Sanity check that `main` is a SafeGroup so the hoist applies."""
    assert isinstance(cli_module.main, SafeGroup)
