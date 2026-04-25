"""Tests for ~/.papayya/config.json migration + envs CLI subcommands."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from papayya import _config as cfg_module
from papayya import cli as cli_module
from papayya._config import (
    current_env,
    env_config,
    list_envs,
    load_cli_config,
    save_cli_config,
    set_env_config,
)


@pytest.fixture
def tmp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the CONFIG_FILE module constants into a tmp dir."""
    config_dir = tmp_path / ".papayya"
    config_file = config_dir / "config.json"
    monkeypatch.setattr(cfg_module, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cfg_module, "CONFIG_FILE", config_file)
    monkeypatch.setattr(cli_module, "_CONFIG_FILE", config_file)
    return config_file


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def test_legacy_config_migrates_into_envs_dev(tmp_config: Path) -> None:
    tmp_config.parent.mkdir(parents=True, exist_ok=True)
    tmp_config.write_text(json.dumps({
        "api_key": "cpk_legacy",
        "base_url": "https://control.papayya.io",
        "project_id": "proj-123",
        "email": "me@example.com",
    }))

    cfg = load_cli_config()

    assert cfg["version"] == 2
    assert cfg["current_env"] == "dev"
    assert cfg["envs"]["dev"]["api_key"] == "cpk_legacy"
    assert cfg["envs"]["dev"]["project_id"] == "proj-123"
    assert cfg["envs"]["dev"]["email"] == "me@example.com"
    assert cfg["_migrated_from_v1"] is True


def test_legacy_jwt_lifts_to_auth(tmp_config: Path) -> None:
    tmp_config.parent.mkdir(parents=True, exist_ok=True)
    tmp_config.write_text(json.dumps({
        "api_key": "cpk_x",
        "jwt": "ey.legacy.token",
        "email": "me@example.com",
    }))

    cfg = load_cli_config()

    assert cfg.get("auth", {}).get("jwt") == "ey.legacy.token"
    assert cfg.get("auth", {}).get("email") == "me@example.com"


def test_v2_config_round_trips(tmp_config: Path) -> None:
    original = {
        "version": 2,
        "current_env": "prod",
        "envs": {
            "dev": {"api_key": "cpk_dev", "project_id": "p-dev"},
            "prod": {"api_key": "cpk_prod", "project_id": "p-prod"},
        },
    }
    tmp_config.parent.mkdir(parents=True, exist_ok=True)
    tmp_config.write_text(json.dumps(original))

    cfg = load_cli_config()
    assert "_migrated_from_v1" not in cfg
    assert cfg == original


def test_save_strips_private_markers(tmp_config: Path) -> None:
    cfg = {
        "version": 2,
        "current_env": "dev",
        "envs": {"dev": {"api_key": "cpk_x"}},
        "_migrated_from_v1": True,
    }
    save_cli_config(cfg)

    on_disk = json.loads(tmp_config.read_text())
    assert "_migrated_from_v1" not in on_disk
    assert on_disk["envs"]["dev"]["api_key"] == "cpk_x"


def test_missing_file_returns_empty(tmp_config: Path) -> None:
    assert load_cli_config() == {}


def test_set_env_config_merges(tmp_config: Path) -> None:
    cfg = {"version": 2, "current_env": "dev", "envs": {"dev": {"api_key": "old", "project_id": "p1"}}}
    set_env_config(cfg, "dev", {"api_key": "new"})
    assert cfg["envs"]["dev"] == {"api_key": "new", "project_id": "p1"}


def test_env_config_returns_copy(tmp_config: Path) -> None:
    cfg = {"version": 2, "current_env": "dev", "envs": {"dev": {"api_key": "k"}}}
    snapshot = env_config(cfg)
    snapshot["api_key"] = "mutated"
    assert cfg["envs"]["dev"]["api_key"] == "k"


def test_current_env_default(tmp_config: Path) -> None:
    assert current_env({}) == "dev"


# ---------------------------------------------------------------------------
# `papayya envs` CLI
# ---------------------------------------------------------------------------


def _invoke(*args: str) -> tuple[int, str]:
    runner = CliRunner()
    result = runner.invoke(cli_module.main, list(args))
    return result.exit_code, result.output


def test_envs_list_empty_shows_hint(tmp_config: Path) -> None:
    code, out = _invoke("envs", "list")
    assert code == 0
    assert "No envs configured" in out


def test_envs_list_marks_current(tmp_config: Path) -> None:
    save_cli_config({
        "version": 2,
        "current_env": "prod",
        "envs": {
            "dev": {"api_key": "cpk_dev", "project_id": "p-dev"},
            "prod": {"api_key": "cpk_prod", "project_id": "p-prod"},
        },
    })
    code, out = _invoke("envs", "list")
    assert code == 0
    # current env marked with '*', other with space
    assert "* prod" in out
    assert "  dev" in out


def test_envs_use_switches_current(tmp_config: Path) -> None:
    save_cli_config({
        "version": 2,
        "current_env": "dev",
        "envs": {
            "dev": {"api_key": "cpk_dev"},
            "prod": {"api_key": "cpk_prod"},
        },
    })
    code, out = _invoke("envs", "use", "prod")
    assert code == 0
    assert "prod" in out
    assert load_cli_config()["current_env"] == "prod"


def test_envs_use_unknown_errors(tmp_config: Path) -> None:
    save_cli_config({
        "version": 2,
        "current_env": "dev",
        "envs": {"dev": {}},
    })
    code, out = _invoke("envs", "use", "ghost")
    assert code != 0
    assert "ghost" in out


def test_envs_link_persists(tmp_config: Path) -> None:
    code, out = _invoke(
        "envs", "link", "staging",
        "--project-id", "p-stg", "--api-key", "cpk_stg",
    )
    assert code == 0, out
    cfg = load_cli_config()
    assert "staging" in list_envs(cfg)
    env_block = env_config(cfg, "staging")
    assert env_block["project_id"] == "p-stg"
    assert env_block["api_key"] == "cpk_stg"


def test_envs_create_requires_jwt(tmp_config: Path) -> None:
    code, out = _invoke("envs", "create", "staging")
    assert code != 0
    assert "papayya login" in out


def test_envs_create_calls_api_and_persists(tmp_config: Path) -> None:
    save_cli_config({
        "version": 2,
        "current_env": "dev",
        "envs": {"dev": {"api_key": "cpk_x"}},
        "auth": {"jwt": "ey.fake.jwt", "email": "me@example.com"},
    })

    # Mock APIClient so no network calls happen.
    fake_project = {"id": "p-new"}
    fake_key = {"key": "cpk_new_key"}
    with patch.object(cli_module, "APIClient") as MockClient:
        instance = MockClient.return_value
        instance.create_project.return_value = fake_project
        instance.create_api_key.return_value = fake_key

        code, out = _invoke("envs", "create", "staging")

    assert code == 0, out
    instance.create_project.assert_called_once()
    instance.create_api_key.assert_called_once_with("p-new", name="cli-env-staging")
    cfg = load_cli_config()
    assert cfg["current_env"] == "staging"
    env_block = env_config(cfg, "staging")
    assert env_block["project_id"] == "p-new"
    assert env_block["api_key"] == "cpk_new_key"


def test_envs_create_duplicate_errors(tmp_config: Path) -> None:
    save_cli_config({
        "version": 2,
        "current_env": "dev",
        "envs": {"dev": {"api_key": "cpk_x"}, "staging": {}},
        "auth": {"jwt": "ey.fake.jwt"},
    })
    code, out = _invoke("envs", "create", "staging")
    assert code != 0
    assert "already exists" in out


# ---------------------------------------------------------------------------
# Legacy flat-key regression guard
# ---------------------------------------------------------------------------


_LEGACY_FLAT_KEY_PATTERN = re.compile(
    r"_?load_cli_config\s*\(\s*\)\s*\.\s*get\s*\(\s*['\"]"
    r"(?:project_id|api_key|base_url|email)['\"]"
)


def test_no_legacy_flat_key_reads() -> None:
    """Top-level reads of env-scoped keys must go through `_env_config()`.

    A Phase 3 bug class came from code paths still reading
    `_load_cli_config().get("project_id")` after the v1→v2 migration. The v2
    config nests api_key/base_url/project_id/email under `envs.<name>`; the
    only sanctioned reader is `_env_config()`.
    """
    src_root = Path(__file__).resolve().parent.parent / "src" / "papayya"
    offenders: list[str] = []
    for py_file in src_root.rglob("*.py"):
        text = py_file.read_text()
        for match in _LEGACY_FLAT_KEY_PATTERN.finditer(text):
            line_no = text[: match.start()].count("\n") + 1
            offenders.append(
                f"{py_file.relative_to(src_root)}:{line_no}: {match.group()}"
            )

    assert not offenders, (
        "Found legacy flat-key reads on config root. Use `_env_config()`:\n"
        + "\n".join(offenders)
    )
