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


def test_login_pastes_key_validates_and_persists(tmp_config: Path) -> None:
    # Dashboard-first login: paste a key, validate it via list_projects, and
    # persist it — no email/password, no JWT.
    with patch.object(cli_module, "APIClient") as MockClient:
        instance = MockClient.return_value
        instance.list_projects.return_value = [{"id": "p-only", "name": "Only"}]

        runner = CliRunner()
        result = runner.invoke(cli_module.main, ["login", "--key", "cpk_pasted"])

    assert result.exit_code == 0, result.output
    cfg = load_cli_config()
    env_block = env_config(cfg, current_env(cfg))
    assert env_block["api_key"] == "cpk_pasted"
    assert env_block["project_id"] == "p-only"
    assert "auth" not in cfg  # no JWT stored anymore


def test_login_rejects_bad_key(tmp_config: Path) -> None:
    from papayya.api import PapayyaAPIError

    with patch.object(cli_module, "APIClient") as MockClient:
        instance = MockClient.return_value
        instance.list_projects.side_effect = PapayyaAPIError(401, "nope")

        runner = CliRunner()
        result = runner.invoke(cli_module.main, ["login", "--key", "bad"])

    assert result.exit_code != 0
    assert "rejected" in result.output
    assert load_cli_config() == {}  # nothing persisted


def test_relogin_keeps_the_envs_base_url(tmp_config: Path) -> None:
    """Plan 53 S8, same defect one command over. `login` cannot read its
    endpoint from _env_scope — it is the command that WRITES an env — so it
    reads the --base-url flag. But the flag's DEFAULT is not a user's choice,
    and re-pasting a key for an env configured against localhost must not
    silently repoint it at the public default."""
    save_cli_config({
        "version": 2, "current_env": "dev",
        "envs": {"dev": {"api_key": "cpk_old", "base_url": "http://localhost:8090",
                         "project_id": "p-only"}},
    })
    with patch.object(cli_module, "APIClient") as MockClient:
        instance = MockClient.return_value
        instance.list_projects.return_value = [{"id": "p-only", "name": "Only"}]
        result = CliRunner().invoke(cli_module.main, ["login", "--key", "cpk_new"])

    assert result.exit_code == 0, result.output
    env_block = env_config(load_cli_config(), "dev")
    assert env_block["base_url"] == "http://localhost:8090"
    assert env_block["api_key"] == "cpk_new"
    # And the client it validated against was that endpoint, not the default.
    assert MockClient.call_args.args[0].base_url == "http://localhost:8090"


def test_login_with_an_explicit_base_url_does_repoint(tmp_config: Path) -> None:
    """The escape hatch stays: an EXPLICIT --base-url is a choice and wins."""
    save_cli_config({
        "version": 2, "current_env": "dev",
        "envs": {"dev": {"api_key": "cpk_old", "base_url": "http://localhost:8090",
                         "project_id": "p-only"}},
    })
    with patch.object(cli_module, "APIClient") as MockClient:
        instance = MockClient.return_value
        instance.list_projects.return_value = [{"id": "p-only", "name": "Only"}]
        result = CliRunner().invoke(
            cli_module.main,
            ["--base-url", "https://moved.test", "login", "--key", "cpk_new"],
        )

    assert result.exit_code == 0, result.output
    assert env_config(load_cli_config(), "dev")["base_url"] == "https://moved.test"


def test_login_multi_project_requires_project_id(tmp_config: Path) -> None:
    with patch.object(cli_module, "APIClient") as MockClient:
        instance = MockClient.return_value
        instance.list_projects.return_value = [
            {"id": "p-a", "name": "A"}, {"id": "p-b", "name": "B"},
        ]

        runner = CliRunner()
        result = runner.invoke(cli_module.main, ["login", "--key", "cpk_multi"])

    assert result.exit_code != 0
    assert "--project-id" in result.output
    assert load_cli_config() == {}


def test_envs_create_guides_to_dashboard(tmp_config: Path) -> None:
    # Dashboard-first: `envs create` no longer provisions; it points you at the
    # dashboard + `envs link`. Succeeds without any account session.
    code, out = _invoke("envs", "create", "staging")
    assert code == 0, out
    assert "app.getpapayya.com" in out
    assert "papayya envs link staging" in out


def test_envs_create_does_not_provision(tmp_config: Path) -> None:
    # It must not call the API or persist a half-formed env.
    with patch.object(cli_module, "APIClient") as MockClient:
        code, out = _invoke("envs", "create", "staging")

    assert code == 0, out
    MockClient.assert_not_called()
    cfg = load_cli_config()
    assert "staging" not in cfg.get("envs", {})


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
