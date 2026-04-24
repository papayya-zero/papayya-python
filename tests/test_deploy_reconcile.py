"""CLI-level tests for `papayya deploy` with papayya.yaml reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from papayya import _config as cfg_module
from papayya import cli as cli_module


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@dataclass
class _FakeReg:
    name: str
    model: str = "gpt-4o-mini"
    instructions: str = ""
    fn: Any = None
    tools: list = None
    max_steps: int = 10
    budget_usd: float | None = 1.0


@pytest.fixture
def tmp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config_dir = tmp_path / ".papayya"
    config_file = config_dir / "config.json"
    monkeypatch.setattr(cfg_module, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cfg_module, "CONFIG_FILE", config_file)
    monkeypatch.setattr(cli_module, "_CONFIG_FILE", config_file)
    return config_file


@pytest.fixture
def deploy_env(
    tmp_path: Path,
    tmp_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Set up a working directory with a stub agent file and canned build loop.

    The fixture returns the MagicMock for APIClient (so tests can wire per-call
    responses), plus the instance the CLI will use.
    """
    monkeypatch.chdir(tmp_path)
    # Pretend there's an agent.py so the --no-file branch works.
    (tmp_path / "agent.py").write_text("# stub — patched below\n")

    # Seed config so _resolve_api_key + project id resolution succeed.
    tmp_config.parent.mkdir(parents=True, exist_ok=True)
    tmp_config.write_text(
        '{"version": 2, "current_env": "dev", '
        '"envs": {"dev": {"api_key": "cpk_test", "base_url": "https://x", "project_id": "proj"}, '
        '"prod": {"api_key": "cpk_prod", "base_url": "https://x", "project_id": "proj-prod"}}}'
    )

    monkeypatch.setattr(
        cli_module,
        "_discover_agents",
        lambda _path: [_FakeReg(name="ops-bot")],
    )
    # Avoid real bundling / sleep.
    monkeypatch.setattr("papayya.bundler.bundle_project", lambda *_a, **_k: (b"tar", "sha"))
    monkeypatch.setattr("papayya.cli.time.sleep", lambda _s: None)

    api_patcher = patch.object(cli_module, "APIClient")
    MockClass = api_patcher.start()
    instance = MockClass.return_value

    # Defaults that satisfy the deploy loop.
    instance.list_agents.return_value = [{"id": "agt1", "slug": "ops-bot"}]
    instance.upload_deployment.return_value = {"id": "dep1", "version": "1"}
    instance.get_deployment.return_value = {"status": "ready", "image_ref": "img:1"}
    instance.list_schedules.return_value = []
    instance.list_webhooks.return_value = []
    instance.create_webhook.return_value = {
        "id": "wh1",
        "name": "trigger",
        "secret": "whs_abcdef",
        "trigger_url": "/v1/webhooks/wh1/trigger",
    }
    instance.create_schedule.return_value = {"id": "s1"}
    instance.delete_schedule.return_value = None
    instance.delete_webhook.return_value = None

    yield {"MockClass": MockClass, "api": instance, "tmp_path": tmp_path}

    api_patcher.stop()


def _write_yaml(tmp_path: Path, body: str) -> None:
    (tmp_path / "papayya.yaml").write_text(body)


def _invoke(*args: str) -> tuple[int, str, str]:
    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(cli_module.main, list(args))
    return result.exit_code, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_deploy_without_yaml_is_unchanged(deploy_env: dict[str, Any]) -> None:
    exit_code, stdout, _stderr = _invoke("deploy")
    assert exit_code == 0, stdout
    api = deploy_env["api"]
    api.list_schedules.assert_not_called()
    api.list_webhooks.assert_not_called()
    api.create_schedule.assert_not_called()
    api.create_webhook.assert_not_called()


def test_deploy_multi_env_yaml_without_env_flag_errors(
    deploy_env: dict[str, Any],
) -> None:
    _write_yaml(deploy_env["tmp_path"], """\
version: 1
envs:
  dev:
    agents:
      ops-bot:
        schedules: [{cron: "0 * * * *"}]
  prod:
    agents:
      ops-bot:
        schedules: [{cron: "0 * * * *"}]
""")
    exit_code, _stdout, stderr = _invoke("deploy")
    assert exit_code != 0
    assert "multiple envs" in stderr
    assert "dev" in stderr and "prod" in stderr
    # Must fail before any server calls.
    api = deploy_env["api"]
    api.list_schedules.assert_not_called()
    api.upload_deployment.assert_not_called()


def test_deploy_dry_run_prints_plan_without_mutation(
    deploy_env: dict[str, Any],
) -> None:
    _write_yaml(deploy_env["tmp_path"], """\
version: 1
envs:
  dev:
    agents:
      ops-bot:
        schedules: [{cron: "0 9 * * *"}]
        webhooks: [{name: trigger, secret_env: MY_SECRET}]
""")
    exit_code, stdout, _stderr = _invoke("deploy", "--dry-run")
    assert exit_code == 0, stdout
    assert "Dry run — no changes applied." in stdout
    api = deploy_env["api"]
    api.list_schedules.assert_called()
    api.list_webhooks.assert_called()
    api.create_schedule.assert_not_called()
    api.create_webhook.assert_not_called()
    api.delete_schedule.assert_not_called()
    api.delete_webhook.assert_not_called()


def test_deploy_create_schedule_and_webhook_happy_path(
    deploy_env: dict[str, Any],
) -> None:
    _write_yaml(deploy_env["tmp_path"], """\
version: 1
envs:
  dev:
    agents:
      ops-bot:
        schedules: [{cron: "0 9 * * *"}]
        webhooks: [{name: trigger, secret_env: MY_SECRET}]
""")
    exit_code, stdout, _stderr = _invoke("deploy")
    assert exit_code == 0, stdout
    api = deploy_env["api"]
    api.create_schedule.assert_called_once_with("agt1", "0 9 * * *")
    api.create_webhook.assert_called_once_with("agt1", "trigger")
    assert "Applied 2 of 2 operations." in stdout


def test_deploy_prints_webhook_secret_and_url_on_create(
    deploy_env: dict[str, Any],
) -> None:
    _write_yaml(deploy_env["tmp_path"], """\
version: 1
envs:
  dev:
    agents:
      ops-bot:
        webhooks: [{name: trigger, secret_env: MY_SECRET}]
""")
    exit_code, stdout, _stderr = _invoke("--base-url", "https://control.example.com", "deploy")
    assert exit_code == 0, stdout
    assert "whs_abcdef" in stdout
    assert "https://control.example.com/v1/webhooks/wh1/trigger" in stdout
    assert "MY_SECRET" in stdout


def test_deploy_webhook_rename_prints_rotation_warning(
    deploy_env: dict[str, Any],
) -> None:
    api = deploy_env["api"]
    api.list_webhooks.return_value = [{"id": "wh-old", "name": "old-name"}]
    _write_yaml(deploy_env["tmp_path"], """\
version: 1
envs:
  dev:
    agents:
      ops-bot:
        webhooks: [{name: new-name, secret_env: MY_SECRET}]
""")
    exit_code, stdout, _stderr = _invoke("deploy")
    assert exit_code == 0, stdout
    assert "rotating webhook 'new-name'" in stdout
    api.delete_webhook.assert_called_once_with("wh-old")
    api.create_webhook.assert_called_once_with("agt1", "new-name")


def test_deploy_partial_failure_stops_and_reports(
    deploy_env: dict[str, Any],
) -> None:
    from papayya.api import PapayyaAPIError
    api = deploy_env["api"]
    # First create_schedule succeeds; create_webhook raises.
    api.create_webhook.side_effect = PapayyaAPIError(500, "boom")
    _write_yaml(deploy_env["tmp_path"], """\
version: 1
envs:
  dev:
    agents:
      ops-bot:
        schedules: [{cron: "0 9 * * *"}]
        webhooks:
          - {name: a, secret_env: A}
          - {name: b, secret_env: B}
""")
    exit_code, stdout, stderr = _invoke("deploy")
    assert exit_code != 0
    # 1 schedule create succeeded, then the first webhook create raised before
    # the second webhook was attempted.
    api.create_schedule.assert_called_once()
    assert api.create_webhook.call_count == 1
    assert "Applied 1 of 3 operations." in stderr


def test_deploy_yaml_refs_undeployed_slug_errors_before_server(
    deploy_env: dict[str, Any],
) -> None:
    _write_yaml(deploy_env["tmp_path"], """\
version: 1
envs:
  dev:
    agents:
      ghost-agent:
        schedules: [{cron: "0 * * * *"}]
""")
    exit_code, _stdout, stderr = _invoke("deploy")
    assert exit_code != 0
    assert "ghost-agent" in stderr
    api = deploy_env["api"]
    api.list_schedules.assert_not_called()
    api.create_schedule.assert_not_called()


def test_deploy_shows_agent_id_in_output(deploy_env: dict[str, Any]) -> None:
    exit_code, stdout, _stderr = _invoke("deploy")
    assert exit_code == 0, stdout
    assert "Deployed ops-bot → agt1" in stdout
