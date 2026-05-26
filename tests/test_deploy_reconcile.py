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

    # Clear the @agent registry so leftover registrations from other test
    # files don't leak into the synthesis splice. The real _discover_agents
    # clears _registry before importing the agent file; the monkeypatched
    # lambda below skips that import, so we clear here instead. Tests that
    # want decorator-attached metadata populate _registry themselves before
    # invoking the CLI.
    from papayya.agent import _registry
    _registry.clear()

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
    # Post-Plan 12: reconciler now calls put_schedules / put_webhooks once
    # per agent per resource. The old create_*/delete_* methods stay on the
    # client for direct-API users but are not in the apply path.
    instance.put_schedules.return_value = {
        "items": [{"id": "s1", "cron_expression": "0 9 * * *", "managed_by": "code"}],
        "summary": {"created": 1, "updated": 0, "deleted": 0, "unchanged": 0},
    }
    instance.put_webhooks.return_value = {
        "items": [{
            "id": "wh1",
            "name": "trigger",
            "managed_by": "code",
            "secret": "whs_abcdef",
            "trigger_url": "/v1/webhooks/wh1/trigger",
        }],
        "summary": {"created": 1, "updated": 0, "deleted": 0, "unchanged": 0},
    }
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
    api.put_schedules.assert_not_called()
    api.put_webhooks.assert_not_called()
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
    api.put_schedules.assert_not_called()
    api.put_webhooks.assert_not_called()
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
    # Post-Plan 12: one PUT per agent per resource type. The old N-call
    # create/delete loop is gone — create_schedule / create_webhook stay
    # on the client for direct-API users but are not in the apply path.
    api.put_schedules.assert_called_once_with(
        "agt1", [{"cron_expression": "0 9 * * *"}],
    )
    api.put_webhooks.assert_called_once_with("agt1", [{"name": "trigger"}])
    api.create_schedule.assert_not_called()
    api.create_webhook.assert_not_called()
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
    # The old name is code-managed; the new yaml omits it -> rename = delete-
    # then-create. Post-Plan 12 the rename is one PUT call replacing the set.
    api.list_webhooks.return_value = [
        {"id": "wh-old", "name": "old-name", "managed_by": "code"},
    ]
    api.put_webhooks.return_value = {
        "items": [{
            "id": "wh-new",
            "name": "new-name",
            "managed_by": "code",
            "secret": "whs_new",
            "trigger_url": "/v1/webhooks/wh-new/trigger",
        }],
        "summary": {"created": 1, "updated": 0, "deleted": 1, "unchanged": 0},
    }
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
    # Desired set carries only the new name; the old name's absence is
    # what triggers the server-side DELETE inside the same transaction.
    api.put_webhooks.assert_called_once_with("agt1", [{"name": "new-name"}])
    api.delete_webhook.assert_not_called()
    api.create_webhook.assert_not_called()


def test_deploy_partial_failure_stops_and_reports(
    deploy_env: dict[str, Any],
) -> None:
    from papayya.api import PapayyaAPIError
    api = deploy_env["api"]
    # put_schedules succeeds; put_webhooks raises. Both webhook ops
    # belong to the failed PUT — applied counts only the schedule op
    # that landed before the failure.
    api.put_webhooks.side_effect = PapayyaAPIError(500, "boom")
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
    api.put_schedules.assert_called_once()
    assert api.put_webhooks.call_count == 1
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
    api.put_schedules.assert_not_called()
    api.create_schedule.assert_not_called()


def test_deploy_shows_agent_id_in_output(deploy_env: dict[str, Any]) -> None:
    exit_code, stdout, _stderr = _invoke("deploy")
    assert exit_code == 0, stdout
    assert "Deployed ops-bot → agt1" in stdout


# ---------------------------------------------------------------------------
# Plan 12: decorator-harvest splice — the deploy flow fuses yaml + decorator
# metadata via _decorator_synthesis.env_spec_from_registry_and_yaml before
# diff_env runs.
# ---------------------------------------------------------------------------


def test_deploy_calls_synthesis_helper_before_diff_env(
    deploy_env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Splice contract: the deploy flow must hand its yaml_env + registry
    to env_spec_from_registry_and_yaml, then pass the helper's RETURN
    value to diff_env (not the raw yaml block)."""
    from papayya._config import AgentSpec, EnvSpec, ScheduleSpec
    from papayya import _decorator_synthesis as synth_mod
    from papayya import _reconcile as reconcile_mod

    sentinel_env = EnvSpec(agents={
        "ops-bot": AgentSpec(schedules=[ScheduleSpec(cron="*/10 * * * *")]),
    })

    synth_calls: list[Any] = []

    def fake_synth(yaml_env: Any, registry: Any) -> EnvSpec:
        synth_calls.append((yaml_env, registry))
        return sentinel_env

    monkeypatch.setattr(
        synth_mod, "env_spec_from_registry_and_yaml", fake_synth,
    )

    diff_calls: list[Any] = []
    real_diff_env = reconcile_mod.diff_env

    def fake_diff_env(env_spec: Any, deployed: Any, api: Any) -> Any:
        diff_calls.append((env_spec, deployed))
        return real_diff_env(env_spec, deployed, api)

    monkeypatch.setattr(reconcile_mod, "diff_env", fake_diff_env)

    _write_yaml(deploy_env["tmp_path"], """\
version: 1
envs:
  dev:
    agents:
      ops-bot:
        schedules: [{cron: "0 9 * * *"}]
""")
    exit_code, stdout, _stderr = _invoke("deploy")
    assert exit_code == 0, stdout
    # Synthesis was called exactly once with the parsed yaml env + the
    # registry dict.
    assert len(synth_calls) == 1
    yaml_env_arg, registry_arg = synth_calls[0]
    assert yaml_env_arg is not None
    assert "ops-bot" in yaml_env_arg.agents
    assert isinstance(registry_arg, dict)
    # diff_env received the SENTINEL env_spec the helper returned, not
    # the raw yaml block.
    assert len(diff_calls) == 1
    diff_env_arg, _deployed = diff_calls[0]
    assert diff_env_arg is sentinel_env


def test_deploy_yaml_only_continues_to_work(deploy_env: dict[str, Any]) -> None:
    """Regression guard: yaml-only customers (no @schedule / @trigger
    decorators) must produce the same plan they did pre-synthesis.
    Exercised via the dry-run path so the test asserts shape, not
    side-effects."""
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
    # The plan was printed (yaml entries surfaced as creates).
    assert "schedule 0 9 * * *" in stdout
    assert "webhook  trigger" in stdout
    api = deploy_env["api"]
    api.put_schedules.assert_not_called()
    api.put_webhooks.assert_not_called()


def test_deploy_decorator_only_no_yaml_succeeds(
    deploy_env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A project with no papayya.yaml but @schedule / @trigger decorators
    on its @agent functions deploys cleanly. The registry-only path
    must reach the reconciler."""
    from papayya._config import ScheduleSpec, WebhookSpec
    from papayya.agent import AgentRegistration, _registry

    # Populate the registry as if decorator-harvest had run inside
    # _discover_agents (which is monkeypatched out in the fixture).
    _registry.clear()
    decorated = AgentRegistration(
        name="ops-bot",
        model="gpt-4o-mini",
        instructions="",
        fn=lambda *_a, **_k: None,
        tools=[],
        max_steps=10,
        budget_usd=1.0,
        schedules=[ScheduleSpec(cron="0 9 * * *")],
        webhooks=[WebhookSpec(name="trigger", secret_env="MY_SECRET")],
    )
    _registry[("ops-bot", "v1")] = decorated

    # No papayya.yaml file written.
    exit_code, stdout, _stderr = _invoke("deploy")
    assert exit_code == 0, stdout
    # The decorator-attached schedule + webhook reached the reconciler
    # and produced one PUT each.
    api = deploy_env["api"]
    api.put_schedules.assert_called_once_with(
        "agt1", [{"cron_expression": "0 9 * * *"}],
    )
    api.put_webhooks.assert_called_once_with("agt1", [{"name": "trigger"}])

    # Clean up the global registry so this test doesn't leak.
    _registry.clear()
