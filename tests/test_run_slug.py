"""Tests for slug-based `papayya run` resolution."""

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
            "dev": {"api_key": "cpk_dev", "project_id": "p-dev"},
            "staging": {"api_key": "cpk_staging", "project_id": "p-staging"},
        },
    })


AGENT_DEV_UUID = "11111111-1111-4111-8111-111111111111"
AGENT_STAGING_UUID = "22222222-2222-4222-8222-222222222222"


class _FakeAgentsAPI:
    """Records list_agents calls and returns env-scoped fixtures."""

    instances: list["_FakeAgentsAPI"] = []

    def __init__(self, config: Any) -> None:
        self.config = config
        self.closed = False
        self.list_agents_calls: list[str] = []
        self.trigger_calls: list[dict[str, Any]] = []
        _FakeAgentsAPI.instances.append(self)

    def list_agents(self, project_id: str) -> list[dict[str, Any]]:
        self.list_agents_calls.append(project_id)
        if project_id == "p-dev":
            return [{"id": AGENT_DEV_UUID, "slug": "ops-bot"}]
        if project_id == "p-staging":
            return [{"id": AGENT_STAGING_UUID, "slug": "ops-bot"}]
        return []

    def trigger_run(self, **kwargs: Any) -> dict[str, Any]:
        self.trigger_calls.append(kwargs)
        return {"id": "run-1", "status": "completed"}

    def get_run(self, run_id: str) -> dict[str, Any]:
        return {"id": run_id, "status": "completed", "current_step": 0, "total_cost_cents": 0}

    def get_steps(self, run_id: str) -> list[dict[str, Any]]:
        return []

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_instances() -> None:
    _FakeAgentsAPI.instances.clear()


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> type[_FakeAgentsAPI]:
    monkeypatch.setattr(cli_module, "APIClient", _FakeAgentsAPI)
    # Skip the polling sleep so tests are fast.
    monkeypatch.setattr(cli_module.time, "sleep", lambda *_: None)
    return _FakeAgentsAPI


@pytest.fixture
def agent_py(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a minimal agent.py that registers one @agent and chdir into it."""
    agent_file = tmp_path / "agent.py"
    agent_file.write_text(
        "from papayya import agent\n"
        "@agent(name='ops-bot', model='gpt-4o-mini', instructions='do the thing', max_steps=5, budget_usd=0.10)\n"
        "def fn(input_data):\n"
        "    return input_data\n"
    )
    monkeypatch.chdir(tmp_path)
    return agent_file


def _invoke(*args: str) -> Any:
    return CliRunner().invoke(cli_module.main, list(args))


# ---------------------------------------------------------------------------
# Unit tests for _resolve_agent_id
# ---------------------------------------------------------------------------


def test_uuid_positional_passthrough_no_api_call(two_env_config: None, fake_api) -> None:
    result = cli_module._resolve_agent_id(
        AGENT_DEV_UUID,
        None,
        {"api_key": None, "base_url": cli_module.DEFAULT_BASE_URL, "env": None, "base_url_source": "DEFAULT"},
    )
    assert result == AGENT_DEV_UUID
    # No APIClient constructed — no list_agents round-trip.
    assert _FakeAgentsAPI.instances == []


def test_slug_hit_resolves_via_list_agents(two_env_config: None, fake_api) -> None:
    result = cli_module._resolve_agent_id(
        "ops-bot",
        None,
        {"api_key": None, "base_url": cli_module.DEFAULT_BASE_URL, "env": "staging", "base_url_source": "DEFAULT"},
    )
    assert result == AGENT_STAGING_UUID
    inst = _FakeAgentsAPI.instances[-1]
    assert inst.list_agents_calls == ["p-staging"]
    assert inst.config.api_key == "cpk_staging"


def test_slug_miss_errors_with_available_slugs(two_env_config: None, fake_api, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_module._resolve_agent_id(
            "ghost",
            None,
            {"api_key": None, "base_url": cli_module.DEFAULT_BASE_URL, "env": "dev", "base_url_source": "DEFAULT"},
        )
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "no agent 'ghost'" in err
    assert "env 'dev'" in err
    assert "ops-bot" in err


def test_agent_id_flag_wins_over_positional(two_env_config: None, fake_api) -> None:
    result = cli_module._resolve_agent_id(
        "ignored-slug",
        AGENT_DEV_UUID,
        {"api_key": None, "base_url": cli_module.DEFAULT_BASE_URL, "env": "dev", "base_url_source": "DEFAULT"},
    )
    assert result == AGENT_DEV_UUID
    # flag path skips the API entirely
    assert _FakeAgentsAPI.instances == []


def test_no_agent_at_all_errors(fake_api, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_module._resolve_agent_id(
            None,
            None,
            {"api_key": None, "base_url": cli_module.DEFAULT_BASE_URL, "env": None, "base_url_source": "DEFAULT"},
        )
    assert exc.value.code == 1
    assert "agent required" in capsys.readouterr().err


def test_slug_resolution_is_env_scoped(two_env_config: None, fake_api) -> None:
    """Same slug 'ops-bot' resolves to different UUIDs in dev vs staging."""
    dev_ctx = {"api_key": None, "base_url": cli_module.DEFAULT_BASE_URL, "env": "dev", "base_url_source": "DEFAULT"}
    staging_ctx = {"api_key": None, "base_url": cli_module.DEFAULT_BASE_URL, "env": "staging", "base_url_source": "DEFAULT"}
    assert cli_module._resolve_agent_id("ops-bot", None, dev_ctx) == AGENT_DEV_UUID
    assert cli_module._resolve_agent_id("ops-bot", None, staging_ctx) == AGENT_STAGING_UUID


# ---------------------------------------------------------------------------
# End-to-end: `papayya run` command surface
# ---------------------------------------------------------------------------


def test_run_slug_positional_triggers_in_env(
    two_env_config: None, fake_api, agent_py: Path
) -> None:
    result = _invoke("--env", "staging", "run", "ops-bot", "hello")
    assert result.exit_code == 0, result.output
    # There will be at least 2 APIClients: one for list_agents, one for trigger.
    assert len(_FakeAgentsAPI.instances) >= 2
    trigger_inst = _FakeAgentsAPI.instances[-1]
    assert trigger_inst.trigger_calls, "trigger_run was not called"
    call = trigger_inst.trigger_calls[0]
    assert call["agent_id"] == AGENT_STAGING_UUID
    assert call["input_data"] == {"message": "hello"}


def test_run_agent_id_flag_still_works(
    two_env_config: None, fake_api, agent_py: Path
) -> None:
    result = _invoke("run", "--agent-id", AGENT_DEV_UUID, "--input", "hi")
    assert result.exit_code == 0, result.output
    # No list_agents call because --agent-id wins.
    for inst in _FakeAgentsAPI.instances:
        assert inst.list_agents_calls == []
    trigger_inst = _FakeAgentsAPI.instances[-1]
    assert trigger_inst.trigger_calls[0]["agent_id"] == AGENT_DEV_UUID


def test_run_slug_miss_errors(two_env_config: None, fake_api, agent_py: Path) -> None:
    result = _invoke("run", "ghost", "hello")
    assert result.exit_code != 0
    assert "no agent 'ghost'" in result.output


def test_run_rejects_dual_input(two_env_config: None, fake_api, agent_py: Path) -> None:
    result = _invoke("run", "ops-bot", "positional", "--input", "also-flag")
    assert result.exit_code != 0
    assert "provided twice" in result.output


def test_run_legacy_flag_style_still_works(
    two_env_config: None, fake_api, agent_py: Path
) -> None:
    """Scripts that pass --file + --input + --agent-id continue to work."""
    result = _invoke(
        "run",
        "--file", "agent.py",
        "--input", "hi",
        "--agent-id", AGENT_DEV_UUID,
    )
    assert result.exit_code == 0, result.output
