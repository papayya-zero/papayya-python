"""Plan 57 D8 — PAPAYYA_API_KEY was honored for auth and ignored for scope.

The walk could not start. From a clean directory with the key exported:

    $ papayya deploy
    Error: HTTP 403: this API key is scoped to a different project

The key was scoped to Ada's project. The agent was in Ada's project. The 403
was real: deploy authenticated with the env-var key and took `project_id` from
`current_env` in ~/.papayya/config.json — left over from an earlier walk and
pointing at a project on a DIFFERENT ACCOUNT. Two sources, nothing checking
that they agree, and the message then blamed the key, which was the correct
half.

It survived success too. With PAPAYYA_PROJECT_ID set as a workaround, deploy
worked and signed off `Env: alpha` / `papayya --env alpha logs <run-id>` — it
deployed to one account and told the user to go look in the other.
"""

from __future__ import annotations

from pathlib import Path

import click
import pytest

from papayya import _config as cfg_module
from papayya import cli as cli_module
from papayya._config import save_cli_config
from papayya._defaults import DEFAULT_BASE_URL

# Captured at IMPORT time, before conftest's autouse stub replaces the module
# attribute for each test. This is the only handle on the real function once
# the stub is installed, and one test below needs it.
_REAL_PROJECTS_FOR_API_KEY = cli_module._projects_for_api_key


@pytest.fixture
def tmp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config_dir = tmp_path / ".papayya"
    config_file = config_dir / "config.json"
    monkeypatch.setattr(cfg_module, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cfg_module, "CONFIG_FILE", config_file)
    monkeypatch.setattr(cli_module, "_CONFIG_FILE", config_file)
    return config_file


def _stale_env_pointing_at_another_account(_: Path) -> None:
    """The exact config the walk started from: env 'alpha' remembers a project
    on an account the exported key has nothing to do with."""
    save_cli_config({
        "version": 2,
        "current_env": "alpha",
        "envs": {"alpha": {"project_id": "46fa0852-other-account"}},
    })


def _key_reaches(monkeypatch: pytest.MonkeyPatch, *ids: str) -> None:
    monkeypatch.setattr(cli_module, "_projects_for_api_key",
                        lambda api_key, base_url: list(ids))


def _ctx() -> dict:
    return {"api_key": None, "base_url": DEFAULT_BASE_URL, "env": None,
            "base_url_source": "DEFAULT"}


# ---------------------------------------------------------------------------
# The headline: an explicit key outranks a remembered env.
# ---------------------------------------------------------------------------

def test_explicit_key_outranks_a_stale_remembered_env(
    tmp_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stale_env_pointing_at_another_account(tmp_config)
    monkeypatch.setenv("PAPAYYA_API_KEY", "cpk_ada")
    _key_reaches(monkeypatch, "c5934959-ada")

    scope = cli_module._env_scope(_ctx())

    assert scope.project_id == "c5934959-ada", (
        "the key names its project; the env is a note the CLI wrote to itself"
    )
    assert scope.project_source == "key"
    assert not scope.env_decided_the_project


def test_a_key_read_out_of_the_env_does_not_outrank_that_env(
    tmp_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The precedence flip is scoped to an EXPLICIT key. An env's own key and
    an env's own project belong together, and asking the server about them
    would be a round trip to confirm what the file already says."""
    save_cli_config({
        "version": 2,
        "current_env": "dev",
        "envs": {"dev": {"api_key": "cpk_dev", "project_id": "p-dev"}},
    })
    _key_reaches(monkeypatch, "p-somewhere-else")

    scope = cli_module._env_scope(_ctx())

    assert scope.project_id == "p-dev"
    assert scope.project_source == "env-config"


# ---------------------------------------------------------------------------
# Two EXPLICIT sources that disagree: refuse, and name both.
# ---------------------------------------------------------------------------

def test_project_id_env_var_the_key_cannot_reach_is_refused(
    tmp_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PAPAYYA_API_KEY", "cpk_ada")
    monkeypatch.setenv("PAPAYYA_PROJECT_ID", "46fa0852-other-account")
    _key_reaches(monkeypatch, "c5934959-ada")

    with pytest.raises(click.ClickException) as exc:
        cli_module._env_scope(_ctx())

    msg = str(exc.value)
    assert "46fa0852-other-account" in msg, "name what the user asked for"
    assert "c5934959-ada" in msg, "and name what the key can actually reach"


def test_project_id_env_var_the_key_can_reach_is_honored(
    tmp_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An account-scoped key with several projects: PAPAYYA_PROJECT_ID is how
    the user picks, and it is not a conflict."""
    monkeypatch.setenv("PAPAYYA_API_KEY", "cpk_account")
    monkeypatch.setenv("PAPAYYA_PROJECT_ID", "p-two")
    _key_reaches(monkeypatch, "p-one", "p-two")

    scope = cli_module._env_scope(_ctx())
    assert scope.project_id == "p-two"
    assert scope.project_source == "flag"


def test_ambiguous_key_falls_back_to_a_remembered_env_it_can_reach(
    tmp_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_cli_config({
        "version": 2,
        "current_env": "dev",
        "envs": {"dev": {"project_id": "p-two"}},
    })
    monkeypatch.setenv("PAPAYYA_API_KEY", "cpk_account")
    _key_reaches(monkeypatch, "p-one", "p-two")

    scope = cli_module._env_scope(_ctx())
    assert scope.project_id == "p-two"
    assert scope.project_source == "env-config"


def test_ambiguous_key_and_an_unreachable_remembered_env_refuses(
    tmp_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing here can be preferred: the key names two projects and the env
    names a third. Guessing is how the original 403 came to be misleading."""
    _stale_env_pointing_at_another_account(tmp_config)
    monkeypatch.setenv("PAPAYYA_API_KEY", "cpk_account")
    _key_reaches(monkeypatch, "p-one", "p-two")

    with pytest.raises(click.ClickException) as exc:
        cli_module._env_scope(_ctx())
    msg = str(exc.value)
    assert "46fa0852-other-account" in msg
    assert "p-one" in msg and "p-two" in msg


# ---------------------------------------------------------------------------
# The lookup is best effort. An unreachable control plane is not evidence.
# ---------------------------------------------------------------------------

def test_unreachable_control_plane_never_becomes_a_refusal(
    tmp_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stale_env_pointing_at_another_account(tmp_config)
    monkeypatch.setenv("PAPAYYA_API_KEY", "cpk_ada")
    monkeypatch.setattr(cli_module, "_projects_for_api_key",
                        lambda api_key, base_url: None)

    scope = cli_module._env_scope(_ctx())

    assert scope.project_id == "46fa0852-other-account"
    assert scope.project_source == "env-config"


def test_projects_for_api_key_returns_none_not_an_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real function, with the network gone. None means 'could not ask'
    and is distinct from [] — 'asked, and the answer was none'."""
    cli_module._PROJECTS_BY_KEY.clear()

    class Boom:
        def __init__(self, *a, **k):
            raise OSError("no route to host")

    monkeypatch.setattr(cli_module, "APIClient", Boom)
    assert _REAL_PROJECTS_FOR_API_KEY("cpk_x", "http://nowhere.invalid") is None
    # No key at all short-circuits before it can raise.
    assert _REAL_PROJECTS_FOR_API_KEY(None, "http://nowhere.invalid") is None
