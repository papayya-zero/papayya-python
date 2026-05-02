"""Schema + loader tests for papayya.yaml."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from papayya._config import PapayyaYaml, PapayyaYamlError, load_yaml


def _write(dir: Path, body: str) -> Path:
    path = dir / "papayya.yaml"
    path.write_text(textwrap.dedent(body))
    return path


def test_minimal_valid_parses(tmp_path: Path) -> None:
    cfg = load_yaml(_write(tmp_path, "version: 1\n"))
    assert cfg.version == 1
    assert cfg.envs == {}


def test_full_example_parses(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """\
        version: 1
        envs:
          dev:
            agents:
              ops-bot:
                schedules:
                  - cron: "0 */6 * * *"
          prod:
            agents:
              ops-bot:
                schedules:
                  - cron: "0 * * * *"
                webhooks:
                  - name: github-issues
                    secret_env: GITHUB_WEBHOOK_SECRET
        """,
    )
    cfg = load_yaml(path)
    assert set(cfg.envs.keys()) == {"dev", "prod"}
    prod = cfg.envs["prod"].agents["ops-bot"]
    assert prod.schedules[0].cron == "0 * * * *"
    assert prod.webhooks[0].name == "github-issues"
    assert prod.webhooks[0].secret_env == "GITHUB_WEBHOOK_SECRET"


def test_missing_file_message(tmp_path: Path) -> None:
    with pytest.raises(PapayyaYamlError, match="not found"):
        load_yaml(tmp_path / "nope.yaml")


def test_empty_file_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "")
    with pytest.raises(PapayyaYamlError, match="empty"):
        load_yaml(path)


def test_non_mapping_top_level_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "- just-a-list\n")
    with pytest.raises(PapayyaYamlError, match="mapping"):
        load_yaml(path)


def test_malformed_yaml_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "version: 1\nenvs: [unclosed\n")
    with pytest.raises(PapayyaYamlError, match="Malformed"):
        load_yaml(path)


def test_unknown_version_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "version: 2\n")
    with pytest.raises(PapayyaYamlError) as excinfo:
        load_yaml(path)
    msg = str(excinfo.value)
    assert "version" in msg
    assert "version: 1" in msg


def test_missing_version_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "envs: {}\n")
    with pytest.raises(PapayyaYamlError, match="version"):
        load_yaml(path)


def test_extra_top_level_field_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "version: 1\nenvs: {}\nbogus_field: 1\n")
    with pytest.raises(PapayyaYamlError, match="bogus_field"):
        load_yaml(path)


def test_extra_agent_field_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """\
        version: 1
        envs:
          dev:
            agents:
              ops-bot:
                schedule:
                  - cron: "0 * * * *"
        """,
    )
    # `schedule` (singular) is a typo for `schedules`; must fail loud.
    with pytest.raises(PapayyaYamlError, match="schedule"):
        load_yaml(path)


def test_webhook_missing_secret_env_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """\
        version: 1
        envs:
          prod:
            agents:
              bot:
                webhooks:
                  - name: incoming
        """,
    )
    with pytest.raises(PapayyaYamlError) as excinfo:
        load_yaml(path)
    msg = str(excinfo.value)
    assert "secret_env" in msg
    # Path-aware message gives the user the exact location.
    assert "prod" in msg and "bot" in msg


def test_schedule_missing_cron_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """\
        version: 1
        envs:
          dev:
            agents:
              bot:
                schedules:
                  - {}
        """,
    )
    with pytest.raises(PapayyaYamlError, match="cron"):
        load_yaml(path)


def test_returns_pydantic_model(tmp_path: Path) -> None:
    cfg = load_yaml(_write(tmp_path, "version: 1\n"))
    assert isinstance(cfg, PapayyaYaml)


def test_partition_key_absent_defaults_none(tmp_path: Path) -> None:
    cfg = load_yaml(_write(tmp_path, "version: 1\n"))
    assert cfg.partition_key is None


def test_partition_key_string_accepted(tmp_path: Path) -> None:
    cfg = load_yaml(
        _write(tmp_path, "version: 1\npartition_key: organization_id\n")
    )
    assert cfg.partition_key == "organization_id"


def test_partition_key_empty_string_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, 'version: 1\npartition_key: ""\n')
    with pytest.raises(PapayyaYamlError, match="non-empty"):
        load_yaml(path)


def test_partition_key_non_string_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "version: 1\npartition_key: 42\n")
    with pytest.raises(PapayyaYamlError):
        load_yaml(path)
