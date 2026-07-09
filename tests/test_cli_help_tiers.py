"""Tiered --help (Plan 34 Unit 4).

`papayya --help` must read as a quickstart: the rung-0 loop
(init → example → dev → replay, plus deploy/login) listed first under
"Getting started", hosted/ops groups grouped below, and the deprecated
hidden aliases (`batch`, `webhooks`) absent.
"""

from __future__ import annotations

import re
from typing import Any

from click.testing import CliRunner

from papayya import cli as cli_module


def _help() -> Any:
    return CliRunner().invoke(cli_module.main, ["--help"], catch_exceptions=False)


def test_sections_appear_in_tier_order() -> None:
    result = _help()
    assert result.exit_code == 0
    out = result.output
    i_start = out.index("Getting started:")
    i_run = out.index("Run agents & inspect results:")
    i_ops = out.index("Account & platform ops:")
    assert i_start < i_run < i_ops


def test_rung_zero_commands_lead() -> None:
    result = _help()
    out = result.output
    started = out[out.index("Getting started:"):out.index("Run agents & inspect results:")]
    for cmd in ("init", "example", "dev", "deploy", "replay", "login"):
        assert re.search(rf"^  {cmd}\b", started, flags=re.M), f"{cmd} missing from Getting started"
    # Ops commands must NOT be in the first section.
    for cmd in ("envs", "secrets", "rate-card", "usage"):
        assert not re.search(rf"^  {cmd}\b", started, flags=re.M)


def test_ops_groups_listed_below() -> None:
    result = _help()
    out = result.output
    ops = out[out.index("Account & platform ops:"):]
    for cmd in ("envs", "secrets", "projects", "project", "deployments",
                "api-keys", "usage", "rate-card"):
        assert re.search(rf"^  {cmd}\b", ops, flags=re.M), f"{cmd} missing from ops tier"


def test_hidden_aliases_absent() -> None:
    result = _help()
    out = result.output
    assert not re.search(r"^  batch\b", out, flags=re.M)
    assert not re.search(r"^  webhooks\b", out, flags=re.M)
    assert re.search(r"^  triggers\b", out, flags=re.M)
    assert re.search(r"^  runs\b", out, flags=re.M)
    assert re.search(r"^  items\b", out, flags=re.M)


def test_every_visible_command_is_tiered() -> None:
    """No command silently falls out of the tiers: if a new command isn't
    registered in a section, TieredGroup shows it under 'Other commands' —
    which should stay empty on purpose."""
    result = _help()
    assert "Other commands:" not in result.output
