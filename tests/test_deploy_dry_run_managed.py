"""CLI-level tests for the managed_by='code' PUT-dry-run preview
(Plan 13). Probes `papayya deploy [--dry-run]`'s splice that calls
`api.put_schedules(..., dry_run=True)` / `api.put_webhooks(...,
dry_run=True)` before the apply attempt, and renders the diff below
the legacy `_print_reconcile_plan` output.

Fixture pattern mirrors `tests/test_deploy_reconcile.py` — `tmp_config`,
a `deploy_env` analog wired with MagicMock APIClient. Shared fixtures
are imported from that file to avoid drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from papayya import cli as cli_module

# Reuse the heavy-lifting fixtures from test_deploy_reconcile rather than
# duplicating them — keeps both files in lockstep when the deploy flow
# evolves and is the convention this suite already follows.
from tests.test_deploy_reconcile import deploy_env, tmp_config  # noqa: F401


def _write_yaml(tmp_path: Path, body: str) -> None:
    (tmp_path / "papayya.yaml").write_text(body)


def _invoke(*args: str) -> tuple[int, str, str]:
    runner = CliRunner()  # click>=8.2: stderr is separate by default
    result = runner.invoke(cli_module.main, list(args))
    return result.exit_code, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Test 1 — dry-run calls put_* with dry_run=True; legacy create/delete idle
# ---------------------------------------------------------------------------

def test_dry_run_calls_put_endpoints_with_dry_run_flag(
    deploy_env: dict[str, Any],
) -> None:
    _write_yaml(deploy_env["tmp_path"], """\
version: 1
envs:
  dev:
    agents:
      ops-bot:
        schedules:
          - {cron: "0 9 * * *"}
          - {cron: "0 17 * * *"}
        webhooks:
          - {name: trigger, secret_env: MY_SECRET}
""")
    exit_code, stdout, _stderr = _invoke("deploy", "--dry-run")
    assert exit_code == 0, stdout
    api = deploy_env["api"]
    # PUT preview called exactly once with dry_run=True per resource type.
    api.put_schedules.assert_called_once_with(
        "agt1",
        [
            {"cron_expression": "0 9 * * *", "timezone": "UTC"},
            {"cron_expression": "0 17 * * *", "timezone": "UTC"},
        ],
        dry_run=True,
    )
    api.put_webhooks.assert_called_once_with(
        "agt1",
        [{"name": "trigger", "secret_env": "MY_SECRET"}],
        dry_run=True,
    )
    # Legacy per-call mutators stay idle.
    api.create_schedule.assert_not_called()
    api.delete_schedule.assert_not_called()
    api.create_webhook.assert_not_called()
    api.delete_webhook.assert_not_called()


# ---------------------------------------------------------------------------
# Test 2 — renderer surfaces counts + per-row detail + unmanaged-skipped
# ---------------------------------------------------------------------------

def test_dry_run_renders_managed_section_with_counts(
    deploy_env: dict[str, Any],
) -> None:
    api = deploy_env["api"]
    api.put_schedules.return_value = {
        "managed_by": "code",
        "create": [{"cron_expression": "0 9 * * *"}],
        "update": [],
        "delete": [{"id": "x", "cron_expression": "0 10 * * *"}],
        "unmanaged_skipped": 2,
    }
    api.put_webhooks.return_value = {
        "managed_by": "code",
        "create": [],
        "update": [],
        "delete": [],
        "unmanaged_skipped": 0,
    }
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
    assert "managed_by='code' diff (PUT-replace preview):" in stdout
    assert "agent: ops-bot" in stdout
    assert (
        "managed_by='code' schedules: 1 to create, 0 to update, 1 to delete "
        "(2 unmanaged rows untouched)"
    ) in stdout
    assert "+ schedule 0 9 * * *" in stdout
    assert "- schedule 0 10 * * *" in stdout
    assert (
        "managed_by='code' webhooks: 0 to create, 0 to update, 0 to delete "
        "(0 unmanaged rows untouched)"
    ) in stdout


# ---------------------------------------------------------------------------
# Test 3 — update rows render with `~` + field-changes summary
# ---------------------------------------------------------------------------

def test_dry_run_renders_update_with_field_changes(
    deploy_env: dict[str, Any],
) -> None:
    api = deploy_env["api"]
    api.put_schedules.return_value = {
        "managed_by": "code",
        "create": [],
        "update": [
            {
                "id": "abc",
                "before": {"timezone": "UTC"},
                "after": {
                    "timezone": "America/Toronto",
                    "cron_expression": "0 9 * * *",
                },
            }
        ],
        "delete": [],
        "unmanaged_skipped": 0,
    }
    api.put_webhooks.return_value = {
        "managed_by": "code",
        "create": [], "update": [], "delete": [], "unmanaged_skipped": 0,
    }
    _write_yaml(deploy_env["tmp_path"], """\
version: 1
envs:
  dev:
    agents:
      ops-bot:
        schedules: [{cron: "0 9 * * *"}]
""")
    exit_code, stdout, _stderr = _invoke("deploy", "--dry-run")
    assert exit_code == 0, stdout
    assert "~ schedule 0 9 * * *" in stdout
    # The before/after summary uses repr() so strings carry quotes.
    assert "timezone: 'UTC' → 'America/Toronto'" in stdout
    # cron_expression was not in `before`; .get returns None on the missing
    # side, so the summary includes it as a `None → '0 9 * * *'` transition.
    assert "cron_expression: None → '0 9 * * *'" in stdout


# ---------------------------------------------------------------------------
# Test 4 — legacy section bytes are unchanged when the managed section is
# appended (additive-only regression guard).
# ---------------------------------------------------------------------------

def test_dry_run_does_not_modify_legacy_reconcile_output(
    deploy_env: dict[str, Any],
) -> None:
    api = deploy_env["api"]
    api.put_schedules.return_value = {
        "managed_by": "code",
        "create": [], "update": [], "delete": [], "unmanaged_skipped": 0,
    }
    api.put_webhooks.return_value = {
        "managed_by": "code",
        "create": [], "update": [], "delete": [], "unmanaged_skipped": 0,
    }
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
    # The legacy renderer's per-agent line and the per-op lines must be
    # present unchanged — these are the bytes existing operators rely on.
    assert "agent: ops-bot (agt1)" in stdout
    assert "+ schedule 0 9 * * *" in stdout
    assert "+ webhook  trigger" in stdout
    # The new section appears below the legacy one — find their positions.
    legacy_idx = stdout.index("agent: ops-bot (agt1)")
    managed_idx = stdout.index("managed_by='code' diff (PUT-replace preview):")
    assert managed_idx > legacy_idx


# ---------------------------------------------------------------------------
# Test 5 — preview also renders on a real (non-dry-run) deploy
# ---------------------------------------------------------------------------

def test_real_deploy_also_prints_managed_preview(
    deploy_env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The preview shows on every deploy that hits reconcile — both
    --dry-run and apply. That way the operator sees the same diff
    before the apply lands."""
    api = deploy_env["api"]
    api.put_schedules.return_value = {
        "managed_by": "code",
        "create": [{"cron_expression": "0 9 * * *"}],
        "update": [], "delete": [], "unmanaged_skipped": 0,
    }
    api.put_webhooks.return_value = {
        "managed_by": "code",
        "create": [], "update": [], "delete": [], "unmanaged_skipped": 0,
    }

    from papayya import _reconcile as reconcile_mod
    apply_calls: list[Any] = []

    def fake_apply(plan: Any, api_arg: Any) -> Any:
        apply_calls.append((plan, api_arg))
        # Mirror the ApplyResult shape the renderer reads.
        from papayya._reconcile import ApplyResult
        return ApplyResult(
            applied=plan.total_ops, total=plan.total_ops,
            failed_op=None, error=None, created_webhooks=[],
        )

    monkeypatch.setattr(reconcile_mod, "apply_plan", fake_apply)

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
    # Preview rendered AND apply was invoked.
    assert "managed_by='code' diff (PUT-replace preview):" in stdout
    assert "+ schedule 0 9 * * *" in stdout
    assert len(apply_calls) == 1


# ---------------------------------------------------------------------------
# Test 6 — dry-run probe raises -> CLI exits with a managed_by-tagged error
# ---------------------------------------------------------------------------

def test_dry_run_api_error_exits_with_message(
    deploy_env: dict[str, Any],
) -> None:
    from papayya.api import PapayyaAPIError
    api = deploy_env["api"]
    api.put_schedules.side_effect = PapayyaAPIError(500, "boom")
    _write_yaml(deploy_env["tmp_path"], """\
version: 1
envs:
  dev:
    agents:
      ops-bot:
        schedules: [{cron: "0 9 * * *"}]
""")
    exit_code, _stdout, stderr = _invoke("deploy")
    assert exit_code != 0
    assert "Error (managed_by preview):" in stderr
    # Apply must not have proceeded — put_schedules was called once
    # (the dry-run probe) and put_webhooks never reached.
    assert api.put_schedules.call_count == 1
    api.put_webhooks.assert_not_called()


# ---------------------------------------------------------------------------
# Test 7 — empty desired set still probes both endpoints
# ---------------------------------------------------------------------------

def test_zero_managed_resources_still_probes_both_endpoints(
    deploy_env: dict[str, Any],
) -> None:
    """Within an agent, every resource type is probed — even ones the
    yaml declares as empty. An empty desired set is still a valid state
    and the operator needs to see what the next deploy would DELETE
    from leftover managed_by='code' rows.

    (The CLI's outer `has_triggers` gate still short-circuits when the
    whole env declares zero triggers across every agent — that's an
    optimisation Plan 13 does not regress. Here we satisfy the gate by
    declaring one schedule, then assert put_webhooks is still probed
    with an empty list.)
    """
    api = deploy_env["api"]
    api.put_schedules.return_value = {
        "managed_by": "code",
        "create": [{"cron_expression": "0 9 * * *"}],
        "update": [], "delete": [], "unmanaged_skipped": 0,
    }
    api.put_webhooks.return_value = {
        "managed_by": "code",
        "create": [], "update": [],
        "delete": [{"id": "ghost", "name": "old-trigger"}],
        "unmanaged_skipped": 0,
    }
    _write_yaml(deploy_env["tmp_path"], """\
version: 1
envs:
  dev:
    agents:
      ops-bot:
        schedules: [{cron: "0 9 * * *"}]
""")
    exit_code, stdout, _stderr = _invoke("deploy", "--dry-run")
    assert exit_code == 0, stdout
    # put_webhooks was probed even though the yaml declares zero webhooks.
    api.put_webhooks.assert_called_once_with("agt1", [], dry_run=True)
    # The proposed delete from the webhooks dry-run shows up below the
    # legacy plan — proving the preview catches leftover code-managed
    # rows the operator would otherwise wipe blindly.
    assert "- webhook old-trigger" in stdout
