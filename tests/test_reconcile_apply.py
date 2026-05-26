"""Apply-path tests for papayya._reconcile.apply_plan (Plan 12).

The reconciler post-Plan 12 issues one ``put_schedules`` / ``put_webhooks``
call per agent per resource type instead of N create/delete round-trips.
Tests here exercise:

- The PUT-once-per-agent shape (the wire-format change).
- Skip when only ``unchanged`` ops exist (no-op write avoidance).
- Secret threading on newly created webhook rows (CLI rotation copy
  keeps working).
- Fail-fast ordering (one failure stops the agent's resource pipeline
  AND skips the next agent).
"""

from __future__ import annotations

from typing import Any

from papayya import _reconcile
from papayya._config import AgentSpec, EnvSpec, ScheduleSpec, WebhookSpec
from papayya.api import PapayyaAPIError


class RecordingAPI:
    """API stub that records put_*/list_* calls and returns canned responses."""

    def __init__(
        self,
        *,
        schedules_by_agent: dict[str, list[dict[str, Any]]] | None = None,
        webhooks_by_agent: dict[str, list[dict[str, Any]]] | None = None,
        put_schedules_responses: dict[str, dict[str, Any]] | None = None,
        put_webhooks_responses: dict[str, dict[str, Any]] | None = None,
        put_schedules_errors: dict[str, PapayyaAPIError] | None = None,
        put_webhooks_errors: dict[str, PapayyaAPIError] | None = None,
    ) -> None:
        self._schedules = schedules_by_agent or {}
        self._webhooks = webhooks_by_agent or {}
        self._put_sched_resp = put_schedules_responses or {}
        self._put_wh_resp = put_webhooks_responses or {}
        self._put_sched_err = put_schedules_errors or {}
        self._put_wh_err = put_webhooks_errors or {}
        self.calls: list[tuple[str, Any]] = []

    def list_schedules(self, agent_id: str) -> list[dict[str, Any]]:
        self.calls.append(("list_schedules", agent_id))
        return self._schedules.get(agent_id, [])

    def list_webhooks(self, agent_id: str) -> list[dict[str, Any]]:
        self.calls.append(("list_webhooks", agent_id))
        return self._webhooks.get(agent_id, [])

    def put_schedules(
        self, agent_id: str, schedules: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.calls.append(("put_schedules", (agent_id, schedules)))
        if agent_id in self._put_sched_err:
            raise self._put_sched_err[agent_id]
        return self._put_sched_resp.get(agent_id, {
            "items": [],
            "summary": {"created": 0, "updated": 0, "deleted": 0, "unchanged": 0},
        })

    def put_webhooks(
        self, agent_id: str, webhooks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.calls.append(("put_webhooks", (agent_id, webhooks)))
        if agent_id in self._put_wh_err:
            raise self._put_wh_err[agent_id]
        return self._put_wh_resp.get(agent_id, {
            "items": [],
            "summary": {"created": 0, "updated": 0, "deleted": 0, "unchanged": 0},
        })


def _env(agents: dict[str, AgentSpec]) -> EnvSpec:
    return EnvSpec(agents=agents)


# ---------------------------------------------------------------------------
# One PUT per agent per resource
# ---------------------------------------------------------------------------


def test_apply_calls_put_schedules_once_per_agent() -> None:
    """3 schedule ops (2 create + 1 delete) on one agent -> exactly ONE
    put_schedules call. The desired body carries the 2 wanted crons;
    the delete is implicit (its cron is absent from desired)."""
    env = _env({
        "ops-bot": AgentSpec(
            schedules=[
                ScheduleSpec(cron="0 9 * * *"),
                ScheduleSpec(cron="*/15 * * * *"),
            ],
        ),
    })
    api = RecordingAPI(
        schedules_by_agent={
            "agt1": [
                # Existing code-managed row with a cron that is NOT in yaml.
                {"id": "s-stale", "cron_expression": "30 2 * * *", "managed_by": "code"},
            ],
        },
    )
    plan = _reconcile.diff_env(env, {"ops-bot": "agt1"}, api)
    # 2 creates + 1 delete = 3 ops.
    assert plan.total_ops == 3

    result = _reconcile.apply_plan(plan, api)
    put_calls = [c for c in api.calls if c[0] == "put_schedules"]
    assert len(put_calls) == 1
    agent_id, desired = put_calls[0][1]
    assert agent_id == "agt1"
    # Desired set = the union of create/unchanged ops, in diff order.
    desired_crons = [d["cron_expression"] for d in desired]
    assert set(desired_crons) == {"0 9 * * *", "*/15 * * * *"}
    assert result.applied == 3
    assert result.total == 3
    assert result.error is None


def test_apply_does_not_call_put_when_only_unchanged() -> None:
    """All yaml entries match existing code-managed rows -> apply_plan
    skips the PUT entirely. Avoids a no-op write that would bump
    updated_at on every row."""
    env = _env({
        "ops-bot": AgentSpec(
            schedules=[ScheduleSpec(cron="0 9 * * *")],
            webhooks=[WebhookSpec(name="trigger", secret_env="SECRET")],
        ),
    })
    api = RecordingAPI(
        schedules_by_agent={
            "agt1": [{"id": "s1", "cron_expression": "0 9 * * *", "managed_by": "code"}],
        },
        webhooks_by_agent={
            "agt1": [{"id": "w1", "name": "trigger", "managed_by": "code"}],
        },
    )
    plan = _reconcile.diff_env(env, {"ops-bot": "agt1"}, api)
    # Both ops are `unchanged`. total_ops counts only non-unchanged ops -> 0.
    assert plan.total_ops == 0
    assert plan.is_noop

    result = _reconcile.apply_plan(plan, api)
    put_calls = [c for c in api.calls if c[0].startswith("put_")]
    assert put_calls == []
    assert result.applied == 0
    assert result.total == 0


def test_apply_surfaces_new_webhook_secrets() -> None:
    """PUT response carries a `secret` field on newly created rows;
    apply_plan threads it into ApplyResult.created_webhooks so the
    CLI's rotation copy keeps working."""
    env = _env({
        "ops-bot": AgentSpec(
            webhooks=[WebhookSpec(name="trigger", secret_env="MY_SECRET")],
        ),
    })
    api = RecordingAPI(
        webhooks_by_agent={"agt1": []},
        put_webhooks_responses={
            "agt1": {
                "items": [{
                    "id": "wh-new", "name": "trigger", "managed_by": "code",
                    "secret": "sk_test_abc", "trigger_url": "/v1/webhooks/wh-new/trigger",
                }],
                "summary": {"created": 1, "updated": 0, "deleted": 0, "unchanged": 0},
            },
        },
    )
    plan = _reconcile.diff_env(env, {"ops-bot": "agt1"}, api)

    result = _reconcile.apply_plan(plan, api)
    assert result.error is None
    assert len(result.created_webhooks) == 1
    created = result.created_webhooks[0]
    assert created["name"] == "trigger"
    assert created["secret"] == "sk_test_abc"
    assert created["trigger_url"] == "/v1/webhooks/wh-new/trigger"
    # Wedge for the CLI's "store in $X" copy.
    assert created["secret_env"] == "MY_SECRET"
    assert created["agent_slug"] == "ops-bot"


def test_apply_does_not_surface_secret_for_unchanged_or_renamed_existing_rows() -> None:
    """When the server returns a row without a secret (UNCHANGED or UPDATED
    row), apply_plan must not invent one. Only secret-bearing rows show
    up in created_webhooks."""
    env = _env({
        "ops-bot": AgentSpec(
            webhooks=[
                WebhookSpec(name="trigger-new", secret_env="A"),
                WebhookSpec(name="trigger-stable", secret_env="B"),
            ],
        ),
    })
    api = RecordingAPI(
        webhooks_by_agent={
            "agt1": [{"id": "w-stable", "name": "trigger-stable", "managed_by": "code"}],
        },
        put_webhooks_responses={
            "agt1": {
                "items": [
                    {
                        "id": "wh-new", "name": "trigger-new", "managed_by": "code",
                        "secret": "sk_test_new", "trigger_url": "/v1/webhooks/wh-new/trigger",
                    },
                    # No secret on stable row.
                    {"id": "w-stable", "name": "trigger-stable", "managed_by": "code"},
                ],
                "summary": {"created": 1, "updated": 0, "deleted": 0, "unchanged": 1},
            },
        },
    )
    plan = _reconcile.diff_env(env, {"ops-bot": "agt1"}, api)
    result = _reconcile.apply_plan(plan, api)
    # Only the secret-bearing row threads through.
    assert len(result.created_webhooks) == 1
    assert result.created_webhooks[0]["name"] == "trigger-new"


def test_apply_fails_fast_on_api_error_does_not_call_next_put() -> None:
    """When put_schedules raises, the subsequent put_webhooks for the
    SAME agent is not called. ApplyResult.error is populated and
    failed_op is the first non-unchanged schedule op."""
    env = _env({
        "ops-bot": AgentSpec(
            schedules=[ScheduleSpec(cron="0 9 * * *")],
            webhooks=[WebhookSpec(name="trigger", secret_env="A")],
        ),
    })
    api = RecordingAPI(
        put_schedules_errors={"agt1": PapayyaAPIError(500, "boom")},
    )
    plan = _reconcile.diff_env(env, {"ops-bot": "agt1"}, api)

    result = _reconcile.apply_plan(plan, api)
    assert result.error is not None
    assert result.failed_op is not None
    assert isinstance(result.failed_op, _reconcile.ScheduleOp)
    assert result.failed_op.kind == "create"
    assert result.failed_op.cron == "0 9 * * *"
    put_calls = [c[0] for c in api.calls if c[0].startswith("put_")]
    assert put_calls == ["put_schedules"]  # put_webhooks never called
    assert result.applied == 0


def test_apply_two_agents_isolated_failures() -> None:
    """When agent A's put_schedules raises, agent B is never called.
    Fail-fast is across the whole plan, not just within one agent —
    matches the pre-Plan-12 contract the CLI relies on."""
    env = _env({
        "agent-a": AgentSpec(schedules=[ScheduleSpec(cron="0 9 * * *")]),
        "agent-b": AgentSpec(schedules=[ScheduleSpec(cron="0 10 * * *")]),
    })
    api = RecordingAPI(
        put_schedules_errors={"agt-a": PapayyaAPIError(500, "boom")},
    )
    plan = _reconcile.diff_env(
        env, {"agent-a": "agt-a", "agent-b": "agt-b"}, api,
    )
    result = _reconcile.apply_plan(plan, api)
    assert result.error is not None
    put_calls = [c[1][0] for c in api.calls if c[0] == "put_schedules"]
    # Only agt-a's PUT was attempted.
    assert put_calls == ["agt-a"]


def test_apply_sends_managed_by_marker_via_api_client_path() -> None:
    """Verifies the contract that the wire-format carries managed_by='code'
    in every item. Exercised at the APIClient layer, which is where the
    marker is attached (the reconciler hands desired specs without the
    field; api.put_*() adds it). This test pins that integration."""
    from unittest.mock import MagicMock
    from papayya.api import APIClient, APIConfig

    client = APIClient(APIConfig(api_key="cpk_test"))
    client._request = MagicMock(return_value={"items": [], "summary": {}})  # type: ignore[method-assign]
    client.put_schedules("agt1", [{"cron_expression": "0 9 * * *"}])
    _method, _path, kwargs = client._request.call_args[0][0], client._request.call_args[0][1], client._request.call_args[1]
    body = kwargs["json"]
    assert body["items"][0]["managed_by"] == "code"
    assert body["items"][0]["cron_expression"] == "0 9 * * *"

    client._request.reset_mock()
    client.put_webhooks("agt1", [{"name": "trigger"}])
    kwargs = client._request.call_args[1]
    body = kwargs["json"]
    assert body["items"][0]["managed_by"] == "code"
    assert body["items"][0]["name"] == "trigger"
    client.close()
