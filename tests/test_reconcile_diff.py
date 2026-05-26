"""Pure-diff tests for papayya._reconcile.diff_env."""

from __future__ import annotations

from typing import Any

import pytest

from papayya import _reconcile
from papayya._config import AgentSpec, EnvSpec, ScheduleSpec, WebhookSpec


class FakeAPI:
    """APIClient stub that records calls and returns canned list responses.

    Post-Plan 12: also stubs the PUT methods so apply-path tests can
    record what the reconciler sends. The PUT methods record args and
    return a default `{items: [], summary: {...}}` envelope; tests that
    need to surface secrets override the return values explicitly.
    """

    def __init__(
        self,
        *,
        schedules_by_agent: dict[str, list[dict[str, Any]]] | None = None,
        webhooks_by_agent: dict[str, list[dict[str, Any]]] | None = None,
        put_schedules_response: dict[str, Any] | None = None,
        put_webhooks_response: dict[str, Any] | None = None,
    ) -> None:
        self._schedules = schedules_by_agent or {}
        self._webhooks = webhooks_by_agent or {}
        self._put_schedules_response = put_schedules_response
        self._put_webhooks_response = put_webhooks_response
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def list_schedules(self, agent_id: str) -> list[dict[str, Any]]:
        self.calls.append(("list_schedules", (agent_id,)))
        return self._schedules.get(agent_id, [])

    def list_webhooks(self, agent_id: str) -> list[dict[str, Any]]:
        self.calls.append(("list_webhooks", (agent_id,)))
        return self._webhooks.get(agent_id, [])

    def put_schedules(
        self, agent_id: str, schedules: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.calls.append(("put_schedules", (agent_id, schedules)))
        if self._put_schedules_response is not None:
            return self._put_schedules_response
        # Default: echo desired set as the post-replace list, no diff counts.
        return {
            "items": [
                {**item, "id": f"sched-{i}", "managed_by": "code"}
                for i, item in enumerate(schedules)
            ],
            "summary": {
                "created": len(schedules), "updated": 0,
                "deleted": 0, "unchanged": 0,
            },
        }

    def put_webhooks(
        self, agent_id: str, webhooks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.calls.append(("put_webhooks", (agent_id, webhooks)))
        if self._put_webhooks_response is not None:
            return self._put_webhooks_response
        return {
            "items": [
                {**item, "id": f"wh-{i}", "managed_by": "code"}
                for i, item in enumerate(webhooks)
            ],
            "summary": {
                "created": len(webhooks), "updated": 0,
                "deleted": 0, "unchanged": 0,
            },
        }

    # The old per-call mutating methods must not be hit by apply_plan
    # post-Plan 12. Kept as guard assertions.
    def create_schedule(self, *a: Any, **k: Any) -> dict[str, Any]:  # pragma: no cover
        raise AssertionError("create_schedule must not be called post-Plan 12")

    def delete_schedule(self, *a: Any, **k: Any) -> None:  # pragma: no cover
        raise AssertionError("delete_schedule must not be called post-Plan 12")

    def create_webhook(self, *a: Any, **k: Any) -> dict[str, Any]:  # pragma: no cover
        raise AssertionError("create_webhook must not be called post-Plan 12")

    def delete_webhook(self, *a: Any, **k: Any) -> None:  # pragma: no cover
        raise AssertionError("delete_webhook must not be called post-Plan 12")


def _env(agents: dict[str, AgentSpec]) -> EnvSpec:
    return EnvSpec(agents=agents)


# ---------------------------------------------------------------------------
# Existing diff coverage — extended to carry managed_by='code' on every
# server row (post-Plan 12 the diff filters out anything that isn't
# code-managed; pre-existing tests therefore need explicit markers).
# ---------------------------------------------------------------------------


def test_diff_noop_when_yaml_matches_server() -> None:
    env = _env({
        "ops-bot": AgentSpec(
            schedules=[ScheduleSpec(cron="0 * * * *")],
            webhooks=[WebhookSpec(name="trigger", secret_env="SECRET")],
        ),
    })
    api = FakeAPI(
        schedules_by_agent={
            "agt1": [{"id": "s1", "cron_expression": "0 * * * *", "managed_by": "code"}],
        },
        webhooks_by_agent={
            "agt1": [{"id": "w1", "name": "trigger", "managed_by": "code"}],
        },
    )
    plan = _reconcile.diff_env(env, {"ops-bot": "agt1"}, api)
    # Both rows match -> two unchanged ops, but plan.is_noop is True because
    # unchanged ops carry no mutation.
    assert plan.is_noop
    assert plan.total_ops == 0
    agent_plan = plan.agents[0]
    assert all(op.kind == "unchanged" for op in agent_plan.schedule_ops)
    assert all(op.kind == "unchanged" for op in agent_plan.webhook_ops)


def test_diff_creates_schedule_absent_on_server() -> None:
    env = _env({
        "ops-bot": AgentSpec(schedules=[ScheduleSpec(cron="0 9 * * *")]),
    })
    api = FakeAPI(schedules_by_agent={"agt1": []})
    plan = _reconcile.diff_env(env, {"ops-bot": "agt1"}, api)
    agent_plan = plan.agents[0]
    assert len(agent_plan.schedule_ops) == 1
    op = agent_plan.schedule_ops[0]
    assert op.kind == "create"
    assert op.cron == "0 9 * * *"
    assert op.remote_id is None


def test_diff_deletes_schedule_absent_from_yaml() -> None:
    env = _env({"ops-bot": AgentSpec(schedules=[])})
    api = FakeAPI(
        schedules_by_agent={
            "agt1": [{
                "id": "s-stale", "cron_expression": "*/5 * * * *",
                "managed_by": "code",
            }],
        },
    )
    plan = _reconcile.diff_env(env, {"ops-bot": "agt1"}, api)
    agent_plan = plan.agents[0]
    assert len(agent_plan.schedule_ops) == 1
    op = agent_plan.schedule_ops[0]
    assert op.kind == "delete"
    assert op.remote_id == "s-stale"


def test_diff_cron_normalized_whitespace_is_noop() -> None:
    env = _env({"ops-bot": AgentSpec(schedules=[ScheduleSpec(cron="0 * * * *")])})
    api = FakeAPI(
        schedules_by_agent={
            "agt1": [{
                "id": "s1", "cron_expression": "0  *  *  *  *",
                "managed_by": "code",
            }],
        },
    )
    plan = _reconcile.diff_env(env, {"ops-bot": "agt1"}, api)
    assert plan.is_noop


def test_diff_webhook_rename_is_delete_plus_create() -> None:
    env = _env({
        "ops-bot": AgentSpec(
            webhooks=[WebhookSpec(name="new-name", secret_env="SECRET")],
        ),
    })
    api = FakeAPI(
        webhooks_by_agent={
            "agt1": [{"id": "w-old", "name": "old-name", "managed_by": "code"}],
        },
    )
    plan = _reconcile.diff_env(env, {"ops-bot": "agt1"}, api)
    ops = plan.agents[0].webhook_ops
    kinds = [(op.kind, op.name, op.reason) for op in ops]
    assert ("delete", "old-name", "removed") in kinds
    assert ("create", "new-name", "rename") in kinds


def test_diff_unknown_slug_raises_before_server() -> None:
    env = _env({
        "ghost-agent": AgentSpec(schedules=[ScheduleSpec(cron="0 * * * *")]),
    })
    api = FakeAPI()
    with pytest.raises(_reconcile.ReconcileError) as exc_info:
        _reconcile.diff_env(env, {"ops-bot": "agt1"}, api)
    assert "ghost-agent" in str(exc_info.value)
    # Must fail before any server call.
    assert api.calls == []


# ---------------------------------------------------------------------------
# Plan 12: managed_by filter — code rows are visible, api rows are not.
# ---------------------------------------------------------------------------


def test_diff_ignores_api_managed_schedule_on_server() -> None:
    """managed_by='api' rows are invisible to the diff — yaml that omits
    them must NOT emit a delete op."""
    env = _env({"ops-bot": AgentSpec(schedules=[])})
    api = FakeAPI(
        schedules_by_agent={
            "agt1": [
                # Dashboard-created row. Yaml doesn't mention it, but the
                # reconciler is forbidden from touching it.
                {"id": "s-dash", "cron_expression": "*/5 * * * *", "managed_by": "api"},
            ],
        },
    )
    plan = _reconcile.diff_env(env, {"ops-bot": "agt1"}, api)
    agent_plan = plan.agents[0]
    assert agent_plan.schedule_ops == []


def test_diff_ignores_api_managed_webhook_on_server() -> None:
    env = _env({"ops-bot": AgentSpec(webhooks=[])})
    api = FakeAPI(
        webhooks_by_agent={
            "agt1": [
                {"id": "wh-dash", "name": "dashboard-hook", "managed_by": "api"},
            ],
        },
    )
    plan = _reconcile.diff_env(env, {"ops-bot": "agt1"}, api)
    agent_plan = plan.agents[0]
    assert agent_plan.webhook_ops == []


def test_diff_emits_unchanged_for_matching_code_row() -> None:
    """A yaml row matching an existing code-managed remote row emits one
    `unchanged` op. Plan 13's dry-run renders this — the previous shape
    was "zero ops", which lost the signal."""
    env = _env({
        "ops-bot": AgentSpec(
            schedules=[ScheduleSpec(cron="0 9 * * *")],
            webhooks=[WebhookSpec(name="trigger", secret_env="SECRET")],
        ),
    })
    api = FakeAPI(
        schedules_by_agent={
            "agt1": [{"id": "s1", "cron_expression": "0 9 * * *", "managed_by": "code"}],
        },
        webhooks_by_agent={
            "agt1": [{"id": "w1", "name": "trigger", "managed_by": "code"}],
        },
    )
    plan = _reconcile.diff_env(env, {"ops-bot": "agt1"}, api)
    agent_plan = plan.agents[0]
    assert len(agent_plan.schedule_ops) == 1
    assert agent_plan.schedule_ops[0].kind == "unchanged"
    assert agent_plan.schedule_ops[0].remote_id == "s1"
    assert len(agent_plan.webhook_ops) == 1
    assert agent_plan.webhook_ops[0].kind == "unchanged"
    assert agent_plan.webhook_ops[0].remote_id == "w1"


def test_diff_treats_missing_managed_by_as_api() -> None:
    """Defensive: a pre-migration row with no managed_by key is treated as
    'api' (invisible to the diff). Protects against an old server
    accidentally surfacing the wedge-blocker bug."""
    env = _env({"ops-bot": AgentSpec(schedules=[])})
    api = FakeAPI(
        schedules_by_agent={
            "agt1": [{"id": "s-legacy", "cron_expression": "*/5 * * * *"}],
        },
    )
    plan = _reconcile.diff_env(env, {"ops-bot": "agt1"}, api)
    agent_plan = plan.agents[0]
    assert agent_plan.schedule_ops == []


def test_diff_code_and_api_rows_coexist_on_same_agent() -> None:
    """One agent can carry both code-managed and api-managed rows. The
    diff sees only the code rows; the api rows are inert."""
    env = _env({
        "ops-bot": AgentSpec(schedules=[ScheduleSpec(cron="0 9 * * *")]),
    })
    api = FakeAPI(
        schedules_by_agent={
            "agt1": [
                {"id": "s-code", "cron_expression": "0 9 * * *", "managed_by": "code"},
                {"id": "s-api", "cron_expression": "*/5 * * * *", "managed_by": "api"},
            ],
        },
    )
    plan = _reconcile.diff_env(env, {"ops-bot": "agt1"}, api)
    agent_plan = plan.agents[0]
    # Only the code row matched the yaml -> one unchanged op. The api row
    # is invisible (not deleted, not surfaced).
    assert len(agent_plan.schedule_ops) == 1
    assert agent_plan.schedule_ops[0].kind == "unchanged"


# ---------------------------------------------------------------------------
# apply_plan: fail-fast semantics — kept as a sanity check on the rewritten
# PUT-based apply. Detailed apply tests live in test_reconcile_apply.py.
# ---------------------------------------------------------------------------


class ApplyAPI(FakeAPI):
    """Extends FakeAPI with PUT failure injection."""

    def __init__(
        self,
        *,
        fail_on_put: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._fail_on_put = fail_on_put

    def put_schedules(self, agent_id: str, schedules: list[dict[str, Any]]) -> dict[str, Any]:
        if self._fail_on_put == "schedules":
            self.calls.append(("put_schedules", (agent_id, schedules)))
            from papayya.api import PapayyaAPIError
            raise PapayyaAPIError(500, "boom")
        return super().put_schedules(agent_id, schedules)

    def put_webhooks(self, agent_id: str, webhooks: list[dict[str, Any]]) -> dict[str, Any]:
        if self._fail_on_put == "webhooks":
            self.calls.append(("put_webhooks", (agent_id, webhooks)))
            from papayya.api import PapayyaAPIError
            raise PapayyaAPIError(500, "boom")
        return super().put_webhooks(agent_id, webhooks)


def test_apply_plan_fails_fast_and_reports_counts() -> None:
    env = _env({
        "ops-bot": AgentSpec(
            schedules=[ScheduleSpec(cron="0 9 * * *")],
            webhooks=[
                WebhookSpec(name="a", secret_env="A"),
                WebhookSpec(name="b", secret_env="B"),
            ],
        ),
    })
    api = ApplyAPI(fail_on_put="webhooks")
    plan = _reconcile.diff_env(env, {"ops-bot": "agt1"}, api)
    assert plan.total_ops == 3  # 1 schedule + 2 webhooks, all creates

    result = _reconcile.apply_plan(plan, api)
    # put_schedules landed (1 op applied); put_webhooks raised.
    assert result.applied == 1
    assert result.total == 3
    assert result.error is not None
    assert result.failed_op is not None
    # Exactly one put_schedules + one put_webhooks call (the latter
    # being the failing one). No legacy create/delete calls.
    put_calls = [c[0] for c in api.calls if c[0].startswith("put_")]
    assert put_calls == ["put_schedules", "put_webhooks"]
