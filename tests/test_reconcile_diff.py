"""Pure-diff tests for papayya._reconcile.diff_env."""

from __future__ import annotations

from typing import Any

import pytest

from papayya import _reconcile
from papayya._config import AgentSpec, EnvSpec, ScheduleSpec, WebhookSpec


class FakeAPI:
    """APIClient stub that records calls and returns canned list responses."""

    def __init__(
        self,
        *,
        schedules_by_agent: dict[str, list[dict[str, Any]]] | None = None,
        webhooks_by_agent: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self._schedules = schedules_by_agent or {}
        self._webhooks = webhooks_by_agent or {}
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def list_schedules(self, agent_id: str) -> list[dict[str, Any]]:
        self.calls.append(("list_schedules", (agent_id,)))
        return self._schedules.get(agent_id, [])

    def list_webhooks(self, agent_id: str) -> list[dict[str, Any]]:
        self.calls.append(("list_webhooks", (agent_id,)))
        return self._webhooks.get(agent_id, [])

    # Create/delete only appear in apply_plan tests; not needed here.
    def create_schedule(self, *a: Any, **k: Any) -> dict[str, Any]:  # pragma: no cover
        raise AssertionError("create_schedule must not be called during diff")

    def delete_schedule(self, *a: Any, **k: Any) -> None:  # pragma: no cover
        raise AssertionError("delete_schedule must not be called during diff")

    def create_webhook(self, *a: Any, **k: Any) -> dict[str, Any]:  # pragma: no cover
        raise AssertionError("create_webhook must not be called during diff")

    def delete_webhook(self, *a: Any, **k: Any) -> None:  # pragma: no cover
        raise AssertionError("delete_webhook must not be called during diff")


def _env(agents: dict[str, AgentSpec]) -> EnvSpec:
    return EnvSpec(agents=agents)


def test_diff_noop_when_yaml_matches_server() -> None:
    env = _env({
        "ops-bot": AgentSpec(
            schedules=[ScheduleSpec(cron="0 * * * *")],
            webhooks=[WebhookSpec(name="trigger", secret_env="SECRET")],
        ),
    })
    api = FakeAPI(
        schedules_by_agent={"agt1": [{"id": "s1", "cron_expression": "0 * * * *"}]},
        webhooks_by_agent={"agt1": [{"id": "w1", "name": "trigger"}]},
    )
    plan = _reconcile.diff_env(env, {"ops-bot": "agt1"}, api)
    assert plan.is_noop
    assert plan.total_ops == 0


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
            "agt1": [{"id": "s-stale", "cron_expression": "*/5 * * * *"}]
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
            "agt1": [{"id": "s1", "cron_expression": "0  *  *  *  *"}],
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
            "agt1": [{"id": "w-old", "name": "old-name"}],
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
# apply_plan: fail-fast semantics (not strictly a diff test, but pairs well)
# ---------------------------------------------------------------------------


class ApplyAPI(FakeAPI):
    """Extends FakeAPI with mutating ops, and can be configured to fail on the Nth call."""

    def __init__(
        self,
        *,
        fail_on_call: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._fail_on_call = fail_on_call
        self._mutating_calls = 0

    def _tick(self) -> None:
        self._mutating_calls += 1
        if self._fail_on_call is not None and self._mutating_calls == self._fail_on_call:
            from papayya.api import PapayyaAPIError
            raise PapayyaAPIError(500, "boom")

    def create_schedule(self, agent_id: str, cron_expression: str, timezone: str = "UTC") -> dict[str, Any]:
        self.calls.append(("create_schedule", (agent_id, cron_expression, timezone)))
        self._tick()
        return {"id": f"sched-{self._mutating_calls}", "cron_expression": cron_expression}

    def delete_schedule(self, schedule_id: str) -> None:
        self.calls.append(("delete_schedule", (schedule_id,)))
        self._tick()

    def create_webhook(self, agent_id: str, name: str) -> dict[str, Any]:
        self.calls.append(("create_webhook", (agent_id, name)))
        self._tick()
        return {
            "id": f"wh-{self._mutating_calls}",
            "name": name,
            "secret": f"whs_{name}",
            "trigger_url": f"/v1/webhooks/wh-{self._mutating_calls}/trigger",
        }

    def delete_webhook(self, webhook_id: str) -> None:
        self.calls.append(("delete_webhook", (webhook_id,)))
        self._tick()


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
    api = ApplyAPI(fail_on_call=2)  # 1st create succeeds, 2nd raises
    plan = _reconcile.diff_env(env, {"ops-bot": "agt1"}, api)
    assert plan.total_ops == 3  # 1 schedule + 2 webhooks, all creates

    result = _reconcile.apply_plan(plan, api)
    assert result.applied == 1
    assert result.total == 3
    assert result.error is not None
    assert result.failed_op is not None
    # Third op must not have been attempted.
    mutating = [c for c in api.calls if c[0] in
                {"create_schedule", "delete_schedule", "create_webhook", "delete_webhook"}]
    assert len(mutating) == 2
