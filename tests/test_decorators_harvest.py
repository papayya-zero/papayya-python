"""Tests for ``harvest_decorator_specs`` (Plan 11).

Plan 12 will consume the harvest output to synthesise an EnvSpec the
reconciler reads. The harvester is tested in isolation against a
hand-built registry dict so we don't depend on registry global state
between tests.
"""

from __future__ import annotations

import pytest

from papayya import agent, get_registry, schedule, trigger
from papayya._config import AgentSpec, ScheduleSpec, WebhookSpec
from papayya.agent import AgentRegistration
from papayya.decorators import (
    DecoratorConflictError,
    harvest_decorator_specs,
)


def _bare_registration(name: str, version: str = "v1") -> AgentRegistration:
    """Minimal AgentRegistration for harvest tests — no fn, no tools."""
    return AgentRegistration(
        name=name,
        model="gpt-4o-mini",
        instructions="",
        fn=lambda: None,
        tools=[],
        max_steps=10,
        budget_usd=None,
        agent_version=version,
    )


def test_harvest_empty_registry_returns_empty():
    assert harvest_decorator_specs({}) == {}


def test_harvest_agent_with_no_decorators_returns_empty_lists():
    reg = _bare_registration("plain", "v1")
    out = harvest_decorator_specs({("plain", "v1"): reg})
    assert out == {"plain": ([], [])}


def test_harvest_collapses_versions_latest_wins():
    reg_v1 = _bare_registration("dup", "v1")
    reg_v1.schedules.append(ScheduleSpec(cron="0 0 * * *"))
    reg_v2 = _bare_registration("dup", "v2")
    reg_v2.schedules.append(ScheduleSpec(cron="0 12 * * *"))

    out = harvest_decorator_specs({
        ("dup", "v1"): reg_v1,
        ("dup", "v2"): reg_v2,
    })
    # Insertion order: v2 inserted after v1, so v2 wins.
    schedules, _webhooks = out["dup"]
    assert [s.cron for s in schedules] == ["0 12 * * *"]


def test_harvest_slug_form():
    reg = _bare_registration("My Reports", "v1")
    out = harvest_decorator_specs({("My Reports", "v1"): reg})
    assert list(out.keys()) == ["my-reports"]


def test_harvest_raises_on_duplicate_cron():
    reg = _bare_registration("dup_cron", "v1")
    reg.schedules.append(ScheduleSpec(cron="0 * * * *"))
    reg.schedules.append(ScheduleSpec(cron="0 * * * *"))

    with pytest.raises(DecoratorConflictError, match="duplicate @schedule"):
        harvest_decorator_specs({("dup_cron", "v1"): reg})


def test_harvest_raises_on_duplicate_webhook_name():
    reg = _bare_registration("dup_hook", "v1")
    reg.webhooks.append(WebhookSpec(name="hook", secret_env="ENV_A"))
    reg.webhooks.append(WebhookSpec(name="hook", secret_env="ENV_B"))

    with pytest.raises(DecoratorConflictError, match="duplicate @trigger"):
        harvest_decorator_specs({("dup_hook", "v1"): reg})


def test_harvest_returns_specs_shaped_for_envspec_consumption():
    """Output specs must drop straight into AgentSpec without further
    transformation — that's the Plan 12 wire contract."""
    reg = _bare_registration("ready", "v1")
    reg.schedules.append(ScheduleSpec(cron="0 0 * * *", timezone="UTC"))
    reg.webhooks.append(WebhookSpec(name="ingest", secret_env="ING_SECRET"))

    out = harvest_decorator_specs({("ready", "v1"): reg})
    schedules, webhooks = out["ready"]

    # If AgentSpec validation accepts these, Plan 12's synthesis step
    # has a clean wire shape to consume.
    spec = AgentSpec(schedules=schedules, webhooks=webhooks)
    assert len(spec.schedules) == 1
    assert spec.schedules[0].cron == "0 0 * * *"
    assert len(spec.webhooks) == 1
    assert spec.webhooks[0].name == "ingest"


def test_harvest_end_to_end_through_decorators():
    """End-to-end: decorate, then harvest from the real registry."""

    @schedule("0 */6 * * *")
    @trigger(name="harvest-e2e-hook", secret_env="HARVEST_E2E_SECRET")
    @agent(name="harvest_e2e_target")
    def fn(payload):
        return None

    out = harvest_decorator_specs(get_registry())
    assert "harvest_e2e_target" in out
    schedules, webhooks = out["harvest_e2e_target"]
    assert [s.cron for s in schedules] == ["0 */6 * * *"]
    assert [w.name for w in webhooks] == ["harvest-e2e-hook"]
