"""Tests for papayya._decorator_synthesis.env_spec_from_registry_and_yaml.

The synthesis helper fuses yaml-sourced EnvSpec with the @schedule /
@trigger decorator-attached registry produced by Plan 11. Exercised
five ways: empty, yaml-only, decorator-only, fused-overlap, and
fused-disjoint.
"""

from __future__ import annotations

import pytest

from papayya._config import AgentSpec, EnvSpec, ScheduleSpec, WebhookSpec
from papayya._decorator_synthesis import env_spec_from_registry_and_yaml
from papayya.agent import AgentRegistration


def _reg(
    name: str,
    *,
    schedules: list[ScheduleSpec] | None = None,
    webhooks: list[WebhookSpec] | None = None,
) -> AgentRegistration:
    return AgentRegistration(
        name=name,
        model="gpt-4o-mini",
        instructions="",
        fn=lambda *_a, **_k: None,
        tools=[],
        max_steps=10,
        budget_usd=1.0,
        schedules=list(schedules or []),
        webhooks=list(webhooks or []),
    )


def test_synthesis_returns_empty_envspec_when_no_yaml_no_registry() -> None:
    out = env_spec_from_registry_and_yaml(None, {})
    assert isinstance(out, EnvSpec)
    assert out.agents == {}


def test_synthesis_yaml_only_returns_yaml() -> None:
    """No decorator metadata -> the yaml passes through unchanged."""
    yaml_env = EnvSpec(agents={
        "ops-bot": AgentSpec(
            schedules=[
                ScheduleSpec(cron="0 9 * * *"),
                ScheduleSpec(cron="*/15 * * * *"),
            ],
        ),
    })
    out = env_spec_from_registry_and_yaml(yaml_env, {})
    assert set(out.agents.keys()) == {"ops-bot"}
    assert len(out.agents["ops-bot"].schedules) == 2
    crons = [s.cron for s in out.agents["ops-bot"].schedules]
    assert crons == ["0 9 * * *", "*/15 * * * *"]
    assert out.agents["ops-bot"].webhooks == []


def test_synthesis_decorator_only_returns_decorator() -> None:
    """No yaml at all -> result has only decorator-attached agents."""
    registry: dict[tuple[str, str], AgentRegistration] = {
        ("ops-bot", "v1"): _reg(
            "ops-bot",
            schedules=[ScheduleSpec(cron="0 9 * * *")],
        ),
    }
    out = env_spec_from_registry_and_yaml(None, registry)
    assert set(out.agents.keys()) == {"ops-bot"}
    assert len(out.agents["ops-bot"].schedules) == 1
    assert out.agents["ops-bot"].schedules[0].cron == "0 9 * * *"
    assert out.agents["ops-bot"].webhooks == []


def test_synthesis_fuses_yaml_and_decorator_for_same_slug() -> None:
    """Agent in both yaml AND registry -> union of schedules + webhooks."""
    yaml_env = EnvSpec(agents={
        "ops-bot": AgentSpec(
            schedules=[ScheduleSpec(cron="0 9 * * *")],
            webhooks=[WebhookSpec(name="yaml-hook", secret_env="A")],
        ),
    })
    registry: dict[tuple[str, str], AgentRegistration] = {
        ("ops-bot", "v1"): _reg(
            "ops-bot",
            schedules=[ScheduleSpec(cron="*/15 * * * *")],
            webhooks=[WebhookSpec(name="decorator-hook", secret_env="B")],
        ),
    }
    out = env_spec_from_registry_and_yaml(yaml_env, registry)
    agent = out.agents["ops-bot"]
    crons = sorted(s.cron for s in agent.schedules)
    assert crons == ["*/15 * * * *", "0 9 * * *"]
    names = sorted(w.name for w in agent.webhooks)
    assert names == ["decorator-hook", "yaml-hook"]


def test_synthesis_decorator_only_slug_not_in_yaml() -> None:
    """yaml has agent A; registry has agent B with decorators -> result
    carries BOTH agents, A from yaml unmodified, B from decorators."""
    yaml_env = EnvSpec(agents={
        "agent-a": AgentSpec(schedules=[ScheduleSpec(cron="0 9 * * *")]),
    })
    registry: dict[tuple[str, str], AgentRegistration] = {
        ("agent-b", "v1"): _reg(
            "agent-b",
            webhooks=[WebhookSpec(name="b-hook", secret_env="B")],
        ),
    }
    out = env_spec_from_registry_and_yaml(yaml_env, registry)
    assert set(out.agents.keys()) == {"agent-a", "agent-b"}
    # A: yaml schedule, no webhooks.
    assert [s.cron for s in out.agents["agent-a"].schedules] == ["0 9 * * *"]
    assert out.agents["agent-a"].webhooks == []
    # B: decorator webhook, no schedules.
    assert out.agents["agent-b"].schedules == []
    assert [w.name for w in out.agents["agent-b"].webhooks] == ["b-hook"]


def test_synthesis_preserves_immutability_of_yaml_envspec() -> None:
    """The synthesis must not mutate the yaml EnvSpec's AgentSpec lists.
    Pydantic frozen-model `model_copy(update=...)` produces a NEW
    AgentSpec — the original's schedules/webhooks must be untouched
    after fusion."""
    original_schedules = [ScheduleSpec(cron="0 9 * * *")]
    yaml_env = EnvSpec(agents={
        "ops-bot": AgentSpec(schedules=original_schedules),
    })
    registry: dict[tuple[str, str], AgentRegistration] = {
        ("ops-bot", "v1"): _reg(
            "ops-bot",
            schedules=[ScheduleSpec(cron="*/15 * * * *")],
        ),
    }
    out = env_spec_from_registry_and_yaml(yaml_env, registry)
    # Original yaml block is untouched.
    assert len(yaml_env.agents["ops-bot"].schedules) == 1
    # Output has the union.
    assert len(out.agents["ops-bot"].schedules) == 2
