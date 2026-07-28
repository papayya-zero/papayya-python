"""Tests for papayya._decorator_synthesis.env_spec_from_registry.

The synthesis helper shapes the @schedule / @trigger decorator-attached
registry produced by Plan 11 into the EnvSpec the reconciler reads.
Decorators are the sole source of truth (the papayya.yaml surface was
removed from the CLI). Exercised: empty, single-agent, multi-agent, and
immutability of the harvested specs.
"""

from __future__ import annotations

from papayya._config import EnvSpec, ScheduleSpec, WebhookSpec
from papayya._decorator_synthesis import env_spec_from_registry
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


def test_synthesis_returns_empty_envspec_when_no_registry() -> None:
    out = env_spec_from_registry({})
    assert isinstance(out, EnvSpec)
    assert out.agents == {}


def test_synthesis_single_agent_schedules() -> None:
    registry: dict[tuple[str, str], AgentRegistration] = {
        ("ops-bot", "v1"): _reg(
            "ops-bot",
            schedules=[
                ScheduleSpec(cron="0 9 * * *"),
                ScheduleSpec(cron="*/15 * * * *"),
            ],
        ),
    }
    out = env_spec_from_registry(registry)
    assert set(out.agents.keys()) == {"ops-bot"}
    crons = [s.cron for s in out.agents["ops-bot"].schedules]
    assert crons == ["0 9 * * *", "*/15 * * * *"]
    assert out.agents["ops-bot"].webhooks == []


def test_synthesis_single_agent_schedules_and_webhooks() -> None:
    registry: dict[tuple[str, str], AgentRegistration] = {
        ("ops-bot", "v1"): _reg(
            "ops-bot",
            schedules=[ScheduleSpec(cron="*/15 * * * *")],
            webhooks=[WebhookSpec(name="decorator-hook", secret_env="B")],
        ),
    }
    out = env_spec_from_registry(registry)
    agent = out.agents["ops-bot"]
    assert [s.cron for s in agent.schedules] == ["*/15 * * * *"]
    assert [w.name for w in agent.webhooks] == ["decorator-hook"]


def test_synthesis_multiple_agents() -> None:
    registry: dict[tuple[str, str], AgentRegistration] = {
        ("agent-a", "v1"): _reg(
            "agent-a", schedules=[ScheduleSpec(cron="0 9 * * *")],
        ),
        ("agent-b", "v1"): _reg(
            "agent-b", webhooks=[WebhookSpec(name="b-hook", secret_env="B")],
        ),
    }
    out = env_spec_from_registry(registry)
    assert set(out.agents.keys()) == {"agent-a", "agent-b"}
    assert [s.cron for s in out.agents["agent-a"].schedules] == ["0 9 * * *"]
    assert out.agents["agent-a"].webhooks == []
    assert out.agents["agent-b"].schedules == []
    assert [w.name for w in out.agents["agent-b"].webhooks] == ["b-hook"]


def test_synthesis_does_not_mutate_registration_lists() -> None:
    """The synthesis must not alias the registration's schedule list —
    the built AgentSpec carries a copy."""
    original_schedules = [ScheduleSpec(cron="0 9 * * *")]
    registry: dict[tuple[str, str], AgentRegistration] = {
        ("ops-bot", "v1"): _reg("ops-bot", schedules=original_schedules),
    }
    out = env_spec_from_registry(registry)
    out.agents["ops-bot"].schedules.append(ScheduleSpec(cron="*/15 * * * *"))
    # Original registration list is untouched.
    assert len(original_schedules) == 1
