"""Tests for ``@schedule`` (Plan 11).

Decoration-time validation (syntactic), attachment to AgentRegistration,
and stacking behaviour. Decorator order is enforced (``@schedule`` must
wrap a function that's already been wrapped by ``@agent``).

Tests use unique agent names per test so the module-level registry
doesn't carry state between tests — mirroring the existing
``test_agent_decorator_*`` convention.
"""

from __future__ import annotations

import pytest

from papayya import agent, get_agent, schedule
from papayya._config import ScheduleSpec
from papayya.decorators import DecoratorTargetError, DecoratorValidationError


def test_schedule_attaches_to_registration():
    @schedule("0 * * * *")
    @agent(name="schedule_attaches_x")
    def fn():
        return None

    reg = get_agent("schedule_attaches_x")
    assert reg is not None
    assert reg.schedules == [ScheduleSpec(cron="0 * * * *", timezone="UTC")]


def test_schedule_with_explicit_timezone():
    @schedule("0 9 * * MON-FRI", timezone="America/Toronto")
    @agent(name="schedule_with_tz_x")
    def fn():
        return None

    reg = get_agent("schedule_with_tz_x")
    assert reg is not None
    assert len(reg.schedules) == 1
    assert reg.schedules[0].cron == "0 9 * * MON-FRI"
    assert reg.schedules[0].timezone == "America/Toronto"


def test_schedule_multiple_decorators_stack():
    @schedule("0 */2 * * *")
    @schedule("0 9 * * MON-FRI", timezone="America/Toronto")
    @schedule("30 0 1 * *", timezone="UTC")
    @agent(name="schedule_stacks_y")
    def fn():
        return None

    reg = get_agent("schedule_stacks_y")
    assert reg is not None
    crons = [s.cron for s in reg.schedules]
    # Bottom-up application order: innermost decorator runs first.
    assert crons == ["30 0 1 * *", "0 9 * * MON-FRI", "0 */2 * * *"]


def test_schedule_invalid_cron_raises_at_decoration_time():
    with pytest.raises(DecoratorValidationError, match="not valid"):
        @schedule("not a cron")
        @agent(name="schedule_bad_cron")
        def fn():
            return None


def test_schedule_empty_cron_raises():
    with pytest.raises(DecoratorValidationError, match="non-empty"):
        @schedule("")
        @agent(name="schedule_empty_cron")
        def fn():
            return None


def test_schedule_invalid_timezone_raises_at_decoration_time():
    with pytest.raises(DecoratorValidationError, match="IANA"):
        @schedule("0 * * * *", timezone="Mars/Olympus")
        @agent(name="schedule_bad_tz")
        def fn():
            return None


def test_schedule_above_agent_required():
    # Wrong order: @agent OUTSIDE @schedule. The inner @schedule receives
    # a plain function that has no _papayya_agent attribute.
    with pytest.raises(DecoratorTargetError, match="ABOVE @agent"):
        @agent(name="schedule_wrong_order")
        @schedule("0 * * * *")
        def fn():
            return None


def test_schedule_on_undecorated_function_raises():
    def bare_fn():
        return None

    with pytest.raises(DecoratorTargetError):
        schedule("0 * * * *")(bare_fn)


def test_schedule_does_not_break_legacy_yaml_path():
    @agent(name="schedule_legacy_unaffected")
    def fn():
        return None

    reg = get_agent("schedule_legacy_unaffected")
    assert reg is not None
    assert reg.schedules == []
    assert reg.webhooks == []
