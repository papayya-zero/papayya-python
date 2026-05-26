"""Tests for ``@trigger`` (Plan 11). Parallel to test_decorators_schedule.py."""

from __future__ import annotations

import pytest

from papayya import agent, get_agent, trigger
from papayya._config import WebhookSpec
from papayya.decorators import DecoratorTargetError, DecoratorValidationError


def test_trigger_attaches_to_registration():
    @trigger(name="hook-x", secret_env="INGEST_HOOK_SECRET")
    @agent(name="trigger_attaches_x")
    def fn(payload):
        return None

    reg = get_agent("trigger_attaches_x")
    assert reg is not None
    assert reg.webhooks == [WebhookSpec(name="hook-x", secret_env="INGEST_HOOK_SECRET")]


def test_trigger_multiple_decorators_stack():
    @trigger(name="hook-a", secret_env="SECRET_A")
    @trigger(name="hook-b", secret_env="SECRET_B")
    @trigger(name="hook-c", secret_env="SECRET_C")
    @agent(name="trigger_stacks_y")
    def fn(payload):
        return None

    reg = get_agent("trigger_stacks_y")
    assert reg is not None
    names = [w.name for w in reg.webhooks]
    # Bottom-up application order: innermost decorator runs first.
    assert names == ["hook-c", "hook-b", "hook-a"]


def test_trigger_empty_name_raises():
    with pytest.raises(DecoratorValidationError, match="must match"):
        @trigger(name="", secret_env="OK_ENV")
        @agent(name="trigger_empty_name")
        def fn(payload):
            return None


def test_trigger_name_with_spaces_raises():
    with pytest.raises(DecoratorValidationError, match="must match"):
        @trigger(name="bad name with spaces", secret_env="OK_ENV")
        @agent(name="trigger_spaces_name")
        def fn(payload):
            return None


def test_trigger_too_long_name_raises():
    with pytest.raises(DecoratorValidationError, match="must match"):
        @trigger(name="x" * 65, secret_env="OK_ENV")
        @agent(name="trigger_long_name")
        def fn(payload):
            return None


def test_trigger_lowercase_secret_env_raises():
    with pytest.raises(DecoratorValidationError, match="env var name"):
        @trigger(name="hook-y", secret_env="lowercase")
        @agent(name="trigger_lower_env")
        def fn(payload):
            return None


def test_trigger_dash_in_secret_env_raises():
    with pytest.raises(DecoratorValidationError, match="env var name"):
        @trigger(name="hook-z", secret_env="WITH-DASH")
        @agent(name="trigger_dash_env")
        def fn(payload):
            return None


def test_trigger_leading_digit_secret_env_raises():
    with pytest.raises(DecoratorValidationError, match="env var name"):
        @trigger(name="hook-w", secret_env="123_LEADING_DIGIT")
        @agent(name="trigger_digit_env")
        def fn(payload):
            return None


def test_trigger_above_agent_required():
    with pytest.raises(DecoratorTargetError, match="ABOVE @agent"):
        @agent(name="trigger_wrong_order")
        @trigger(name="inner-hook", secret_env="OK_ENV")
        def fn(payload):
            return None


def test_trigger_on_undecorated_function_raises():
    def bare_fn(payload):
        return None

    with pytest.raises(DecoratorTargetError):
        trigger(name="some-hook", secret_env="OK_ENV")(bare_fn)
