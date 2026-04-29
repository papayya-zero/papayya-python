"""``@agent(max_duration_seconds=...)`` validation and registration.

The watchdog itself is integration-tested in
``tests/integration/test_per_item_timeout.py``. This file pins the
decorator-side contract: the field lands on AgentRegistration, and
invalid values fail fast at decoration time so customers see the
problem before the worker runs.
"""

from __future__ import annotations

import pytest

from papayya import agent
from papayya.agent import AgentRegistration, get_registry


def test_max_duration_defaults_to_none():
    @agent(name="default_dur")
    def fn(item_id: str) -> dict:
        return {"id": item_id}

    reg = get_registry()["default_dur"]
    assert isinstance(reg, AgentRegistration)
    assert reg.max_duration_seconds is None


def test_max_duration_persists_on_registration():
    @agent(name="explicit_dur", max_duration_seconds=12.5)
    def fn(item_id: str) -> dict:
        return {"id": item_id}

    reg = get_registry()["explicit_dur"]
    assert reg.max_duration_seconds == 12.5


def test_max_duration_zero_rejects_at_decoration_time():
    """Zero would arm the watchdog with an immediate timer — almost
    certainly a misconfiguration. Fail fast so the customer sees it
    when running locally, not after deploy when items start failing."""
    with pytest.raises(ValueError, match="max_duration_seconds"):
        @agent(name="zero_dur", max_duration_seconds=0)
        def fn(item_id: str) -> None:
            pass


def test_max_duration_negative_rejects_at_decoration_time():
    with pytest.raises(ValueError, match="max_duration_seconds"):
        @agent(name="neg_dur", max_duration_seconds=-1)
        def fn(item_id: str) -> None:
            pass


def test_max_duration_fractional_seconds_accepted():
    """``signal.setitimer(ITIMER_REAL, ...)`` supports sub-second
    intervals — make sure decoration doesn't reject them."""
    @agent(name="frac_dur", max_duration_seconds=0.25)
    def fn(item_id: str) -> None:
        pass

    assert get_registry()["frac_dur"].max_duration_seconds == 0.25
