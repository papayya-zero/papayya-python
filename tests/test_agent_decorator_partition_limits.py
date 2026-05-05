"""``@agent(concurrency_per_key=, rate_limit=)`` decorator-side contract.

Layer 3 #1 + #2. The dispatcher-side enforcement lives in the Go
control-pane (`internal/enforcement/partition_limits.go`); here we pin
just the SDK contract:

  - kwargs land on AgentRegistration as integer caps
  - rate_limit string parser accepts "N/min" + "N/sec", normalises to RPM
  - bad inputs fail fast at decoration time so a typo doesn't ship
    silently disabled
"""

from __future__ import annotations

import pytest

from papayya import agent
from papayya.agent import AgentRegistration, _parse_rate_limit, get_agent


# ---- AgentRegistration field plumbing ----

def test_caps_default_to_none():
    @agent(name="default_caps")
    def fn(item_id: str) -> dict:
        return {"id": item_id}

    reg = get_agent("default_caps")
    assert isinstance(reg, AgentRegistration)
    assert reg.concurrency_per_key is None
    assert reg.rate_limit_per_min is None


def test_concurrency_per_key_persists():
    @agent(name="explicit_conc", concurrency_per_key=5)
    def fn(item_id: str) -> dict:
        return {"id": item_id}

    assert get_agent("explicit_conc").concurrency_per_key == 5


def test_rate_limit_per_min_persists_from_minutes_form():
    @agent(name="explicit_rate_min", rate_limit="100/min")
    def fn(item_id: str) -> dict:
        return {"id": item_id}

    assert get_agent("explicit_rate_min").rate_limit_per_min == 100


def test_rate_limit_per_min_normalised_from_seconds_form():
    """``"5/sec"`` should normalise to 300 RPM. The control-pane stores
    one unit (RPM) so the lease-time check has a single comparison."""
    @agent(name="explicit_rate_sec", rate_limit="5/sec")
    def fn(item_id: str) -> dict:
        return {"id": item_id}

    assert get_agent("explicit_rate_sec").rate_limit_per_min == 300


# ---- Validation: fail fast at decoration time ----

def test_concurrency_per_key_zero_rejected():
    with pytest.raises(ValueError, match="concurrency_per_key"):
        @agent(name="zero_conc", concurrency_per_key=0)
        def fn(item_id: str) -> None:
            pass


def test_concurrency_per_key_negative_rejected():
    with pytest.raises(ValueError, match="concurrency_per_key"):
        @agent(name="neg_conc", concurrency_per_key=-1)
        def fn(item_id: str) -> None:
            pass


def test_rate_limit_missing_slash_rejected():
    with pytest.raises(ValueError, match="N/min.*N/sec"):
        @agent(name="bad_rate_no_slash", rate_limit="100min")
        def fn(item_id: str) -> None:
            pass


def test_rate_limit_unknown_unit_rejected():
    with pytest.raises(ValueError, match="unit"):
        @agent(name="bad_rate_unit", rate_limit="100/hour")
        def fn(item_id: str) -> None:
            pass


def test_rate_limit_non_integer_numerator_rejected():
    with pytest.raises(ValueError, match="numerator"):
        @agent(name="bad_rate_n", rate_limit="abc/min")
        def fn(item_id: str) -> None:
            pass


def test_rate_limit_zero_rejected():
    with pytest.raises(ValueError, match="must be > 0"):
        @agent(name="zero_rate", rate_limit="0/min")
        def fn(item_id: str) -> None:
            pass


# ---- Parser unit tests (catch shapes not exercised by decorator path) ----

@pytest.mark.parametrize(
    "value,expected",
    [
        ("1/min", 1),
        ("60/min", 60),
        ("1/sec", 60),
        ("10/sec", 600),
        ("  100/min  ", 100),     # whitespace tolerated
        ("100 / min", 100),        # spaces around slash tolerated
        ("100/MINUTE", 100),       # case-insensitive
        ("100/minute", 100),
        ("100/m", 100),            # short form
        ("100/s", 6000),
    ],
)
def test_parser_happy_paths(value: str, expected: int):
    assert _parse_rate_limit(value) == expected
