"""Tests for the @agent registration log line.

Verifies that decoration-time emits a log so the resolved
``agent_version`` (and its source) is visible immediately, rather than
only surfacing later at replay time when a mismatch trips the gate.
"""

from __future__ import annotations

import logging
import sys

import pytest

from papayya import agent
from papayya.agent import _clear_agent_version_cache, _registry

# The package's __init__.py re-exports `agent` (the function), which
# shadows the `papayya.agent` submodule attribute on the package. Reach
# into sys.modules directly when a test needs to monkeypatch a name on
# the submodule itself.
agent_module = sys.modules["papayya.agent"]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    _registry.clear()
    _clear_agent_version_cache()
    monkeypatch.delenv("PAPAYYA_AGENT_VERSION", raising=False)


def _format(records: list[logging.LogRecord]) -> list[tuple[int, str]]:
    return [(r.levelno, r.getMessage()) for r in records]


def test_logs_decorator_source_at_info(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="papayya.agent"):
        @agent(name="ops", agent_version="2.3.1")
        def ops(input_data):
            return input_data

    assert (logging.INFO, "registered 'ops' v=2.3.1 (source=decorator)") in _format(
        caplog.records
    )


def test_logs_env_source_at_info(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("PAPAYYA_AGENT_VERSION", "build-7f2a")

    with caplog.at_level(logging.INFO, logger="papayya.agent"):
        @agent(name="ingest")
        def ingest(input_data):
            return input_data

    assert (logging.INFO, "registered 'ingest' v=build-7f2a (source=env)") in _format(
        caplog.records
    )


def test_logs_git_source_at_info(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Force the env layer to fall through; git layer returns a fixed sha.
    monkeypatch.delenv("PAPAYYA_AGENT_VERSION", raising=False)
    monkeypatch.setattr(agent_module, "_resolve_git_version", lambda: "abc1234")

    with caplog.at_level(logging.INFO, logger="papayya.agent"):
        @agent(name="enrich")
        def enrich(input_data):
            return input_data

    assert (logging.INFO, "registered 'enrich' v=abc1234 (source=git)") in _format(
        caplog.records
    )


def test_logs_unknown_at_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # No explicit, no env, no git — should warn rather than silently
    # register at "unknown" version.
    monkeypatch.delenv("PAPAYYA_AGENT_VERSION", raising=False)
    monkeypatch.setattr(agent_module, "_resolve_git_version", lambda: None)

    with caplog.at_level(logging.WARNING, logger="papayya.agent"):
        @agent(name="loose")
        def loose(input_data):
            return input_data

    messages = _format(caplog.records)
    assert any(
        level == logging.WARNING
        and msg.startswith("registered 'loose' v=unknown")
        for level, msg in messages
    ), messages
