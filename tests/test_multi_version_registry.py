"""Multi-version registry — ADR-0003 § Worker #4.

Slice 3 re-keys ``papayya.agent._registry`` from
``dict[str, AgentRegistration]`` to
``dict[tuple[str, str], AgentRegistration]`` so a hosted worker can
hold v1 and v2 of the same agent slug resident at once. ``get_agent``
takes an optional ``version`` kwarg: ``None`` preserves single-resident
behaviour for ``papayya dev`` / LocalDispatcher; a concrete string does
the multi-version lookup.
"""

from __future__ import annotations

import pytest

from papayya import agent as agent_decorator
from papayya.agent import _clear_agent_version_cache, _registry, get_agent


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the module-level registry + version-resolver cache.

    The version resolver memoizes its env+git answers across the
    process; without the clear, an explicit ``agent_version`` set in
    one test could leak into the resolver state read by another.
    """
    _registry.clear()
    _clear_agent_version_cache()
    monkeypatch.delenv("PAPAYYA_AGENT_VERSION", raising=False)


def test_two_versions_resolve_independently():
    """``@agent(name="foo", agent_version="v1")`` and ``("foo", "v2")``
    coexist; ``get_agent`` routes by tuple."""

    @agent_decorator(name="foo", agent_version="v1")
    def foo_v1(item_id: str) -> dict:
        return {"version": "v1", "item_id": item_id}

    @agent_decorator(name="foo", agent_version="v2")
    def foo_v2(item_id: str) -> dict:
        return {"version": "v2", "item_id": item_id}

    reg_v1 = get_agent("foo", "v1")
    reg_v2 = get_agent("foo", "v2")

    assert reg_v1 is not None and reg_v2 is not None
    assert reg_v1 is not reg_v2
    assert reg_v1.fn("x")["version"] == "v1"
    assert reg_v2.fn("x")["version"] == "v2"


def test_no_version_returns_latest_registered():
    """``get_agent("foo", None)`` preserves the legacy single-resident
    semantics: latest insertion wins. ``papayya dev`` and tests that
    register one agent per slug rely on this branch."""

    @agent_decorator(name="bar", agent_version="v1")
    def bar_v1(item_id: str) -> str:
        return "v1"

    @agent_decorator(name="bar", agent_version="v2")
    def bar_v2(item_id: str) -> str:
        return "v2"

    # Insertion order: v1 first, v2 second → v2 wins on no-version lookup.
    reg = get_agent("bar")
    assert reg is not None
    assert reg.fn("x") == "v2"


def test_unknown_name_or_version_returns_none():
    """Both miss paths return None without raising. The worker maps a
    None registration to a failed completion with an unknown-agent
    error message, so distinguishing miss-by-name vs. miss-by-version
    is the worker's job, not the registry's."""

    @agent_decorator(name="baz", agent_version="v1")
    def baz_v1(item_id: str) -> str:
        return "ok"

    assert get_agent("missing", "v1") is None
    assert get_agent("baz", "v99") is None
    assert get_agent("missing") is None


def test_same_version_re_registration_overwrites():
    """If the same ``(name, version)`` is decorated twice (e.g., a
    bundle is re-imported in-process), the second registration replaces
    the first. This matches the legacy slug-keyed dict behaviour and is
    what hot-reload of the same version expects."""

    @agent_decorator(name="qux", agent_version="v1")
    def first(item_id: str) -> str:
        return "first"

    @agent_decorator(name="qux", agent_version="v1")
    def second(item_id: str) -> str:
        return "second"

    reg = get_agent("qux", "v1")
    assert reg is not None
    assert reg.fn("x") == "second"
    # Only one entry in the registry under this tuple.
    assert sum(1 for k in _registry if k == ("qux", "v1")) == 1
