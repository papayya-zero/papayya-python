"""The empty-agent footgun.

``@agent`` injects a run only when the function declares ``run`` as its first
positional parameter. Drop that parameter — the obvious edit, since the body
often never touches it — and the body still runs, still returns, and records
nothing. Nothing appears in the dashboard and nothing says why.

That is the same failure the product exists to catch: it ran, and it didn't
work. These tests pin that the SDK now says so, and — just as important —
that it stays quiet in the cases where silence is correct.
"""

from __future__ import annotations

import sys
import warnings

import pytest

import papayya
from papayya import iterators


@pytest.fixture(autouse=True)
def _reset_warned_agents():
    """The warning fires once per agent name; tests must not leak into each other."""
    iterators._EMPTY_AGENT_WARNED.clear()
    yield
    iterators._EMPTY_AGENT_WARNED.clear()


def _clear_cli_login(monkeypatch):
    """Neutralise a real `papayya login` on the machine running the tests.

    Two traps here. The resolver imports `load_cli_config` at module scope, so
    patching `papayya._config` does not reach it. And `papayya.papayya`
    resolves to the exported factory FUNCTION, not the submodule — the package
    shadows its own module name — so a dotted-string setattr silently patches
    an attribute on a function object and changes nothing. Go via sys.modules.
    """
    mod = sys.modules["papayya.papayya"]
    monkeypatch.setattr(mod, "load_cli_config", lambda *a, **k: {})
    monkeypatch.setattr(
        "papayya._config.load_cli_config", lambda *a, **k: {}, raising=False
    )


@pytest.fixture
def configured(tmp_path, monkeypatch):
    """The caller has opted in, but runs land in a throwaway SQLite file.

    Setting PAPAYYA_API_KEY for real would also switch the store to CloudStore
    and send these bodies at a live control plane, so the opt-in signal is
    stubbed and the credentials are cleared to keep the store on SQLite.
    `test_sdk_is_configured_reads_the_env` covers the real resolver.

    The cleared env also has to include any `papayya login` on the developer's
    machine, or the store resolves to Cloud and the run mint hangs on DNS.
    """
    monkeypatch.setattr(iterators, "_sdk_is_configured", lambda: True)
    monkeypatch.delenv("PAPAYYA_API_KEY", raising=False)
    monkeypatch.delenv("PAPAYYA_BASE_URL", raising=False)
    _clear_cli_login(monkeypatch)
    monkeypatch.setenv("PAPAYYA_LOCAL_DB_PATH", str(tmp_path / "w.db"))


def test_sdk_is_configured_reads_the_env(monkeypatch):
    """The opt-in gate itself: a key present means configured, absent means not."""
    _clear_cli_login(monkeypatch)
    monkeypatch.delenv("PAPAYYA_API_KEY", raising=False)
    assert iterators._sdk_is_configured() is False

    monkeypatch.setenv("PAPAYYA_API_KEY", "cpk_test")
    assert iterators._sdk_is_configured() is True


def _warnings_from(fn, *args):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args)
    papayya_warnings = [
        w for w in caught if "finished without recording a run" in str(w.message)
    ]
    return result, papayya_warnings


def test_warns_when_body_records_nothing(configured):
    """The footgun itself: no `run` parameter, no papayya verbs, no record."""

    @papayya.agent(name="silent")
    def silent(name: str) -> str:
        return f"hello {name}"

    result, caught = _warnings_from(silent, "world")

    assert result == "hello world"  # behaviour is unchanged — we only warn
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "silent" in message
    # It must name the actual fix, not just report the symptom.
    assert "def silent(run, ...)" in message


def test_warning_fires_once_per_agent(configured):
    """A loop over 500 items must not emit 500 warnings."""

    @papayya.agent(name="chatty")
    def chatty(name: str) -> str:
        return name

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            chatty("x")

    relevant = [w for w in caught if "finished without recording a run" in str(w.message)]
    assert len(relevant) == 1


def test_silent_when_run_is_injected(configured):
    """`def f(run, ...)` opens a run, so there is nothing to warn about."""

    @papayya.agent(name="injected")
    def injected(run, name: str) -> str:
        run.complete({"name": name})
        return name

    _, caught = _warnings_from(injected, "world")
    assert caught == []


def test_silent_when_body_marks_an_outcome(configured):
    """An ambient verb mints the run lazily — that body recorded something."""

    @papayya.agent(name="ambient")
    def ambient(name: str) -> str:
        papayya.mark_degraded("looked wrong")
        return name

    _, caught = _warnings_from(ambient, "world")
    assert caught == []


def test_silent_when_not_configured(tmp_path, monkeypatch):
    """No credentials means the caller never opted in.

    Calling a decorated function as a plain function is supported; adoption is
    rewarded, not required. Warning here would nag people who are not using
    the product yet.
    """
    monkeypatch.delenv("PAPAYYA_API_KEY", raising=False)
    monkeypatch.delenv("PAPAYYA_BASE_URL", raising=False)
    monkeypatch.setenv("PAPAYYA_LOCAL_DB_PATH", str(tmp_path / "w.db"))
    _clear_cli_login(monkeypatch)

    @papayya.agent(name="unconfigured")
    def unconfigured(name: str) -> str:
        return name

    _, caught = _warnings_from(unconfigured, "world")
    assert caught == []


def test_silent_on_the_hosted_worker(configured):
    """The pool owns terminal status, so an empty body there is not the footgun.

    Guarded via `own_completion`, which the decorator sets False when a
    bootstrap run id is present. Without this a worker pool processing a
    million items would emit a warning storm about runs it does not own.
    """
    _, caught = _warnings_from(
        lambda: iterators.drive_ambient_sync(
            "hosted", None, None, lambda: "result", own_completion=False
        )
    )
    assert caught == []
