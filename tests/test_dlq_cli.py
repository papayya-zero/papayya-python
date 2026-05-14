"""Tests for the hosted ``papayya dlq`` CLI commands.

Three operator-triage commands hit ``client.runs.dlq_{skip,acknowledge,replay}``
against the hosted control-pane. We mock the Papayya client so these tests
don't need a live backend — the SDK's resource methods have their own
HTTP-level coverage.
"""

from __future__ import annotations

from click.testing import CliRunner

from papayya import cli as cli_module


class _FakeRuns:
    """Captures which SDK method was called with which run_id."""

    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def dlq_skip(self, run_id: str) -> dict:
        self.calls.append(("dlq_skip", run_id))
        return self.response

    def dlq_acknowledge(self, run_id: str) -> dict:
        self.calls.append(("dlq_acknowledge", run_id))
        return self.response

    def dlq_replay(self, run_id: str) -> dict:
        self.calls.append(("dlq_replay", run_id))
        return self.response


class _FakeClient:
    def __init__(self, runs: _FakeRuns) -> None:
        self.runs = runs
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _patch_client(monkeypatch, runs: _FakeRuns) -> _FakeClient:
    """Swap in a fake client so the CLI invocation never hits the network."""
    client = _FakeClient(runs)
    monkeypatch.setattr(cli_module, "_make_papayya_client", lambda ctx: client)
    return client


def test_dlq_skip_calls_sdk_with_run_id(monkeypatch) -> None:
    runs = _FakeRuns({"id": "r1", "dlq_disposition": "skipped"})
    client = _patch_client(monkeypatch, runs)

    result = CliRunner().invoke(cli_module.main, ["dlq", "skip", "r1"])

    assert result.exit_code == 0, result.output
    assert runs.calls == [("dlq_skip", "r1")]
    assert '"dlq_disposition": "skipped"' in result.output
    assert client.closed


def test_dlq_acknowledge_calls_sdk_with_run_id(monkeypatch) -> None:
    runs = _FakeRuns({"id": "r2", "dlq_disposition": "acknowledged"})
    client = _patch_client(monkeypatch, runs)

    result = CliRunner().invoke(cli_module.main, ["dlq", "acknowledge", "r2"])

    assert result.exit_code == 0, result.output
    assert runs.calls == [("dlq_acknowledge", "r2")]
    assert '"dlq_disposition": "acknowledged"' in result.output
    assert client.closed


def test_dlq_replay_calls_sdk_with_run_id(monkeypatch) -> None:
    runs = _FakeRuns({"id": "r3-new", "replayed_from": "r3", "status": "queued"})
    client = _patch_client(monkeypatch, runs)

    result = CliRunner().invoke(cli_module.main, ["dlq", "replay", "r3"])

    assert result.exit_code == 0, result.output
    assert runs.calls == [("dlq_replay", "r3")]
    assert '"replayed_from": "r3"' in result.output
    assert client.closed
