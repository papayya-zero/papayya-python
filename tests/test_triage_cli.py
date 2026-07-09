"""Tests for the hosted ``papayya triage`` CLI commands.

v1→v2 cutover: the read feed (``triage list``) is durable-backed; the
disposition commands act on the quarantine lane only (``retry`` → ``release``,
``dismiss`` → ``discard``). DLQ-lane disposition (the old dlq_skip/replay/
acknowledge dispatch) retired with the v1 DROP and is a deferred follow-up, so
non-quarantine rows exit 2 with a "not yet available" message. We mock the
Papayya client so these tests stay network-free — the SDK's resource methods
have their own HTTP-level coverage in ``test_triage_resource.py``.
"""

from __future__ import annotations

import json
from typing import Any

from click.testing import CliRunner

from papayya import cli as cli_module


class _FakeRuns:
    """Records dispatch decisions made by the CLI."""

    def __init__(self, get_response: dict | None = None) -> None:
        self.get_response = get_response or {}
        self.calls: list[tuple[str, str, dict]] = []

    def _record(self, name: str, run_id: str, **kwargs: Any) -> dict:
        self.calls.append((name, run_id, kwargs))
        return {"id": run_id, "via": name}

    def get(self, run_id: str) -> dict:
        self.calls.append(("get", run_id, {}))
        return self.get_response

    def release(self, run_id: str) -> dict:
        return self._record("release", run_id)

    def discard(self, run_id: str) -> dict:
        return self._record("discard", run_id)

    def quarantine(self, run_id: str, reason: str) -> dict:
        return self._record("quarantine", run_id, reason=reason)


class _FakeTriage:
    def __init__(self, pages: list[list[dict]]) -> None:
        # Caller passes pre-paginated rows; iter() yields them flat.
        self.pages = pages
        self.iter_calls: list[dict] = []

    def iter(self, **kwargs: Any):
        self.iter_calls.append(kwargs)
        for page in self.pages:
            for row in page:
                yield row


class _FakeClient:
    def __init__(self, runs: _FakeRuns, triage: _FakeTriage | None = None) -> None:
        self.runs = runs
        # Plan 34: quarantine ops live on the per-item resource (client.items).
        self.items = runs
        self.triage = triage or _FakeTriage([])
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _patch_client(monkeypatch, client: _FakeClient) -> _FakeClient:
    monkeypatch.setattr(cli_module, "_make_papayya_client", lambda ctx: client)
    return client


# ── list ──

def test_triage_list_streams_ndjson(monkeypatch) -> None:
    triage = _FakeTriage([
        [{"kind": "quarantine", "run_id": "q1"}, {"kind": "dlq", "run_id": "d1"}],
    ])
    client = _patch_client(monkeypatch, _FakeClient(_FakeRuns(), triage))

    result = CliRunner().invoke(
        cli_module.main, ["triage", "list", "--kind", "dlq", "--limit", "25"]
    )

    assert result.exit_code == 0, result.output
    assert triage.iter_calls == [{
        "partition_key": None,
        "tenant": None,
        "kind": "dlq",
        "page_size": 25,
    }]
    lines = [line for line in result.output.splitlines() if line.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["run_id"] == "q1"
    assert json.loads(lines[1])["run_id"] == "d1"
    assert client.closed


def test_triage_list_forwards_partition_key_and_tenant(monkeypatch) -> None:
    triage = _FakeTriage([[]])
    client = _patch_client(monkeypatch, _FakeClient(_FakeRuns(), triage))

    result = CliRunner().invoke(
        cli_module.main,
        ["triage", "list", "--partition-key", "shard-1", "--tenant", "acme"],
    )

    assert result.exit_code == 0, result.output
    assert triage.iter_calls == [{
        "partition_key": "shard-1",
        "tenant": "acme",
        "kind": "all",
        "page_size": 50,
    }]


# ── retry (quarantine lane only) ──

def test_triage_retry_dispatches_release_when_quarantine(monkeypatch) -> None:
    runs = _FakeRuns(get_response={"status": "quarantine"})
    client = _patch_client(monkeypatch, _FakeClient(runs))

    result = CliRunner().invoke(cli_module.main, ["triage", "retry", "r1"])

    assert result.exit_code == 0, result.output
    names = [c[0] for c in runs.calls]
    assert names == ["get", "release"]
    assert "release" in result.output


def test_triage_retry_defers_dlq_lane(monkeypatch) -> None:
    runs = _FakeRuns(get_response={"status": "failed"})
    _patch_client(monkeypatch, _FakeClient(runs))

    result = CliRunner().invoke(cli_module.main, ["triage", "retry", "r2"])

    assert result.exit_code == 2
    assert "failed" in result.output
    # Only the upfront GET — no dispatch, the DLQ action is deferred.
    assert [c[0] for c in runs.calls] == ["get"]


def test_triage_retry_errors_on_unsupported_status(monkeypatch) -> None:
    runs = _FakeRuns(get_response={"status": "completed"})
    _patch_client(monkeypatch, _FakeClient(runs))

    result = CliRunner().invoke(cli_module.main, ["triage", "retry", "r4"])

    assert result.exit_code == 2
    assert "completed" in result.output
    assert [c[0] for c in runs.calls] == ["get"]


# ── dismiss (quarantine lane only) ──

def test_triage_dismiss_dispatches_discard_when_quarantine(monkeypatch) -> None:
    runs = _FakeRuns(get_response={"status": "quarantine"})
    _patch_client(monkeypatch, _FakeClient(runs))

    result = CliRunner().invoke(cli_module.main, ["triage", "dismiss", "r1"])

    assert result.exit_code == 0, result.output
    assert [c[0] for c in runs.calls] == ["get", "discard"]


def test_triage_dismiss_defers_dlq_lane(monkeypatch) -> None:
    runs = _FakeRuns(get_response={"status": "failed"})
    _patch_client(monkeypatch, _FakeClient(runs))

    result = CliRunner().invoke(cli_module.main, ["triage", "dismiss", "r2"])

    assert result.exit_code == 2
    assert "failed" in result.output
    assert [c[0] for c in runs.calls] == ["get"]
