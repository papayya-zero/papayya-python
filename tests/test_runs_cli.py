"""Tests for `papayya runs` (invocations) and `papayya items` (per-item).

Plan 34 BREAKING shift, covered explicitly here:

- `papayya runs list` reads the LOCAL ledger and lists invocations, one
  NDJSON line per run, carrying the outcome rollup (degraded/failed item
  counts + worst_outcome_status).
- The hosted per-item verbs the old `runs` group carried (list/stream)
  moved to `papayya items` (plus `get`). Those tests swap the Papayya
  client with a recording fake.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from papayya import cli as cli_module
from papayya.api import PapayyaAPIError


class _FakeItems:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.list_return: list[dict[str, Any]] = []
        self.get_return: dict[str, Any] = {"id": "i1", "status": "completed"}
        self.stream_events: list[dict[str, Any]] = []
        self.raise_on: str | None = None

    def _maybe_raise(self, method: str) -> None:
        if self.raise_on == method:
            raise PapayyaAPIError(500, "boom")

    def list(self) -> list[dict[str, Any]]:
        self.calls.append(("list", {}))
        self._maybe_raise("list")
        return self.list_return

    def get(self, item_id: str) -> dict[str, Any]:
        self.calls.append(("get", {"item_id": item_id}))
        self._maybe_raise("get")
        return self.get_return

    def stream(self, item_id: str, *, from_step: int | None = None):
        self.calls.append(("stream", {"item_id": item_id, "from_step": from_step}))
        self._maybe_raise("stream")
        yield from self.stream_events


class _FakeClient:
    def __init__(self) -> None:
        self.items = _FakeItems()
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> _FakeClient:
    client = _FakeClient()
    monkeypatch.setattr(cli_module, "_make_papayya_client", lambda ctx: client)
    return client


def _run(args: list[str]) -> Any:
    return CliRunner().invoke(cli_module.main, args, catch_exceptions=False)


# ---------------------------------------------------------------------------
# runs list — LOCAL invocations with the outcome rollup
# ---------------------------------------------------------------------------


def _seed_local_db(db_path: Path) -> None:
    from datetime import datetime, timezone

    from papayya.durable.sqlite_store import SQLiteStore
    from papayya.durable.types import RunCheckpoint, TaskEntry

    store = SQLiteStore(str(db_path))
    store.create_run("inv-1", agent="triage", total_items=2)
    now = datetime.now(timezone.utc).isoformat()
    for i, outcome in enumerate(("ok", "degraded"), start=1):
        store.create(RunCheckpoint(
            run_id=f"item-{i}", agent="triage", tasks=[], status="running",
            created_at=now, updated_at=now, invocation_id="inv-1",
        ))
        store.save_task(f"item-{i}", TaskEntry(
            label="step", result="x", duration_ms=1, completed_at=now,
            outcome_status=outcome,
            outcome_reason="empty_string" if outcome != "ok" else None,
        ))
        store.set_status(f"item-{i}", "completed", output=None)
    store.close()


def test_runs_list_reads_local_ledger_with_outcome_rollup(tmp_path: Path) -> None:
    db = tmp_path / "local.db"
    _seed_local_db(db)

    result = _run(["runs", "list", "--db", str(db)])
    assert result.exit_code == 0, result.output
    lines = [json.loads(ln) for ln in result.output.splitlines() if ln.strip()]
    row = next(r for r in lines if r["run_id"] == "inv-1")
    assert row["agent"] == "triage"
    assert row["item_count"] == 2
    assert row["degraded_items"] == 1
    assert row["failed_items"] == 0
    assert row["worst_outcome_status"] == "degraded"


def test_runs_list_errors_without_local_db(tmp_path: Path) -> None:
    result = _run(["runs", "list", "--db", str(tmp_path / "missing.db")])
    assert result.exit_code == 1
    assert "No local database" in result.output


# ---------------------------------------------------------------------------
# items — hosted per-item verbs (the pre-0.3.0 `runs` surface, renamed)
# ---------------------------------------------------------------------------


def test_items_list_outputs_ndjson(fake_client: _FakeClient) -> None:
    fake_client.items.list_return = [{"id": "r1"}, {"id": "r2"}]
    result = _run(["items", "list"])
    assert result.exit_code == 0, result.output
    assert ("list", {}) in fake_client.items.calls
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert [json.loads(ln)["id"] for ln in lines] == ["r1", "r2"]


def test_items_get_pretty_prints(fake_client: _FakeClient) -> None:
    result = _run(["items", "get", "i1"])
    assert result.exit_code == 0, result.output
    assert ("get", {"item_id": "i1"}) in fake_client.items.calls
    assert json.loads(result.output)["id"] == "i1"


def test_items_stream_emits_one_event_per_line(fake_client: _FakeClient) -> None:
    fake_client.items.stream_events = [
        {"event": "step", "data": {"step_type": "llm"}, "id": 1},
        {"event": "terminal", "data": {"status": "completed"}},
    ]
    result = _run(["items", "stream", "r1", "--from-step", "5"])
    assert result.exit_code == 0, result.output
    assert ("stream", {"item_id": "r1", "from_step": 5}) in fake_client.items.calls
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert [json.loads(ln)["event"] for ln in lines] == ["step", "terminal"]


def test_runs_stream_is_gone(fake_client: _FakeClient) -> None:
    """BREAKING (0.3.0): streaming a per-item record lives at
    `items stream`; the old `runs stream` spelling errors rather than
    silently meaning something new."""
    result = _run(["runs", "stream", "r1"])
    assert result.exit_code != 0
