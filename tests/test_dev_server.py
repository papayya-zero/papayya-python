"""Tests for Slice 3 — the dev-server API endpoints.

Each endpoint is exercised end-to-end through a real HTTP request against
an ephemeral ``ThreadingHTTPServer`` instance. That's slightly heavier than
calling the handler functions directly, but it catches routing bugs,
JSON serialisation problems, and 500-on-bad-input regressions that the
execution plan explicitly calls out.
"""

from __future__ import annotations

import json
import socket
import threading
import urllib.request
import urllib.error
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

from papayya.dev.server import DevHandler
from papayya.durable.sqlite_store import SQLiteStore
from papayya.durable.types import RunCheckpoint, TaskEntry


# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _checkpoint(run_id: str, agent: str = "t") -> RunCheckpoint:
    now = datetime.now(timezone.utc).isoformat()
    return RunCheckpoint(
        run_id=run_id, agent=agent, tasks=[], status="running",
        created_at=now, updated_at=now,
    )


def _task() -> TaskEntry:
    return TaskEntry(
        label="t", result="ok", duration_ms=100,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )


def _seed(store: SQLiteStore) -> None:
    """Populate a representative local DB for the endpoint tests."""
    # Explicit batch with 3 runs — 2 successes, 1 failure
    store.create_batch("b-explicit", agent="enrich", total_items=3)
    for i, status in enumerate(("completed", "completed", "failed"), start=1):
        chk = _checkpoint(f"run-{i}")
        chk.agent = "enrich"
        store.create(chk)
        # Rewrite the implicit-batch linkage so these 3 belong to b-explicit
        store._conn.execute(
            "UPDATE runs SET batch_id='b-explicit' WHERE run_id=?", (f"run-{i}",)
        )
        store._conn.commit()
        store.save_task(f"run-{i}", _task())
        # Two runs share a tool + input (cluster of 2); run-3 fails provider
        if i < 3:
            store.record_step(
                f"run-{i}", task_label="search",
                tool_calls=[{"name": "search_web", "arguments": {"q": "x"}}],
                duration_ms=50,
            )
        else:
            store.record_step(
                f"run-{i}", task_label="search",
                tool_calls=[{"name": "search_web", "arguments": {"q": "y"}}],
                duration_ms=50,
                error_message="HTTP 429: rate limit",
            )
        store.set_status(f"run-{i}", status, output=None)

    # A separate single-run outside the batch for /api/runs breadth
    store.create(_checkpoint("run-single"))
    store.save_task("run-single", _task())
    store.set_status("run-single", "completed", output=None)


@pytest.fixture
def seeded_server(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    db_path = tmp_path / "local.db"
    store = SQLiteStore(str(db_path))
    _seed(store)
    store.close()

    DevHandler.db_path = str(db_path)
    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), DevHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", db_path
    finally:
        server.shutdown()
        server.server_close()


def _get(base: str, path: str) -> tuple[int, Any]:
    try:
        with urllib.request.urlopen(base + path, timeout=5) as resp:
            body = json.loads(resp.read().decode())
            return (resp.status, body)
    except urllib.error.HTTPError as e:
        body = json.loads(e.read().decode())
        return (e.code, body)


def _post(base: str, path: str) -> tuple[int, Any]:
    req = urllib.request.Request(base + path, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode())
            return (resp.status, body)
    except urllib.error.HTTPError as e:
        body = json.loads(e.read().decode())
        return (e.code, body)


# --------------------------------------------------------------------------- #
#  Endpoints                                                                   #
# --------------------------------------------------------------------------- #


class TestStats:
    def test_includes_batch_counts(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, body = _get(base, "/api/stats")
        assert status == 200
        assert "total_batches" in body
        assert body["total_batches"] >= 2  # b-explicit + implicit for run-single
        assert body["total_runs"] == 4


class TestBatches:
    def test_list(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, body = _get(base, "/api/batches")
        assert status == 200
        assert any(b["batch_id"] == "b-explicit" for b in body)

    def test_detail(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, body = _get(base, "/api/batches/b-explicit")
        assert status == 200
        assert body["total_items"] == 3
        assert body["completed"] == 2
        assert body["failed"] == 1

    def test_detail_unknown_404(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, body = _get(base, "/api/batches/nope")
        assert status == 404
        assert "error" in body

    def test_runs(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, body = _get(base, "/api/batches/b-explicit/runs")
        assert status == 200
        assert len(body) == 3

    def test_runs_filtered_by_status(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, body = _get(base, "/api/batches/b-explicit/runs?status=failed")
        assert status == 200
        assert len(body) == 1
        assert body[0]["status"] == "failed"

    def test_clusters(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, body = _get(base, "/api/batches/b-explicit/clusters")
        assert status == 200
        # Exactly one failed step cluster (provider_rate_limit / hash of q=y)
        assert len(body) == 1
        assert body[0]["error_code"] == "provider_rate_limit"
        assert body[0]["count"] == 1

    def test_outliers_sorted_by_duration(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, body = _get(base, "/api/batches/b-explicit/outliers")
        assert status == 200
        durations = [r["duration_ms"] for r in body]
        assert durations == sorted(durations, reverse=True)


class TestStepSearch:
    def test_by_tool_name(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, body = _get(base, "/api/steps/search?tool_name=search_web")
        assert status == 200
        assert len(body) == 3
        assert all(s["tool_name"] == "search_web" for s in body)

    def test_by_error_code(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, body = _get(base, "/api/steps/search?error_code=provider_rate_limit")
        assert status == 200
        assert len(body) == 1


class TestThrashing:
    def test_requires_scope(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, _body = _get(base, "/api/thrashing")
        assert status == 400

    def test_run_with_no_thrash_returns_empty(
        self, seeded_server: tuple[str, Path]
    ) -> None:
        base, _ = seeded_server
        # The seed has at most 1 call per (run, tool, hash) — no thrash
        status, body = _get(base, "/api/thrashing?run_id=run-1")
        assert status == 200
        assert body == []

    def test_detects_repeated_identical_calls(
        self, seeded_server: tuple[str, Path], tmp_path: Path
    ) -> None:
        base, db_path = seeded_server
        store = SQLiteStore(str(db_path))
        for _ in range(5):
            store.record_step(
                "run-1", task_label="search",
                tool_calls=[{"name": "thrash_tool", "arguments": {"x": 1}}],
            )
        store.close()

        status, body = _get(base, "/api/thrashing?run_id=run-1")
        assert status == 200
        assert any(r["tool_name"] == "thrash_tool" and r["repeat_count"] == 5 for r in body)


class TestProjection:
    def test_returns_rolling_window(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, body = _get(base, "/api/projection")
        assert status == 200
        assert body["window_days"] == 30
        assert body["total_runs"] >= 4
        assert body["total_batches"] >= 2
        assert body["compute_minutes"] >= 0


class TestRunEndpoints:
    def test_run_detail(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, body = _get(base, "/api/runs/run-1")
        assert status == 200
        assert body["run_id"] == "run-1"

    def test_run_not_found(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, _body = _get(base, "/api/runs/does-not-exist")
        assert status == 404

    def test_run_steps(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, body = _get(base, "/api/runs/run-1/steps")
        assert status == 200
        assert len(body) >= 1


class TestCancelEndpoint:
    def test_cancel_running_batch(
        self, seeded_server: tuple[str, Path], tmp_path: Path
    ) -> None:
        base, db_path = seeded_server
        # Seed a new running batch that isn't yet terminal
        store = SQLiteStore(str(db_path))
        store.create_batch("b-live", agent="t", total_items=5)
        store.close()

        status, body = _post(base, "/api/batches/b-live/cancel")
        assert status == 200
        assert body == {"noop": False, "status": "cancelled"}

    def test_cancel_completed_batch_is_noop(
        self, seeded_server: tuple[str, Path]
    ) -> None:
        base, _ = seeded_server
        # b-explicit is terminal already because all 3 runs finished
        status, body = _post(base, "/api/batches/b-explicit/cancel")
        assert status == 200
        assert body["noop"] is True

    def test_cancel_unknown_404(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, _body = _post(base, "/api/batches/nope/cancel")
        assert status == 404


class TestBadInput:
    """Guard: never return 500 for malformed input."""

    def test_unknown_api_path(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, _body = _get(base, "/api/wat")
        assert status == 404

    def test_bad_limit(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, _body = _get(base, "/api/runs?limit=abc")
        assert status == 400

    def test_negative_limit(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, _body = _get(base, "/api/runs?limit=-1")
        assert status == 400


class TestStaticFallback:
    def test_api_404_is_json_not_html(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        url = base + "/api/nothing-here"
        try:
            urllib.request.urlopen(url, timeout=5)
        except urllib.error.HTTPError as e:
            assert e.code == 404
            assert e.headers.get("Content-Type", "").startswith("application/json")
