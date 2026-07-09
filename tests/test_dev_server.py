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


def _checkpoint(
    run_id: str,
    agent: str = "t",
    invocation_id: str | None = None,
    partition_key: str | None = None,
    item_id: str | None = None,
) -> RunCheckpoint:
    now = datetime.now(timezone.utc).isoformat()
    return RunCheckpoint(
        run_id=run_id, agent=agent, tasks=[], status="running",
        created_at=now, updated_at=now, invocation_id=invocation_id,
        partition_key=partition_key, item_id=item_id,
    )


def _task() -> TaskEntry:
    return TaskEntry(
        label="t", result="ok", duration_ms=100,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )


def _seed(store: SQLiteStore) -> None:
    """Populate a representative local DB for the endpoint tests."""
    # Explicit run (invocation) with 3 items — 2 successes, 1 failure.
    # Item 2 completes but its LLM step is DEGRADED (the wedge case: ran,
    # didn't work) and carries a tenant so /tenants has a blast radius.
    store.create_run("b-explicit", agent="enrich", total_items=3)
    for i, status in enumerate(("completed", "completed", "failed"), start=1):
        chk = _checkpoint(
            f"run-{i}",
            agent="enrich",
            invocation_id="b-explicit",
            partition_key="acme" if i == 2 else "globex",
            item_id=f"co_{i}",
        )
        store.create(chk)
        store.save_task(f"run-{i}", _task())
        if i == 2:
            store.save_task(
                f"run-{i}",
                TaskEntry(
                    label="call-model",
                    result={"content": ""},
                    duration_ms=80,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    kind="llm",
                    llm_total_tokens=130,
                    outcome_status="degraded",
                    outcome_reason="empty_string",
                    partition_key="acme",
                ),
            )
        # Item 3 carries a classified provider failure so the clusters
        # endpoint (grouping on steps.error_category) has one bucket.
        if i == 3:
            store.save_task(
                f"run-{i}",
                TaskEntry(
                    label="search",
                    result=None,
                    duration_ms=50,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    error_category="provider",
                ),
            )
        store.set_status(f"run-{i}", status, output=None)

    # A separate direct-call item outside the run for /api/items breadth
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
        # New-noun keys, incl. the wedge counter the runs page shows.
        assert body["items_total"] == 4
        assert body["items_degraded"] == 1
        assert body["runs_total"] >= 2


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
        # Exactly one cluster: the seeded provider-classified step. The v12
        # clusters endpoint groups on steps.error_category and emits it under
        # the old error_code key for the shipped UI.
        assert len(body) == 1
        assert body[0]["error_code"] == "provider"
        assert body[0]["count"] == 1

    def test_outliers_sorted_by_duration(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, body = _get(base, "/api/batches/b-explicit/outliers")
        assert status == 200
        durations = [r["duration_ms"] for r in body]
        assert durations == sorted(durations, reverse=True)


class TestBatchDlq:
    def test_lists_unresolved_failures(
        self, seeded_server: tuple[str, Path]
    ) -> None:
        """b-explicit has 2 completed + 1 failed; the failed run should show
        up as a dead letter until an operator disposes of it."""
        base, _ = seeded_server
        status, body = _get(base, "/api/batches/b-explicit/dlq")
        assert status == 200
        assert len(body) == 1
        dl = body[0]
        assert dl["run_id"] == "run-3"
        assert dl["agent"] == "enrich"

    def test_excludes_resolved_failures(
        self, seeded_server: tuple[str, Path], tmp_path: Path
    ) -> None:
        base, db_path = seeded_server
        store = SQLiteStore(str(db_path))
        from papayya.durable import _schema as schema
        store.mark_dlq_disposition("run-3", schema.DLQ_SKIPPED)
        store.close()

        status, body = _get(base, "/api/batches/b-explicit/dlq")
        assert status == 200
        assert body == []

    def test_unknown_batch_404(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, _body = _get(base, "/api/batches/nope/dlq")
        assert status == 404


class TestDlqActions:
    def test_skip_marks_disposition(
        self, seeded_server: tuple[str, Path]
    ) -> None:
        base, _ = seeded_server
        status, body = _post(base, "/api/batches/b-explicit/dlq/run-3/skip")
        assert status == 200
        assert body["disposition"] == "skipped"
        assert body["noop"] is False

        # Re-issuing is a no-op, not an error.
        status, body = _post(base, "/api/batches/b-explicit/dlq/run-3/skip")
        assert status == 200
        assert body["noop"] is True
        assert body["disposition"] == "skipped"

    def test_acknowledge_marks_disposition(
        self, seeded_server: tuple[str, Path], tmp_path: Path
    ) -> None:
        # Make a second failed item in the same run so we can acknowledge
        # it without interfering with the skip test's run-3.
        base, db_path = seeded_server
        store = SQLiteStore(str(db_path))
        store.create(_checkpoint("run-ack", invocation_id="b-explicit"))
        store.set_status("run-ack", "failed", output="nope")
        store.close()

        status, body = _post(base, "/api/batches/b-explicit/dlq/run-ack/acknowledge")
        assert status == 200
        assert body["disposition"] == "acknowledged"

    def test_dispose_run_not_in_batch_404(
        self, seeded_server: tuple[str, Path]
    ) -> None:
        base, _ = seeded_server
        # run-single exists but is in a different batch
        status, _body = _post(base, "/api/batches/b-explicit/dlq/run-single/skip")
        assert status == 404

    def test_dispose_non_failed_run_409(
        self, seeded_server: tuple[str, Path]
    ) -> None:
        base, _ = seeded_server
        status, _body = _post(base, "/api/batches/b-explicit/dlq/run-1/skip")
        assert status == 409

    def test_dispose_unknown_run_404(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, _body = _post(base, "/api/batches/b-explicit/dlq/nope/skip")
        assert status == 404

    def test_replay_without_snapshot_rejects(
        self, seeded_server: tuple[str, Path]
    ) -> None:
        """The seeded b-explicit/run-3 has no input_snapshot (the seed
        predates v6 semantics), so the replay endpoint must refuse at the
        validation step rather than spawn a CLI that would fail anyway."""
        base, _ = seeded_server
        status, body = _post(base, "/api/batches/b-explicit/dlq/run-3/replay")
        assert status == 409
        assert "input_snapshot" in body["error"]


class TestStepSearch:
    def test_by_tool_name_returns_empty(self, seeded_server: tuple[str, Path]) -> None:
        """tool_name searched the dead legacy LLM-call log; v12 has no such
        column, so the filter matches nothing (Unit 3 redesigns the page)."""
        base, _ = seeded_server
        status, body = _get(base, "/api/steps/search?tool_name=search_web")
        assert status == 200
        assert body == []

    def test_by_error_code(self, seeded_server: tuple[str, Path]) -> None:
        """error_code is the legacy spelling of error_category."""
        base, _ = seeded_server
        status, body = _get(base, "/api/steps/search?error_code=provider")
        assert status == 200
        assert len(body) == 1

    def test_by_label_and_outcome(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, body = _get(base, "/api/steps/search?label=call&outcome=degraded")
        assert status == 200
        assert len(body) == 1
        assert body[0]["label"] == "call-model"
        assert body[0]["outcome_status"] == "degraded"


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

    def test_batch_scope_also_empty(
        self, seeded_server: tuple[str, Path]
    ) -> None:
        """Legacy batch_id scope still accepted; the seed has no repeated
        identical steps so nothing is flagged."""
        base, _ = seeded_server
        status, body = _get(base, "/api/thrashing?batch_id=b-explicit")
        assert status == 200
        assert body == []

    def test_detects_repeated_identical_steps(
        self, seeded_server: tuple[str, Path]
    ) -> None:
        """Rebuilt on v12 step rows: the same step journaled 3+ times with
        the same input snapshot is a thrash. Repeated live calls store as
        label / label#2 / label#3 (occurrence suffix), so the detector
        groups on the bare label."""
        base, db_path = seeded_server
        store = SQLiteStore(str(db_path))
        store.create(_checkpoint("run-thrash"))
        for label in ("search", "search#2", "search#3"):
            store.save_task(
                "run-thrash",
                TaskEntry(
                    label=label,
                    result=None,
                    duration_ms=10,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    input_snapshot='{"q": "same"}',
                ),
            )
        store.close()

        status, body = _get(base, "/api/thrashing?item=run-thrash")
        assert status == 200
        assert len(body) == 1
        assert body[0]["label"] == "search"
        assert body[0]["repeat_count"] == 3

        # Legacy spelling of the record scope still works.
        status, body = _get(base, "/api/thrashing?run_id=run-thrash")
        assert status == 200
        assert body[0]["repeat_count"] == 3


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
    """The new-noun surface: /api/runs lists INVOCATIONS with the outcome
    rollup, /api/runs/:id/items lists the records, and
    /api/runs/:id/items/:record is per-record detail."""

    def test_runs_list_carries_worst_outcome_status(
        self, seeded_server: tuple[str, Path]
    ) -> None:
        """The wedge must read off the runs list without clicking anything:
        every run row carries worst_outcome_status + degraded/failed counts."""
        base, _ = seeded_server
        status, body = _get(base, "/api/runs")
        assert status == 200
        run = next(r for r in body if r["run_id"] == "b-explicit")
        assert run["worst_outcome_status"] == "failed"  # 1 failed beats 1 degraded
        assert run["degraded_items"] == 1
        assert run["failed_items"] == 1
        assert run["item_count"] == 3
        assert run["degraded_tenants"] == 1
        assert run["total_tokens"] == 130
        # Legacy alias kept one release.
        assert run["batch_id"] == "b-explicit"

    def test_runs_list_agent_filter(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, body = _get(base, "/api/runs?agent=enrich")
        assert status == 200
        assert all(r["agent"] == "enrich" for r in body)
        assert any(r["run_id"] == "b-explicit" for r in body)

    def test_run_detail_is_invocation(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, body = _get(base, "/api/runs/b-explicit")
        assert status == 200
        assert body["run_id"] == "b-explicit"
        assert body["total_items"] == 3
        assert body["worst_outcome_status"] == "failed"

    def test_run_detail_falls_back_to_record(
        self, seeded_server: tuple[str, Path]
    ) -> None:
        """A pre-0.3.0 'run id' was a record uuid — the endpoint falls back
        to the item row so old deep links land somewhere useful."""
        base, _ = seeded_server
        status, body = _get(base, "/api/runs/run-1")
        assert status == 200
        assert body["id"] == "run-1"
        assert body["run_id"] == "b-explicit"  # NEW meaning: the invocation

    def test_run_not_found(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, _body = _get(base, "/api/runs/does-not-exist")
        assert status == 404

    def test_run_items_carry_outcome_and_tokens(
        self, seeded_server: tuple[str, Path]
    ) -> None:
        base, _ = seeded_server
        status, body = _get(base, "/api/runs/b-explicit/items")
        assert status == 200
        assert len(body) == 3
        degraded = next(i for i in body if i["id"] == "run-2")
        assert degraded["worst_outcome_status"] == "degraded"
        assert degraded["degraded_count"] == 1
        assert degraded["total_tokens"] == 130
        assert degraded["step_count"] == 2
        assert degraded["partition_key"] == "acme"

    def test_run_items_outcome_filter(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, body = _get(base, "/api/runs/b-explicit/items?outcome=degraded")
        assert status == 200
        assert [i["id"] for i in body] == ["run-2"]

    def test_record_detail(self, seeded_server: tuple[str, Path]) -> None:
        """Per-RECORD keyspace: /api/runs/:run/items/:record returns the
        item row + its step trace."""
        base, _ = seeded_server
        status, body = _get(base, "/api/runs/b-explicit/items/run-2")
        assert status == 200
        assert body["item"]["id"] == "run-2"
        assert body["item"]["item_id"] == "co_2"  # customer identity
        assert len(body["steps"]) == 2
        degraded_step = body["steps"][1]
        assert degraded_step["outcome_status"] == "degraded"
        assert degraded_step["outcome_reason"] == "empty_string"

    def test_record_detail_wrong_run_404(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, _body = _get(base, "/api/runs/b-explicit/items/run-single")
        assert status == 404

    def test_run_tenants_blast_radius(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, body = _get(base, "/api/runs/b-explicit/tenants")
        assert status == 200
        by_tenant = {t["tenant"]: t for t in body}
        assert by_tenant["acme"]["degraded_items"] == 1
        assert by_tenant["globex"]["failed_items"] == 1
        # Sorted worst-first: the degraded tenant leads.
        assert body[0]["tenant"] == "acme"

    def test_legacy_steps_endpoint_gone(self, seeded_server: tuple[str, Path]) -> None:
        """/api/runs/<record>/steps served a dead pre-v12 table via a []
        stub; the stub is removed along with the page that fetched it."""
        base, _ = seeded_server
        status, _body = _get(base, "/api/runs/run-1/steps")
        assert status == 404


class TestAgentsEndpoint:
    def test_lists_agents_with_rollups(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, body = _get(base, "/api/agents")
        assert status == 200
        enrich = next(a for a in body if a["agent"] == "enrich")
        assert enrich["run_count"] == 1
        assert enrich["item_count"] == 3
        assert enrich["degraded_items"] == 1
        assert enrich["failed_items"] == 1


class TestItemsCollection:
    def test_lists_latest_records(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, body = _get(base, "/api/items")
        assert status == 200
        assert len(body) == 4
        assert {i["id"] for i in body} == {"run-1", "run-2", "run-3", "run-single"}

    def test_outcome_filter(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, body = _get(base, "/api/items?outcome=degraded")
        assert status == 200
        assert [i["id"] for i in body] == ["run-2"]


class TestItemLineage:
    def test_customer_keyspace(self, seeded_server: tuple[str, Path]) -> None:
        """/api/items/<customer id> is the lineage view — records + steps
        for that identity across runs (the OTHER keyspace)."""
        base, _ = seeded_server
        status, body = _get(base, "/api/items/co_2")
        assert status == 200
        assert body["item_id"] == "co_2"
        assert len(body["records"]) == 1
        assert body["records"][0]["id"] == "run-2"
        # Pre-0.3.0 keys kept one release.
        assert body["runs"] == body["records"]

    def test_unknown_item_404(self, seeded_server: tuple[str, Path]) -> None:
        base, _ = seeded_server
        status, _body = _get(base, "/api/items/co_nope")
        assert status == 404

    def test_run_tasks_exposes_llm_fields(
        self, seeded_server: tuple[str, Path], tmp_path: Path
    ) -> None:
        """The /tasks endpoint returns the v5 BYOF observability fields.

        pageRun in app.js depends on these being present on the row so
        renderLlmBadges can decide what to draw. If the handler ever
        stops returning them (e.g. a SELECT narrowed beyond `*`), the
        dashboard silently loses the badges.
        """
        base, db_path = seeded_server
        store = SQLiteStore(str(db_path))
        store.create(_checkpoint("run-llm"))
        store.save_task(
            "run-llm",
            TaskEntry(
                label="call-gemini",
                result={"ok": True},
                duration_ms=240,
                completed_at=datetime.now(timezone.utc).isoformat(),
                kind="llm",
                llm_prompt_tokens=40,
                llm_completion_tokens=10,
                llm_total_tokens=50,
                llm_model="gemini-2.0-flash",
                llm_stop_reason="STOP",
                llm_provider_shape="gemini",
            ),
        )
        store.close()

        status, body = _get(base, "/api/runs/run-llm/tasks")
        assert status == 200
        assert len(body) == 1
        row = body[0]
        assert row["kind"] == "llm"
        assert row["llm_provider_shape"] == "gemini"
        assert row["llm_total_tokens"] == 50
        assert row["llm_model"] == "gemini-2.0-flash"
        assert row["llm_stop_reason"] == "STOP"


class TestCancelEndpoint:
    def test_cancel_running_batch(
        self, seeded_server: tuple[str, Path], tmp_path: Path
    ) -> None:
        base, db_path = seeded_server
        # Seed a new running batch that isn't yet terminal
        store = SQLiteStore(str(db_path))
        store.create_run("b-live", agent="t", total_items=5)
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


class TestServePortConflict:
    """Phase 1.5 finding: a stale `papayya dev` on the default port produced
    a bare `OSError: [Errno 48] Address already in use` with no port number
    in the message. ``serve()`` should exit 1 with a clear message instead."""

    def test_exits_with_clear_message_when_port_in_use(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from papayya.dev.server import serve

        db_path = tmp_path / "local.db"
        SQLiteStore(str(db_path)).close()

        squatter = socket.socket()
        squatter.bind(("127.0.0.1", 0))
        squatter.listen(1)
        port = squatter.getsockname()[1]

        try:
            with pytest.raises(SystemExit) as excinfo:
                serve(host="127.0.0.1", port=port, db_path=str(db_path))
        finally:
            squatter.close()

        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert str(port) in captured.err
        assert "already in use" in captured.err
        assert "--port" in captured.err
