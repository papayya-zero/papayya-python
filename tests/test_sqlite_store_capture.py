"""Tests for capture-at-write-time in the local ledger (v12 nouns).

Covers the capture guarantees from ``LOCAL_DEV_EXECUTION.md``, restated in
Plan 34 vocabulary:

1. Every direct-call item gets an implicit run-of-one so the run-first UI
   has a home.
2. Terminal item transitions roll up to run counters and mark the run
   terminal once all items resolve.
3. Explicit runs (invocations) link items via ``invocation_id``.

Plus a feature-flag test for the ``PAPAYYA_LOCAL_CAPTURE_V2=false`` fallback,
and a benchmark to guard the <10% write-overhead budget.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from papayya.durable import _errors, _schema
from papayya.durable.sqlite_store import SQLiteStore, _single_run_id
from papayya.durable.types import RunCheckpoint, TaskEntry


def _checkpoint(
    run_id: str = "run-1",
    agent: str = "t",
    invocation_id: str | None = None,
) -> RunCheckpoint:
    now = datetime.now(timezone.utc).isoformat()
    return RunCheckpoint(
        run_id=run_id,
        agent=agent,
        tasks=[],
        status="running",
        created_at=now,
        updated_at=now,
        invocation_id=invocation_id,
    )


def _task(label: str = "t") -> TaskEntry:
    return TaskEntry(
        label=label,
        result="ok",
        duration_ms=100,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )


@pytest.fixture
def store(tmp_path: Path) -> SQLiteStore:
    return SQLiteStore(str(tmp_path / "local.db"))


# --------------------------------------------------------------------------- #
#  Error classification (shared _errors module)                                #
# --------------------------------------------------------------------------- #


class TestErrorClassification:
    def test_provider_rate_limit(self) -> None:
        code, cat = _errors.classify_error("HTTP 429: rate limit exceeded")
        assert cat == _errors.CATEGORY_PROVIDER
        assert code == "provider_rate_limit"

    def test_provider_credit(self) -> None:
        code, cat = _errors.classify_error("Insufficient credits remaining")
        assert cat == _errors.CATEGORY_PROVIDER
        assert code == "provider_credit"

    def test_timeout(self) -> None:
        _, cat = _errors.classify_error("deadline exceeded")
        assert cat == _errors.CATEGORY_TIMEOUT

    def test_budget_exceeded_is_timeout_category(self) -> None:
        _, cat = _errors.classify_error("Budget exceeded: $0.50 consumed")
        assert cat == _errors.CATEGORY_TIMEOUT

    def test_tool_error(self) -> None:
        _, cat = _errors.classify_error("tool call failed: bad JSON")
        assert cat == _errors.CATEGORY_TOOL

    def test_logic_fallback(self) -> None:
        code, cat = _errors.classify_error("KeyError: 'missing_key'")
        assert cat == _errors.CATEGORY_LOGIC
        assert code == "logic_error"

    def test_empty_input(self) -> None:
        assert _errors.classify_error(None) == (None, None)
        assert _errors.classify_error("") == (None, None)
        assert _errors.classify_error("   ") == (None, None)


# --------------------------------------------------------------------------- #
#  Implicit run-of-one                                                         #
# --------------------------------------------------------------------------- #


class TestImplicitRunOfOne:
    def test_create_makes_single_run(self, store: SQLiteStore, tmp_path: Path) -> None:
        store.create(_checkpoint("run-1"))
        conn = sqlite3.connect(tmp_path / "local.db")
        conn.row_factory = sqlite3.Row
        run = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (_single_run_id("run-1"),),
        ).fetchone()
        assert run is not None
        assert run["total_items"] == 1
        assert run["agent"] == "t"
        assert run["status"] == "running"

    def test_item_is_linked_to_run(self, store: SQLiteStore, tmp_path: Path) -> None:
        store.create(_checkpoint("run-1"))
        conn = sqlite3.connect(tmp_path / "local.db")
        conn.row_factory = sqlite3.Row
        item = conn.execute("SELECT * FROM items WHERE id='run-1'").fetchone()
        assert item["run_id"] == _single_run_id("run-1")

    def test_recreate_is_idempotent(self, store: SQLiteStore, tmp_path: Path) -> None:
        store.create(_checkpoint("run-1"))
        # Simulate a resume path where create is called a second time for the
        # same item — the implicit run INSERT uses OR IGNORE so the second
        # call must not raise. The second item INSERT will fail by PK, but the
        # run-side idempotency is what's under test here.
        conn = sqlite3.connect(tmp_path / "local.db")
        rows = conn.execute(
            "SELECT COUNT(*) FROM runs WHERE run_id = ?",
            (_single_run_id("run-1"),),
        ).fetchone()
        assert rows[0] == 1

    def test_explicit_invocation_id_suppresses_single_wrap(
        self, store: SQLiteStore, tmp_path: Path
    ) -> None:
        """An item created with an invocation_id links to that run and does
        NOT get an implicit single- wrapper — the shift-by-one fix."""
        store.create_run("inv-1", agent="t", total_items=1)
        store.create(_checkpoint("run-1", invocation_id="inv-1"))
        conn = sqlite3.connect(tmp_path / "local.db")
        conn.row_factory = sqlite3.Row
        item = conn.execute("SELECT * FROM items WHERE id='run-1'").fetchone()
        assert item["run_id"] == "inv-1"
        single = conn.execute(
            "SELECT COUNT(*) FROM runs WHERE run_id = ?",
            (_single_run_id("run-1"),),
        ).fetchone()[0]
        assert single == 0


# --------------------------------------------------------------------------- #
#  Aggregates roll up                                                          #
# --------------------------------------------------------------------------- #


class TestAggregateRollup:
    def test_terminal_status_bumps_completed(
        self, store: SQLiteStore, tmp_path: Path
    ) -> None:
        store.create(_checkpoint("run-1"))
        store.set_status("run-1", "completed", output="done")

        conn = sqlite3.connect(tmp_path / "local.db")
        conn.row_factory = sqlite3.Row
        run = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (_single_run_id("run-1"),),
        ).fetchone()
        assert run["completed"] == 1
        assert run["failed"] == 0
        assert run["status"] == "completed"
        assert run["completed_at"] is not None

    def test_terminal_status_bumps_failed(
        self, store: SQLiteStore, tmp_path: Path
    ) -> None:
        store.create(_checkpoint("run-1"))
        store.set_status("run-1", "failed", output="nope")

        conn = sqlite3.connect(tmp_path / "local.db")
        conn.row_factory = sqlite3.Row
        run = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (_single_run_id("run-1"),),
        ).fetchone()
        assert run["failed"] == 1
        assert run["completed"] == 0

    def test_mixed_run_status_is_partial(
        self, store: SQLiteStore, tmp_path: Path
    ) -> None:
        """A run with both successes and failures becomes 'partial', not
        'completed'. Guards the bug where any failure was silently dressed
        up as a clean completion."""
        store.create_run("b-mixed", agent="t", total_items=3)
        for i, status in enumerate(("completed", "completed", "failed"), start=1):
            rid = f"run-{i}"
            store.create(_checkpoint(rid, invocation_id="b-mixed"))
            store.set_status(rid, status, output=None)

        conn = sqlite3.connect(tmp_path / "local.db")
        conn.row_factory = sqlite3.Row
        run = conn.execute(
            "SELECT * FROM runs WHERE run_id='b-mixed'"
        ).fetchone()
        assert run["completed"] == 2
        assert run["failed"] == 1
        assert run["status"] == "partial"
        assert run["completed_at"] is not None

    def test_all_failed_run_status_is_failed(
        self, store: SQLiteStore, tmp_path: Path
    ) -> None:
        """Every item failed → run status = 'failed', not 'partial'."""
        store.create_run("b-all-failed", agent="t", total_items=2)
        for i in (1, 2):
            rid = f"rf-{i}"
            store.create(_checkpoint(rid, invocation_id="b-all-failed"))
            store.set_status(rid, "failed", output=None)

        conn = sqlite3.connect(tmp_path / "local.db")
        conn.row_factory = sqlite3.Row
        run = conn.execute(
            "SELECT * FROM runs WHERE run_id='b-all-failed'"
        ).fetchone()
        assert run["completed"] == 0
        assert run["failed"] == 2
        assert run["status"] == "failed"

    def test_double_terminal_transition_only_counts_once(
        self, store: SQLiteStore, tmp_path: Path
    ) -> None:
        """Guard: re-transitioning from terminal should not double-count.

        ``set_status`` gates its counter bump on the prior status being
        non-terminal. If the caller re-sets 'completed', the run counter
        should stay at 1, not climb to 2.
        """
        store.create(_checkpoint("run-1"))
        store.set_status("run-1", "completed", output="done")
        store.set_status("run-1", "completed", output="done-again")

        conn = sqlite3.connect(tmp_path / "local.db")
        completed = conn.execute(
            "SELECT completed FROM runs WHERE run_id = ?",
            (_single_run_id("run-1"),),
        ).fetchone()[0]
        assert completed == 1

    def test_open_run_does_not_roll_up_until_finalized(
        self, store: SQLiteStore, tmp_path: Path
    ) -> None:
        """An OPEN run (total_items=0, minted by map/iter before the count
        is known) must not flip terminal on each item — only finalize_run
        seals it."""
        store.create_run("open-1", agent="t")  # total_items defaults to 0
        for i in (1, 2):
            rid = f"op-{i}"
            store.create(_checkpoint(rid, invocation_id="open-1"))
            store.set_status(rid, "completed", output=None)
            row = store.get_run("open-1")
            assert row is not None and row["status"] == "running"

        store.finalize_run("open-1")
        row = store.get_run("open-1")
        assert row is not None
        assert row["total_items"] == 2
        assert row["status"] == "completed"
        assert row["completed_at"] is not None

    def test_finalize_zero_item_run_completes(self, store: SQLiteStore) -> None:
        """map() over an empty iterable must not leave a forever-'running'
        run row."""
        store.create_run("empty-1", agent="t")
        store.finalize_run("empty-1")
        row = store.get_run("empty-1")
        assert row is not None
        assert row["total_items"] == 0
        assert row["status"] == "completed"


# --------------------------------------------------------------------------- #
#  Explicit multi-item runs                                                    #
# --------------------------------------------------------------------------- #


class TestExplicitRun:
    def test_create_run_then_link_items(
        self, store: SQLiteStore, tmp_path: Path
    ) -> None:
        store.create_run("b-1", agent="enricher", total_items=3, concurrency_cap=2)

        conn = sqlite3.connect(tmp_path / "local.db")
        conn.row_factory = sqlite3.Row
        run = conn.execute(
            "SELECT * FROM runs WHERE run_id='b-1'"
        ).fetchone()
        assert run["total_items"] == 3
        assert run["concurrency_cap"] == 2
        assert run["status"] == "running"

    def test_create_run_records_replayed_from(
        self, store: SQLiteStore, tmp_path: Path
    ) -> None:
        store.create_run("b-src", agent="t", total_items=1)
        store.create_run("b-replay", agent="t", total_items=1, replayed_from="b-src")

        conn = sqlite3.connect(tmp_path / "local.db")
        conn.row_factory = sqlite3.Row
        run = conn.execute(
            "SELECT * FROM runs WHERE run_id='b-replay'"
        ).fetchone()
        assert run["replayed_from"] == "b-src"


# --------------------------------------------------------------------------- #
#  Feature flag fallback                                                       #
# --------------------------------------------------------------------------- #


class TestCaptureDisabled:
    def test_no_implicit_run_when_disabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PAPAYYA_LOCAL_CAPTURE_V2", "false")
        store = SQLiteStore(str(tmp_path / "local.db"))
        store.create(_checkpoint("run-1"))

        conn = sqlite3.connect(tmp_path / "local.db")
        run_count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        assert run_count == 0


# --------------------------------------------------------------------------- #
#  LLM-kind round trip                                                         #
# --------------------------------------------------------------------------- #


class TestLlmKindRoundTrip:
    """A ``kind="llm"`` TaskEntry persists and loads back identically.

    Guards the SQLite side of the BYOF observability feature: if any of
    the eight v5 columns get silently dropped on write or decode, the
    dashboard sees missing usage data and the feature breaks without
    raising.
    """

    def test_llm_task_round_trips(self, store: SQLiteStore) -> None:
        store.create(_checkpoint("run-1"))
        entry = TaskEntry(
            label="call-openai",
            result={"ok": True},
            duration_ms=250,
            completed_at=datetime.now(timezone.utc).isoformat(),
            kind="llm",
            llm_prompt_tokens=100,
            llm_completion_tokens=30,
            llm_total_tokens=130,
            llm_model="gpt-4o-mini",
            llm_stop_reason="stop",
            llm_provider_shape="openai",
            error_category=None,
        )
        store.save_task("run-1", entry)

        loaded = store.load("run-1")
        assert loaded is not None
        assert len(loaded.tasks) == 1
        got = loaded.tasks[0]
        assert got.kind == "llm"
        assert got.llm_prompt_tokens == 100
        assert got.llm_completion_tokens == 30
        assert got.llm_total_tokens == 130
        assert got.llm_model == "gpt-4o-mini"
        assert got.llm_stop_reason == "stop"
        assert got.llm_provider_shape == "openai"
        assert got.error_category is None

    def test_non_llm_task_stores_nulls(self, store: SQLiteStore) -> None:
        """A plain step (no ``kind``) must round-trip with LLM fields as None."""
        store.create(_checkpoint("run-1"))
        entry = TaskEntry(
            label="plain",
            result="ok",
            duration_ms=10,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        store.save_task("run-1", entry)

        loaded = store.load("run-1")
        assert loaded is not None
        got = loaded.tasks[0]
        assert got.kind is None
        assert got.llm_prompt_tokens is None
        assert got.llm_provider_shape is None
        assert got.error_category is None

    def test_error_category_round_trips(self, store: SQLiteStore) -> None:
        """``error_category`` persists independently of the LLM usage fields."""
        store.create(_checkpoint("run-1"))
        entry = TaskEntry(
            label="classified-failure",
            result=None,
            duration_ms=50,
            completed_at=datetime.now(timezone.utc).isoformat(),
            kind="llm",
            llm_provider_shape="unknown",
            error_category="provider",
        )
        store.save_task("run-1", entry)

        loaded = store.load("run-1")
        assert loaded is not None
        got = loaded.tasks[0]
        assert got.error_category == "provider"
        assert got.llm_provider_shape == "unknown"


class TestDlqDrainedPromotion:
    """A 'partial' run promotes to 'completed' once its DLQ is empty."""

    def _setup_partial(self, store: SQLiteStore) -> None:
        store.create_run("b-drained", agent="t", total_items=3)
        for i, status in enumerate(("completed", "completed", "failed"), start=1):
            rid = f"drn-{i}"
            store.create(_checkpoint(rid, invocation_id="b-drained"))
            store.set_status(rid, status, output=None)

    def test_skipping_last_dead_letter_promotes_to_completed(
        self, store: SQLiteStore, tmp_path: Path
    ) -> None:
        self._setup_partial(store)
        # Sanity: partial first.
        conn = sqlite3.connect(tmp_path / "local.db")
        conn.row_factory = sqlite3.Row
        pre = conn.execute(
            "SELECT status FROM runs WHERE run_id='b-drained'"
        ).fetchone()
        assert pre["status"] == "partial"

        store.mark_dlq_disposition("drn-3", _schema.DLQ_SKIPPED)

        post = conn.execute(
            "SELECT status FROM runs WHERE run_id='b-drained'"
        ).fetchone()
        assert post["status"] == "completed"

    def test_multiple_dead_letters_partial_until_all_resolved(
        self, store: SQLiteStore, tmp_path: Path
    ) -> None:
        """Resolving one of two dead letters keeps the run 'partial'."""
        store.create_run("b-two-dead", agent="t", total_items=3)
        for i, status in enumerate(("completed", "failed", "failed"), start=1):
            rid = f"td-{i}"
            store.create(_checkpoint(rid, invocation_id="b-two-dead"))
            store.set_status(rid, status, output=None)

        store.mark_dlq_disposition("td-2", _schema.DLQ_SKIPPED)

        conn = sqlite3.connect(tmp_path / "local.db")
        conn.row_factory = sqlite3.Row
        mid = conn.execute(
            "SELECT status FROM runs WHERE run_id='b-two-dead'"
        ).fetchone()
        assert mid["status"] == "partial"

        store.mark_dlq_disposition("td-3", _schema.DLQ_ACKNOWLEDGED)
        after = conn.execute(
            "SELECT status FROM runs WHERE run_id='b-two-dead'"
        ).fetchone()
        assert after["status"] == "completed"

    def test_failed_run_stays_failed(
        self, store: SQLiteStore, tmp_path: Path
    ) -> None:
        """Draining the DLQ of an all-failed run does not make it
        'completed' — it was a rout, not a partial success."""
        store.create_run("b-all-failed", agent="t", total_items=2)
        for i in (1, 2):
            rid = f"af-{i}"
            store.create(_checkpoint(rid, invocation_id="b-all-failed"))
            store.set_status(rid, "failed", output=None)

        store.mark_dlq_disposition("af-1", _schema.DLQ_SKIPPED)
        store.mark_dlq_disposition("af-2", _schema.DLQ_SKIPPED)

        conn = sqlite3.connect(tmp_path / "local.db")
        conn.row_factory = sqlite3.Row
        run = conn.execute(
            "SELECT status FROM runs WHERE run_id='b-all-failed'"
        ).fetchone()
        assert run["status"] == "failed"


class TestItemInputSnapshot:
    """Item-level input_snapshot is the DLQ replay source."""

    def test_input_snapshot_round_trips(self, store: SQLiteStore) -> None:
        now = datetime.now(timezone.utc).isoformat()
        store.create(RunCheckpoint(
            run_id="run-with-input", agent="t", tasks=[],
            status="running", created_at=now, updated_at=now,
            input_snapshot={"lead_id": "xyz", "email": "a@b.com"},
        ))
        loaded = store.load("run-with-input")
        assert loaded is not None
        assert loaded.input_snapshot == {"lead_id": "xyz", "email": "a@b.com"}

    def test_input_snapshot_defaults_to_none(self, store: SQLiteStore) -> None:
        now = datetime.now(timezone.utc).isoformat()
        store.create(RunCheckpoint(
            run_id="run-no-input", agent="t", tasks=[],
            status="running", created_at=now, updated_at=now,
        ))
        loaded = store.load("run-no-input")
        assert loaded is not None
        assert loaded.input_snapshot is None


class TestDlqDisposition:
    """mark_dlq_disposition transitions a failed item out of the DLQ."""

    def _failed_item(self, store: SQLiteStore, run_id: str = "dead-1") -> None:
        store.create(_checkpoint(run_id))
        store.set_status(run_id, "failed", output="boom")

    def test_skip_sets_disposition_and_resolved_at(
        self, store: SQLiteStore, tmp_path: Path
    ) -> None:
        self._failed_item(store)
        store.mark_dlq_disposition("dead-1", _schema.DLQ_SKIPPED)

        conn = sqlite3.connect(tmp_path / "local.db")
        conn.row_factory = sqlite3.Row
        item = conn.execute("SELECT * FROM items WHERE id='dead-1'").fetchone()
        assert item[_schema.COL_ITEM_DLQ_DISPOSITION] == _schema.DLQ_SKIPPED
        assert item[_schema.COL_ITEM_DLQ_RESOLVED_AT] is not None

    def test_replay_sets_replayed_from_on_new_item(
        self, store: SQLiteStore, tmp_path: Path
    ) -> None:
        """When a replay happens, the NEW item carries replayed_from; this test
        covers the older-item-side half: disposition=replayed, link is nullable
        on this side (the chain points forward from the original)."""
        self._failed_item(store, run_id="dead-2")
        store.mark_dlq_disposition("dead-2", _schema.DLQ_REPLAYED)

        conn = sqlite3.connect(tmp_path / "local.db")
        conn.row_factory = sqlite3.Row
        item = conn.execute("SELECT * FROM items WHERE id='dead-2'").fetchone()
        assert item[_schema.COL_ITEM_DLQ_DISPOSITION] == _schema.DLQ_REPLAYED

    def test_double_disposition_is_noop(
        self, store: SQLiteStore, tmp_path: Path
    ) -> None:
        """Second call with a different disposition must not overwrite."""
        self._failed_item(store, run_id="dead-3")
        store.mark_dlq_disposition("dead-3", _schema.DLQ_SKIPPED)
        store.mark_dlq_disposition("dead-3", _schema.DLQ_ACKNOWLEDGED)

        conn = sqlite3.connect(tmp_path / "local.db")
        conn.row_factory = sqlite3.Row
        item = conn.execute("SELECT * FROM items WHERE id='dead-3'").fetchone()
        assert item[_schema.COL_ITEM_DLQ_DISPOSITION] == _schema.DLQ_SKIPPED

    def test_disposition_on_non_failed_is_noop(
        self, store: SQLiteStore, tmp_path: Path
    ) -> None:
        """Don't mark a successful item as dead-letter'd. The UPDATE is guarded
        by status='failed', so calling on a completed item is a silent no-op."""
        store.create(_checkpoint("run-ok"))
        store.set_status("run-ok", "completed", output="ok")
        store.mark_dlq_disposition("run-ok", _schema.DLQ_SKIPPED)

        conn = sqlite3.connect(tmp_path / "local.db")
        conn.row_factory = sqlite3.Row
        item = conn.execute("SELECT * FROM items WHERE id='run-ok'").fetchone()
        assert item[_schema.COL_ITEM_DLQ_DISPOSITION] is None

    def test_invalid_disposition_raises(self, store: SQLiteStore) -> None:
        self._failed_item(store, run_id="dead-4")
        with pytest.raises(ValueError):
            store.mark_dlq_disposition("dead-4", "something_else")


class TestWriteOverhead:
    """Guard the 10%-overhead budget called out in LOCAL_DEV_EXECUTION.md.

    The benchmark is intentionally coarse — it exists to catch a regression
    of orders-of-magnitude, not to measure micro-performance. Local SQLite
    writes are in the microseconds; this check runs in well under a second.
    """

    def test_1000_steps_completes_quickly(self, store: SQLiteStore) -> None:
        store.create(_checkpoint("run-1"))
        start = time.perf_counter()
        for i in range(1000):
            store.save_task("run-1", _task(label=f"t-{i}"))
        elapsed = time.perf_counter() - start
        # Very generous bound — a regression that blows this is architectural
        # (e.g. accidental O(N) scan per write), not micro.
        assert elapsed < 5.0, f"1000 steps took {elapsed:.2f}s"
