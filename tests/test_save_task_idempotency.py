"""Workstream D parity: local save_task is idempotent on (run_id, label).

Mirrors the control-plane SaveCheckpoint xmax=0 guard — a re-delivery of the
same step must not insert a duplicate task row or double-count the run-level
worst_outcome_status / degraded_count aggregates. The local step cache
normally prevents re-execution, so this is defensive; first-writer-wins.
"""

from __future__ import annotations

from datetime import datetime, timezone

from papayya.durable.run import PapayyaRun, DurableRunConfig, MemoryStore
from papayya.durable.sqlite_store import SQLiteStore
from papayya.durable.types import RunCheckpoint, TaskEntry


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_run(store: SQLiteStore, run_id: str = "r1") -> None:
    store.create(
        RunCheckpoint(
            run_id=run_id, agent="a", tasks=[], status="running",
            created_at=_now(), updated_at=_now(),
        )
    )


def _task(label: str, status: str = "degraded", reason: str | None = "empty_none") -> TaskEntry:
    return TaskEntry(
        label=label, result={"v": label}, duration_ms=1, completed_at=_now(),
        outcome_status=status, outcome_reason=reason,
    )


def test_redelivery_does_not_double_count_aggregates(tmp_path):
    store = SQLiteStore(str(tmp_path / "idem.db"))
    try:
        _make_run(store)
        store.save_task("r1", _task("retrieve"))
        # Re-deliver the same (run_id, label) twice more.
        store.save_task("r1", _task("retrieve"))
        store.save_task("r1", _task("retrieve"))

        loaded = store.load("r1")
        assert loaded is not None
        assert loaded.degraded_count == 1, "re-delivery must not bump degraded_count"
        assert loaded.worst_outcome_status == "degraded"
        assert len(loaded.tasks) == 1, "re-delivery must not insert a duplicate task row"
    finally:
        store.close()


def test_idempotency_key_is_deterministic_per_run_and_label():
    run = PapayyaRun(DurableRunConfig(agent="a", run_id="run-123", store=MemoryStore()))
    assert run.idempotency_key("draft") == "run-123:draft"
    # Stable across calls (so a re-executed step yields the same provider key).
    assert run.idempotency_key("draft") == run.idempotency_key("draft")
    assert run.idempotency_key("draft") != run.idempotency_key("polish")
