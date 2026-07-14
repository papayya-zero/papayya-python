"""Plan 33 — auto-pause on degradation (SDK side, Unit 3).

Covers the local fences in SQLiteStore and the WorkloadPaused step-boundary
signal: the run-level consecutive-K fence, the workload-level rate fence
enforced at the next item boundary in papayya.map, and the ambient-lifecycle
exemption that keeps a paused run 'paused' rather than flipping it to 'failed'.

The synchronous SQLite path lets us assert the strict N+1 boundary (the pause
raises before the very next step and never loses the completed step's
checkpoint); the fire-and-forget cloud store deliberately can't promise that.
"""

from __future__ import annotations

import pytest

import papayya
from papayya import WorkloadPaused, CreditExhausted
from papayya.durable import _schema
from papayya.durable.run import Item
from papayya.durable.sqlite_store import SQLiteStore
from papayya.durable.types import DurableRunConfig, RunCheckpoint, TaskEntry
from papayya.iterators import drive_ambient_sync
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task(label: str, *, status: str, reason: str | None = None) -> TaskEntry:
    return TaskEntry(
        label=label, result={"v": label}, duration_ms=1, completed_at=_now(),
        outcome_status=status, outcome_reason=reason,
    )


# --- 1. store-level run-level fence ------------------------------------- #

def test_store_run_level_fence_sets_pending_pause(tmp_path):
    store = SQLiteStore(str(tmp_path / "rl.db"))
    try:
        store.create(RunCheckpoint(run_id="r1", agent="a", tasks=[], status="running",
                                   created_at=_now(), updated_at=_now()))
        store.set_run_fence("r1", 3)
        store.save_task("r1", _task("s1", status="degraded", reason="empty_none"))
        store.save_task("r1", _task("s2", status="degraded", reason="empty_none"))
        assert store.pending_pause("r1") is None  # streak of 2 < K=3
        store.save_task("r1", _task("s3", status="degraded", reason="empty_none"))
        assert store.pending_pause("r1") == "3 consecutive degraded steps: empty_none"
    finally:
        store.close()


def test_store_run_level_fence_resets_on_ok(tmp_path):
    store = SQLiteStore(str(tmp_path / "rl2.db"))
    try:
        store.create(RunCheckpoint(run_id="r1", agent="a", tasks=[], status="running",
                                   created_at=_now(), updated_at=_now()))
        store.set_run_fence("r1", 2)
        store.save_task("r1", _task("s1", status="degraded", reason="x"))
        store.save_task("r1", _task("s2", status="ok"))  # resets the streak
        store.save_task("r1", _task("s3", status="degraded", reason="x"))
        assert store.pending_pause("r1") is None  # only 1 trailing degraded
        store.save_task("r1", _task("s4", status="degraded", reason="x"))
        assert store.pending_pause("r1") is not None  # now 2 trailing → trip
    finally:
        store.close()


def test_store_run_level_fence_disabled_when_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPAYYA_PAUSE_AFTER_DEGRADED", "0")
    store = SQLiteStore(str(tmp_path / "rl3.db"))
    try:
        store.create(RunCheckpoint(run_id="r1", agent="a", tasks=[], status="running",
                                   created_at=_now(), updated_at=_now()))
        for i in range(5):
            store.save_task("r1", _task(f"s{i}", status="degraded", reason="x"))
        assert store.pending_pause("r1") is None  # 0 disables the fence
    finally:
        store.close()


# --- 2. run/step path: pause raises before the next step (strict N+1) ---- #

def test_run_step_pauses_before_next_step(tmp_path, monkeypatch):
    monkeypatch.delenv("PAPAYYA_PAUSE_AFTER_DEGRADED", raising=False)
    store = SQLiteStore(str(tmp_path / "step.db"))
    try:
        run = Item(DurableRunConfig(agent="a", store=store, pause_after_degraded=2))
        step = run.step("think", lambda: None)  # None → degraded 'empty_none'

        step()  # degraded #1
        step()  # degraded #2 → fence trips, pending_pause set

        with pytest.raises(WorkloadPaused) as ei:
            step()  # 3rd call: _pre_call raises BEFORE the body runs
        assert "2 consecutive degraded steps" in ei.value.reason
        assert ei.value.run_id == run.run_id

        # Exactly the two completed steps are checkpointed — the paused (N+1)
        # step never saved, and neither completed step was lost.
        loaded = store.load(run.run_id)
        assert loaded is not None
        assert len(loaded.tasks) == 2
    finally:
        store.close()


def test_resume_replay_picks_up_after_pause(tmp_path, monkeypatch):
    monkeypatch.delenv("PAPAYYA_PAUSE_AFTER_DEGRADED", raising=False)
    store = SQLiteStore(str(tmp_path / "resume.db"))
    try:
        calls: list[str] = []

        def body(x):
            calls.append(x)
            return None  # degraded

        run = Item(DurableRunConfig(agent="a", store=store, pause_after_degraded=2))
        step = run.step("think", body)
        step("a")
        step("b")  # 2 degraded → fence trips
        with pytest.raises(WorkloadPaused):
            step("c")  # raises before the body runs
        assert calls == ["a", "b"]

        # Operator resume clears the run-level pause; replay of the same run
        # skips the two saved steps (cache hits, body not re-run) and picks up
        # exactly at the step where the pause landed.
        store.clear_pending_pause(run.run_id)
        run2 = Item(DurableRunConfig(agent="a", store=store, run_id=run.run_id,
                                     pause_after_degraded=2))
        step2 = run2.step("think", body)
        step2("a")  # cache hit
        step2("b")  # cache hit
        assert calls == ["a", "b"]  # neither cached step re-executed
        step2("c")  # the paused step now executes
        assert calls == ["a", "b", "c"]
    finally:
        store.close()


# --- 3. map: workload fence pauses at an item boundary ------------------- #

def test_map_pauses_at_item_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPAYYA_WORKLOAD_PAUSE_MIN_DEGRADED", "3")
    monkeypatch.setenv("PAPAYYA_WORKLOAD_PAUSE_PCT", "50")
    monkeypatch.delenv("PAPAYYA_PAUSE_AFTER_DEGRADED", raising=False)
    store = SQLiteStore(str(tmp_path / "wl.db"))
    try:
        seen: list[int] = []

        def process(item: int) -> int:
            seen.append(item)
            papayya.mark_degraded("bad_output")  # every item's run completes degraded
            return item

        with pytest.raises(WorkloadPaused):
            papayya.map(process, list(range(10)), agent="wl",
                        item_id=str, partition_key=lambda x: "t", store=store)

        # Trips after the 3rd degraded item completes; the 4th item's run is
        # never opened — completed items intact, remaining items unstarted.
        assert seen == [0, 1, 2]
        assert store.workload_paused("wl") is not None

        # Local resume clears the flag so a re-drive can proceed.
        store.resume_workload("wl")
        assert store.workload_paused("wl") is None
    finally:
        store.close()


# --- 4. ambient lifecycle exemption: paused stays paused, not failed ----- #

def _step_reasons_and_item_statuses(db_path):
    store = SQLiteStore(str(db_path))
    try:
        reasons = [
            r["r"] for r in store._conn.execute(
                f"SELECT {_schema.COL_STEP_OUTCOME_REASON} AS r FROM {_schema.TBL_STEPS}"
            ).fetchall()
        ]
        statuses = [
            r["status"] for r in store._conn.execute(
                f"SELECT status FROM {_schema.TBL_ITEMS}"
            ).fetchall()
        ]
    finally:
        store.close()
    return reasons, statuses


def test_ambient_workload_paused_not_flipped_to_failed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # auto-resolved store lands under tmp_path/.papayya

    def body():
        papayya.mark_degraded("some_reason")  # mints the isolate run
        raise WorkloadPaused("budget", "rid")

    with pytest.raises(WorkloadPaused):
        drive_ambient_sync("amb", "item-1", None, body, own_completion=True)

    reasons, statuses = _step_reasons_and_item_statuses(tmp_path / ".papayya" / "local.db")
    assert "agent_body_exception" not in reasons  # no synthetic failed entry
    assert "failed" not in statuses               # run not flipped to failed


def test_ambient_credit_exhausted_also_exempt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def body():
        papayya.mark_degraded("some_reason")
        raise CreditExhausted("out of credits")

    with pytest.raises(CreditExhausted):
        drive_ambient_sync("amb", "item-1", None, body, own_completion=True)

    reasons, statuses = _step_reasons_and_item_statuses(tmp_path / ".papayya" / "local.db")
    assert "agent_body_exception" not in reasons
    assert "failed" not in statuses


def test_ambient_normal_exception_still_flips_to_failed(tmp_path, monkeypatch):
    # Control: the exemption is targeted — an ordinary body exception STILL
    # writes the synthetic entry and fails the run.
    monkeypatch.chdir(tmp_path)

    def body():
        papayya.mark_degraded("some_reason")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        drive_ambient_sync("amb", "item-1", None, body, own_completion=True)

    reasons, statuses = _step_reasons_and_item_statuses(tmp_path / ".papayya" / "local.db")
    assert "agent_body_exception" in reasons
    assert "failed" in statuses
