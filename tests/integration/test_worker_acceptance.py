"""Worker-model acceptance test — the foundation test for Phase 1.

This test specifies the customer-facing contract of the new worker model:

  • a worker is a long-lived subprocess (not one container per item)
  • it imports the agent module exactly once on boot
  • it pulls items from a dispatcher and runs the @agent function
  • lineage rows are written through the existing CheckpointStore protocol
  • run.step("name", fn, kind="llm") is preserved end-to-end
  • run.complete(value) finalizes the run

The test fails until:
  1. papayya.runtime worker module exists
  2. fake_dispatcher fixture is implemented
  3. in_memory_store fixture is implemented
  4. worker_subprocess fixture is implemented

This is the acceptance gate for Phase 1. When this test goes green the
prototype is done; the next gate is Kingsley running examples/enrich_companies.py
against the local worker and confirming the iteration loop feels right.

Reference:
  RUNTIME_VISION.md                            (canonical direction)
  adr/0001-worker-pool-design-decisions.md     (runtime semantics)
  next_session_2026_04_25_continue.md          (launch tail context)
"""

from __future__ import annotations

import time as _t

import pytest

from .conftest import write_test_agent
from ._store import SharedSQLiteStore


# --------------------------------------------------------------------------- #
#  Test-scoped customer-shaped agent.                                          #
#                                                                              #
#  Mirrors what examples/enrich_companies.py looks like in real customer code, #
#  minus the OpenAI dependency. Uses run.step(..., kind="llm") so we verify    #
#  the kind hint propagates through the worker → store path.                   #
# --------------------------------------------------------------------------- #

_AGENT_SOURCE = '''\
"""Acceptance-test agent: two-step enrich, no real LLM.

This file gets written to tmp_path and imported by the worker subprocess.
The worker process is responsible for invoking the @agent function once
per dispatched item, with the item_id passed as the first arg.
"""

import os
from pathlib import Path

from papayya import agent
from papayya.durable import papayya


# Module-import counter — the worker subprocess exports an env var pointing
# at a file we increment on every import. The test asserts this stays at 1
# across all dispatched items.
_counter_path = os.environ.get("PAPAYYA_TEST_IMPORT_COUNTER")
if _counter_path:
    p = Path(_counter_path)
    n = int(p.read_text() or "0") if p.exists() else 0
    p.write_text(str(n + 1))


@agent(name="enrich")
def enrich(item_id: str) -> dict:
    run = papayya().run("enrich", item_id=item_id)

    fetch = run.step("fetch", lambda: f"snippet-for-{item_id}")
    extract = run.step(
        "extract",
        lambda data: {"id": item_id, "data": data, "score": 0.42},
        kind="llm",
    )

    snippet = fetch()
    extracted = extract(snippet)
    run.complete(extracted)
    return extracted
'''


# --------------------------------------------------------------------------- #
#  Acceptance test                                                             #
# --------------------------------------------------------------------------- #

def test_worker_processes_batch_with_correct_lineage(
    tmp_path,
    fake_dispatcher,
    in_memory_store,
    worker_subprocess,
):
    """Foundation test: 10 items go through one worker subprocess.

    Once green, this proves:
      - workers are subprocesses, not in-process loops (import counter)
      - workers serve many items without re-importing (counter stays at 1)
      - the SDK contract is unchanged (customer code is identical to today)
      - lineage flows through the existing store protocol
      - kind="llm" hints survive the worker round-trip
    """
    agent_path = write_test_agent(tmp_path, _AGENT_SOURCE)

    items = [f"co_{i:02d}" for i in range(10)]
    for item_id in items:
        fake_dispatcher.enqueue(agent="enrich", item_id=item_id)

    worker = worker_subprocess(
        agent_module=agent_path,
        dispatcher=fake_dispatcher,
        store=in_memory_store,
    )

    fake_dispatcher.wait_until_drained(timeout=10.0)

    # 1. Module imported exactly once across all 10 items.
    assert worker.module_import_count == 1, (
        f"agent module imported {worker.module_import_count} times; "
        "the worker should import the customer module on boot only. "
        "If this is >1 the warm-worker promise is broken."
    )

    # 2. Every item produced one completed run.
    #
    # Poll for up to 2s after dispatcher drain. The worker's
    # ``_report_complete`` posts /complete after ``run.complete()``'s
    # SQLite commit returns, but cross-process WAL visibility can lag
    # the in-process commit by a few milliseconds when many small
    # commits pipeline. ``wait_until_drained`` already gave the test a
    # reason to expect 10 rows; the poll just absorbs the WAL-replication
    # gap without masking a real bug — a genuine miss would still fail
    # at the deadline.
    deadline = _t.monotonic() + 2.0
    last_count = in_memory_store.completed_run_count()
    while last_count < len(items) and _t.monotonic() < deadline:
        _t.sleep(0.02)
        last_count = in_memory_store.completed_run_count()
    assert last_count == len(items), (
        f"expected {len(items)} completed runs after 2s poll, got {last_count}"
    )

    # 3. Per-item lineage shape is identical to what a single-process
    #    `papayya dev` would produce. The worker is a deployment detail;
    #    the customer-facing data shape must not change.
    for item_id in items:
        run = in_memory_store.run_for_item(item_id)
        assert run is not None, f"no run found for item_id={item_id}"

        labels = [t.label for t in run.tasks]
        assert labels == ["fetch", "extract"], (
            f"item {item_id}: expected ['fetch', 'extract'], got {labels}"
        )

        kinds = [t.kind for t in run.tasks]
        assert kinds == [None, "llm"], (
            f"item {item_id}: kind='llm' did not propagate through the worker. "
            f"Got {kinds}; cost metering and dashboard rely on this hint."
        )

        last = run.tasks[-1]
        assert last.result == {"id": item_id, "data": f"snippet-for-{item_id}", "score": 0.42}

        # input_snapshot must be populated from the @agent call args so
        # `runs.replay()` and dlq replay can re-issue this run. Captured
        # by the @agent wrapper via inspect.signature.bind().
        assert run.input_snapshot == {"item_id": item_id}, (
            f"item {item_id}: expected input_snapshot={{'item_id': {item_id!r}}}, "
            f"got {run.input_snapshot!r}. Replay paths read this column; "
            "if it's None, replay surfaces silently break."
        )

    # 4. Worker exits cleanly when stopped.
    worker.stop(timeout=5.0)
    assert worker.exit_code == 0, (
        f"worker exited with code {worker.exit_code}; expected clean shutdown"
    )


# --------------------------------------------------------------------------- #
#  Async lineage equivalence — Phase 2 of the async-support unit.              #
#                                                                              #
#  An async @agent registered against the same fixtures must produce the same  #
#  lineage shape as the sync agent above: same labels, same kind="llm"         #
#  propagation, same input_snapshot, same last-step result. If this test       #
#  fails but the sync acceptance test passes, the bug is in either the         #
#  @agent async wrapper (input snapshot didn't survive `await`) or the         #
#  worker's async dispatch (event loop didn't drive the coroutine).            #
# --------------------------------------------------------------------------- #

_ASYNC_AGENT_SOURCE = '''\
"""Async acceptance-test agent: same lineage as the sync version.

Imported by the worker subprocess once on boot. ``run.step`` accepts a
coroutine and returns an async wrapper (Phase 1); the @agent decorator
detects the coroutine and produces an async wrapper (Phase 2); the
worker drives it through asyncio.run (Phase 2).
"""

import asyncio
import os
from pathlib import Path

from papayya import agent
from papayya.durable import papayya


_counter_path = os.environ.get("PAPAYYA_TEST_IMPORT_COUNTER")
if _counter_path:
    p = Path(_counter_path)
    n = int(p.read_text() or "0") if p.exists() else 0
    p.write_text(str(n + 1))


@agent(name="enrich")
async def enrich(item_id: str) -> dict:
    run = papayya().run("enrich", item_id=item_id)

    async def fetch():
        await asyncio.sleep(0)
        return f"snippet-for-{item_id}"

    async def extract(data):
        await asyncio.sleep(0)
        return {"id": item_id, "data": data, "score": 0.42}

    fetch_step = run.step("fetch", fetch)
    extract_step = run.step("extract", extract, kind="llm")

    snippet = await fetch_step()
    extracted = await extract_step(snippet)
    run.complete(extracted)
    return extracted
'''


def test_worker_processes_batch_with_async_agent_lineage(
    tmp_path,
    fake_dispatcher,
    in_memory_store,
    worker_subprocess,
):
    """An async ``@agent`` produces lineage identical to the sync agent.

    Same 10-item batch, same fixtures, same assertions as the sync
    acceptance test — the difference is the agent module is async all
    the way down (`async def @agent`, awaited `run.step` calls).
    """
    agent_path = write_test_agent(tmp_path, _ASYNC_AGENT_SOURCE)

    items = [f"co_{i:02d}" for i in range(10)]
    for item_id in items:
        fake_dispatcher.enqueue(agent="enrich", item_id=item_id)

    worker = worker_subprocess(
        agent_module=agent_path,
        dispatcher=fake_dispatcher,
        store=in_memory_store,
    )

    fake_dispatcher.wait_until_drained(timeout=10.0)

    assert worker.module_import_count == 1, (
        f"agent module imported {worker.module_import_count} times; "
        "the warm-worker promise is broken on the async path"
    )

    deadline = _t.monotonic() + 2.0
    last_count = in_memory_store.completed_run_count()
    while last_count < len(items) and _t.monotonic() < deadline:
        _t.sleep(0.02)
        last_count = in_memory_store.completed_run_count()
    assert last_count == len(items), (
        f"expected {len(items)} completed runs after 2s poll, got {last_count}"
    )

    for item_id in items:
        run = in_memory_store.run_for_item(item_id)
        assert run is not None, f"no run found for item_id={item_id}"

        labels = [t.label for t in run.tasks]
        assert labels == ["fetch", "extract"], (
            f"item {item_id}: expected ['fetch', 'extract'], got {labels}"
        )

        kinds = [t.kind for t in run.tasks]
        assert kinds == [None, "llm"], (
            f"item {item_id}: kind='llm' did not propagate through the "
            f"async worker path. Got {kinds}."
        )

        last = run.tasks[-1]
        assert last.result == {"id": item_id, "data": f"snippet-for-{item_id}", "score": 0.42}

        assert run.input_snapshot == {"item_id": item_id}, (
            f"item {item_id}: expected input_snapshot={{'item_id': {item_id!r}}}, "
            f"got {run.input_snapshot!r}. The @agent async wrapper must "
            "set _AGENT_INPUT before await and reset in finally."
        )

    worker.stop(timeout=5.0)
    assert worker.exit_code == 0, (
        f"worker exited with code {worker.exit_code}; expected clean shutdown"
    )


# --------------------------------------------------------------------------- #
#  Crash durability — committed runs survive a hard worker kill.               #
#                                                                              #
#  Regression guard for the local-store WAL frame-loss bug (fixed by           #
#  journal_mode=DELETE + busy_timeout in sqlite_store.py). Under WAL the        #
#  worker minted one short-lived store connection per item while the           #
#  long-lived dev/test reader blocked checkpointing, so committed frames       #
#  stranded in the -wal file and never reached main.db — runs the SDK had      #
#  already reported complete silently vanished on shutdown. That is the        #
#  product's own wedge failure mode (silent, non-raising data loss) firing     #
#  inside the durable substrate.                                               #
#                                                                              #
#  The invariant under test, stated independently of journal mode:             #
#    a run whose /complete the worker reported MUST be durable — readable       #
#    through a freshly opened store after the worker is SIGKILLed with no       #
#    chance to flush, checkpoint, or close cleanly.                            #
#                                                                              #
#  The acceptance test caught the old bug only by luck (slice-10's             #
#  import-graph timing shifted GC enough to expose it). This asserts the       #
#  guarantee directly, so any regression fails loudly instead of latently.     #
# --------------------------------------------------------------------------- #

def test_committed_runs_survive_worker_sigkill(
    tmp_path,
    fake_dispatcher,
    in_memory_store,
    worker_subprocess,
):
    """Every run the worker reported complete is on disk after a hard kill.

    Enqueue more items than the worker drains in an instant, let it commit
    a few, then SIGKILL it mid-batch. Reopen the store from scratch and
    assert no reported completion was lost. Under the old WAL code these
    committed-but-stranded frames disappeared; under journal_mode=DELETE
    every commit reaches main.db before /complete is posted.
    """
    agent_path = write_test_agent(tmp_path, _AGENT_SOURCE)

    # Larger than the worker can finish before we kill it, so the kill lands
    # mid-batch: some runs committed, some never started.
    items = [f"co_{i:02d}" for i in range(50)]
    item_for_lease = {
        fake_dispatcher.enqueue(agent="enrich", item_id=item_id): item_id
        for item_id in items
    }

    worker = worker_subprocess(
        agent_module=agent_path,
        dispatcher=fake_dispatcher,
        store=in_memory_store,
    )

    # Wait until the worker has reported a few completions, then crash it.
    # A reported /complete means run.complete()'s commit already returned —
    # these are commits the SDK promised were durable.
    deadline = _t.monotonic() + 10.0
    while fake_dispatcher.completed_count() < 3 and _t.monotonic() < deadline:
        _t.sleep(0.01)

    worker.kill()
    exit_code = worker.wait(timeout=5.0)
    assert exit_code is not None, "worker did not die after SIGKILL"

    # The process is dead, so this set can no longer change. Each lease here
    # is a run the SDK acknowledged as complete.
    reported = fake_dispatcher.completed_leases()
    assert reported, (
        "worker reported zero completions before the kill, so the durability "
        "path was never exercised — raise the wait deadline and retry"
    )

    # Reopen the store from a clean connection: this proves the rows are on
    # disk, not lingering in a connection that happened to be alive during
    # the run. (run_for_item opens a fresh connection per lookup.)
    reopened = SharedSQLiteStore(in_memory_store.db_path)
    lost = sorted(
        item_for_lease[lease_id]
        for lease_id in reported
        if reopened.run_for_item(item_for_lease[lease_id]) is None
    )
    assert not lost, (
        f"{len(lost)}/{len(reported)} runs the worker reported complete were "
        f"missing after SIGKILL: {lost}. A committed run the SDK acknowledged "
        "must survive a crash — silent loss here is the exact failure mode "
        "Papayya exists to detect.\n"
        f"worker log tail:\n{worker.stderr_tail()}"
    )


# --------------------------------------------------------------------------- #
#  Sentinel: this test must never be skipped or xfailed without a flag.        #
#                                                                              #
#  The acceptance test is the gate for Phase 1. If someone marks it skipped    #
#  to ship faster, this sentinel screams. Remove it only when Phase 1 is       #
#  green and you are intentionally retiring the gate.                          #
# --------------------------------------------------------------------------- #

def test_acceptance_gate_sentinel():
    """Reminds future-us not to silently skip the gate above.

    If you are reading this because pytest told you so: don't skip the
    acceptance test. Either build the fixtures (Phase 1) or, if Phase 1
    is already green, delete this sentinel as part of the green commit.
    """
    import inspect
    src = inspect.getsource(test_worker_processes_batch_with_correct_lineage)
    assert "@pytest.mark.skip" not in src
    assert "@pytest.mark.xfail" not in src
