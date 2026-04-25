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

import pytest

from .conftest import write_test_agent


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
    assert in_memory_store.completed_run_count() == len(items), (
        f"expected {len(items)} completed runs, got "
        f"{in_memory_store.completed_run_count()}"
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

    # 4. Worker exits cleanly when stopped.
    worker.stop(timeout=5.0)
    assert worker.exit_code == 0, (
        f"worker exited with code {worker.exit_code}; expected clean shutdown"
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
