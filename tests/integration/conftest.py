"""Fixture skeletons for the worker-model integration tests.

These fixtures are deliberately unimplemented — they raise NotImplementedError
with explicit messages so a pytest run produces a useful navigation aid:
"the test that's failing wants this fixture, here's where to build it."

Each fixture's docstring is the contract Phase 1 implementation must honor.
Once Phase 1 lands, replace the NotImplementedError bodies with real impls.

See:
- tribe-agents/RUNTIME_VISION.md      (canonical direction)
- tribe-agents/adr/0001-worker-pool-design-decisions.md  (locked decisions)
"""

from __future__ import annotations

import pytest


@pytest.fixture
def fake_dispatcher():
    """In-memory dispatcher that mimics the control-pane → worker protocol.

    Contract for the eventual implementation:
      .enqueue(*, agent: str, item_id: str, payload: Any = None) -> str
          Add an item to the queue, return a run_id.
      .wait_until_drained(timeout: float) -> None
          Block until every enqueued item has reached terminal status.
          Raises TimeoutError if not drained within timeout.
      .leased_items() -> list[LeasedItem]
          Snapshot of items currently leased to a worker (for chaos tests).
      .release_lease(run_id: str) -> None
          Force-release a lease (simulates worker death).

    Lifecycle: created on test entry, torn down on exit. Workers connect
    to it over a unix socket or an in-process channel — Phase 1 picks one.
    """
    raise NotImplementedError(
        "fake_dispatcher fixture not yet implemented.\n"
        "Phase 1 work — see tribe-agents/adr/0001-worker-pool-design-decisions.md\n"
        "Build target: papayya-python/tests/integration/_dispatcher.py"
    )


@pytest.fixture
def in_memory_store():
    """Lineage store the worker writes through; tests assert against it.

    Wraps papayya.durable.MemoryStore (already in the SDK) plus a few
    helpers the assertions need:

    Contract:
      .completed_run_count() -> int
      .run_for_item(item_id: str) -> RunCheckpoint | None
          The most recent completed run carrying this item_id.
      .all_runs() -> list[RunCheckpoint]

    Workers connect to this store via a CheckpointStore instance handed
    in at boot — same way SDK code already uses it. The fixture exposes
    the *handle* for assertions plus the *connection-string-or-config*
    workers receive.
    """
    raise NotImplementedError(
        "in_memory_store fixture not yet implemented.\n"
        "Should wrap papayya.durable.MemoryStore + add the assertion helpers above.\n"
        "Build target: papayya-python/tests/integration/_store.py"
    )


@pytest.fixture
def worker_subprocess(tmp_path):
    """Factory: spawns a real subprocess running the worker module.

    The subprocess shape is non-negotiable. Running the worker in-process
    makes 'imports module once on boot' unverifiable — the test process
    has already imported everything. The whole point of this test layer
    is to exercise the realistic shape.

    Contract:
      worker = worker_subprocess(
          agent_module: Path,        # path to a .py file with @agent decoration
          dispatcher: FakeDispatcher,  # how the worker finds work
          store: CheckpointStore,    # where the worker writes lineage
      )

      worker.module_import_count -> int
          Counted by an env-var hook in the agent module (the module
          increments a counter file on import). Tests assert == 1.

      worker.stop(timeout: float = 5) -> None
          Send SIGTERM, wait up to timeout for clean exit, SIGKILL if not.

      worker.exit_code -> int | None
          None if still running; integer once stopped.

    Workers run as `python -m papayya.runtime --agent-module <path>` once
    Phase 1 lands. Pre-Phase-1 the subprocess command itself doesn't
    exist, which is why this test fails red until the worker module is built.
    """
    raise NotImplementedError(
        "worker_subprocess fixture not yet implemented.\n"
        "Spawns `python -m papayya.runtime` (which is also Phase 1 work).\n"
        "Build target: papayya-python/tests/integration/_worker_proc.py\n"
        "Critical: must be a real subprocess, NOT in-process — see fixture docstring."
    )


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #

def write_test_agent(tmp_path, source: str, *, name: str = "agent_module.py"):
    """Write an agent module source to tmp_path and return the path.

    Used by acceptance tests to materialize a customer-shaped agent.py
    file the worker subprocess can import.
    """
    path = tmp_path / name
    path.write_text(source)
    return path
