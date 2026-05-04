"""Fixtures for worker-model integration tests.

Phase 1 prototype implementations. Each fixture's contract is documented
in its docstring; the build targets live in sibling _<name>.py modules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ._dispatcher import FakeDispatcher
from ._store import SharedSQLiteStore
from ._worker_proc import WorkerSubprocess


@pytest.fixture
def fake_dispatcher():
    """In-memory HTTP dispatcher; serves the worker subprocess over localhost."""
    d = FakeDispatcher()
    try:
        yield d
    finally:
        d.shutdown()


@pytest.fixture
def in_memory_store(tmp_path):
    """SQLiteStore at tmp_path, with assertion helpers (run_for_item, etc.).

    Backed by a real SQLite file because workers run in subprocesses; the
    file is the IPC channel between the worker's writes and the test's
    reads. Lifetime is the test (tmp_path is per-test).
    """
    db = tmp_path / "store.db"
    return SharedSQLiteStore(str(db))


@pytest.fixture
def worker_subprocess(tmp_path):
    """Factory that spawns `python -m papayya.runtime` as a real subprocess.

    Returns a WorkerSubprocess instance. The fixture stops the subprocess
    on test teardown if the test didn't call .stop() itself.
    """
    procs: list[WorkerSubprocess] = []

    def _spawn(
        *,
        agent_module: Path | None = None,
        dispatcher,
        store,
        api_key: str | None = None,
        bundle_url_base: str | None = None,
        env_overrides: dict[str, str] | None = None,
        bootstrap: bool = False,
    ) -> WorkerSubprocess:
        counter = tmp_path / "import_counter"
        proc = WorkerSubprocess(
            agent_module=agent_module,
            dispatcher_url=dispatcher.url,
            store_path=store.db_path,
            counter_path=counter,
            api_key=api_key,
            bundle_url_base=bundle_url_base,
            env_overrides=env_overrides,
            bootstrap=bootstrap,
        )
        procs.append(proc)
        return proc

    try:
        yield _spawn
    finally:
        for p in procs:
            p.stop(timeout=2)


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
