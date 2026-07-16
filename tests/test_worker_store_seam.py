"""Plan 37 Unit 1/2 — the worker's store seam.

A hosted bootstrap worker (dispatcher_url carrying the /v1/runtime prefix)
must point the customer's in-process papayya() client at the platform runtime
lane, NOT worker-local SQLite: it sets PAPAYYA_RUNTIME_STORE_BASE +
PAPAYYA_PLATFORM_WORKER_KEY and clears PAPAYYA_LOCAL_DB_PATH, so _auto_store()
selects the runtime CheckpointStore. The local prototype (no /runtime prefix)
keeps SQLite until Unit 4.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

from papayya.durable.cloud_store import CloudStore
from papayya.papayya import Papayya
from papayya.runtime.worker import Worker

_SEAM_KEYS = (
    "PAPAYYA_RUNTIME_STORE_BASE",
    "PAPAYYA_PLATFORM_WORKER_KEY",
    "PAPAYYA_LOCAL_DB_PATH",
    "PAPAYYA_API_KEY",
)


@contextmanager
def _isolated_env():
    saved = {k: os.environ.get(k) for k in _SEAM_KEYS}
    try:
        for k in _SEAM_KEYS:
            os.environ.pop(k, None)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _make_worker(dispatcher_url: str) -> Worker:
    w = Worker(
        dispatcher_url=dispatcher_url,
        store_path="/tmp/papayya-seam-test.db",
        agent_module_path=None,  # bootstrap mode: no import side effects
        api_key="plat-secret",
        heartbeat_interval_seconds=3600,
    )
    # The constructor starts a daemon heartbeat thread; stop it immediately.
    w.stop()
    w._hb_stop.set()
    return w


def test_hosted_worker_seam_selects_runtime_store():
    with _isolated_env():
        os.environ["PAPAYYA_API_KEY"] = "cpk_stray"  # must not survive
        _make_worker("http://control-pane-api:8090/v1/runtime")

        assert os.environ["PAPAYYA_RUNTIME_STORE_BASE"] == "http://control-pane-api:8090/v1"
        assert os.environ["PAPAYYA_PLATFORM_WORKER_KEY"] == "plat-secret"
        assert "PAPAYYA_LOCAL_DB_PATH" not in os.environ
        assert "PAPAYYA_API_KEY" not in os.environ  # popped so it can't shadow

        # The in-process client the @agent body would build now resolves to
        # the runtime lane.
        store = Papayya()._auto_store()
        assert isinstance(store, CloudStore)
        assert store._runs_base == "/v1/runtime/runs"


def test_local_worker_seam_keeps_sqlite():
    with _isolated_env():
        _make_worker("http://127.0.0.1:8765")  # LocalDispatcher, no /runtime

        assert os.environ["PAPAYYA_LOCAL_DB_PATH"] == "/tmp/papayya-seam-test.db"
        assert "PAPAYYA_RUNTIME_STORE_BASE" not in os.environ
