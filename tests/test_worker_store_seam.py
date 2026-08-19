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
    # Plan 56 F2: the constructor now starts a heartbeat PROCESS (a
    # thread cannot heartbeat through a step that holds the GIL). These
    # tests never run the poll loop, whose finally block would stop it,
    # so they have to tear it down themselves or leak one child per test.
    w._stop_heartbeat()
    return w


def test_hosted_worker_seam_selects_runtime_store():
    with _isolated_env():
        os.environ["PAPAYYA_API_KEY"] = "cpk_stray"  # must not survive
        _make_worker("http://control-pane-api:8090/v1/runtime")

        # The ROOT, not …/v1 — make_runtime_store() re-adds the /v1/runtime
        # path, so leaving a stray /v1 here doubles it into /v1/v1 (Plan 37).
        assert os.environ["PAPAYYA_RUNTIME_STORE_BASE"] == "http://control-pane-api:8090"
        # PLAN 60 S1 — THIS ASSERTION IS INVERTED FROM WHAT IT USED TO BE, and
        # the old form was asserting the defect. The worker used to export the
        # platform worker key into the environment of the process that then
        # imports and calls the customer's @agent: one os.environ.get in any
        # customer module, and they held the credential that leases across
        # every tenant and downloads any account's deployment bundle. The base
        # URL alone now selects the lane; the credential is the per-run token
        # on the lease, carried in a contextvar.
        assert "PAPAYYA_PLATFORM_WORKER_KEY" not in os.environ
        assert "PAPAYYA_LOCAL_DB_PATH" not in os.environ
        assert "PAPAYYA_API_KEY" not in os.environ  # popped so it can't shadow

        # The in-process client the @agent body would build now resolves to
        # the runtime lane.
        store = Papayya()._auto_store()
        assert isinstance(store, CloudStore)
        assert store._runs_base == "/v1/runtime/runs"

        # Cross the seam: compose the base_url the worker actually wired up
        # with the store's route and assert the request hits the REAL
        # control-plane path — no doubled /v1. This is the assertion the
        # original tests lacked: they checked the env var and _runs_base in
        # isolation, so the /v1/v1 404 slipped through green (Plan 37 Unit 1).
        req = store._client.build_request(
            "POST", f"{store._runs_base}/run-1/checkpoints"
        )
        assert (
            str(req.url)
            == "http://control-pane-api:8090/v1/runtime/runs/run-1/checkpoints"
        )
        # And the client carries NO credential of its own (plan 60 S1). A key
        # baked in at construction would be process-wide and item-independent,
        # which is exactly the property that made the old one dangerous.
        assert "X-Api-Key" not in store._client.headers


def test_local_worker_seam_keeps_sqlite():
    with _isolated_env():
        _make_worker("http://127.0.0.1:8765")  # LocalDispatcher, no /runtime

        assert os.environ["PAPAYYA_LOCAL_DB_PATH"] == "/tmp/papayya-seam-test.db"
        assert "PAPAYYA_RUNTIME_STORE_BASE" not in os.environ
