"""Plan 37 Unit 1 — the platform-authed runtime CheckpointStore.

The hosted worker pool runs customer @agent code in-process and must write
its checkpoints + run status through the platform lane
(``/v1/runtime/runs/...`` with the shared platform worker key), NOT the
tenant-scoped ``/v1/durable`` API (which the worker can't reach — it holds
no ``cpk_`` project key). These tests pin the route prefix, the auth header,
and the ``_auto_store`` selection order that wires it up.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from papayya.durable.cloud_store import make_runtime_store
from papayya.durable.types import RunCheckpoint, TaskEntry


def _capture(store, handler) -> None:
    """Route the store's client through a MockTransport WITHOUT rebuilding it,
    so the base_url + auth headers the store actually configured are exercised.
    """
    store._client._transport = httpx.MockTransport(handler)


def test_save_task_hits_runtime_lane_with_platform_auth() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["x_api_key"] = request.headers.get("x-api-key")
        captured["authorization"] = request.headers.get("authorization")
        return httpx.Response(201, json={})

    store = make_runtime_store("http://mock", "platform-secret")
    _capture(store, handler)
    store.save_task(
        "run-1",
        TaskEntry(label="classify", result={"ok": True}, duration_ms=5, completed_at=""),
    )

    assert captured["path"] == "/v1/runtime/runs/run-1/checkpoints"
    # Platform key rides X-Api-Key (auth.go matches it there), NOT Bearer,
    # even though it isn't a cpk_ key.
    assert captured["x_api_key"] == "platform-secret"
    assert captured["authorization"] is None


def test_set_status_and_load_hit_runtime_lane() -> None:
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.method == "GET":
            return httpx.Response(404)
        return httpx.Response(200, json={})

    store = make_runtime_store("http://mock", "platform-secret")
    _capture(store, handler)
    store.set_status("run-1", "completed", {"result": 1})
    assert store.load("run-1") is None

    assert ("PATCH", "/v1/runtime/runs/run-1") in seen
    assert ("GET", "/v1/runtime/runs/run-1") in seen


def test_auto_store_selects_runtime_store_over_cpk(monkeypatch: pytest.MonkeyPatch) -> None:
    from papayya.durable.cloud_store import CloudStore
    from papayya.papayya import Papayya

    monkeypatch.setenv("PAPAYYA_RUNTIME_STORE_BASE", "http://control-pane-api:8090")
    monkeypatch.setenv("PAPAYYA_PLATFORM_WORKER_KEY", "platform-secret")

    # A stray cpk_ key must NOT win: the worker path takes precedence so hosted
    # runs land on the platform lane and never on a mis-scoped tenant API.
    store = Papayya(api_key="cpk_stray")._auto_store()
    assert isinstance(store, CloudStore)
    assert store._runs_base == "/v1/runtime/runs"


def test_auto_store_falls_back_when_runtime_env_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    from papayya.durable.cloud_store import CloudStore
    from papayya.papayya import Papayya

    monkeypatch.delenv("PAPAYYA_RUNTIME_STORE_BASE", raising=False)
    monkeypatch.delenv("PAPAYYA_PLATFORM_WORKER_KEY", raising=False)

    store = Papayya(api_key="cpk_real")._auto_store()
    assert isinstance(store, CloudStore)
    # Tenant-scoped durable API, not the runtime lane.
    assert store._runs_base == "/v1/durable/runs"
