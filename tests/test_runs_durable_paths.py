"""v1→v2 cutover: the run read/poll surface targets /v1/durable/runs/*.

After the cutover a triggered run is a durable_run, and the v1 runs/steps
tables are unfed. These tests pin the HTTP contract so a regression that
points polling back at /v1/runs/* (which would silently 404) is caught.

Uses httpx.MockTransport to stub the control-pane — no backend needed.
"""

from __future__ import annotations

from typing import Callable

import httpx

from papayya.api import APIClient, APIConfig
from papayya.resources.runs import Runs


def _make(handler: Callable[[httpx.Request], httpx.Response]) -> tuple[Runs, APIClient]:
    transport = httpx.MockTransport(handler)
    config = APIConfig(api_key="cpk_test_key", base_url="http://mock")
    api = APIClient(config)
    api._http = httpx.Client(
        base_url=config.base_url,
        timeout=config.timeout,
        headers=api._http.headers,
        transport=transport,
    )
    return Runs(api), api


def _record(paths: list[str], json_body: object) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json=json_body)

    return handler


def test_runs_resource_read_paths_are_durable() -> None:
    paths: list[str] = []
    runs, _ = _make(_record(paths, {"run_id": "r-1", "status": "queued"}))

    runs.get("r-1")
    runs.list()
    runs.steps("r-1")

    assert paths == [
        "/v1/durable/runs/r-1",
        "/v1/durable/runs",
        "/v1/durable/runs/r-1/checkpoints",
    ]


def test_api_client_poll_paths_are_durable() -> None:
    paths: list[str] = []
    _, api = _make(_record(paths, {"run_id": "r-2", "status": "running", "checkpoints": []}))

    api.get_run("r-2")
    api.get_steps("r-2")

    assert paths == [
        "/v1/durable/runs/r-2",
        "/v1/durable/runs/r-2/checkpoints",
    ]


def test_no_v1_runs_path_on_read() -> None:
    # Guard: none of the read methods should touch the retired v1 surface.
    paths: list[str] = []
    runs, api = _make(_record(paths, {"run_id": "r-3", "status": "completed", "checkpoints": []}))

    runs.get("r-3")
    runs.list()
    runs.steps("r-3")
    api.get_run("r-3")
    api.get_steps("r-3")

    assert all(p.startswith("/v1/durable/runs") for p in paths), paths
