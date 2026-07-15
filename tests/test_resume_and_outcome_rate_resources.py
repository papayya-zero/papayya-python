"""SDK client wrappers for the Plan 33 resume + Plan 35 outcome-rate endpoints.

These endpoints existed on the control plane but had no SDK method — a
remediation agent had to fall back to raw HTTP. httpx.MockTransport stubs the
control plane so we assert the HTTP contract (method, path, query, body) the
new facade methods produce.
"""

from __future__ import annotations

from typing import Any, Callable

import httpx

from papayya.api import APIClient, APIConfig
from papayya.resources.agents import Agents
from papayya.resources.items import Items


def _clients(handler: Callable[[httpx.Request], httpx.Response]):
    transport = httpx.MockTransport(handler)
    config = APIConfig(api_key="cpk_test", base_url="http://mock")
    api = APIClient(config)
    api._http = httpx.Client(
        base_url=config.base_url,
        timeout=config.timeout,
        headers=api._http.headers,
        transport=transport,
    )
    return Items(api), Agents(api)


def test_item_resume_posts_to_resume_endpoint():
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        return httpx.Response(200, json={"run_id": "r1", "status": "running"})

    items, _ = _clients(handler)
    out = items.resume("r1")
    assert seen["method"] == "POST"
    assert seen["path"] == "/v1/durable/runs/r1/resume"
    assert out["status"] == "running"


def test_agent_resume_posts_to_agent_resume_endpoint():
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        return httpx.Response(200, json={"id": "a1", "config": {"auto_paused": False}})

    _, agents = _clients(handler)
    out = agents.resume("a1")
    assert seen["method"] == "POST"
    assert seen["path"] == "/v1/agents/a1/resume"
    assert out["config"]["auto_paused"] is False


def test_agent_outcome_rate_gets_with_agent_and_window():
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json={"agent": "support-reply", "total": 20, "ok_count": 12})

    _, agents = _clients(handler)
    out = agents.outcome_rate("support-reply", window=7)
    assert seen["method"] == "GET"
    assert seen["path"] == "/v1/durable/runs/outcome-rate"
    assert seen["query"] == {"agent": "support-reply", "window": "7"}
    assert out["total"] == 20


def test_item_clusters_narrows_by_partition():
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json={"clusters": [], "total_failures": 0})

    items, _ = _clients(handler)
    items.clusters(partition_key="acme")
    assert seen["path"] == "/v1/durable/runs/clusters"
    assert seen["query"] == {"partition_key": "acme"}
