"""Ergonomics fixes for the hosted off-cycle path (submit → poll/stream).

Covers three developer-facing guarantees added alongside Plan 37:
  1. ``resolve_config`` falls back to the CLI config file (auth parity with
     the durable path) — a ``papayya login`` is enough, no env var required.
  2. ``APIClient._request`` retries ONLY provably-safe transient failures
     (connect errors + 429/502/503/504) and raises 4xx/500 immediately, so a
     non-idempotent submit is never blindly replayed.
  3. ``Items.create`` guarantees a stable ``run_id`` key regardless of which
     field the control-plane spells the id under.

All stubbed with httpx.MockTransport — no running backend needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import httpx
import pytest

from papayya.api import APIClient, APIConfig, PapayyaAPIError, resolve_config
from papayya.resources.items import Items, _ensure_run_id


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> APIClient:
    api = APIClient(APIConfig(api_key="cpk_test", base_url="http://mock"))
    api._http = httpx.Client(
        base_url="http://mock",
        timeout=api._http.timeout,
        headers=api._http.headers,
        transport=httpx.MockTransport(handler),
    )
    return api


# -- Fix 1: CLI-config auth fallback ---------------------------------------- #


def test_resolve_config_reads_cli_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A developer who ran `papayya login` (no env var) resolves cleanly."""
    monkeypatch.delenv("PAPAYYA_API_KEY", raising=False)
    monkeypatch.delenv("PAPAYYA_BASE_URL", raising=False)
    cfg_file = tmp_path / ".papayya" / "config.json"
    cfg_file.parent.mkdir(parents=True)
    cfg_file.write_text(
        json.dumps(
            {
                "version": 2,
                "current_env": "prod",
                "envs": {
                    "prod": {"api_key": "cpk_from_cli", "base_url": "http://cli"}
                },
            }
        )
    )
    monkeypatch.setattr("papayya._config.CONFIG_FILE", cfg_file)

    config = resolve_config()
    assert config.api_key == "cpk_from_cli"
    # base_url is paired from the SAME env block as the key.
    assert config.base_url == "http://cli"


def test_resolve_config_raises_without_any_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PAPAYYA_API_KEY", raising=False)
    monkeypatch.setattr(
        "papayya._config.CONFIG_FILE", tmp_path / "nope" / "config.json"
    )
    with pytest.raises(PapayyaAPIError, match="No API key"):
        resolve_config()


def test_resolve_config_error_does_not_mention_nonexistent_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SDK has no --api-key flag; the message must not suggest one."""
    monkeypatch.delenv("PAPAYYA_API_KEY", raising=False)
    monkeypatch.setattr(
        "papayya._config.CONFIG_FILE", tmp_path / "nope" / "config.json"
    )
    with pytest.raises(PapayyaAPIError) as exc:
        resolve_config()
    assert "--api-key" not in str(exc.value)
    assert "papayya login" in str(exc.value)


# -- Fix 2: conservative retry --------------------------------------------- #


def test_request_retries_transient_status_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("papayya.api.time.sleep", lambda _: None)
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json={"ok": True})

    api = _client(handler)
    assert api._request("POST", "/v1/agents/a/runs", json={}) == {"ok": True}
    assert calls["n"] == 3  # two 503s retried, third succeeded


def test_request_does_not_retry_4xx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("papayya.api.time.sleep", lambda _: None)
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    api = _client(handler)
    with pytest.raises(PapayyaAPIError) as exc:
        api._request("POST", "/v1/agents/a/runs", json={})
    assert exc.value.status == 400
    assert calls["n"] == 1  # raised on first attempt, no retry


def test_request_does_not_retry_500(monkeypatch: pytest.MonkeyPatch) -> None:
    """500 may have partially applied — never blindly replay a submit."""
    monkeypatch.setattr("papayya.api.time.sleep", lambda _: None)
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, text="boom")

    api = _client(handler)
    with pytest.raises(PapayyaAPIError) as exc:
        api._request("POST", "/v1/agents/a/runs", json={})
    assert exc.value.status == 500
    assert calls["n"] == 1


def test_request_exhausts_retries_and_raises_last_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("papayya.api.time.sleep", lambda _: None)
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, text="slow down")

    api = _client(handler)
    with pytest.raises(PapayyaAPIError) as exc:
        api._request("GET", "/v1/durable/runs/r1")
    assert exc.value.status == 429
    assert calls["n"] == 5  # _MAX_ATTEMPTS


# -- Fix 3: stable run_id key ---------------------------------------------- #


def test_ensure_run_id_maps_id_alias() -> None:
    assert _ensure_run_id({"id": "run_123", "status": "queued"})["run_id"] == "run_123"


def test_ensure_run_id_maps_durable_run_id_alias() -> None:
    assert _ensure_run_id({"durable_run_id": "run_9"})["run_id"] == "run_9"


def test_ensure_run_id_preserves_existing_run_id() -> None:
    out = _ensure_run_id({"run_id": "keep", "id": "other"})
    assert out["run_id"] == "keep"


def test_items_create_returns_stable_run_id() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/agents/agt_1/runs"
        return httpx.Response(200, json={"id": "run_abc", "status": "queued"})

    items = Items(_client(handler))
    resp = items.create(agent_id="agt_1", input={"x": 1})
    assert resp["run_id"] == "run_abc"  # reliable regardless of server field
