"""CloudStore Slice 6 wire-format tests.

Verifies that the SDK's HTTP client sends and receives `item_id`,
`input_snapshot`, and `output_snapshot` using the exact JSON field names
expected by the Go control-plane handler. Uses an httpx MockTransport so
no real server is required; the request bodies and response bodies are
inspected directly.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from papayya.durable.cloud_store import CloudStore, CloudStoreConfig
from papayya.durable.types import RunCheckpoint, TaskEntry


def _make_store(handler) -> CloudStore:
    config = CloudStoreConfig(api_key="cpk_test", base_url="http://mock")
    store = CloudStore(config)
    # Swap the real transport for an in-memory mock.
    store._client = httpx.Client(
        base_url="http://mock",
        headers={"X-Api-Key": "cpk_test"},
        transport=httpx.MockTransport(handler),
    )
    return store


class TestCreateSendsItemId:
    def test_create_posts_item_id(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(201, json={})

        store = _make_store(handler)
        checkpoint = RunCheckpoint(
            run_id="r1", agent="enrich", tasks=[], status="running",
            item_id="co_42", created_at="", updated_at="",
        )
        store.create(checkpoint)

        assert captured["body"]["run_id"] == "r1"
        assert captured["body"]["item_id"] == "co_42"


class TestSaveTaskSendsSnapshots:
    def test_save_task_posts_all_slice6_fields(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(201, json={})

        store = _make_store(handler)
        entry = TaskEntry(
            label="enrich",
            result={"out": 1},
            duration_ms=50,
            completed_at="2026-04-15T00:00:00+00:00",
            item_id="co_42",
            input_snapshot={"in": 0},
            output_snapshot={"out": 1},
        )
        store.save_task("r1", entry)

        body = captured["body"]
        assert body["label"] == "enrich"
        assert body["item_id"] == "co_42"
        assert body["input_snapshot"] == {"in": 0}
        assert body["output_snapshot"] == {"out": 1}


class TestLoadRestoresSnapshots:
    def test_load_reads_snapshots_from_response(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "run_id": "r1",
                    "agent": "enrich",
                    "status": "running",
                    "item_id": "co_42",
                    "checkpoints": [
                        {
                            "label": "enrich",
                            "result": {"out": 1},
                            "duration_ms": 50,
                            "completed_at": "2026-04-15T00:00:00+00:00",
                            "item_id": "co_42",
                            "input_snapshot": {"in": 0},
                            "output_snapshot": {"out": 1},
                        }
                    ],
                    "created_at": "2026-04-15T00:00:00+00:00",
                    "updated_at": "2026-04-15T00:00:00+00:00",
                },
            )

        store = _make_store(handler)
        checkpoint = store.load("r1")
        assert checkpoint is not None
        assert checkpoint.item_id == "co_42"
        (entry,) = checkpoint.tasks
        assert entry.item_id == "co_42"
        assert entry.input_snapshot == {"in": 0}
        assert entry.output_snapshot == {"out": 1}


class TestNonJsonNativePayloads:
    """User-provided values (LLM SDK responses, dataclasses) must land as
    structured JSON, not crash httpx's internal ``json.dumps``.

    Mirrors the shim-side contract proven by
    ``runtime-images/python/papayya_shim/checkpoint_store.py``.
    """

    def test_save_task_encodes_simplenamespace_result(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(201, json={})

        store = _make_store(handler)
        fake_provider_response = SimpleNamespace(
            model="gemini-2.5-flash",
            usage=SimpleNamespace(total_tokens=50),
        )
        entry = TaskEntry(
            label="enrich",
            result=fake_provider_response,
            duration_ms=50,
            completed_at="2026-04-21T00:00:00+00:00",
        )
        store.save_task("r1", entry)

        body = captured["body"]
        assert body["result"]["model"] == "gemini-2.5-flash"
        assert body["result"]["usage"]["total_tokens"] == 50

    def test_set_status_encodes_simplenamespace_output(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={})

        store = _make_store(handler)
        store.set_status(
            "r1",
            "completed",
            output=SimpleNamespace(answer="42", confidence=0.9),
        )

        body = captured["body"]
        assert body["status"] == "completed"
        assert body["output"]["answer"] == "42"
        assert body["output"]["confidence"] == 0.9

    def test_save_task_rejects_non_json_snapshot(self) -> None:
        """Snapshots use strict=True to match SQLite's _encode_snapshot
        contract — silent degradation to repr would poison the /item/:id
        diff view."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(201, json={})

        store = _make_store(handler)
        entry = TaskEntry(
            label="enrich",
            result={"ok": True},
            duration_ms=50,
            completed_at="2026-04-21T00:00:00+00:00",
            input_snapshot=SimpleNamespace(not_json=True),
        )
        with pytest.raises(ValueError):
            store.save_task("r1", entry)


class TestBackwardCompat:
    def test_load_tolerates_missing_slice6_fields(self) -> None:
        """Pre-Slice-6 control-plane responses don't include these keys."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "run_id": "r1",
                    "agent": "enrich",
                    "status": "running",
                    "checkpoints": [
                        {
                            "label": "enrich",
                            "result": {"out": 1},
                            "duration_ms": 50,
                            "completed_at": "2026-04-15T00:00:00+00:00",
                        }
                    ],
                    "created_at": "2026-04-15T00:00:00+00:00",
                    "updated_at": "2026-04-15T00:00:00+00:00",
                },
            )

        store = _make_store(handler)
        checkpoint = store.load("r1")
        assert checkpoint is not None
        assert checkpoint.item_id is None
        (entry,) = checkpoint.tasks
        assert entry.item_id is None
        assert entry.input_snapshot is None
        assert entry.output_snapshot is None
