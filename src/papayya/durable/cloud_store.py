"""Checkpoint store backed by the Papayya control plane API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from papayya._defaults import DEFAULT_BASE_URL

from .types import CheckpointStore, RunCheckpoint, TaskEntry


@dataclass
class CloudStoreConfig:
    """Configuration for the cloud checkpoint store."""

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 15.0


class CloudStore:
    """Checkpoint store that persists to the Papayya control plane via HTTP."""

    def __init__(self, config: CloudStoreConfig) -> None:
        headers: dict[str, str] = {"Content-Type": "application/json", "Accept": "application/json"}
        if config.api_key.startswith("cpk_"):
            headers["X-Api-Key"] = config.api_key
        else:
            headers["Authorization"] = f"Bearer {config.api_key}"

        self._client = httpx.Client(
            base_url=config.base_url,
            headers=headers,
            timeout=config.timeout,
        )

    def load(self, run_id: str) -> RunCheckpoint | None:
        resp = self._client.get(f"/v1/durable/runs/{run_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        body = resp.json()
        return RunCheckpoint(
            run_id=body["run_id"],
            agent=body["agent"],
            status=body["status"],
            tasks=[
                TaskEntry(
                    label=cp["label"],
                    result=cp["result"],
                    duration_ms=cp["duration_ms"],
                    completed_at=cp["completed_at"],
                    item_id=cp.get("item_id"),
                    input_snapshot=cp.get("input_snapshot"),
                    output_snapshot=cp.get("output_snapshot"),
                )
                for cp in body.get("checkpoints") or []
            ],
            item_id=body.get("item_id"),
            created_at=body["created_at"],
            updated_at=body["updated_at"],
        )

    def create(self, checkpoint: RunCheckpoint) -> None:
        resp = self._client.post(
            "/v1/durable/runs",
            json={
                "run_id": checkpoint.run_id,
                "agent": checkpoint.agent,
                "item_id": checkpoint.item_id,
            },
        )
        resp.raise_for_status()

    def save_task(self, run_id: str, entry: TaskEntry) -> None:
        resp = self._client.post(
            f"/v1/durable/runs/{run_id}/checkpoints",
            json={
                "label": entry.label,
                "result": entry.result,
                "duration_ms": entry.duration_ms,
                "item_id": entry.item_id,
                "input_snapshot": entry.input_snapshot,
                "output_snapshot": entry.output_snapshot,
            },
        )
        resp.raise_for_status()

    def set_status(self, run_id: str, status: str, output: Any = None) -> None:
        resp = self._client.patch(
            f"/v1/durable/runs/{run_id}",
            json={"status": status, "output": output},
        )
        resp.raise_for_status()

    def close(self) -> None:
        self._client.close()
