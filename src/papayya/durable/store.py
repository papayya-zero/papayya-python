"""Checkpoint store implementations: MemoryStore and FileStore."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import CheckpointStore, RunCheckpoint, TaskEntry


class MemoryStore:
    """In-memory checkpoint store. Useful for testing."""

    def __init__(self) -> None:
        self._runs: dict[str, RunCheckpoint] = {}

    def load(self, run_id: str) -> RunCheckpoint | None:
        return self._runs.get(run_id)

    def save_task(self, run_id: str, entry: TaskEntry) -> None:
        run = self._runs.get(run_id)
        if run is None:
            raise RuntimeError(f"Run {run_id} not found — call create() first")
        run.tasks.append(entry)

    def set_status(self, run_id: str, status: str, output: Any = None) -> None:
        run = self._runs.get(run_id)
        if run is None:
            raise RuntimeError(f"Run {run_id} not found")
        run.status = status

    def create(self, checkpoint: RunCheckpoint) -> None:
        self._runs[checkpoint.run_id] = checkpoint


class FileStore:
    """File-based checkpoint store. Persists as JSON in a local directory."""

    def __init__(self, directory: str = ".papayya/checkpoints") -> None:
        self._dir = Path(directory)

    def _path(self, run_id: str) -> Path:
        return self._dir / f"{run_id}.json"

    def load(self, run_id: str) -> RunCheckpoint | None:
        path = self._path(run_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return RunCheckpoint(
            run_id=data["run_id"],
            agent=data["agent"],
            tasks=[
                TaskEntry(
                    label=t["label"],
                    result=t["result"],
                    duration_ms=t["duration_ms"],
                    completed_at=t["completed_at"],
                )
                for t in data.get("tasks", [])
            ],
            status=data["status"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )

    def save_task(self, run_id: str, entry: TaskEntry) -> None:
        run = self.load(run_id)
        if run is None:
            raise RuntimeError(f"Run {run_id} not found — call create() first")
        run.tasks.append(entry)
        self._write(run)

    def set_status(self, run_id: str, status: str, output: Any = None) -> None:
        run = self.load(run_id)
        if run is None:
            raise RuntimeError(f"Run {run_id} not found")
        run.status = status
        self._write(run)

    def create(self, checkpoint: RunCheckpoint) -> None:
        self._write(checkpoint)

    def _write(self, run: RunCheckpoint) -> None:
        path = self._path(run.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "run_id": run.run_id,
            "agent": run.agent,
            "tasks": [
                {
                    "label": t.label,
                    "result": t.result,
                    "duration_ms": t.duration_ms,
                    "completed_at": t.completed_at,
                }
                for t in run.tasks
            ],
            "status": run.status,
            "created_at": run.created_at,
            "updated_at": run.updated_at,
        }
        path.write_text(json.dumps(data, indent=2))
