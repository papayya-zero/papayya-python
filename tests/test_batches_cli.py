"""Unit tests for the `papayya batch` CLI group.

The CLI is thin — it resolves auth, translates flags, then delegates to
the Batches SDK resource. These tests swap the Papayya client with a
recording fake so we can assert the CLI's translation layer in isolation
(dollar→cents, file→stream, --failed required, etc.) without needing a
running backend or even a real httpx transport.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from papayya import cli as cli_module
from papayya.api import PapayyaAPIError


class _FakeBatches:
    """Records each method call with its arguments."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        # Swappable return values / side effects per method.
        self.create_stream_return: dict[str, Any] = {
            "id": "batch-xyz",
            "status": "queued",
            "total_items": 3,
        }
        self.get_return: dict[str, Any] = {
            "id": "batch-xyz",
            "status": "running",
            "agent_id": "agent-1",
            "total_items": 10,
            "completed": 4,
            "failed": 1,
            "paused": 0,
            "aggregate_cost_cents": 250,
            "budget_cents_cap": 2000,
            "name": "lead enrichment",
            "created_at": "2026-04-17T00:00:00Z",
        }
        self.cancel_return: dict[str, Any] = {"id": "batch-xyz", "status": "cancelled"}
        self.retry_failed_return: dict[str, Any] = {"id": "batch-xyz", "total_items": 12}
        self.stream_results_items: list[dict[str, Any]] = []
        self.raise_on: str | None = None

    def _maybe_raise(self, method: str) -> None:
        if self.raise_on == method:
            raise PapayyaAPIError(500, "boom")

    def create_stream(self, **kwargs: Any) -> dict[str, Any]:
        # Materialise the items iterator so tests can inspect the payload —
        # the real SDK consumes it lazily while streaming, but for assertion
        # we need the concrete list.
        items = list(kwargs.pop("items"))
        self.calls.append(("create_stream", {"items": items, **kwargs}))
        self._maybe_raise("create_stream")
        return self.create_stream_return

    def get(self, batch_id: str) -> dict[str, Any]:
        self.calls.append(("get", {"batch_id": batch_id}))
        self._maybe_raise("get")
        return self.get_return

    def cancel(self, batch_id: str) -> dict[str, Any]:
        self.calls.append(("cancel", {"batch_id": batch_id}))
        self._maybe_raise("cancel")
        return self.cancel_return

    def retry_failed(self, batch_id: str) -> dict[str, Any]:
        self.calls.append(("retry_failed", {"batch_id": batch_id}))
        self._maybe_raise("retry_failed")
        return self.retry_failed_return

    def stream_results(
        self, batch_id: str, *, poll_interval: float = 2.0, include_failed: bool = False
    ):
        self.calls.append(
            (
                "stream_results",
                {
                    "batch_id": batch_id,
                    "poll_interval": poll_interval,
                    "include_failed": include_failed,
                },
            )
        )
        self._maybe_raise("stream_results")
        yield from self.stream_results_items


class _FakeClient:
    def __init__(self) -> None:
        self.batches = _FakeBatches()
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> _FakeClient:
    client = _FakeClient()
    monkeypatch.setattr(cli_module, "_make_papayya_client", lambda ctx: client)
    return client


def _run(args: list[str]) -> Any:
    return CliRunner().invoke(cli_module.main, args, catch_exceptions=False)


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

def _write_jsonl(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    p = tmp_path / "items.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_submit_converts_budget_dollars_to_cents_and_streams_file(
    fake_client: _FakeClient, tmp_path: Path
) -> None:
    file_ = _write_jsonl(tmp_path, [{"input": "a"}, {"input": "b"}, {"input": "c"}])

    result = _run([
        "batch", "submit",
        "--agent", "agent-1",
        "--file", str(file_),
        "--budget", "20",
        "--concurrency", "10",
        "--name", "lead enrichment",
    ])
    assert result.exit_code == 0, result.output

    method, kwargs = fake_client.batches.calls[-1]
    assert method == "create_stream"
    assert kwargs["agent_id"] == "agent-1"
    assert kwargs["items"] == [{"input": "a"}, {"input": "b"}, {"input": "c"}]
    assert kwargs["budget_cents_cap"] == 2000  # $20 → 2000¢
    assert kwargs["concurrency_cap"] == 10
    assert kwargs["name"] == "lead enrichment"
    assert kwargs["callback_url"] is None
    assert kwargs["idempotency_key"] is None
    assert "Batch submitted: batch-xyz" in result.output
    assert fake_client.closed


def test_submit_without_optional_flags_passes_none(
    fake_client: _FakeClient, tmp_path: Path
) -> None:
    file_ = _write_jsonl(tmp_path, [{"input": "a"}])
    result = _run(["batch", "submit", "--agent", "a", "--file", str(file_)])
    assert result.exit_code == 0

    _, kwargs = fake_client.batches.calls[-1]
    assert kwargs["budget_cents_cap"] is None
    assert kwargs["concurrency_cap"] is None
    assert kwargs["name"] is None


def test_submit_rounds_fractional_dollars_to_nearest_cent(
    fake_client: _FakeClient, tmp_path: Path
) -> None:
    """$0.015 → 2¢ (banker's-rounded via round()); makes sure we don't
    silently truncate fractional budgets."""
    file_ = _write_jsonl(tmp_path, [{"input": "a"}])
    result = _run([
        "batch", "submit",
        "--agent", "a",
        "--file", str(file_),
        "--budget", "0.5",
    ])
    assert result.exit_code == 0
    _, kwargs = fake_client.batches.calls[-1]
    assert kwargs["budget_cents_cap"] == 50


def test_submit_errors_on_missing_file(fake_client: _FakeClient) -> None:
    result = _run([
        "batch", "submit",
        "--agent", "a",
        "--file", "/definitely/does/not/exist.jsonl",
    ])
    assert result.exit_code == 1
    assert "File not found" in result.output
    assert fake_client.batches.calls == []  # never got as far as the SDK


def test_submit_errors_on_invalid_json_line(
    fake_client: _FakeClient, tmp_path: Path
) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text('{"input": "a"}\nnot-json\n', encoding="utf-8")
    result = _run(["batch", "submit", "--agent", "a", "--file", str(p)])
    assert result.exit_code == 1
    assert "invalid JSON" in result.output


def test_submit_skips_blank_lines(
    fake_client: _FakeClient, tmp_path: Path
) -> None:
    p = tmp_path / "items.jsonl"
    p.write_text('{"input": "a"}\n\n   \n{"input": "b"}\n', encoding="utf-8")
    result = _run(["batch", "submit", "--agent", "a", "--file", str(p)])
    assert result.exit_code == 0
    _, kwargs = fake_client.batches.calls[-1]
    assert kwargs["items"] == [{"input": "a"}, {"input": "b"}]


def test_submit_surfaces_api_error(
    fake_client: _FakeClient, tmp_path: Path
) -> None:
    fake_client.batches.raise_on = "create_stream"
    file_ = _write_jsonl(tmp_path, [{"input": "a"}])
    result = _run(["batch", "submit", "--agent", "a", "--file", str(file_)])
    assert result.exit_code == 1
    assert "HTTP 500" in result.output


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def test_status_prints_aggregate(fake_client: _FakeClient) -> None:
    result = _run(["batch", "status", "batch-xyz"])
    assert result.exit_code == 0, result.output
    assert ("get", {"batch_id": "batch-xyz"}) in fake_client.batches.calls
    assert "Batch:" in result.output
    assert "running" in result.output
    assert "4/10 completed" in result.output
    assert "1 failed" in result.output
    assert "250¢ / 2000¢" in result.output


def test_status_handles_no_budget_cap(fake_client: _FakeClient) -> None:
    fake_client.batches.get_return = {
        "id": "b", "status": "completed", "agent_id": "a",
        "total_items": 1, "completed_items": 1, "failed_items": 0, "paused_items": 0,
        "aggregate_cost_cents": 42, "budget_cents_cap": None,
    }
    result = _run(["batch", "status", "b"])
    assert result.exit_code == 0
    assert "42¢ (no cap)" in result.output


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

def test_results_writes_jsonl_file(
    fake_client: _FakeClient, tmp_path: Path
) -> None:
    fake_client.batches.stream_results_items = [
        {"id": "run-1", "status": "completed"},
        {"id": "run-2", "status": "completed"},
    ]
    out = tmp_path / "out.jsonl"
    result = _run([
        "batch", "results", "batch-xyz",
        "-o", str(out),
        "--poll-interval", "0.01",
    ])
    assert result.exit_code == 0, result.output

    call = [c for c in fake_client.batches.calls if c[0] == "stream_results"][0]
    assert call[1]["batch_id"] == "batch-xyz"
    assert call[1]["include_failed"] is False
    assert call[1]["poll_interval"] == 0.01

    lines = out.read_text().splitlines()
    assert [json.loads(x) for x in lines] == [
        {"id": "run-1", "status": "completed"},
        {"id": "run-2", "status": "completed"},
    ]
    assert "Wrote 2 run(s)" in result.output


def test_results_streams_to_stdout_without_output_flag(
    fake_client: _FakeClient,
) -> None:
    fake_client.batches.stream_results_items = [
        {"id": "run-1"},
    ]
    result = _run(["batch", "results", "batch-xyz", "--poll-interval", "0.01"])
    assert result.exit_code == 0
    # The CLI streams JSON to stdout; no trailing summary in the non-file path.
    assert '{"id": "run-1"}' in result.output
    assert "Wrote" not in result.output


def test_results_include_failed_flag_forwards(fake_client: _FakeClient) -> None:
    result = _run([
        "batch", "results", "batch-xyz",
        "--include-failed", "--poll-interval", "0.01",
    ])
    assert result.exit_code == 0
    call = [c for c in fake_client.batches.calls if c[0] == "stream_results"][0]
    assert call[1]["include_failed"] is True


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------

def test_cancel_invokes_sdk_and_prints(fake_client: _FakeClient) -> None:
    result = _run(["batch", "cancel", "batch-xyz"])
    assert result.exit_code == 0, result.output
    assert ("cancel", {"batch_id": "batch-xyz"}) in fake_client.batches.calls
    assert "cancellation accepted" in result.output
    assert "cancelled" in result.output


# ---------------------------------------------------------------------------
# retry
# ---------------------------------------------------------------------------

def test_retry_requires_failed_flag(fake_client: _FakeClient) -> None:
    # --failed is the only retry mode today, but marking it required now
    # prevents an ambiguous `papayya batch retry <id>` from meaning "retry
    # all" in the future. Assert click rejects the call without it.
    result = _run(["batch", "retry", "batch-xyz"])
    assert result.exit_code != 0
    assert "failed" in result.output.lower()
    assert fake_client.batches.calls == []


def test_retry_with_failed_flag_calls_retry_failed(
    fake_client: _FakeClient,
) -> None:
    result = _run(["batch", "retry", "batch-xyz", "--failed"])
    assert result.exit_code == 0, result.output
    assert ("retry_failed", {"batch_id": "batch-xyz"}) in fake_client.batches.calls
    assert "re-enqueued" in result.output
    assert "total_items now 12" in result.output
