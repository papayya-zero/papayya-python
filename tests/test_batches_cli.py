"""Unit tests for `papayya runs submit` (pre-0.3.0: `papayya batch submit`).

The CLI is thin — it resolves auth, translates flags, then delegates to
the Runs (invocation) SDK resource. These tests swap the Papayya client with a
recording fake so we can assert the CLI's translation layer in isolation
(dollar→cents, file→stream, etc.) without needing a running backend.

v1→v2 cutover: only submit (→ create_stream → POST /v1/batches, wire
frozen until Plan 34 Unit 5) survives; the v1 batch read + lifecycle
commands retired with the v1 DROP. Plan 34: the command moved to
`runs submit`; `batch submit` stays as a hidden alias sharing the same
command object.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from papayya import cli as cli_module
from papayya.api import PapayyaAPIError


class _FakeRuns:
    """Records each method call with its arguments."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.create_stream_return: dict[str, Any] = {
            "id": "batch-xyz",
            "status": "queued",
            "total_items": 3,
        }
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


class _FakeClient:
    def __init__(self) -> None:
        self.runs = _FakeRuns()
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
        "runs", "submit",
        "--agent", "agent-1",
        "--file", str(file_),
        "--budget", "20",
        "--concurrency", "10",
        "--name", "lead enrichment",
    ])
    assert result.exit_code == 0, result.output

    method, kwargs = fake_client.runs.calls[-1]
    assert method == "create_stream"
    assert kwargs["agent_id"] == "agent-1"
    assert kwargs["items"] == [{"input": "a"}, {"input": "b"}, {"input": "c"}]
    assert kwargs["budget_cents_cap"] == 2000  # $20 → 2000¢
    assert kwargs["concurrency_cap"] == 10
    assert kwargs["name"] == "lead enrichment"
    assert kwargs["callback_url"] is None
    assert kwargs["idempotency_key"] is None
    assert "Run submitted: batch-xyz" in result.output
    assert fake_client.closed


def test_submit_without_optional_flags_passes_none(
    fake_client: _FakeClient, tmp_path: Path
) -> None:
    file_ = _write_jsonl(tmp_path, [{"input": "a"}])
    result = _run(["runs", "submit", "--agent", "a", "--file", str(file_)])
    assert result.exit_code == 0

    _, kwargs = fake_client.runs.calls[-1]
    assert kwargs["budget_cents_cap"] is None
    assert kwargs["concurrency_cap"] is None
    assert kwargs["name"] is None


def test_submit_rounds_fractional_dollars_to_nearest_cent(
    fake_client: _FakeClient, tmp_path: Path
) -> None:
    """$0.5 → 50¢; makes sure we don't silently truncate fractional budgets."""
    file_ = _write_jsonl(tmp_path, [{"input": "a"}])
    result = _run([
        "runs", "submit",
        "--agent", "a",
        "--file", str(file_),
        "--budget", "0.5",
    ])
    assert result.exit_code == 0
    _, kwargs = fake_client.runs.calls[-1]
    assert kwargs["budget_cents_cap"] == 50


def test_submit_errors_on_missing_file(fake_client: _FakeClient) -> None:
    result = _run([
        "runs", "submit",
        "--agent", "a",
        "--file", "/definitely/does/not/exist.jsonl",
    ])
    assert result.exit_code == 1
    assert "File not found" in result.output
    assert fake_client.runs.calls == []  # never got as far as the SDK


def test_submit_errors_on_invalid_json_line(
    fake_client: _FakeClient, tmp_path: Path
) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text('{"input": "a"}\nnot-json\n', encoding="utf-8")
    result = _run(["runs", "submit", "--agent", "a", "--file", str(p)])
    assert result.exit_code == 1
    assert "invalid JSON" in result.output


def test_submit_skips_blank_lines(
    fake_client: _FakeClient, tmp_path: Path
) -> None:
    p = tmp_path / "items.jsonl"
    p.write_text('{"input": "a"}\n\n   \n{"input": "b"}\n', encoding="utf-8")
    result = _run(["runs", "submit", "--agent", "a", "--file", str(p)])
    assert result.exit_code == 0
    _, kwargs = fake_client.runs.calls[-1]
    assert kwargs["items"] == [{"input": "a"}, {"input": "b"}]


def test_submit_surfaces_api_error(
    fake_client: _FakeClient, tmp_path: Path
) -> None:
    fake_client.runs.raise_on = "create_stream"
    file_ = _write_jsonl(tmp_path, [{"input": "a"}])
    result = _run(["runs", "submit", "--agent", "a", "--file", str(file_)])
    assert result.exit_code == 1
    assert "HTTP 500" in result.output


# ---------------------------------------------------------------------------
# hidden alias: `papayya batch submit` (removed one release after 0.3.0)
# ---------------------------------------------------------------------------

def test_batch_submit_hidden_alias_still_works(
    fake_client: _FakeClient, tmp_path: Path
) -> None:
    file_ = _write_jsonl(tmp_path, [{"input": "a"}])
    result = _run(["batch", "submit", "--agent", "a", "--file", str(file_)])
    assert result.exit_code == 0, result.output
    method, kwargs = fake_client.runs.calls[-1]
    assert method == "create_stream"
    assert kwargs["agent_id"] == "a"


def test_batch_group_is_hidden_from_help() -> None:
    result = _run(["--help"])
    assert result.exit_code == 0
    assert "\n  batch " not in result.output
