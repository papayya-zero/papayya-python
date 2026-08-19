"""Tests for `papayya runs` (invocations) and `papayya items` (per-item).

Plan 34 BREAKING shift, covered explicitly here:

- `papayya runs list` reads the LOCAL ledger and lists invocations, one
  NDJSON line per run, carrying the outcome rollup (degraded/failed item
  counts + worst_outcome_status).
- The hosted per-item verbs the old `runs` group carried (list/stream)
  moved to `papayya items` (plus `get`). Those tests swap the Papayya
  client with a recording fake.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from papayya import cli as cli_module
from papayya.api import PapayyaAPIError

# The file-level skip that was here is gone. It was written when this file's
# SUBJECT was the deactivated local surface (iter/map / local SQLite CLI), and
# it kept covering the file as hosted tests were added underneath it — so
# `items list`, `items get` and `items stream`, all live commands, had tests
# that never ran. The local tests it was written for have been deleted along
# with the command they exercised.



class _FakeItems:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.list_return: list[dict[str, Any]] = []
        self.get_return: dict[str, Any] = {"id": "i1", "status": "completed"}
        self.resolve_return: list[dict[str, Any]] = []
        self.stream_events: list[dict[str, Any]] = []
        self.raise_on: str | None = None

    def _maybe_raise(self, method: str) -> None:
        if self.raise_on == method:
            raise PapayyaAPIError(500, "boom")

    def list(
        self,
        *,
        run_id: str | None = None,
        agent: str | None = None,
        partition_key: str | None = None,
        item_id: str | None = None,
        item_id_prefix: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        # Records the scope so a test can assert `--run` actually reached the
        # resource. Plan 56 F6: the server has always accepted parent_run_id,
        # agent and partition_key — only the client never passed them. Plan 57
        # D4: it ACCEPTED the three below and ignored them, which is worse.
        self.calls.append((
            "list",
            {"run_id": run_id, "agent": agent, "partition_key": partition_key,
             "item_id": item_id, "item_id_prefix": item_id_prefix,
             "status": status},
        ))
        self._maybe_raise("list")
        return self.list_return

    def get(self, item_id: str) -> dict[str, Any]:
        self.calls.append(("get", {"item_id": item_id}))
        self._maybe_raise("get")
        return self.get_return

    def resolve(self, item_id: str) -> list[dict[str, Any]]:
        self.calls.append(("resolve", {"item_id": item_id}))
        self._maybe_raise("resolve")
        return self.resolve_return

    def stream(self, item_id: str, *, from_step: int | None = None):
        self.calls.append(("stream", {"item_id": item_id, "from_step": from_step}))
        self._maybe_raise("stream")
        yield from self.stream_events


class _FakeRuns:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.list_return: list[dict[str, Any]] = []

    def list(self, *, agent: str | None = None, item_id: str | None = None,
             limit: int | None = None):
        self.calls.append({"agent": agent, "item_id": item_id, "limit": limit})
        return self.list_return


class _FakeClient:
    def __init__(self) -> None:
        self.items = _FakeItems()
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


# The local-ledger `runs list` tests that were here are gone with the command.
# Both were SKIPPED, never passing: they exercised a click command that was
# never registered on the `runs` group, reading a `.papayya/local.db` that
# `papayya dev` stopped creating. The hosted replacement is covered below.


# ---------------------------------------------------------------------------
# items — hosted per-item verbs (the pre-0.3.0 `runs` surface, renamed)
# ---------------------------------------------------------------------------


def test_items_list_outputs_ndjson(fake_client: _FakeClient) -> None:
    fake_client.items.list_return = [{"id": "r1"}, {"id": "r2"}]
    result = _run(["items", "list"])
    assert result.exit_code == 0, result.output
    # Unscoped stays unscoped: every filter None is "the whole project", which
    # is what this command has always done.
    assert ("list", {"run_id": None, "agent": None, "partition_key": None,
                     "item_id": None, "item_id_prefix": None, "status": None}) \
        in fake_client.items.calls
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert [json.loads(ln)["id"] for ln in lines] == ["r1", "r2"]


def test_items_list_scopes_to_a_run(fake_client: _FakeClient) -> None:
    """Plan 56 F6 / plan 55 D6.3.

    `papayya status` prints "Items in this run: papayya items list", and that
    command took no options — so the only route from a submitted run id to one
    wedged item was dumping every item in the project as NDJSON and reading for
    "running". The server accepted parent_run_id the whole time.
    """
    fake_client.items.list_return = [{"id": "r1"}]
    result = _run(["items", "list", "--run", "run-42", "--tenant", "northwind"])
    assert result.exit_code == 0, result.output
    assert ("list", {"run_id": "run-42", "agent": None, "partition_key": "northwind",
                     "item_id": None, "item_id_prefix": None, "status": None}) \
        in fake_client.items.calls


def test_items_get_pretty_prints(fake_client: _FakeClient) -> None:
    result = _run(["items", "get", "i1"])
    assert result.exit_code == 0, result.output
    assert ("get", {"item_id": "i1"}) in fake_client.items.calls
    assert json.loads(result.output)["id"] == "i1"


def test_items_stream_emits_one_event_per_line(fake_client: _FakeClient) -> None:
    fake_client.items.stream_events = [
        {"event": "step", "data": {"step_type": "llm"}, "id": 1},
        {"event": "terminal", "data": {"status": "completed"}},
    ]
    result = _run(["items", "stream", "r1", "--from-step", "5"])
    assert result.exit_code == 0, result.output
    assert ("stream", {"item_id": "r1", "from_step": 5}) in fake_client.items.calls
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert [json.loads(ln)["event"] for ln in lines] == ["step", "terminal"]


def test_runs_stream_is_gone(fake_client: _FakeClient) -> None:
    """BREAKING (0.3.0): streaming a per-item record lives at
    `items stream`; the old `runs stream` spelling errors rather than
    silently meaning something new."""
    result = _run(["runs", "stream", "r1"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# runs list — "what have I run?" (plan 52 G1)
# ---------------------------------------------------------------------------
#
# The CLI had no answer. `items list` prints per-record NDJSON, and `status` /
# `logs` both need a run id the user had to keep — so losing the id
# `papayya run` printed left no route back to their own work. The endpoint did
# not exist either: ListRunsV2 always took a nullable agent, but only
# /v2/agents/{slug}/runs was mounted.


def _a_run(**over: Any) -> dict[str, Any]:
    base = {
        "id": "1a097e38-f49b-4ff0-a072-665046960e64",
        "agent": "triage",
        "status": "completed",
        "item_count": 5,
        "failed_count": 0,
        "cost_usd": 0.0123,
        "unpriced_item_count": 0,
        "created_at": "2026-08-17T18:00:00Z",
    }
    base.update(over)
    return base


def test_runs_list_shows_a_table_a_person_can_read(fake_client: _FakeClient) -> None:
    fake_client.runs.list_return = [_a_run()]
    result = _run(["runs", "list"])

    assert result.exit_code == 0, result.output
    assert "RUN" in result.output and "AGENT" in result.output
    assert "1a097e38-f49b-4ff0-a072-665046960e64" in result.output
    assert "triage" in result.output
    assert "$0.01" in result.output


def test_runs_list_counts_the_failures_beside_the_items(fake_client: _FakeClient) -> None:
    """The number a person is scanning for is how many went wrong."""
    fake_client.runs.list_return = [_a_run(item_count=5, failed_count=2)]
    result = _run(["runs", "list"])

    assert "2 failed" in result.output


def test_runs_list_says_when_every_item_was_unpriced(fake_client: _FakeClient) -> None:
    """A run whose every item spent unpriced tokens is not a $0.00 run — the
    same asymmetry formatRunCost encodes in the dashboard (plan 51)."""
    fake_client.runs.list_return = [
        _a_run(cost_usd=0, item_count=3, unpriced_item_count=3)]
    result = _run(["runs", "list"])

    assert "$0.00" not in result.output


def test_runs_list_marks_a_partly_unpriced_total_as_a_lower_bound(
    fake_client: _FakeClient,
) -> None:
    fake_client.runs.list_return = [
        _a_run(cost_usd=0.5, item_count=3, unpriced_item_count=1)]
    result = _run(["runs", "list"])

    assert "$0.50*" in result.output


def test_runs_list_says_so_when_it_stopped_at_the_limit(
    fake_client: _FakeClient,
) -> None:
    """A list that silently stops at its cap answers the question with a
    subset and looks like the whole answer."""
    fake_client.runs.list_return = [_a_run(id=f"r-{i}") for i in range(3)]
    result = _run(["runs", "list", "--limit", "3"])

    assert "showing the newest 3" in result.output


def test_runs_list_empty_says_what_to_type_next(fake_client: _FakeClient) -> None:
    fake_client.runs.list_return = []
    result = _run(["runs", "list"])

    assert result.exit_code == 0
    assert "No runs yet" in result.output
    assert "papayya run" in result.output


def test_runs_list_passes_the_agent_filter_through(fake_client: _FakeClient) -> None:
    fake_client.runs.list_return = []
    _run(["runs", "list", "--agent", "triage"])

    assert fake_client.runs.calls == [{"agent": "triage", "item_id": None, "limit": 20}]


def test_runs_list_json_is_one_run_per_line(fake_client: _FakeClient) -> None:
    """The machine surface stays available — the table is the default, not the
    only option."""
    fake_client.runs.list_return = [_a_run(id="r-1"), _a_run(id="r-2")]
    result = _run(["runs", "list", "--json"])

    lines = [json.loads(ln) for ln in result.output.splitlines() if ln.strip()]
    assert [r["id"] for r in lines] == ["r-1", "r-2"]


# ---------------------------------------------------------------------------
# Plan 57 D4/D5 — the document becomes addressable from the CLI.
# ---------------------------------------------------------------------------

def test_items_list_passes_the_item_filters_through(fake_client: _FakeClient) -> None:
    """These three names were ACCEPTED by the server and ignored, which is
    worse than unsupported: `?item_id=DOC-2001` returned every run in the
    project, so a client that trusted it rendered one document's history under
    another's heading."""
    fake_client.items.list_return = []
    result = _run(["items", "list", "--item-prefix", "DOC-2001#", "--status", "failed"])
    assert result.exit_code == 0, result.output
    assert ("list", {"run_id": None, "agent": None, "partition_key": None,
                     "item_id": None, "item_id_prefix": "DOC-2001#",
                     "status": "failed"}) in fake_client.items.calls


def test_runs_list_passes_the_item_filter_through(fake_client: _FakeClient) -> None:
    fake_client.runs.list_return = []
    _run(["runs", "list", "--item", "DOC-2001"])
    assert fake_client.runs.calls == [
        {"agent": None, "item_id": "DOC-2001", "limit": 20}]


def test_runs_list_empty_under_item_names_the_filter(fake_client: _FakeClient) -> None:
    """An empty list under a narrowing flag reads as "you have never run
    anything" — the zero-returning default this codebase keeps meeting."""
    fake_client.runs.list_return = []
    result = _run(["runs", "list", "--item", "DOC-9999"])
    assert result.exit_code == 0
    assert "DOC-9999" in result.output
    assert "items get" in result.output


def test_items_get_names_the_other_runs_of_the_same_document(
    fake_client: _FakeClient,
) -> None:
    """THE plan 57 D5 case. After a re-drive the document lives under two run
    ids and no CLI surface joined them, so a customer who fixed a document
    could not get from the run that failed to the run that fixed it."""
    fake_client.items.get_return = {
        "run_id": "9be12e39", "item_id": "DOC-2001", "status": "completed"}
    fake_client.items.resolve_return = [
        {"run_id": "9be12e39", "item_id": "DOC-2001", "status": "completed",
         "created_at": "2026-08-19T10:04:00Z"},
        {"run_id": "0ac9eb00", "item_id": "DOC-2001", "status": "failed",
         "created_at": "2026-08-19T10:00:00Z"},
    ]
    result = _run(["items", "get", "DOC-2001"])
    assert result.exit_code == 0, result.output
    assert "0ac9eb00" in result.output, "the earlier run must be named"
    assert "1 earlier run" in result.output


def test_items_get_on_a_lone_record_says_nothing_extra(
    fake_client: _FakeClient,
) -> None:
    fake_client.items.get_return = {
        "run_id": "9be12e39", "item_id": "DOC-2000", "status": "completed"}
    fake_client.items.resolve_return = [
        {"run_id": "9be12e39", "item_id": "DOC-2000", "status": "completed"}]
    result = _run(["items", "get", "DOC-2000"])
    assert result.exit_code == 0
    assert "earlier run" not in result.output
