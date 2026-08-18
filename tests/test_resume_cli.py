"""Plan 53 — `papayya resume`, and the states `run --wait` stops on.

Two gaps found by driving one fenced run:

  1. `POST /v1/durable/runs/{id}/resume` has always existed. The CLI had no
     verb for it. So a run something DECIDED to stop had no way out through the
     CLI at all — `papayya replay` answers 409 "run is paused; only a terminal
     run can be replayed", and there was nothing else to type.

  2. `papayya run --wait` polled until `completed/failed/cancelled/
     budget_exceeded`. A run a fence pauses mid-flight is none of those and
     never becomes one without a human, so the command printed
     "4 step(s) — paused" every two seconds until it was killed.

Both fixtures below are verbatim server bodies.
"""

from __future__ import annotations

from typing import Any

import pytest
from click.testing import CliRunner

from papayya import cli as cli_module
from papayya.api import PapayyaAPIError
from papayya.cli import main


RUN_ID = "dd017aba-c293-4677-b3e4-9dc9f8c4ae35"

# POST /v1/durable/runs/{id}/resume — the run row plus the two fields only
# resume reports.
RESUME_REDRIVEN = {
    "run_id": RUN_ID,
    "agent": "fenceprobe",
    "status": "running",
    "checkpoints": [],
    "redriven": True,
    "reexecuting": 3,
}


class _FakeAPI:
    def __init__(self, resume: dict[str, Any] | Exception):
        self._resume = resume
        self.closed = False

    def resume_run(self, run_id: str) -> dict[str, Any]:
        if isinstance(self._resume, Exception):
            raise self._resume
        return self._resume

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPAYYA_API_KEY", "cpk_test")
    monkeypatch.delenv("PAPAYYA_ENV", raising=False)


def _resume_cli(monkeypatch, resume):
    fake = _FakeAPI(resume)
    monkeypatch.setattr(cli_module, "APIClient", lambda _config: fake)
    return CliRunner().invoke(main, ["resume", RUN_ID]), fake


def test_resume_exists_at_all(monkeypatch):
    result, _ = _resume_cli(monkeypatch, RESUME_REDRIVEN)

    assert result.exit_code == 0, result.output
    assert RUN_ID in result.output


def test_resume_reports_how_many_steps_will_actually_run_again(monkeypatch):
    result, _ = _resume_cli(monkeypatch, RESUME_REDRIVEN)

    assert "3 step(s)" in result.output


def test_resume_says_plainly_when_nothing_will_re_execute(monkeypatch):
    """`reexecuting=0` means this will produce exactly what it produced before,
    which is almost never what the person resuming wants."""
    result, _ = _resume_cli(monkeypatch, {**RESUME_REDRIVEN, "reexecuting": 0})

    assert result.exit_code == 0, result.output
    assert "produced before" in result.output
    assert "papayya replay" in result.output


def test_resume_says_plainly_when_nothing_was_re_queued(monkeypatch):
    """`redriven=false` is the case a fence-on-the-last-step produces: the
    pause is cleared but the lease was already completed, so no parked item
    exists and the run will not move. Reporting a cheerful "resumed" here
    leaves an operator waiting on work nothing is doing."""
    result, _ = _resume_cli(monkeypatch, {**RESUME_REDRIVEN, "redriven": False})

    assert result.exit_code == 0, result.output
    assert "Nothing was re-queued" in result.output
    assert "papayya replay" in result.output


def test_resume_409_names_the_other_verb(monkeypatch):
    """Being told "this is not resumable" without being told what IS is how a
    dead end at 2am gets built. Driven: `replay` on a paused run 409s, and
    before this command existed that was where the trail stopped."""
    err = PapayyaAPIError(409, "run is completed; only paused durable runs can be resumed")
    result, _ = _resume_cli(monkeypatch, err)

    assert result.exit_code != 0
    assert "papayya replay" in result.output


# --- run --wait stop states -------------------------------------------- #


def test_wait_stops_on_operator_gated_states():
    """`paused` and `quarantine` are non-terminal on the run row and terminal
    for a client waiting on it: nothing moves them without a human."""
    from papayya.cli import _RUN_WAIT_STOP_STATES

    for state in ("completed", "failed", "cancelled", "budget_exceeded",
                  "paused", "quarantine"):
        assert state in _RUN_WAIT_STOP_STATES, state


def test_wait_does_not_stop_on_states_that_move_by_themselves():
    from papayya.cli import _RUN_WAIT_STOP_STATES

    for state in ("queued", "running"):
        assert state not in _RUN_WAIT_STOP_STATES, state
