"""Plan 48 W4 + W6 — `papayya logs` and `papayya status`, against real shapes.

Both commands read the pre-cutover v1 field names. `status` indexed
`result['id']` where the durable API returns `run_id`, so it raised
`KeyError: 'id'` on every hosted run there has ever been; `logs` read
`step_number` / `step_type` / `status` / `input_tokens` / `output_tokens` /
`output`, six names for fields that are actually `seq` / `label` /
`outcome_status` / `llm_prompt_tokens` / `llm_completion_tokens` / `result`.

**Every fixture below is a verbatim server response.** The two tests that
already covered `status` handed it `{"id": ..., "current_step": 0,
"total_cost_cents": 0}` — a fake answering with the fields the CLI *wanted*,
which is the whole reason a `KeyError` on the product's primary inspection
command stayed green for four months. A fixture that manufactures the shape
the code expects tests nothing but the code's own opinion.
"""

from __future__ import annotations

from typing import Any

import pytest
from click.testing import CliRunner

from papayya import cli as cli_module
from papayya.cli import _fmt_usd, main


# A real GET /v1/durable/runs/{id} body, fields and all.
RUN_OK = {
    "run_id": "cdeba672-104a-44b6-8b9b-6e9a7a082447",
    "tenant_id": "b5e4dbe6-4664-4a5b-82c1-31ea8ba9d23a",
    "agent": "hello",
    "status": "completed",
    "project_id": "29e7434f-5bf8-45e6-825f-421e71635efb",
    "budget_consumed_usd": 0,
    "budget_limit_usd": 5,
    "output": {"greeting": "Hello, world!"},
    "input": "world",
    "agent_version": "2",
    "worst_outcome_status": "ok",
    "llm_tokens_total": 0,
    "cost_priced": True,
    "degraded_count": 0,
    "checkpoints": None,
}

# A real GET /v1/durable/runs/{id}/checkpoints item.
STEP_OK = {
    "id": "ea618b89-a7c0-50cd-a7b5-3c1e3bfc9e68",
    "run_id": RUN_OK["run_id"],
    "agent": "hello",
    "label": "greet",
    "result": "Hello, world!",
    "cost_usd": 0,
    "duration_ms": 12,
    "outcome_status": "ok",
    "attempt": 1,
    "seq": 1071796,
}


RUN_V2 = {
    "id": "ff32b881-a40c-42ba-8c2a-a93f4ecd7ab1",
    "agent": "hello",
    "status": "completed",
    "item_count": 5,
    "failed_count": 0,
    "queued_count": 0,
    "running_count": 0,
    "completed_count": 5,
    "worst_outcome_status": "ok",
    "degraded_count": 0,
    "cost_usd": 1.995,
    "unpriced_item_count": 0,
}


class _FakeAPI:
    def __init__(self, run: dict[str, Any] | Exception, steps: list[dict[str, Any]],
                 run_v2: dict[str, Any] | None = None):
        self._run = run
        self._steps = steps
        self._run_v2 = run_v2
        self.closed = False

    def get_run_v2(self, run_id: str) -> dict[str, Any] | None:
        return self._run_v2

    def get_run(self, run_id: str) -> dict[str, Any]:
        if isinstance(self._run, Exception):
            raise self._run
        return self._run

    def get_steps(self, run_id: str) -> list[dict[str, Any]]:
        return self._steps

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPAYYA_API_KEY", "cpk_test")
    monkeypatch.delenv("PAPAYYA_ENV", raising=False)


def _run_cli(monkeypatch, argv, *, run=RUN_OK, steps=(), run_v2=None):
    fake = _FakeAPI(run, list(steps), run_v2)
    monkeypatch.setattr(cli_module, "APIClient", lambda _config: fake)
    return CliRunner().invoke(main, argv), fake


# --- status ------------------------------------------------------------ #


def test_status_reads_run_id_not_id(monkeypatch):
    """The defect, exactly: `result['id']` against a body keyed `run_id`."""
    result, _ = _run_cli(monkeypatch, ["status", RUN_OK["run_id"]])

    assert result.exit_code == 0, result.output
    assert "KeyError" not in result.output
    assert RUN_OK["run_id"] in result.output
    assert "completed" in result.output


def test_status_does_not_invent_a_step_count_or_a_cost(monkeypatch):
    """`current_step` and `total_cost_cents` are on no response, ever."""
    result, _ = _run_cli(monkeypatch, ["status", RUN_OK["run_id"]])

    assert "Step:" not in result.output
    assert "cents" not in result.output


def test_status_says_why_a_failed_run_failed(monkeypatch):
    """plan 50 put the error on the row; this is the command that shows it."""
    run = {**RUN_OK, "status": "failed",
           "error": "KeyError: 'missing'", "error_category": "customer_code"}
    result, _ = _run_cli(monkeypatch, ["status", RUN_OK["run_id"]], run=run)

    assert "KeyError: 'missing'" in result.output
    assert "customer_code" in result.output


def test_status_reports_unpriced_rather_than_free(monkeypatch):
    """A zero beside real tokens is 'we could not price it', not '$0.00'."""
    run = {**RUN_OK, "llm_tokens_total": 4210, "cost_priced": False,
           "cost_unpriced_reason": "unpriced — no rate card entry for gpt-4o-mini"}
    result, _ = _run_cli(monkeypatch, ["status", RUN_OK["run_id"]], run=run)

    assert "no rate card entry for gpt-4o-mini" in result.output
    assert "$0.00 of" not in result.output


def test_status_surfaces_a_completed_run_that_did_not_work(monkeypatch):
    """The wedge: `completed` and `degraded` are not the same answer."""
    run = {**RUN_OK, "worst_outcome_status": "degraded", "degraded_count": 2}
    result, _ = _run_cli(monkeypatch, ["status", RUN_OK["run_id"]], run=run)

    assert "degraded" in result.output


# --- logs -------------------------------------------------------------- #


def test_logs_renders_the_checkpoint_shape(monkeypatch):
    result, _ = _run_cli(monkeypatch, ["logs", RUN_OK["run_id"]], steps=[STEP_OK])

    assert result.exit_code == 0, result.output
    assert "KeyError" not in result.output
    assert "greet" in result.output
    assert "ok" in result.output
    assert "Hello, world!" in result.output


def test_logs_marks_a_second_attempt(monkeypatch):
    """Two rows with one label read as a duplicate unless the attempt shows."""
    steps = [STEP_OK, {**STEP_OK, "attempt": 2}]
    result, _ = _run_cli(monkeypatch, ["logs", RUN_OK["run_id"]], steps=steps)

    assert "attempt 2" in result.output


def test_logs_prints_no_token_line_for_a_step_that_made_no_model_call(monkeypatch):
    """llm_* are omitempty; defaulting them to 0 reports a measurement that
    was never taken."""
    result, _ = _run_cli(monkeypatch, ["logs", RUN_OK["run_id"]], steps=[STEP_OK])

    assert "Tokens" not in result.output


def test_logs_prints_tokens_when_the_step_did_call_a_model(monkeypatch):
    step = {**STEP_OK, "llm_prompt_tokens": 120, "llm_completion_tokens": 34,
            "cost_usd": 0.000034}
    result, _ = _run_cli(monkeypatch, ["logs", RUN_OK["run_id"]], steps=[step])

    assert "120 in / 34 out" in result.output
    # And the charge is not rounded away — see test_sub_cent_costs below.
    assert "$0.00 " not in result.output


def test_logs_names_the_run_when_there_are_no_steps(monkeypatch):
    """'No steps found.' meant three different things and exited 0 for all."""
    result, _ = _run_cli(monkeypatch, ["logs", RUN_OK["run_id"]], steps=[])

    assert RUN_OK["run_id"] in result.output
    assert "completed" in result.output
    assert "run.step" in result.output


def test_logs_does_not_report_a_missing_run_as_an_empty_one(monkeypatch):
    """A typo'd run id used to print 'No steps found.' and exit 0."""
    from papayya.api import PapayyaAPIError

    result, _ = _run_cli(monkeypatch, ["logs", "00000000-0000-0000-0000-000000000000"],
                         run=PapayyaAPIError(404, "durable run not found"),
                         steps=[])

    assert result.exit_code != 0
    assert "404" in str(result.output) + str(result.exception)


def test_logs_says_the_run_failed_under_the_steps_that_worked(monkeypatch):
    """The step that raises never checkpoints, so a failed run's step list is
    a clean list of successes — `logs` was the one place the failure was
    invisible."""
    run = {**RUN_OK, "status": "failed",
           "error": "KeyError: 'missing'", "error_category": "customer_code"}
    result, _ = _run_cli(monkeypatch, ["logs", RUN_OK["run_id"]],
                         run=run, steps=[STEP_OK])

    assert "greet" in result.output
    assert "Run failed" in result.output
    assert "KeyError: 'missing'" in result.output


def test_logs_stays_quiet_on_a_healthy_run(monkeypatch):
    result, _ = _run_cli(monkeypatch, ["logs", RUN_OK["run_id"]], steps=[STEP_OK])

    assert "Run failed" not in result.output
    assert "worst step outcome" not in result.output


# --- money ------------------------------------------------------------- #


def test_sub_cent_costs_are_not_rounded_to_zero():
    """`${:.4f}` rendered a real $0.000034 as `$0.0000` — the product's own
    cost column reporting a charge as free (plan 51)."""
    assert _fmt_usd(0.000034) == "$0.000034"
    assert _fmt_usd(0) == "$0.00"
    assert _fmt_usd(12.5) == "$12.50"


# --- one id, two nouns ------------------------------------------------- #
#
# `runs submit` returns a GROUP id and `run` returns an ITEM id. They live on
# different surfaces (/v2/runs/{id} vs /v1/durable/runs/{id}), so `status`
# 404'd on half the ids the product itself had handed out — including the one
# printed directly above the line telling the user what to run next (plan 48
# W6's second half).


def test_status_falls_back_to_the_submission_for_a_group_id(monkeypatch):
    from papayya.api import PapayyaAPIError

    result, _ = _run_cli(monkeypatch, ["status", RUN_V2["id"]],
                         run=PapayyaAPIError(404, "durable run not found"),
                         run_v2=RUN_V2)

    assert result.exit_code == 0, result.output
    assert RUN_V2["id"] in result.output
    assert "hello" in result.output
    assert "5" in result.output


def test_status_reports_a_submissions_failures(monkeypatch):
    from papayya.api import PapayyaAPIError

    result, _ = _run_cli(monkeypatch, ["status", RUN_V2["id"]],
                         run=PapayyaAPIError(404, "not found"),
                         run_v2={**RUN_V2, "status": "failed", "failed_count": 2})

    assert "2 of 5 item(s) failed" in result.output


def test_status_on_an_id_that_is_neither_keeps_the_original_404(monkeypatch):
    """A typo is a typo. Inventing a second error for it would bury the first."""
    from papayya.api import PapayyaAPIError

    result, _ = _run_cli(monkeypatch, ["status", "00000000-0000-0000-0000-000000000000"],
                         run=PapayyaAPIError(404, "durable run not found"),
                         run_v2=None)

    assert result.exit_code != 0
    assert "404" in str(result.output) + str(result.exception)


def test_status_does_not_swallow_a_non_404_from_the_item_surface(monkeypatch):
    """Only a 404 means "wrong noun". A 500 is a 500."""
    from papayya.api import PapayyaAPIError

    result, _ = _run_cli(monkeypatch, ["status", RUN_OK["run_id"]],
                         run=PapayyaAPIError(500, "boom"), run_v2=RUN_V2)

    assert result.exit_code != 0
    assert "500" in str(result.output) + str(result.exception)
