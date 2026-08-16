"""Plan 41 R4 C7 — `papayya release` and the cohort diff.

The diff is the answer the whole loop exists to give: did the fix work, on
which records, at what cost. What is pinned here is mostly what the diff
refuses to claim — a queued re-drive is not a recovery, a partial release is
not a whole one, and the record the fix BROKE is never folded into a pass.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from papayya import cli as cli_module
from papayya.cohort_diff import (
    NEWLY_BROKEN,
    PENDING,
    RECOVERED,
    STILL_NOT_OK,
    STILL_OK,
    diff_from_release,
    merge_runs,
    pending_run_ids,
)

COST_NOTE = "cost_usd is estimated from token counts x your project rate card"


def _member(source_id, new_run_id, item_id, outcome="degraded", cost=0.0135):
    return {
        "source_id": source_id,
        "new_run_id": new_run_id,
        "item_id": item_id,
        "agent": "enrich",
        "agent_version": "3",
        "source_outcome": outcome,
        "source_cost_usd": cost,
    }


def _release(*members, **over):
    body = {
        "released": len(members),
        "cohort_total": len(members),
        "members": list(members),
        "skipped_not_terminal": 0,
        "skipped_agent_missing": 0,
        "cost_note": COST_NOTE,
    }
    body.update(over)
    return body


def _run(status="completed", outcome="ok", cost=0.0140):
    return {
        "status": status,
        "worst_outcome_status": outcome,
        "budget_consumed_usd": cost,
    }


# --- what the diff refuses to claim ------------------------------------- #


def test_a_queued_redrive_is_pending_not_recovered():
    """`worst_outcome_status` on a fresh run is its initial 'ok'. Reading
    that as a recovery would report a fix that has not run yet."""
    diff = diff_from_release(_release(_member("s1", "n1", "co_1")),
                             {"n1": _run(status="queued", outcome="ok")})
    assert diff.records[0].verdict == PENDING
    assert pending_run_ids(diff) == ["n1"]


def test_a_record_never_polled_is_pending():
    diff = diff_from_release(_release(_member("s1", "n1", "co_1")))
    assert diff.records[0].verdict == PENDING


def test_a_redrive_that_worked_is_recovered():
    diff = diff_from_release(_release(_member("s1", "n1", "co_1")),
                             {"n1": _run(outcome="ok")})
    assert diff.records[0].verdict == RECOVERED
    assert diff.counts == {RECOVERED: 1}


def test_a_redrive_that_did_not_work_is_still_not_ok():
    diff = diff_from_release(_release(_member("s1", "n1", "co_1")),
                             {"n1": _run(outcome="degraded")})
    assert diff.records[0].verdict == STILL_NOT_OK


def test_a_failed_run_is_not_a_recovery_even_with_an_ok_outcome():
    """A run that ended 'failed' never got to a verdict worth trusting;
    worst_outcome_status can still read 'ok' if it died before its first
    step wrote."""
    diff = diff_from_release(_release(_member("s1", "n1", "co_1")),
                             {"n1": _run(status="failed", outcome="ok")})
    assert diff.records[0].verdict == STILL_NOT_OK


def test_newly_broken_is_the_column_that_earns_the_diff():
    """An operator who re-drives 500 records after a fix wants to know the
    fix did not break 30 that were fine. Only reachable via --outcome any,
    which is exactly when it matters."""
    diff = diff_from_release(
        _release(
            _member("s1", "n1", "co_1", outcome="degraded"),
            _member("s2", "n2", "co_2", outcome="ok"),
        ),
        {"n1": _run(outcome="ok"), "n2": _run(outcome="degraded")},
    )
    assert diff.records[0].verdict == RECOVERED
    assert diff.records[1].verdict == NEWLY_BROKEN


def test_a_clean_record_that_stays_clean_is_still_ok():
    diff = diff_from_release(_release(_member("s1", "n1", "co_1", outcome="ok")),
                             {"n1": _run(outcome="ok")})
    assert diff.records[0].verdict == STILL_OK


# --- the cost columns ---------------------------------------------------- #


def test_the_diff_totals_both_sides_of_the_cost():
    diff = diff_from_release(
        _release(
            _member("s1", "n1", "co_1", cost=0.01),
            _member("s2", "n2", "co_2", cost=0.02),
        ),
        {"n1": _run(cost=0.03), "n2": _run(cost=0.04)},
    )
    assert diff.before_cost_usd == pytest.approx(0.03)
    assert diff.after_cost_usd == pytest.approx(0.07)


def test_the_estimate_caveat_survives_the_pipeline():
    """The server sends it; losing it in the middle would leave the reader
    with a rate-card multiplication that looks like a bill."""
    diff = diff_from_release(_release(_member("s1", "n1", "co_1")))
    assert COST_NOTE in diff.cost_note
    assert COST_NOTE in diff.to_dict()["cost_note"]


# --- a partial release is never reported as a whole one ------------------ #


def test_the_named_skips_survive_into_the_diff():
    diff = diff_from_release(_release(
        _member("s1", "n1", "co_1"),
        cohort_total=4, skipped_not_terminal=2, skipped_agent_missing=1,
    ))
    assert diff.cohort_total == 4
    assert diff.skipped_not_terminal == 2
    assert diff.skipped_agent_missing == 1
    assert len(diff.records) == 1


def test_merging_polled_runs_updates_the_verdict_in_place():
    diff = diff_from_release(_release(_member("s1", "n1", "co_1")))
    assert diff.records[0].verdict == PENDING
    merge_runs(diff, [("n1", _run(outcome="ok", cost=0.02))])
    assert diff.records[0].verdict == RECOVERED
    assert diff.records[0].after_cost_usd == pytest.approx(0.02)
    assert pending_run_ids(diff) == []


def test_merging_an_unrelated_run_leaves_the_record_alone():
    diff = diff_from_release(_release(_member("s1", "n1", "co_1")))
    merge_runs(diff, [("someone-else", _run(outcome="ok"))])
    assert diff.records[0].verdict == PENDING


# --- CLI ----------------------------------------------------------------- #


class _FakeAPI:
    """Stands in for APIClient over the three calls `release` makes."""

    def __init__(self, total, release, runs=None, cohort_error=None,
                 release_error=None):
        self._total = total
        self._release = release
        self._runs = runs or {}
        self._cohort_error = cohort_error
        self._release_error = release_error
        self.released_with = None

    def get_cohort(self, **kw):
        if self._cohort_error:
            raise self._cohort_error
        return {"members": [], "total": self._total, "truncated": False}

    def release_cohort(self, **kw):
        if self._release_error:
            raise self._release_error
        self.released_with = kw
        return self._release

    def get_run(self, run_id):
        return self._runs[run_id]

    def close(self):
        pass


@pytest.fixture
def patched_api(monkeypatch):
    holder = {}

    def install(fake):
        holder["fake"] = fake
        monkeypatch.setattr(cli_module, "APIClient", lambda cfg: fake)
        monkeypatch.setattr(cli_module, "_require_api_key", lambda scope: "cpk_test")
        return fake

    return install


def _invoke(*args):
    return CliRunner().invoke(cli_module.main, ["release", *args])


def test_cli_prints_the_diff_and_names_the_newly_broken(patched_api):
    patched_api(_FakeAPI(
        total=2,
        release=_release(
            _member("s1", "n1", "co_1", outcome="degraded"),
            _member("s2", "n2", "co_2", outcome="ok"),
        ),
        runs={"n1": _run(outcome="ok"), "n2": _run(outcome="degraded")},
    ))
    result = _invoke("--agent", "enrich", "--outcome", "any", "-y")
    assert result.exit_code == 0, result.output
    assert "RECOVERED" in result.output
    assert "NEWLY BROKEN" in result.output
    assert "1 recovered" in result.output
    assert "were fine before this re-drive and are not now" in result.output
    # The caveat rides all the way to the terminal.
    assert COST_NOTE in result.output


def test_cli_refuses_to_act_on_an_empty_cohort(patched_api):
    fake = patched_api(_FakeAPI(total=0, release=_release()))
    result = _invoke("--agent", "enrich", "-y")
    assert result.exit_code == 0
    assert "Nothing to release" in result.output
    assert fake.released_with is None, "released against an empty cohort"


def test_cli_confirms_before_re_driving_production(patched_api):
    fake = patched_api(_FakeAPI(total=3, release=_release()))
    result = CliRunner().invoke(cli_module.main, ["release", "--agent", "enrich"],
                                input="n\n")
    assert fake.released_with is None, "re-drove production without confirmation"
    assert "Re-drive 3 record(s)" in result.output
    assert result.exit_code != 0


def test_cli_says_which_version_it_will_re_drive_on(patched_api):
    patched_api(_FakeAPI(total=3, release=_release()))
    original = CliRunner().invoke(
        cli_module.main, ["release", "--agent", "enrich"], input="n\n")
    assert "the version each record originally ran" in original.output
    latest = CliRunner().invoke(
        cli_module.main, ["release", "--agent", "enrich", "--latest"], input="n\n")
    assert "the agent's CURRENT version" in latest.output


def test_cli_passes_latest_through(patched_api):
    fake = patched_api(_FakeAPI(total=1, release=_release(_member("s1", "n1", "co_1")),
                                runs={"n1": _run()}))
    _invoke("--agent", "enrich", "--latest", "-y")
    assert fake.released_with["latest"] is True


def test_cli_surfaces_a_quota_shortfall_as_an_error(patched_api):
    from papayya.api import PapayyaAPIError

    patched_api(_FakeAPI(
        total=500, release=None,
        release_error=PapayyaAPIError(
            429, "cohort is 500 releasable record(s); only 120 trigger "
                 "reservation(s) remain. Nothing was released."),
    ))
    result = _invoke("--agent", "enrich", "-y")
    assert result.exit_code == 1
    assert "Nothing was released" in result.output
    assert "only 120" in result.output


def test_cli_names_what_the_release_skipped(patched_api):
    patched_api(_FakeAPI(
        total=4,
        release=_release(_member("s1", "n1", "co_1"),
                         cohort_total=4, skipped_not_terminal=2,
                         skipped_agent_missing=1),
        runs={"n1": _run(outcome="ok")},
    ))
    result = _invoke("--agent", "enrich", "-y")
    assert "Released 1 of 4 record(s)" in result.output
    assert "still running and were not re-driven" in result.output
    assert "no agent to route to" in result.output


def test_cli_no_wait_returns_the_manifest_without_polling(patched_api):
    class _NoPoll(_FakeAPI):
        def get_run(self, run_id):
            raise AssertionError("polled a run under --no-wait")

    patched_api(_NoPoll(total=1, release=_release(_member("s1", "n1", "co_1"))))
    result = _invoke("--agent", "enrich", "-y", "--no-wait")
    assert result.exit_code == 0
    assert "…running" in result.output


def test_cli_json_mode_carries_the_whole_diff(patched_api):
    patched_api(_FakeAPI(
        total=1, release=_release(_member("s1", "n1", "co_1")),
        runs={"n1": _run(outcome="ok", cost=0.02)},
    ))
    result = _invoke("--agent", "enrich", "-y", "--json")
    assert result.exit_code == 0
    # --json means JSON and nothing else on stdout — caught by piping the
    # real command into a parser against the live stack.
    payload = json.loads(result.stdout)
    assert payload["counts"] == {RECOVERED: 1}
    assert payload["after_cost_usd"] == pytest.approx(0.02)
    assert COST_NOTE in payload["cost_note"]


def test_release_is_registered_in_the_help_tiers():
    result = CliRunner().invoke(cli_module.main, ["--help"])
    assert "Other commands:" not in result.output
    assert "release" in result.output


def test_a_pending_record_shows_no_after_verdict_at_all():
    """Caught by running release against the live stack: the diff printed
    "degraded -> ok" beside "…running", which reads as a recovery that had
    not happened. A queued run's worst_outcome_status is its initial 'ok',
    so it is not shown until the run is terminal."""
    diff = diff_from_release(_release(_member("s1", "n1", "co_1")),
                             {"n1": _run(status="queued", outcome="ok")})
    assert diff.records[0].verdict == PENDING
    assert diff.records[0].after_status is None
    merge_runs(diff, [("n1", _run(status="running", outcome="ok"))])
    assert diff.records[0].after_status is None
    merge_runs(diff, [("n1", _run(status="completed", outcome="ok"))])
    assert diff.records[0].after_status == "ok"


def test_cli_json_mode_puts_nothing_but_json_on_stdout(patched_api):
    """Caught by piping the real command into a parser against the live
    stack: a "Released 3 of 3" progress line ahead of the payload made
    --json unparseable for the caller who asked for it."""
    patched_api(_FakeAPI(
        total=4,
        release=_release(_member("s1", "n1", "co_1"),
                         cohort_total=4, skipped_not_terminal=2,
                         skipped_agent_missing=1),
        runs={"n1": _run(outcome="ok")},
    ))
    # click>=8.2 separates stdout/stderr by default and removed mix_stderr=.
    result = CliRunner().invoke(
        cli_module.main, ["release", "--agent", "enrich", "-y", "--json"])
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["released"] == 1
    # The skips still get said — on stderr, where they don't corrupt the feed.
    assert "still running and were not re-driven" in result.stderr
