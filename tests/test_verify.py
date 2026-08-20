"""Plan 41 R4 C8 — `papayya verify`, offline fixture verification.

The behaviours pinned here are the ones verify is worthless without: that it
reaches no store and no control plane, that "fixed" is computed by the same
inspector pipeline production used, that a fix which breaks a previously-fine
record is reported as NEWLY BROKEN rather than folded into a pass, and that a
fixture it could not answer is never counted as one that passed.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

import papayya
from papayya import cli as cli_module
from papayya.fixtures import (
    INPUT_FROM_FIRST_CHECKPOINT,
    INPUT_MISSING,
    Fixture,
    write_fixtures,
)
from papayya.verify import (
    FIXED,
    NEWLY_BROKEN,
    NO_VERDICT,
    SKIPPED_NO_INPUT,
    STILL_NOT_OK,
    STILL_OK,
    UNRESOLVED_AGENT,
    UNRUNNABLE_INPUT,
    VerifyError,
    verify,
    verify_fixtures,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """The @agent registry is process-global and `Item.init()` merges a
    registration's checks by agent NAME — so a registration left behind by one
    test would change another test's verdict. That is production behaviour,
    not a verify bug; the isolation belongs here."""
    from papayya.agent import _registry

    _registry.clear()
    yield
    _registry.clear()


def _fx(**over):
    base = dict(
        item_id="co_007",
        record_id="rec-1",
        agent="enrich",
        input={"n": 1},
        input_source="item_snapshot",
        recorded_output={"ok": False},
        worst_outcome_status="degraded",
    )
    base.update(over)
    return Fixture(**base)


def _handler(rows):
    """An iter-style body whose leaf step is recorded and inspected — the
    shape a customer actually has. `@papayya.step` is the rung-0 wrap."""

    @papayya.step
    def fetch(n):
        if isinstance(rows, BaseException):
            raise rows
        return rows

    def body(item):
        return fetch(item["n"])

    return body


# --- the verdict is production's, not a second opinion ----------------- #


def test_a_fix_that_works_reports_fixed():
    """The built-in inspectors run: an empty sequence is `degraded` in
    production, so a step that now returns rows is `fixed`."""
    summary = verify_fixtures([_fx()], handler=_handler([{"id": 1}]))
    assert [r.verdict for r in summary.results] == [FIXED]
    assert summary.ok


def test_a_fix_that_does_not_work_reports_still_not_ok_with_the_reason():
    """Still returning the empty shape means still degraded — and the reason
    comes from `outcomes.inspect_empty`, the same code the dashboard shows."""
    summary = verify_fixtures([_fx()], handler=_handler([]))
    result = summary.results[0]
    assert result.verdict == STILL_NOT_OK
    assert result.new_status == "degraded"
    assert result.new_reason == "empty_sequence"
    assert not summary.ok


def test_a_fix_that_breaks_a_clean_record_is_newly_broken_not_a_pass():
    """C7's column, at the fixture grain. A cohort pulled with --outcome any
    carries records that were fine; a fix that breaks one must not disappear
    into 'everything else passed'."""
    summary = verify_fixtures(
        [_fx(worst_outcome_status="ok", recorded_output=[{"id": 1}])],
        handler=_handler([]),
    )
    assert summary.results[0].verdict == NEWLY_BROKEN
    assert not summary.ok


def test_a_record_that_was_ok_and_stays_ok_is_still_ok():
    summary = verify_fixtures(
        [_fx(worst_outcome_status="ok")], handler=_handler([{"id": 1}])
    )
    assert summary.results[0].verdict == STILL_OK
    assert summary.ok


def test_a_raising_body_is_a_failure_not_a_crash():
    """`replay_slice`'s discipline: one record must not cost the operator the
    set. The exception is recorded with production's own reason for the mode
    being verified — `loop_body_exception` is what `papayya.iter` writes."""
    summary = verify_fixtures(
        [_fx(), _fx(record_id="rec-2")],
        handler=_handler(RuntimeError("still broken")),
    )
    assert len(summary.results) == 2
    first = summary.results[0]
    assert first.verdict == STILL_NOT_OK
    assert first.raised == "RuntimeError: still broken"
    assert first.new_status == "failed"
    assert first.new_reason == "loop_body_exception"


def test_an_at_agent_body_exception_carries_the_at_agent_reason(tmp_path):
    """The @agent path writes `agent_body_exception`, not
    `loop_body_exception`. Verify reproduces the reason of the path it is
    verifying, so a before/after comparison of outcome_reason lines up."""
    module = tmp_path / "raises.py"
    module.write_text(
        "import papayya\n"
        "@papayya.agent(name='enrich')\n"
        "def enrich(run, item):\n"
        "    run.step('fetch', lambda: [1])()\n"
        "    raise RuntimeError('nope')\n"
    )
    summary = verify_fixtures([_fx()], agent_module=str(module))
    assert summary.results[0].new_reason == "agent_body_exception"


def test_customer_checks_run_offline_too(tmp_path):
    """@agent(checks=...) is part of the production verdict, so it has to be
    part of verify's — otherwise 'fixed' means something weaker here."""
    module = tmp_path / "checked.py"
    module.write_text(
        "import papayya\n"
        "from papayya import checks\n"
        "def too_short(result):\n"
        "    return checks.degraded('too_short') if len(result) < 2 else None\n"
        "@papayya.agent(name='enrich', checks=[too_short])\n"
        "def enrich(run, item):\n"
        "    return run.step('fetch', lambda: [1])()\n"
    )
    summary = verify_fixtures([_fx()], agent_module=str(module))
    result = summary.results[0]
    assert result.verdict == STILL_NOT_OK
    assert result.new_reason == "user:too_short"


# --- it reaches nothing ------------------------------------------------- #


def test_verify_never_resolves_a_store(monkeypatch):
    """The load-bearing claim of C8. If verify ever fell back to the ambient
    store resolution it would try CloudStore and hit the network — so make
    that resolution explode and assert verify does not care."""
    from papayya.papayya import Papayya

    def explode(self):
        raise AssertionError("verify resolved an ambient store")

    monkeypatch.setattr(Papayya, "_auto_store", explode)
    monkeypatch.delenv("PAPAYYA_API_KEY", raising=False)
    monkeypatch.delenv("PAPAYYA_LOCAL_DB_PATH", raising=False)

    summary = verify_fixtures([_fx()], handler=_handler([{"id": 1}]))
    assert summary.results[0].verdict == FIXED


def test_verify_needs_no_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("PAPAYYA_API_KEY", raising=False)
    write_fixtures([_fx()], tmp_path)
    summary = verify(tmp_path, handler=_handler([{"id": 1}]))
    assert summary.ok


# --- what it could not answer is not a pass ----------------------------- #


def test_a_body_that_inspects_nothing_gets_no_verdict_not_a_pass():
    """A verdict in this product comes from STEPS, never from a return value.
    A body that runs none inspected nothing, so calling it `fixed` would ship
    the operator on an inspection that never ran."""
    summary = verify_fixtures([_fx()], handler=lambda item: [{"id": 1}])
    result = summary.results[0]
    assert result.verdict == NO_VERDICT
    assert "no inspected step" in (result.note or "")
    assert summary.answered == 0
    # `_worst([])` is "ok" — production's own default for a record with no
    # steps — but reporting `degraded -> ok` here would read as a fix.
    assert result.new_status is None
    assert not summary.ok


def test_a_fixture_with_no_input_is_skipped_not_passed():
    """A skipped fixture alongside a real answer is tolerated — pull already
    warned about it at write time — but it fails under --strict."""
    fixtures = [_fx(input=None, input_source=INPUT_MISSING), _fx(record_id="r2")]
    summary = verify_fixtures(fixtures, handler=_handler([{"id": 1}]))
    assert summary.results[0].verdict == SKIPPED_NO_INPUT
    assert summary.answered == 1
    assert summary.ok
    assert not verify_fixtures(
        fixtures, handler=_handler([{"id": 1}]), strict=True
    ).ok


def test_a_run_that_answered_nothing_is_never_a_pass():
    """"Verified" over a set where nothing ran is the silent partial success
    this product sells against. It fails without --strict."""
    summary = verify_fixtures(
        [_fx(input=None, input_source=INPUT_MISSING)],
        handler=_handler([{"id": 1}]),
    )
    assert summary.answered == 0
    assert not summary.ok


def test_an_input_that_does_not_fit_the_signature_is_not_reported_as_a_failed_fix():
    """A hosted fixture's input is a RECONSTRUCTION of the first step's bound
    args (C4). When it doesn't fit the agent's own parameters that is a
    fixture problem, and reporting it as 'your fix failed' would send the
    operator to debug the wrong thing."""
    def two_args(a, b):
        return [a, b]

    summary = verify_fixtures(
        [_fx(input={"n": 1}, input_source=INPUT_FROM_FIRST_CHECKPOINT)],
        handler=two_args,
    )
    result = summary.results[0]
    assert result.verdict == UNRUNNABLE_INPUT
    assert "RECONSTRUCTION" in (result.note or "")
    assert result.raised is None


def test_an_unknown_agent_is_reported_per_fixture_not_raised(tmp_path):
    """A cohort selected by predicate can span agents, so resolution is per
    fixture — one unknown name must not lose the rest of the set."""
    module = tmp_path / "spanning.py"
    module.write_text(
        "import papayya\n"
        "@papayya.agent(name='enrich')\n"
        "def enrich(run, item):\n"
        "    return run.step('fetch', lambda: [{'id': item['n']}])()\n"
    )
    summary = verify_fixtures(
        [_fx(), _fx(record_id="rec-2", agent="other")],
        agent_module=str(module),
    )
    assert summary.results[0].verdict == FIXED
    assert summary.results[1].verdict == UNRESOLVED_AGENT
    assert "other" in (summary.results[1].note or "")


# --- resolution mirrors replay ------------------------------------------ #


def test_handler_and_agent_module_are_mutually_exclusive():
    with pytest.raises(VerifyError, match="not both"):
        verify_fixtures([_fx()], handler=lambda i: i, agent_module="agent.py")


def test_an_injected_run_parameter_does_not_shift_the_binding(tmp_path):
    """`def f(run, item)` — the wrapper supplies `run`, so verify must bind
    the fixture's input to `item`. functools.wraps reports the wrapped
    signature, which still declares `run`; binding against it would put the
    payload in the wrong parameter."""
    module = tmp_path / "injected.py"
    module.write_text(
        "import papayya\n"
        "@papayya.agent(name='enrich')\n"
        "def enrich(run, item):\n"
        "    return run.step('fetch', lambda: [{'id': item['n']}])()\n"
    )
    summary = verify_fixtures([_fx()], agent_module=str(module))
    assert summary.results[0].verdict == FIXED


def test_a_clean_path_agent_body_is_verified_too(tmp_path):
    """`def f(item)` with an ambient leaf — the rung-0 shape, no `run`."""
    module = tmp_path / "clean.py"
    module.write_text(
        "import papayya\n"
        "@papayya.step\n"
        "def fetch(n):\n"
        "    return [{'id': n}]\n"
        "@papayya.agent(name='enrich')\n"
        "def enrich(item):\n"
        "    return fetch(item['n'])\n"
    )
    summary = verify_fixtures([_fx()], agent_module=str(module))
    assert summary.results[0].verdict == FIXED


def test_a_missing_agent_module_says_what_to_pass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(VerifyError, match="No agent.py in cwd"):
        verify_fixtures([_fx()])


# --- the version shift is reported, never refused ----------------------- #


def test_a_version_change_is_reported_not_refused(tmp_path):
    """The inverse of replay's gate. verify exists to run CHANGED code
    against an old failure — refusing on a version mismatch would refuse
    every real use of it."""
    module = tmp_path / "versioned.py"
    module.write_text(
        "import papayya\n"
        "@papayya.agent(name='enrich', agent_version='v2')\n"
        "def enrich(run, item):\n"
        "    return run.step('fetch', lambda: [{'id': item['n']}])()\n"
    )
    summary = verify_fixtures([_fx(agent_version="v1")], agent_module=str(module))
    result = summary.results[0]
    assert result.verdict == FIXED
    assert result.recorded_agent_version == "v1"
    assert result.current_agent_version == "v2"


# --- determinism, which is what makes a fixture a CI case --------------- #


def test_the_same_fixture_verifies_under_the_same_run_id_every_time():
    """`checks.run_checks` gates sampled checks on a hash of the run id. A
    fixture whose sampled judge fires on a different subset each run is not
    the permanent CI case D7b argues for."""
    seen = []

    @papayya.step
    def fetch(n):
        seen.append(papayya.active_run_id())
        return [{"id": n}]

    def body(item):
        return fetch(item["n"])

    for _ in range(2):
        verify_fixtures([_fx()], handler=body)
    assert seen[0] == seen[1], "run id must derive from the fixture, not be minted"


# --- output_changed says more than the verdict does --------------------- #


def test_an_unchanged_output_is_distinguished_from_a_changed_one():
    """Compared on the STEP TRACE, not the return value: `recorded_output` is
    the ITEM's output and on the clean @agent path nothing sets it from the
    return, so a return-vs-output comparison would report 'changed' on every
    fixture and never fire the signal that earns the field."""
    from papayya.fixtures import FixtureStep

    trace = [FixtureStep(label="fetch", outcome_status="degraded",
                         outcome_reason="empty_sequence",
                         input_snapshot={"n": 1}, output_snapshot=[])]
    unchanged = verify_fixtures([_fx(steps=trace)], handler=_handler([]))
    assert unchanged.results[0].output_changed is False
    changed = verify_fixtures([_fx(steps=trace)], handler=_handler([{"id": 1}]))
    assert changed.results[0].output_changed is True


def test_output_comparison_falls_back_to_the_return_when_there_is_no_trace():
    summary = verify_fixtures([_fx(recorded_output=[])], handler=_handler([]))
    assert summary.results[0].output_changed is False


def test_a_local_json_string_snapshot_is_decoded_before_binding():
    """The SQLite ledger keeps input_snapshot as a JSON string; the hosted API
    returns it decoded. A fixture can carry either."""
    summary = verify_fixtures(
        [_fx(input=json.dumps({"n": 1}))], handler=_handler([{"id": 1}])
    )
    assert summary.results[0].verdict == FIXED


def test_a_plain_string_input_stays_a_string():
    @papayya.step
    def shout(text):
        return [text.upper()]

    summary = verify_fixtures([_fx(input="hello")], handler=shout)
    assert summary.results[0].verdict == FIXED


# --- the trace is carried, because that is what a CI case needs --------- #


def test_the_new_step_trace_is_returned(tmp_path):
    module = tmp_path / "stepped.py"
    module.write_text(
        "import papayya\n"
        "@papayya.agent(name='enrich')\n"
        "def enrich(run, item):\n"
        "    return run.step('fetch', lambda: [])()\n"
    )
    summary = verify_fixtures([_fx()], agent_module=str(module))
    result = summary.results[0]
    assert [s["label"] for s in result.steps] == ["fetch"]
    assert result.steps[0]["outcome_status"] == "degraded"


# --- CLI ----------------------------------------------------------------- #


_WORKING_AGENT = (
    "import papayya\n"
    "@papayya.agent(name='enrich')\n"
    "def enrich(run, item):\n"
    "    return run.step('fetch', lambda: [{'id': item['n']}])()\n"
)
_BROKEN_AGENT = (
    "import papayya\n"
    "@papayya.agent(name='enrich')\n"
    "def enrich(run, item):\n"
    "    return run.step('fetch', lambda: [])()\n"
)


def _lay_out(tmp_path, body: str, fixtures):
    (tmp_path / "agent.py").write_text(body)
    write_fixtures(fixtures, tmp_path / "fixtures")
    return tmp_path


def test_cli_exits_zero_when_the_fix_holds(tmp_path, monkeypatch):
    _lay_out(tmp_path, _WORKING_AGENT, [_fx()])
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli_module.main, ["verify", "--fixtures", "fixtures"])
    assert result.exit_code == 0, result.output
    assert "FIXED" in result.output
    assert "1 fixed" in result.output


def test_cli_exits_nonzero_when_it_does_not(tmp_path, monkeypatch):
    _lay_out(tmp_path, _BROKEN_AGENT, [_fx(recorded_output=[])])
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli_module.main, ["verify", "--fixtures", "fixtures"])
    assert result.exit_code == 1
    assert "STILL NOT OK" in result.output
    # The fix never touched this path — a stronger statement than the verdict.
    assert "output is UNCHANGED" in result.output


def test_cli_json_mode_carries_the_whole_result(tmp_path, monkeypatch):
    _lay_out(tmp_path, _WORKING_AGENT, [_fx()])
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        cli_module.main, ["verify", "--fixtures", "fixtures", "--json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["counts"] == {FIXED: 1}
    assert payload["results"][0]["item_id"] == "co_007"


def test_cli_says_what_to_do_about_an_empty_directory(tmp_path, monkeypatch):
    (tmp_path / "fixtures").mkdir()
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli_module.main, ["verify", "--fixtures", "fixtures"])
    assert result.exit_code == 1
    assert "no fixtures found" in result.output.lower()


def test_help_states_the_side_effect_caveat_not_just_the_spend():
    """Plan 59 D4: the help led with "offline, no re-drive... nothing is sent"
    on a command measured writing 190 downstream rows in one invocation. It ran
    the customer's function, so it always did. The caveat it carried was about
    LLM tokens, which is the smaller exposure of the two."""
    result = CliRunner().invoke(cli_module.main, ["verify", "--help"])
    assert result.exit_code == 0
    assert "every side effect that function has still happens" in result.output
    assert "offline" not in result.output.lower()


def test_help_states_the_second_axis():
    result = CliRunner().invoke(cli_module.main, ["verify", "--help"])
    assert "did the inspectors change their mind" in result.output
    assert "did your code produce something different" in result.output


# ---------------------------------------------------------------------------
# Plan 48 R4: the CI gate could not fail on the commonest class of failure
# ---------------------------------------------------------------------------
#
# A record that RAISED before its first checkpoint keeps the
# worst_outcome_status='ok' it was born with while `status` says 'failed' — the
# commonest way a first batch fails, and the exact shape plan 48 walked into.
# `verify` read worst_outcome_status alone, so it called three crashed fixtures
# healthy, printed `[ok -> ok] · 0 fixed` on the fix that repaired them, and
# exited 0 under --strict, "the CI setting".
#
# The `status` these assert on was added to the fixture by plan 50 for this
# call site, and nothing consumed it until now.


def _crashed(**over):
    """The fixture shape a pre-first-checkpoint crash produces."""
    base = dict(status="failed", worst_outcome_status="ok",
                error="KeyError: 'body'", error_category="customer_code")
    base.update(over)
    return _fx(**base)


def test_a_repaired_crash_is_fixed_not_still_ok():
    summary = verify_fixtures([_crashed()], handler=_handler([{"a": 1}]),
                              strict=True)

    verdicts = [r.verdict for r in summary.results]
    assert verdicts == [FIXED], (
        f"a crashed fixture that now succeeds is FIXED; got {verdicts}. "
        "Reading worst_outcome_status alone reports STILL_OK — the fix that "
        "repaired it credited with fixing nothing."
    )


def test_a_crash_that_still_crashes_fails_the_gate():
    """The half that matters for CI: exit non-zero on a fix that didn't."""
    summary = verify_fixtures([_crashed()], handler=_handler(KeyError("body")),
                              strict=True)

    assert [r.verdict for r in summary.results] == [STILL_NOT_OK]
    assert not summary.ok, (
        "verify --strict must fail on a fix that did not repair the crash; "
        "it exited 0 on this for as long as the verdict came from the wrong "
        "column"
    )


def test_the_recorded_side_of_the_transition_says_failed():
    """`[ok -> ok]` on a crashed record was the printed form of the defect."""
    summary = verify_fixtures([_crashed()], handler=_handler([{"a": 1}]),
                              strict=True)

    assert summary.results[0].recorded_status == "failed"


def test_a_genuinely_healthy_record_is_still_ok():
    """The guard on the fix: a record that really was ok must not read as a
    crash just because both columns are now consulted."""
    summary = verify_fixtures(
        [_fx(status="completed", worst_outcome_status="ok")],
        handler=_handler([{"a": 1}]), strict=True)

    assert [r.verdict for r in summary.results] == [STILL_OK]
    assert summary.ok


# --- plan 64 D3: the second axis, and what --strict rests on ------------ #
#
# Plan 59 D3 measured `verify --strict` producing BYTE-IDENTICAL output and
# exit 0 for the broken code and the fix, on a cohort pulled with --flagged.
# It is structural: `pull --flagged` selects items the INSPECTORS called ok,
# verify re-derives the inspectors' verdict, so `ok -> ok` is the only verdict
# that cohort can return. `output_changed` was already computed and was read in
# exactly one branch. These pin the fact that it is now the answer.


def _flagged(**over):
    """A fixture off a `pull --flagged` — the cohort a human, not an inspector,
    said was wrong. No steps, so `output_changed` falls back to comparing the
    return value against `recorded_output`, which is what makes these readable."""
    over.setdefault("cohort", {"agent": "enrich", "flagged": True, "outcome": "any"})
    over.setdefault("worst_outcome_status", "ok")
    return _fx(**over)


def test_a_flagged_cohort_that_did_not_move_fails_strict():
    """The broken code, driven. Every verdict is `still_ok` and nothing moved,
    so there is no evidence the fix does anything — and a release would re-drive
    the cohort to reproduce exactly what was flagged."""
    summary = verify_fixtures(
        [_flagged(recorded_output=[{"id": 1}])],
        handler=_handler([{"id": 1}]),
        strict=True,
    )
    result = summary.results[0]
    assert result.verdict == STILL_OK
    assert result.output_changed is False
    assert result.stalled
    assert summary.moved == 0
    assert summary.stalled == 1
    assert not summary.ok


def test_a_flagged_cohort_that_moved_passes_strict():
    """The fix, driven. Same verdict, same statuses — the only difference in
    the entire result is that the answer changed, and that is the difference."""
    summary = verify_fixtures(
        [_flagged(recorded_output=[{"id": 9}])],
        handler=_handler([{"id": 1}]),
        strict=True,
    )
    result = summary.results[0]
    assert result.verdict == STILL_OK
    assert result.output_changed is True
    assert not result.stalled
    assert summary.moved == 1
    assert summary.ok


def test_an_unflagged_still_ok_record_that_did_not_move_still_passes():
    """The regression guard on the rule above. Outside a flagged cohort,
    `still ok` and unchanged is the DESIRED state — a record that was fine
    before and is fine now. Failing it would break every `--outcome not_ok`
    pull, which is the default."""
    summary = verify_fixtures(
        [_fx(worst_outcome_status="ok", recorded_output=[{"id": 1}])],
        handler=_handler([{"id": 1}]),
        strict=True,
    )
    assert summary.results[0].verdict == STILL_OK
    assert summary.results[0].output_changed is False
    assert not summary.results[0].stalled     # not flagged, so not stalled
    assert summary.stalled == 0
    assert summary.ok


def test_unknown_movement_is_not_counted_as_stalled():
    """`output_changed is None` means verify could not compare — a fixture that
    raised, or one with nothing to compare against. Counting unknown as "did
    not move" would fail a cohort on the absence of evidence, which is the
    silent wrong answer this product exists to catch."""
    summary = verify_fixtures([_flagged()], handler=_handler(RuntimeError("boom")))
    result = summary.results[0]
    assert result.output_changed is None
    assert not result.stalled
    assert summary.stalled == 0


def test_the_flag_is_carried_from_the_fixture_not_re_queried():
    """Every result carries the predicate that selected its fixture, including
    the ones verify could not answer — so a mixed directory cannot silently
    turn one pull's cohort into another's."""
    summary = verify_fixtures(
        [_flagged(record_id="a"), _fx(record_id="b", input=None,
                                      input_source=INPUT_MISSING)],
        handler=_handler([{"id": 1}]),
    )
    by_id = {r.record_id: r for r in summary.results}
    assert by_id["a"].flagged is True
    assert by_id["b"].flagged is False
    assert by_id["b"].verdict == SKIPPED_NO_INPUT
    assert summary.flagged == 1


# --- plan 64 D3, at the CLI: the two runs must not read the same --------- #

_DEPLOYED_AGENT = (
    "import papayya\n"
    "@papayya.agent(name='enrich')\n"
    "def enrich(run, item):\n"
    "    return run.step('fetch', lambda: [{'id': 1, 'damage': 'none'}])()\n"
)
_FIXED_AGENT = (
    "import papayya\n"
    "@papayya.agent(name='enrich')\n"
    "def enrich(run, item):\n"
    "    return run.step('fetch', lambda: [{'id': 1, 'damage': 'dent'}])()\n"
)
_FLAGGED_FIXTURE_OUTPUT = [{"id": 1, "damage": "none"}]


def _flagged_run(tmp_path, monkeypatch, body):
    tmp_path.mkdir(parents=True, exist_ok=True)
    _lay_out(tmp_path, body, [
        _fx(worst_outcome_status="ok",
            recorded_output=_FLAGGED_FIXTURE_OUTPUT,
            cohort={"agent": "enrich", "flagged": True, "outcome": "any"})
    ])
    monkeypatch.chdir(tmp_path)
    return CliRunner().invoke(
        cli_module.main, ["verify", "--fixtures", "fixtures", "--strict"]
    )


def test_cli_refuses_to_recommend_a_release_that_would_reproduce_the_complaint(
    tmp_path, monkeypatch
):
    """Plan 59 D3's exact scenario. The deployed code, verified against the
    cohort a human flagged: `ok -> ok`, nothing moved. The old build printed
    'Verified. Re-drive the cohort with `papayya release`.' and exited 0."""
    result = _flagged_run(tmp_path, monkeypatch, _DEPLOYED_AGENT)
    assert result.exit_code == 1, result.output
    assert "Nothing moved" in result.output
    assert "papayya release" not in result.output
    assert "0 of 1 answered fixture(s) produced a different answer" in result.output


def test_cli_verifies_a_flagged_cohort_the_fix_actually_moved(tmp_path, monkeypatch):
    result = _flagged_run(tmp_path, monkeypatch, _FIXED_AGENT)
    assert result.exit_code == 0, result.output
    assert "output CHANGED" in result.output
    assert "1 of 1 flagged item(s) produced a different answer" in result.output
    assert "papayya release" in result.output


def test_the_broken_code_and_the_fix_do_not_produce_the_same_output(
    tmp_path, monkeypatch
):
    """The measurement itself. `diff` of the two runs was empty; it is the one
    thing verify must never do, and the one thing it did."""
    a = _flagged_run(tmp_path / "a", monkeypatch, _DEPLOYED_AGENT)
    b = _flagged_run(tmp_path / "b", monkeypatch, _FIXED_AGENT)
    assert a.output != b.output
    assert a.exit_code != b.exit_code


# --- plan 64 D2: the receipt, and the rule that it never matters -------- #


def test_no_credentials_still_verifies_and_says_no_receipt(tmp_path, monkeypatch):
    """MEASURED FAILURE, PINNED. `_make_papayya_client` does not raise when
    there is no key — it ends the process. `except Exception` does not catch a
    SystemExit, so verify-with-no-credentials went from exit 0 to exit 1 with
    its own verdict already printed above the error.

    verify is the one recovery verb that works with no account at all: it reads
    fixtures off disk and runs local code. A receipt is a convenience for a
    colleague in a browser and does not get to take the command down."""
    def _no_key(scope):
        # Verbatim shape of what `_require_api_key` does: it ENDS THE PROCESS
        # rather than raising something a caller can see as an Exception.
        raise SystemExit(
            "No API key. Run `papayya login` to paste one, or set PAPAYYA_API_KEY"
        )

    monkeypatch.setattr(cli_module, "_require_api_key", _no_key)
    _lay_out(tmp_path, _WORKING_AGENT, [_fx()])
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli_module.main, ["verify", "--fixtures", "fixtures"])
    assert result.exit_code == 0, result.output
    assert "1 fixed" in result.output
    assert "No receipt recorded" in result.output
    assert "papayya login" in result.output       # the reason, not just the fact


def test_no_record_makes_no_call_at_all(tmp_path, monkeypatch):
    calls = []
    _capture_receipts(monkeypatch, calls)
    _lay_out(tmp_path, _WORKING_AGENT, [_fx()])
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        cli_module.main, ["verify", "--fixtures", "fixtures", "--no-record"]
    )
    assert result.exit_code == 0, result.output
    assert calls == []


class _StubClient:
    """Stands in for APIClient — the object `pull` and `release` are built on,
    and therefore the one `verify` posts its receipt through."""

    def __init__(self, sink):
        self.sink = sink

    def record_verification(self, payload):
        self.sink.append(payload)
        return payload

    def close(self):
        pass


def _capture_receipts(monkeypatch, sink):
    monkeypatch.setattr(cli_module, "APIClient", lambda config: _StubClient(sink))
    monkeypatch.setattr(cli_module, "_require_api_key", lambda scope: "cpk_test")


def test_the_receipt_carries_the_counts_and_the_predicate_and_nothing_else(
    tmp_path, monkeypatch
):
    """What crosses the wire is the whole security argument for the feature.
    A receipt carrying the values it compared would put a copy of the
    customer's extracted documents in a table nobody thinks of as holding
    documents, to render one sentence."""
    sent = []
    _capture_receipts(monkeypatch, sent)
    _lay_out(tmp_path, _WORKING_AGENT, [
        _fx(cohort={"agent": "enrich", "flagged": True, "outcome": "any"})
    ])
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli_module.main, ["verify", "--fixtures", "fixtures"])
    assert result.exit_code == 0, result.output

    payload = sent[0]
    assert payload["agent"] == "enrich"
    assert payload["cohort"] == {"agent": "enrich", "flagged": True, "outcome": "any"}
    assert payload["verdict"] == "passed"
    assert payload["fixtures"] == 1 and payload["fixed"] == 1
    assert payload["moved"] == 1 and payload["flagged"] == 1
    forbidden = {"input", "output", "steps", "results", "recorded_output"}
    assert forbidden.isdisjoint(payload), payload.keys()
    assert "Receipt recorded" in result.output


def test_a_failed_verification_is_recorded_too(tmp_path, monkeypatch):
    """'Someone ran this fix forty seconds ago and it did not hold' is the most
    useful thing the Release button can say. A door that only took passes would
    make it unsayable."""
    sent = []
    _capture_receipts(monkeypatch, sent)
    _lay_out(tmp_path, _BROKEN_AGENT, [_fx(recorded_output=[])])
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli_module.main, ["verify", "--fixtures", "fixtures"])
    assert result.exit_code == 1
    assert sent[0]["verdict"] == "failed"


def test_fixtures_from_two_pulls_record_no_receipt(tmp_path, monkeypatch):
    """Two pulls describe two cohorts, and a receipt claiming one of them would
    be rendered next to a Release button for work it did not cover."""
    sent = []
    _capture_receipts(monkeypatch, sent)
    _lay_out(tmp_path, _WORKING_AGENT, [
        _fx(record_id="a", cohort={"agent": "enrich", "flagged": True}),
        _fx(record_id="b", cohort={"agent": "enrich", "tenant": "acme"}),
    ])
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli_module.main, ["verify", "--fixtures", "fixtures"])
    assert result.exit_code == 0, result.output
    assert sent == []
    assert "more than one pull" in result.output


def test_the_confirmation_line_matches_the_verdict_it_stored(tmp_path, monkeypatch):
    """A failed verification is recorded on purpose, so confirming it with the
    word "verified" would put the summary line at odds with its own row."""
    sent = []
    _capture_receipts(monkeypatch, sent)
    _lay_out(tmp_path, _BROKEN_AGENT, [_fx(recorded_output=[])])
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli_module.main, ["verify", "--fixtures", "fixtures"])
    assert sent[0]["verdict"] == "failed"
    assert "did NOT verify" in result.output
    assert "can now say this cohort was verified" not in result.output
