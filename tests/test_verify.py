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


def test_help_states_the_api_spend_caveat():
    result = CliRunner().invoke(cli_module.main, ["verify", "--help"])
    assert result.exit_code == 0
    assert "does NOT stop YOUR function" in result.output
