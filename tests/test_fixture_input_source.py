"""Where a fixture's input comes from — plan 43 B2a step 5.

Migration 089 gave `durable_runs` an `input` column, and the cohort member wire
now carries it. Before that, `papayya pull` on the hosted path fell back to the
first checkpoint's `input_snapshot` — the bound args of the first durable step —
which a bare @agent function never writes at all. A fixture with no input cannot
be re-run by `papayya verify`, which is the entire point of pulling one.
"""

from papayya.fixtures import (
    INPUT_FROM_FIRST_CHECKPOINT,
    INPUT_FROM_ITEM,
    INPUT_FROM_SUBMISSION,
    INPUT_MISSING,
    fixture_from_record,
)


def test_the_submitted_input_wins_and_is_named_as_itself():
    fx = fixture_from_record(
        {"id": "r-1", "agent": "summarizer", "input": {"order_id": "co_42"}},
        [{"label": "summarize", "input_snapshot": {"reconstructed": True}}],
    )
    assert fx.input == {"order_id": "co_42"}
    assert fx.input_source == INPUT_FROM_SUBMISSION


def test_a_pre_089_record_still_reconstructs_from_the_first_step():
    fx = fixture_from_record(
        {"id": "r-1", "agent": "summarizer"},
        [{"label": "summarize", "input_snapshot": {"reconstructed": True}}],
    )
    assert fx.input == {"reconstructed": True}
    assert fx.input_source == INPUT_FROM_FIRST_CHECKPOINT


def test_the_local_ledgers_own_snapshot_is_unaffected():
    fx = fixture_from_record(
        {"id": "r-1", "agent": "summarizer", "input_snapshot": {"local": True}}, []
    )
    assert fx.input == {"local": True}
    assert fx.input_source == INPUT_FROM_ITEM


# The pre-089 bare-@agent case, which is what motivates preferring the column:
# no submission recorded and no step to reconstruct from.
def test_no_input_anywhere_is_named_rather_than_guessed():
    fx = fixture_from_record({"id": "r-1", "agent": "summarizer"}, [])
    assert fx.input is None
    assert fx.input_source == INPUT_MISSING


# A submitted input is legitimately falsy, and a truthiness test here would
# silently downgrade it to a reconstruction from an unrelated step.
def test_a_falsy_submitted_input_is_still_the_submitted_input():
    for value in (None, "", 0, False, {}, []):
        fx = fixture_from_record(
            {"id": "r-1", "agent": "summarizer", "input": value},
            [{"label": "s", "input_snapshot": {"wrong": True}}],
        )
        assert fx.input == value, value
        assert fx.input_source == INPUT_FROM_SUBMISSION, value
