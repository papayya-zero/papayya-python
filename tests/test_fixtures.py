"""Plan 41 R4 — cohort fixtures (ADR 0009 D7b).

The behaviours pinned here are the ones a fixture set is worthless without:
that it says where its input came from, that it survives a round trip, and
that it refuses a version it does not understand rather than mis-reading it.
"""

from __future__ import annotations

import json

import pytest

from papayya.fixtures import (
    FIXTURE_VERSION,
    INPUT_FROM_FIRST_CHECKPOINT,
    INPUT_FROM_ITEM,
    INPUT_MISSING,
    Fixture,
    FixtureError,
    fixture_from_record,
    read_fixtures,
    write_fixtures,
)


def _record(**over):
    base = {
        "id": "rec-1",
        "item_id": "co_007",
        "agent": "enrich",
        "worst_outcome_status": "degraded",
        "partition_key": "acme",
        "output": {"final": True},
        "created_at": "2026-08-06T00:00:00Z",
    }
    base.update(over)
    return base


def _ckpt(label, **over):
    base = {
        "label": label,
        "attempt": 1,
        "outcome_status": "ok",
        "input_snapshot": {"arg": label},
        "output_snapshot": {"out": label},
        "completed_at": "2026-08-06T00:00:00Z",
        "seq": 1,
    }
    base.update(over)
    return base


# --- where the input comes from --------------------------------------- #


def test_local_record_input_comes_from_its_own_snapshot():
    """Local: the items table keeps its own input_snapshot."""
    fx = fixture_from_record(
        _record(input_snapshot={"the": "request"}), [_ckpt("fetch")]
    )
    assert fx.input == {"the": "request"}
    assert fx.input_source == INPUT_FROM_ITEM


def test_hosted_record_input_is_reconstructed_from_the_first_checkpoint():
    """Hosted: durable_runs has no input column, so the first step's bound
    args are the closest recorded thing. It is a RECONSTRUCTION and the
    fixture must say so — a fixture that silently claimed to be the original
    request would be the same class of lie R2 spent a step removing."""
    fx = fixture_from_record(_record(), [_ckpt("fetch"), _ckpt("score", seq=2)])
    assert fx.input == {"arg": "fetch"}
    assert fx.input_source == INPUT_FROM_FIRST_CHECKPOINT


def test_missing_input_is_named_not_invented():
    fx = fixture_from_record(_record(), [])
    assert fx.input is None
    assert fx.input_source == INPUT_MISSING


def test_first_checkpoint_is_by_execution_order_not_list_order():
    """A journaled write drains late, so it arrives AFTER the execution that
    replaced it and lands at a higher seq. Ordering on arrival — or on seq —
    picks the wrong step as 'first'. Plan 41 R3's ordering, reused."""
    late_drained_first_step = _ckpt(
        "fetch", completed_at="2026-08-06T00:00:00Z", seq=99,
        input_snapshot={"arg": "the-real-first"},
    )
    later_step = _ckpt(
        "score", completed_at="2026-08-06T00:05:00Z", seq=10,
        input_snapshot={"arg": "later"},
    )
    fx = fixture_from_record(_record(), [later_step, late_drained_first_step])
    assert fx.input == {"arg": "the-real-first"}


def test_recorded_output_prefers_the_items_own_output():
    fx = fixture_from_record(_record(output={"final": True}), [_ckpt("score")])
    assert fx.recorded_output == {"final": True}


def test_recorded_output_falls_back_to_the_last_step():
    fx = fixture_from_record(_record(output=None), [_ckpt("a"), _ckpt("b", seq=2)])
    assert fx.recorded_output == {"out": "b"}


# --- the trace ---------------------------------------------------------- #


def test_fixture_carries_the_whole_trace_not_just_the_boundary():
    """D7b's argument is that a fixture becomes a permanent CI case, and a
    boundary-only fixture cannot catch a regression that changes WHICH STEPS
    RUN. So the trace is carried."""
    fx = fixture_from_record(
        _record(),
        [_ckpt("fetch", outcome_status="degraded", outcome_reason="empty_sequence"),
         _ckpt("score", seq=2)],
    )
    assert [s.label for s in fx.steps] == ["fetch", "score"]
    assert fx.steps[0].outcome_status == "degraded"
    assert fx.steps[0].outcome_reason == "empty_sequence"


# --- disk --------------------------------------------------------------- #


def test_round_trips_through_disk(tmp_path):
    fx = fixture_from_record(_record(), [_ckpt("fetch")], cohort={"agent": "enrich"})
    paths = write_fixtures([fx], tmp_path)
    assert len(paths) == 1

    back = read_fixtures(tmp_path)
    assert len(back) == 1
    assert back[0].item_id == "co_007"
    assert back[0].input == {"arg": "fetch"}
    assert back[0].input_source == INPUT_FROM_FIRST_CHECKPOINT
    assert back[0].cohort == {"agent": "enrich"}
    assert [s.label for s in back[0].steps] == ["fetch"]


def test_two_records_sharing_one_item_id_do_not_overwrite_each_other(tmp_path):
    """A retry of the same customer record is a second row. Naming files on
    item_id alone would silently drop one half of the incident."""
    a = fixture_from_record(_record(id="rec-a"), [_ckpt("fetch")])
    b = fixture_from_record(_record(id="rec-b"), [_ckpt("fetch")])
    write_fixtures([a, b], tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_filename_is_greppable_by_the_customers_own_id(tmp_path):
    fx = fixture_from_record(_record(item_id="co_007"), [])
    assert fx.filename().startswith("co_007.")


def test_unsafe_item_id_cannot_escape_the_output_directory(tmp_path):
    fx = fixture_from_record(_record(item_id="../../etc/passwd"), [])
    paths = write_fixtures([fx], tmp_path)
    assert paths[0].parent == tmp_path


def test_a_future_fixture_version_is_refused_not_misread(tmp_path):
    raw = json.loads(fixture_from_record(_record(), []).to_json())
    raw["papayya_fixture_version"] = FIXTURE_VERSION + 1
    (tmp_path / "future.json").write_text(json.dumps(raw))
    with pytest.raises(FixtureError, match="version"):
        read_fixtures(tmp_path)


def test_empty_directory_raises_rather_than_reporting_success(tmp_path):
    with pytest.raises(FixtureError):
        read_fixtures(tmp_path)


def test_partial_write_leaves_no_readable_fixture(tmp_path, monkeypatch):
    """Writes are atomic: an interrupted pull must leave whole fixtures or
    none. A half-written fixture that still parses is worse than a missing
    one, because it looks like evidence."""
    good = fixture_from_record(_record(id="rec-good"), [])

    class Boom(Fixture):
        def to_json(self):  # noqa: D102
            raise RuntimeError("interrupted")

    bad = Boom(**{**good.__dict__, "record_id": "rec-bad"})
    with pytest.raises(RuntimeError):
        write_fixtures([good, bad], tmp_path)

    # The good one landed whole; the bad one left nothing behind — no temp
    # files, no truncated JSON.
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert not list(tmp_path.glob("*.tmp"))
    assert read_fixtures(tmp_path)[0].record_id == "rec-good"
