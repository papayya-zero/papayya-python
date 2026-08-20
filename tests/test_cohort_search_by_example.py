"""Plan 39 S3 — search by example, at the SDK edge (ADR 0009 §4 step 3).

*"Paste one known-bad record, get everything like it across the last 60 days.
Writing a predicate against history is the power-user path; pointing at a bad
one is what actually happens at 2am."*

Most of what is pinned here is the THREE-DOORS rule. A cohort's selection comes
from exactly one of: a frozen by-example probe, a drift episode, or the
descriptive terms. The server parses all three with one shared function so an
operator cannot approve one cohort and re-drive a different one — and a client
that re-encoded the rule per method would hand that guarantee back at the edge.
`api.py` had two copies of the params loop before this unit and would have had
six after, so there is now exactly one builder and these tests are pointed at it.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from papayya import cli as cli_module
from papayya.api import APIClient

params = APIClient._cohort_params


# --- the three doors ---------------------------------------------------------

def test_descriptive_terms_travel_as_themselves():
    assert params(
        agent="enrich", tenant="acme", run_id="r1", outcome="any",
        since="2026-08-01T00:00:00Z", until="2026-08-02T00:00:00Z", limit=5,
    ) == {
        "agent": "enrich", "tenant": "acme", "run_id": "r1", "outcome": "any",
        "from": "2026-08-01T00:00:00Z", "to": "2026-08-02T00:00:00Z", "limit": "5",
    }


def test_a_probe_travels_alone():
    """Every term a probe stands for — bare label, window, condition token,
    borrowed floor — stays server-side on the cohort_probes row. Emitting them
    for the client to pass back would make the preview and the re-drive two
    client-built selections that can disagree, and would put a server-learned
    floor on the wire for a verb that re-executes work."""
    assert params(probe="p-1") == {"probe": "p-1"}
    assert params(drift_episode="e-1") == {"drift_episode": "e-1"}


@pytest.mark.parametrize("selector", [{"probe": "p-1"}, {"drift_episode": "e-1"}])
@pytest.mark.parametrize("term", [
    {"agent": "enrich"}, {"tenant": "acme"}, {"run_id": "r1"}, {"outcome": "any"},
])
def test_a_selector_and_a_description_are_refused_not_merged(selector, term):
    """Someone passing both has two different selections in mind, and quietly
    letting one win is how an operator approves one cohort and re-drives
    another. Refused client-side so it never travels; the server refuses it
    too — this is the earlier, better-worded copy, not the authority."""
    with pytest.raises(ValueError, match="selects its own items"):
        params(**selector, **term)


def test_the_two_selectors_are_refused_together():
    with pytest.raises(ValueError, match="not both"):
        params(probe="p-1", drift_episode="e-1")


# --- the one asymmetry between the two selectors -----------------------------

def test_since_until_NARROW_a_probe_but_describe_a_drift_selection():
    """A by-example probe is unbounded by construction where a drift episode
    inherits its own window. The release cap (1000 records) is reachable on
    real cohorts, and the server's own refusal says "narrow the window
    (from/to) and release in passes" — so without this the advice would name a
    flag that did not exist. They can only shrink the frozen window, never
    widen it, so the preview and the release still cannot disagree in the
    direction that matters."""
    assert params(probe="p-1", since="2026-08-01T00:00:00Z") == {
        "probe": "p-1", "from": "2026-08-01T00:00:00Z",
    }
    with pytest.raises(ValueError, match="remove since"):
        params(drift_episode="e-1", since="2026-08-01T00:00:00Z")


@pytest.mark.parametrize("selector", [{"probe": "p-1"}, {"drift_episode": "e-1"}])
def test_paging_and_the_triage_filter_ride_along_with_any_door(selector):
    """Neither describes WHICH records the predicate picks out — one re-admits
    work someone already dispositioned, the other pages the response."""
    out = params(**selector, include_triaged=True, limit=10)
    assert out["include_triaged"] == "true" and out["limit"] == "10"


# --- the CLI ------------------------------------------------------------------

class _FakeAPI:
    """Records what the CLI asked for. Nothing here talks to a server."""

    def __init__(self, derive=None, cohort=None):
        self._derive = derive or {
            "record": "r-bad", "agent": "enrich",
            "from": "2026-06-14T00:00:00Z", "to": "2026-08-13T00:00:00Z",
            "proposals": [{
                "probe": "pr-1", "label": "extract", "condition": "field_blank",
                "field_key": "summary", "count": 1847,
                "why": 'every record whose "extract" step returned no "summary"',
            }],
            "refusals": [{
                "label": "extract", "condition": "below_size_floor",
                "reason": "the size_floor baseline for this step is still forming "
                          "(4 reference / 2 recent records)",
            }],
            "window_note": "the window is a default, not a retention guarantee",
        }
        self._cohort = cohort or {"members": [], "total": 0, "truncated": False}
        self.derive_calls: list = []
        self.cohort_calls: list = []

    def derive_probes(self, record, *, since=None, until=None):
        self.derive_calls.append((record, since, until))
        return self._derive

    def get_cohort(self, **kw):
        # DELEGATES TO THE REAL BUILDER rather than accepting anything. A double
        # that skipped it would stub out the exact rule these tests exist to
        # assert, and `--probe --agent` would silently "pass".
        params(**kw)
        self.cohort_calls.append(kw)
        return self._cohort

    def close(self):
        pass


@pytest.fixture
def fake_api(monkeypatch):
    api = _FakeAPI()
    monkeypatch.setattr(cli_module, "APIClient", lambda config: api)
    monkeypatch.setenv("PAPAYYA_API_KEY", "cpk_00112233_" + "0" * 32)
    return api


def _run(*args):
    return CliRunner().invoke(cli_module.main, ["--base-url", "http://x", *args])


def test_like_reports_the_blast_radius_and_writes_nothing(fake_api, tmp_path):
    res = _run("pull", "--like", "r-bad", "--out", str(tmp_path))
    assert res.exit_code == 0, res.output
    # The count is what makes a proposal something to decide about.
    assert "1,847" in res.output
    assert "pr-1" in res.output
    # It derives; it does not select, and it does not write.
    assert fake_api.derive_calls == [("r-bad", None, None)]
    assert fake_api.cohort_calls == []
    assert not list(tmp_path.iterdir())


def test_like_prints_refusals_rather_than_swallowing_them(fake_api):
    """"Your baseline is still forming" and "this record looks fine on that
    axis" are completely different answers, and only one of them means come
    back later. A silently missing proposal is indistinguishable from a
    condition the record did not satisfy."""
    res = _run("pull", "--like", "r-bad")
    assert "below_size_floor" in res.output
    assert "still forming" in res.output


def test_like_says_so_plainly_when_no_predicate_fits(monkeypatch):
    """A real answer, not an error: the platform's vocabulary has no word for
    what is wrong with this output. ADR 0009 §5 — it never says "correct"."""
    api = _FakeAPI(derive={
        "record": "r", "agent": "enrich", "from": "a", "to": "b",
        "proposals": [], "refusals": [],
    })
    monkeypatch.setattr(cli_module, "APIClient", lambda config: api)
    monkeypatch.setenv("PAPAYYA_API_KEY", "cpk_00112233_" + "0" * 32)
    res = _run("pull", "--like", "r")
    assert res.exit_code == 0
    assert "No predicate fits this item" in res.output


def test_like_and_probe_together_are_refused(fake_api):
    """One derives a predicate, the other uses one."""
    res = _run("pull", "--like", "r-bad", "--probe", "pr-1")
    assert res.exit_code == 1
    assert "one at a time" in res.output


def test_probe_reaches_the_cohort_verb_as_a_probe(fake_api):
    res = _run("pull", "--probe", "pr-1")
    assert res.exit_code == 0
    assert fake_api.cohort_calls[0]["probe"] == "pr-1"


def test_empty_cohort_advice_matches_the_door(fake_api):
    """On a derived predicate the window is frozen server-side and --outcome is
    refused outright, so the hand-written advice would name two flags that
    cannot help. An empty cohort in a silent-failure product is exactly the
    moment not to send someone down a dead end."""
    res = _run("pull", "--probe", "pr-1")
    assert "selects its own window" in res.output
    assert "--outcome any" not in res.output

    res = _run("pull", "--agent", "enrich")
    assert "--outcome any" in res.output
    assert "selects its own window" not in res.output


def test_the_doors_are_refused_at_the_cli_too(fake_api):
    res = _run("pull", "--probe", "pr-1", "--agent", "enrich")
    assert res.exit_code == 1
    assert "selects its own items" in res.output
