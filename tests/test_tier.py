"""Tests for the tier recommender.

Covers the five scenarios called out in ``LOCAL_DEV_EXECUTION.md`` plus
the constraint-mismatch case that drives the confidence band (primary +
secondary) output.
"""

from __future__ import annotations

from papayya.dev._tier import TIERS, recommend


def _by_name(name: str) -> object:
    return next(t for t in TIERS if t.name == name)


class TestRecommender:
    def test_empty_workload_is_free(self) -> None:
        r = recommend(compute_min=0, peak_concurrency=0)
        assert r.primary.name == "free"
        assert r.secondary is None

    def test_modest_workload_is_starter(self) -> None:
        r = recommend(compute_min=300, peak_concurrency=3)
        assert r.primary.name == "starter"
        assert r.secondary is None

    def test_pro_shaped_workload(self) -> None:
        r = recommend(compute_min=2_500, peak_concurrency=15)
        assert r.primary.name == "pro"
        assert r.secondary is None

    def test_scale_shaped_workload(self) -> None:
        r = recommend(compute_min=10_000, peak_concurrency=60)
        assert r.primary.name == "scale"
        assert r.secondary is None

    def test_mismatch_recommends_higher_tier(self) -> None:
        """Compute fits Starter, concurrency demands Pro → primary is Pro."""
        r = recommend(compute_min=400, peak_concurrency=20)
        assert r.primary.name == "pro"
        assert r.secondary is not None
        assert r.secondary.name == "starter"
        assert "peak concurrency" in r.reason.lower()

    def test_over_scale_is_capped_at_scale(self) -> None:
        r = recommend(compute_min=50_000, peak_concurrency=500)
        assert r.primary.name == "scale"

    def test_to_dict_shape(self) -> None:
        r = recommend(compute_min=100, peak_concurrency=2)
        d = r.to_dict()
        assert "primary" in d and "reason" in d
        assert d["primary"]["name"] == "free"
        assert d["primary"]["price_usd_per_month"] == 0
