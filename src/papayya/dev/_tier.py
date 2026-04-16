"""Tier recommendation for the upgrade page.

Maps a user's last-30-day local workload shape to one of the hosted
tiers. Returns a confidence band (``primary`` + optional ``secondary``)
rather than a point value — if the user's peak concurrency fits Starter
but their compute minutes fit Pro, we tell them both and let them pick.
Over-recommending is a trust hit; under-recommending is worse, since
the cloud product stops working when the cap bites.

The thresholds below are the canonical pricing table from
``memory/pricing_model.md``. If pricing changes, update here and republish
the SDK — there is no runtime config path (deliberately, since the local
tool has no network dependency).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tier:
    name: str
    price_usd_per_month: int
    included_compute_min: int
    concurrent_runs: int

    def fits(self, compute_min: float, peak_concurrency: int) -> bool:
        return (
            compute_min <= self.included_compute_min
            and peak_concurrency <= self.concurrent_runs
        )


# Ordered cheapest -> most expensive. Keep in sync with pricing_model.md.
TIERS: tuple[Tier, ...] = (
    Tier("free",    0,   100,   2),
    Tier("starter", 49,  500,   5),
    Tier("pro",     199, 3_000, 25),
    Tier("scale",   799, 15_000, 100),
)


@dataclass(frozen=True)
class Recommendation:
    primary: Tier
    # Alternative the user could reasonably pick — e.g. if compute fits
    # Starter but peak concurrency demands Pro, surface both.
    secondary: Tier | None
    # Human-readable reason the recommender landed where it did.
    reason: str

    def to_dict(self) -> dict[str, object]:
        def tier_dict(t: Tier) -> dict[str, object]:
            return {
                "name": t.name,
                "price_usd_per_month": t.price_usd_per_month,
                "included_compute_min": t.included_compute_min,
                "concurrent_runs": t.concurrent_runs,
            }

        return {
            "primary": tier_dict(self.primary),
            "secondary": tier_dict(self.secondary) if self.secondary else None,
            "reason": self.reason,
        }


def recommend(
    compute_min: float,
    peak_concurrency: int,
) -> Recommendation:
    """Pick the cheapest tier that contains the workload.

    If two different constraints would push the user into two different
    tiers (e.g. compute fits Starter but concurrency fits Pro), we
    recommend the **more expensive** one as primary (so they don't hit
    a cap) and surface the cheaper one as secondary.
    """
    # Smallest tier whose compute ceiling covers the usage.
    compute_fit = next(
        (t for t in TIERS if compute_min <= t.included_compute_min),
        TIERS[-1],
    )
    # Smallest tier whose concurrency ceiling covers the usage.
    concurrency_fit = next(
        (t for t in TIERS if peak_concurrency <= t.concurrent_runs),
        TIERS[-1],
    )

    # Pick the more expensive of the two — cheaper would throttle.
    primary = compute_fit if compute_fit.price_usd_per_month >= concurrency_fit.price_usd_per_month else concurrency_fit
    other = concurrency_fit if primary is compute_fit else compute_fit
    secondary = other if other.name != primary.name else None

    if secondary is None:
        reason = (
            f"Your last 30 days fit the {primary.name} tier "
            f"(${primary.price_usd_per_month}/mo): "
            f"{compute_min:.0f} compute-minutes used of "
            f"{primary.included_compute_min} included, peak concurrency "
            f"{peak_concurrency} of {primary.concurrent_runs} allowed."
        )
    else:
        reason = (
            f"Your compute usage fits the {compute_fit.name} tier "
            f"(${compute_fit.price_usd_per_month}/mo), but peak concurrency "
            f"{peak_concurrency} requires {concurrency_fit.name} "
            f"(${concurrency_fit.price_usd_per_month}/mo). We recommend "
            f"{primary.name} so you don't hit a cap."
        )
    return Recommendation(primary=primary, secondary=secondary, reason=reason)
