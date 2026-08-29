"""
Deterministic compensation math. No LLM ever computes a pay figure (see CLAUDE.md
non-negotiable 1) -- these are the functions agents call instead.

Every function returns its inputs alongside its output (non-negotiable 2: provenance).
Callers should never need to reconstruct what was passed in from the result alone.
"""

from __future__ import annotations


def compute_pay_metrics(
    salary: float, range_min: float, range_mid: float, range_max: float
) -> dict:
    """Compa-ratio, range penetration, and cost to minimum for one salary against one range.

    compa_ratio: salary / range_mid. 1.0 means paid exactly at midpoint.
    range_penetration: position within [range_min, range_max] as a fraction, where
        0.0 = at minimum and 1.0 = at maximum. Can go negative (below min) or above 1.0
        (above max, e.g. a red-circled employee) -- both are valid results, not errors.
    cost_to_minimum: dollars needed to bring salary up to range_min; 0.0 if already there
        or above.
    """
    if range_mid == 0:
        raise ValueError("range_mid must be nonzero -- compa-ratio is undefined against a zero midpoint")
    if range_max <= range_min:
        raise ValueError(f"range_max ({range_max}) must be greater than range_min ({range_min})")

    compa_ratio = salary / range_mid
    range_penetration = (salary - range_min) / (range_max - range_min)
    cost_to_minimum = max(0.0, range_min - salary)

    return {
        "inputs": {
            "salary": salary,
            "range_min": range_min,
            "range_mid": range_mid,
            "range_max": range_max,
        },
        "compa_ratio": compa_ratio,
        "range_penetration": range_penetration,
        "cost_to_minimum": cost_to_minimum,
    }


def apply_geo_differential(national_midpoint: float, differential: float) -> dict:
    """Local-market midpoint = national midpoint x geo differential (e.g. 1.00 = national,
    0.34 = a lower-cost market). Differential is a fact from geo_locations, not derived here.
    """
    return {
        "inputs": {"national_midpoint": national_midpoint, "differential": differential},
        "local_midpoint": national_midpoint * differential,
    }


def interpolate_percentile(p25: float, p50: float, p75: float, p90: float, target_percentile: float) -> dict:
    """Linear interpolation between the two known percentile points (from survey_data's own
    p25/p50/p75/p90 columns) bracketing target_percentile. Needed because comp_philosophy.md's
    target percentiles (P60, and "target + 5 points" at L6+, e.g. P65) aren't columns
    survey_data carries directly. Never extrapolated: target_percentile outside [25, 90]
    raises rather than guessing past the edges of what the survey actually measured.
    """
    if not 25 <= target_percentile <= 90:
        raise ValueError(f"target_percentile must be within the surveyed range [25, 90]; got {target_percentile}")

    points = [(25, p25), (50, p50), (75, p75), (90, p90)]
    for (lo_p, lo_v), (hi_p, hi_v) in zip(points, points[1:]):
        if lo_p <= target_percentile <= hi_p:
            fraction = 0.0 if hi_p == lo_p else (target_percentile - lo_p) / (hi_p - lo_p)
            value = lo_v + fraction * (hi_v - lo_v)
            return {
                "inputs": {"p25": p25, "p50": p50, "p75": p75, "p90": p90, "target_percentile": target_percentile},
                "value": value,
                "bracket": (lo_p, hi_p),
            }

    raise AssertionError(f"target_percentile {target_percentile} not bracketed -- unreachable given the range check above")


def compute_pay_gap(current_pay: float, target_pay: float) -> dict:
    """The gap between what someone is paid now and a target figure (market target percentile
    for the cost model, or a level's range_mid for a retention award -- same math either way).

    gap: signed (target_pay - current_pay); negative means already at or above target.
    cost: max(0.0, gap) -- what it would actually cost to close the gap. Never negative --
    being paid above target isn't a "negative cost," it's simply nothing to fund.
    """
    gap = target_pay - current_pay
    return {
        "inputs": {"current_pay": current_pay, "target_pay": target_pay},
        "gap": gap,
        "cost": max(0.0, gap),
    }


def phase_amount(total: float, splits: list[float]) -> dict:
    """Split `total` across len(splits) phases (e.g. [0.5, 0.5] for 50/50 over 2 years, per
    comp_philosophy.md's phasing schedule). splits must sum to 1.0 (within floating-point
    tolerance) -- a schedule that doesn't fully fund the total, or overfunds it, is a bug in
    the caller's schedule, not a valid partial plan.
    """
    if not splits:
        raise ValueError("splits must not be empty")
    total_split = sum(splits)
    if abs(total_split - 1.0) > 1e-6:
        raise ValueError(f"splits must sum to 1.0; got {splits} summing to {total_split}")

    return {
        "inputs": {"total": total, "splits": splits},
        "phases": [{"phase": i + 1, "amount": total * s} for i, s in enumerate(splits)],
    }


def flag_underwater(compa_ratio: float, threshold: float) -> dict:
    """Whether compa_ratio falls below the retention-risk threshold (comp_philosophy.md /
    the comp manager's own policy -- this function applies whatever threshold it's given,
    it doesn't own the policy value itself)."""
    return {
        "inputs": {"compa_ratio": compa_ratio, "threshold": threshold},
        "underwater": compa_ratio < threshold,
    }


def age_market_value(value: float, months: float, annual_growth_rate: float) -> dict:
    """Project a market data point forward by `months` at a compounding `annual_growth_rate`.

    Aging is inherently forward-looking: `months` is how far past the data's own
    effective_date we're projecting it, so it must be >= 0. A negative value almost always
    means the caller swapped effective_date and as_of_date -- that's a bug to surface, not a
    result to compute (de-aging isn't a real operation this system needs).
    """
    if months < 0:
        raise ValueError(
            f"months must be >= 0 (aging is forward-only); got {months}. "
            "Check that effective_date and as_of_date weren't swapped."
        )

    aged_value = value * (1 + annual_growth_rate) ** (months / 12)

    return {
        "inputs": {"value": value, "months": months, "annual_growth_rate": annual_growth_rate},
        "aged_value": aged_value,
    }
