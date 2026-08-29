import pandas as pd
import pytest

from tools.comp_math import (
    age_market_value,
    apply_geo_differential,
    compute_pay_gap,
    compute_pay_metrics,
    flag_underwater,
    interpolate_percentile,
    phase_amount,
)
from tools.currency import convert_currency


# ---------------------------------------------------------------- happy path (the four asks)

def test_compa_ratio():
    result = compute_pay_metrics(salary=100_000, range_min=90_000, range_mid=125_000, range_max=160_000)
    assert result["compa_ratio"] == pytest.approx(0.80)


def test_range_penetration_at_minimum_is_zero():
    result = compute_pay_metrics(salary=90_000, range_min=90_000, range_mid=125_000, range_max=160_000)
    assert result["range_penetration"] == pytest.approx(0.0)


def test_age_market_value_18_months_at_3_5_percent():
    result = age_market_value(value=150_000, months=18, annual_growth_rate=0.035)
    assert result["aged_value"] == pytest.approx(158_000, rel=0.001)


def test_geo_differential_034_on_200k():
    result = apply_geo_differential(national_midpoint=200_000, differential=0.34)
    assert result["local_midpoint"] == pytest.approx(68_000)


# ------------------------------------------------------------------------- provenance

def test_compute_pay_metrics_echoes_inputs():
    result = compute_pay_metrics(salary=100_000, range_min=90_000, range_mid=125_000, range_max=160_000)
    assert result["inputs"] == {
        "salary": 100_000, "range_min": 90_000, "range_mid": 125_000, "range_max": 160_000,
    }


# --------------------------------------------------------------------------- edge cases

def test_zero_midpoint_raises():
    with pytest.raises(ValueError, match="range_mid"):
        compute_pay_metrics(salary=100_000, range_min=90_000, range_mid=0, range_max=160_000)


def test_degenerate_range_raises():
    with pytest.raises(ValueError, match="range_max"):
        compute_pay_metrics(salary=100_000, range_min=100_000, range_mid=100_000, range_max=100_000)


def test_salary_above_maximum_is_not_an_error():
    result = compute_pay_metrics(salary=180_000, range_min=90_000, range_mid=125_000, range_max=160_000)
    assert result["range_penetration"] == pytest.approx((180_000 - 90_000) / (160_000 - 90_000))
    assert result["range_penetration"] > 1.0
    assert result["cost_to_minimum"] == 0.0


def test_negative_aging_period_raises():
    with pytest.raises(ValueError, match="months must be >= 0"):
        age_market_value(value=150_000, months=-6, annual_growth_rate=0.035)


def test_zero_aging_period_is_a_noop():
    result = age_market_value(value=150_000, months=0, annual_growth_rate=0.035)
    assert result["aged_value"] == pytest.approx(150_000)


# --------------------------------------------------------------------- convert_currency

@pytest.fixture
def fx_rates():
    return pd.DataFrame([
        {"from_currency": "USD", "to_currency": "INR", "rate": 83.0, "rate_month": pd.Timestamp("2026-08-01")},
        {"from_currency": "INR", "to_currency": "USD", "rate": 1 / 83.0, "rate_month": pd.Timestamp("2026-08-01")},
    ])


def test_convert_currency_happy_path(fx_rates):
    result = convert_currency(1_000, "USD", "INR", "2026-08-15", fx_rates)
    assert result["converted_amount"] == pytest.approx(83_000)
    assert result["rate"] == pytest.approx(83.0)
    assert result["rate_month"] == pd.Timestamp("2026-08-01")
    assert len(result["source_rows"]) == 1


def test_convert_currency_same_currency_is_identity(fx_rates):
    result = convert_currency(500, "USD", "USD", "2026-08-15", fx_rates)
    assert result["converted_amount"] == 500
    assert result["rate"] == 1.0


def test_convert_currency_missing_rate_raises(fx_rates):
    with pytest.raises(ValueError, match="No FX rate"):
        convert_currency(1_000, "USD", "INR", "2026-01-15", fx_rates)


def test_convert_currency_missing_pair_raises(fx_rates):
    with pytest.raises(ValueError, match="No FX rate"):
        convert_currency(1_000, "USD", "EUR", "2026-08-15", fx_rates)


# ------------------------------------------------------------------- interpolate_percentile

def test_interpolate_percentile_at_a_known_point_returns_that_point():
    result = interpolate_percentile(p25=100, p50=150, p75=200, p90=250, target_percentile=50)
    assert result["value"] == pytest.approx(150)


def test_interpolate_percentile_p60_is_40_percent_of_the_way_from_p50_to_p75():
    result = interpolate_percentile(p25=100, p50=150, p75=200, p90=250, target_percentile=60)
    assert result["value"] == pytest.approx(150 + 0.4 * (200 - 150))
    assert result["bracket"] == (50, 75)


def test_interpolate_percentile_p65_for_the_l6_plus_target_plus_5_rule():
    # comp_philosophy.md: "L6 and above, any family: target + 5 points" -- P60 base -> P65.
    result = interpolate_percentile(p25=100, p50=150, p75=200, p90=250, target_percentile=65)
    assert result["value"] == pytest.approx(150 + 0.6 * (200 - 150))


def test_interpolate_percentile_below_25_raises():
    with pytest.raises(ValueError, match="surveyed range"):
        interpolate_percentile(p25=100, p50=150, p75=200, p90=250, target_percentile=10)


def test_interpolate_percentile_above_90_raises():
    with pytest.raises(ValueError, match="surveyed range"):
        interpolate_percentile(p25=100, p50=150, p75=200, p90=250, target_percentile=95)


# ------------------------------------------------------------------------- compute_pay_gap

def test_compute_pay_gap_below_target_has_positive_cost():
    result = compute_pay_gap(current_pay=100_000, target_pay=120_000)
    assert result["gap"] == pytest.approx(20_000)
    assert result["cost"] == pytest.approx(20_000)


def test_compute_pay_gap_above_target_has_zero_cost_not_negative():
    result = compute_pay_gap(current_pay=130_000, target_pay=120_000)
    assert result["gap"] == pytest.approx(-10_000)
    assert result["cost"] == 0.0


# --------------------------------------------------------------------------- phase_amount

def test_phase_amount_50_50_over_two_years():
    result = phase_amount(total=100_000, splits=[0.5, 0.5])
    assert result["phases"] == [{"phase": 1, "amount": 50_000.0}, {"phase": 2, "amount": 50_000.0}]


def test_phase_amount_splits_must_sum_to_one():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        phase_amount(total=100_000, splits=[0.5, 0.4])


def test_phase_amount_empty_splits_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        phase_amount(total=100_000, splits=[])


# ------------------------------------------------------------------------ flag_underwater

def test_flag_underwater_below_threshold():
    result = flag_underwater(compa_ratio=0.80, threshold=0.85)
    assert result["underwater"] is True


def test_flag_underwater_at_or_above_threshold():
    result = flag_underwater(compa_ratio=0.85, threshold=0.85)
    assert result["underwater"] is False
