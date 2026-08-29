"""agents/cost_model.py's deterministic pieces (target percentile resolution, per-employee
cost lines, population totals) run against real committed data, matching
tests/test_equity_gate.py's style. The one model call (CostRecommendation) is faked.
"""

import pytest

from agents.cost_model import _load_level_sort_order, assess_cost, target_percentile_for
from agents.modeling_schemas import CostAssessment, CostRecommendation
from tests.fakes import FakeModel

_SORT_ORDER = _load_level_sort_order()


def test_target_percentile_base_case_no_l6_bonus():
    assert target_percentile_for("Digital Design", "L5", _SORT_ORDER) == 60.0


def test_target_percentile_l6_plus_bonus():
    assert target_percentile_for("Digital Design", "L6", _SORT_ORDER) == 65.0


def test_target_percentile_software_family_is_p50():
    assert target_percentile_for("Embedded Software", "L4", _SORT_ORDER) == 50.0


def test_target_percentile_m5_gets_l6_bonus_via_ic_equivalent_sort_order():
    # M5's sort_order (65) sits between L6 (60) and L7 (70) -- ic_equivalent L6 -- so it
    # should get the same +5 bonus L6 itself does.
    assert target_percentile_for("Digital Design", "M5", _SORT_ORDER) == 65.0


def test_target_percentile_unknown_family_raises():
    with pytest.raises(ValueError, match="No target percentile configured"):
        target_percentile_for("Not A Real Family", "L5", _SORT_ORDER)


def _population():
    return [
        {
            "employee_id": "NYX-011",
            "job_id": "DD-UARCH-L6",
            "family": "Digital Design",
            "level_code": "L6",
            "geo_code": "IN-BLR",
            "currency": "INR",
            "current_pay": 5_000_000.0,
        }
    ]


def test_assess_cost_computes_deterministic_figures_and_uses_the_fake_recommendation():
    fake = FakeModel(
        CostRecommendation(strategy="phased", reasoning="test reasoning"), schema=CostRecommendation
    )
    result = assess_cost(_population(), as_of_date="2026-08-01", model=fake)

    assert isinstance(result, CostAssessment)
    assert result.reporting_currency == "USD"
    assert len(result.employees) == 1
    line = result.employees[0]
    assert line.employee_id == "NYX-011"
    assert line.target_percentile == 65.0  # L6 bonus
    assert line.target_pay > 0
    assert line.cost_gap == max(0.0, line.target_pay - 5_000_000.0)  # in the employee's own currency, INR
    assert line.currency == "INR"
    # cost_gap_reporting_currency is the USD conversion -- a real FX rate, not equal to the
    # raw INR figure (roughly cost_gap / 87, not cost_gap itself).
    assert line.cost_gap_reporting_currency != line.cost_gap
    assert 0 < line.cost_gap_reporting_currency < line.cost_gap
    assert result.total_day_one_cost == line.cost_gap_reporting_currency
    assert len(result.total_phased_by_phase) == 2
    assert result.total_phased_by_phase[0].amount + result.total_phased_by_phase[1].amount == result.total_day_one_cost
    assert result.recommendation.strategy == "phased"


def test_assess_cost_handles_an_empty_population():
    fake = FakeModel(
        CostRecommendation(strategy="day_one", reasoning="Nothing to fund."), schema=CostRecommendation
    )
    result = assess_cost([], as_of_date="2026-08-01", model=fake)
    assert result.employees == []
    assert result.total_day_one_cost == 0.0


def test_assess_cost_rolls_up_correctly_across_three_currencies():
    # The scenario the fix is actually for: a population spanning USD, INR and EUR must not
    # have its total computed as a naive sum of raw, differently-scaled numbers.
    population = [
        {
            "employee_id": "USD-EMP", "job_id": "DD-RTL-L1", "family": "Digital Design",
            "level_code": "L1", "geo_code": "US-SJC", "currency": "USD", "current_pay": 90_000.0,
        },
        {
            "employee_id": "INR-EMP", "job_id": "DD-UARCH-L6", "family": "Digital Design",
            "level_code": "L6", "geo_code": "IN-BLR", "currency": "INR", "current_pay": 5_000_000.0,
        },
        {
            "employee_id": "EUR-EMP", "job_id": "DD-RTL-L3", "family": "Digital Design",
            "level_code": "L3", "geo_code": "EU-EIN", "currency": "EUR", "current_pay": 100_000.0,
        },
    ]
    fake = FakeModel(CostRecommendation(strategy="phased", reasoning="test"), schema=CostRecommendation)
    result = assess_cost(population, as_of_date="2026-08-01", model=fake)

    by_id = {e.employee_id: e for e in result.employees}
    # Every employee's own-currency cost_gap should be well over 1000 in this scenario (real
    # gaps, not zero) -- otherwise this test isn't actually exercising the rollup.
    assert all(e.cost_gap > 1000 for e in result.employees)

    # The naive (wrong) sum mixes currencies and would be dominated by INR's larger raw
    # scale; the correct total must not equal that.
    naive_wrong_sum = sum(e.cost_gap for e in result.employees)
    assert result.total_day_one_cost != naive_wrong_sum

    # The correct total is exactly the sum of each employee's already-converted figure.
    assert result.total_day_one_cost == sum(e.cost_gap_reporting_currency for e in result.employees)

    # And it should land in a plausible USD range given the inputs -- not off by an order of
    # magnitude the way the naive sum (dominated by ~5,000,000+ raw INR) would be.
    assert 1_000 < result.total_day_one_cost < 300_000
    assert result.total_day_one_cost == pytest.approx(
        by_id["USD-EMP"].cost_gap_reporting_currency
        + by_id["INR-EMP"].cost_gap_reporting_currency
        + by_id["EUR-EMP"].cost_gap_reporting_currency
    )
