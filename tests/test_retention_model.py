"""agents/retention_model.py's deterministic pieces run against real committed data. The one
model call (RetentionJudgment) is faked -- and skipped entirely by the agent itself when
nobody is underwater, which is asserted directly below rather than just faked around.
"""

import pytest

from agents.modeling_schemas import RetentionAssessment, RetentionJudgment
from agents.retention_model import assess_retention
from tests.fakes import FakeModel


def _underwater_employee(**overrides) -> dict:
    fields = dict(
        employee_id="NYX-011",
        family_group="engineering",
        level_code="L6",
        geo_code="IN-BLR",
        currency="INR",
        current_pay=5_000_000.0,  # well below IN-BLR/engineering/L6's range -- underwater
        unvested_equity_value=62_750.78,
        role_summary="Company-wide final authority on core PPA tradeoffs.",
    )
    fields.update(overrides)
    return fields


def _comfortable_employee(**overrides) -> dict:
    fields = dict(
        employee_id="NYX-999",
        family_group="engineering",
        level_code="L6",
        geo_code="IN-BLR",
        currency="INR",
        current_pay=9_000_000.0,  # comfortably above range_mid -- not underwater
    )
    fields.update(overrides)
    return fields


def test_assess_retention_flags_underwater_and_computes_award():
    fake = FakeModel(
        RetentionJudgment(critical_employee_ids=["NYX-011"], reasoning="Distinguished-level scope."),
        schema=RetentionJudgment,
    )
    result = assess_retention([_underwater_employee()], as_of_date="2026-08-01", model=fake)

    assert isinstance(result, RetentionAssessment)
    assert result.reporting_currency == "USD"
    line = result.employees[0]
    assert line.underwater is True
    assert line.compa_ratio < 0.85
    assert line.retention_award == max(0.0, line.range_mid - 5_000_000.0)  # in INR, the employee's own currency
    assert line.retention_award > 0
    assert line.retention_award_reporting_currency != line.retention_award
    assert 0 < line.retention_award_reporting_currency < line.retention_award
    assert len(line.award_phased_schedule) == 2
    assert result.total_award_day_one == line.retention_award_reporting_currency
    assert result.judgment.critical_employee_ids == ["NYX-011"]


def test_assess_retention_not_underwater_has_zero_award():
    fake = FakeModel(
        RetentionJudgment(critical_employee_ids=[], reasoning="Nobody underwater."), schema=RetentionJudgment
    )
    result = assess_retention([_comfortable_employee()], as_of_date="2026-08-01", model=fake)

    line = result.employees[0]
    assert line.underwater is False
    assert line.retention_award == 0.0
    assert result.total_award_day_one == 0.0


def test_assess_retention_skips_the_model_call_when_nobody_is_underwater():
    fake = FakeModel(None, schema=RetentionJudgment)  # would raise if actually invoked with a schema mismatch
    result = assess_retention([_comfortable_employee()], as_of_date="2026-08-01", model=fake)

    assert fake.raw_structured_model.call_count == 0
    assert result.judgment.critical_employee_ids == []


def test_assess_retention_mixed_population_only_flags_the_underwater_one():
    fake = FakeModel(
        RetentionJudgment(critical_employee_ids=["NYX-011"], reasoning="test"), schema=RetentionJudgment
    )
    result = assess_retention(
        [_underwater_employee(), _comfortable_employee()], as_of_date="2026-08-01", model=fake
    )
    by_id = {e.employee_id: e for e in result.employees}
    assert by_id["NYX-011"].underwater is True
    assert by_id["NYX-999"].underwater is False
    assert result.total_award_day_one == by_id["NYX-011"].retention_award_reporting_currency


def test_assess_retention_rolls_up_correctly_across_three_currencies():
    # Same scenario as test_cost_model.py's equivalent test: a mixed-currency population's
    # total award must not be a naive sum of raw, differently-scaled currency figures.
    population = [
        _underwater_employee(employee_id="INR-EMP", geo_code="IN-BLR", currency="INR", current_pay=5_000_000.0),
        _underwater_employee(
            employee_id="EUR-EMP", geo_code="EU-EIN", currency="EUR", current_pay=100_000.0,
            role_summary="Second-year analog designer.",
        ),
        _underwater_employee(
            employee_id="USD-EMP", geo_code="US-SJC", currency="USD", current_pay=90_000.0,
            role_summary="Second-year analog designer.",
        ),
    ]
    fake = FakeModel(
        RetentionJudgment(critical_employee_ids=["INR-EMP"], reasoning="test"), schema=RetentionJudgment
    )
    result = assess_retention(population, as_of_date="2026-08-01", model=fake)

    assert all(e.underwater and e.retention_award > 1000 for e in result.employees)

    naive_wrong_sum = sum(e.retention_award for e in result.employees)
    assert result.total_award_day_one != naive_wrong_sum
    assert result.total_award_day_one == sum(e.retention_award_reporting_currency for e in result.employees)
    assert result.total_award_day_one == pytest.approx(
        sum(e.retention_award_reporting_currency for e in result.employees)
    )
