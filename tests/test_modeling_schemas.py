import pytest
from pydantic import ValidationError

from agents.modeling_schemas import (
    CostAssessment,
    CostRecommendation,
    EmployeeCostLine,
    EmployeeRetentionLine,
    PhaseAmount,
    ReconciliationConflict,
    RetentionAssessment,
    RetentionJudgment,
    SynthesisResult,
)


def _cost_line(**overrides) -> EmployeeCostLine:
    fields = dict(
        employee_id="NYX-011",
        job_id="DD-UARCH-L6",
        level_code="L6",
        geo_code="IN-BLR",
        currency="INR",
        current_pay=5_000_000.0,
        target_percentile=65.0,
        target_pay=6_600_000.0,
        cost_gap=1_600_000.0,
        cost_gap_reporting_currency=18_390.80,
        phased_schedule=[PhaseAmount(phase=1, amount=800_000.0), PhaseAmount(phase=2, amount=800_000.0)],
    )
    fields.update(overrides)
    return EmployeeCostLine(**fields)


def test_employee_cost_line_accepts_valid_fields():
    line = _cost_line()
    assert line.level_code == "L6"
    assert line.cost_gap == 1_600_000.0


def test_employee_cost_line_rejects_negative_cost_gap():
    with pytest.raises(ValidationError):
        _cost_line(cost_gap=-100.0)


def test_cost_assessment_accepts_a_full_population():
    assessment = CostAssessment(
        as_of_date="2026-08-01",
        annual_growth_rate=0.035,
        reporting_currency="USD",
        phase_splits=[0.5, 0.5],
        employees=[_cost_line()],
        total_day_one_cost=18_390.80,
        total_phased_by_phase=[PhaseAmount(phase=1, amount=9_195.40), PhaseAmount(phase=2, amount=9_195.40)],
        recommendation=CostRecommendation(strategy="phased", reasoning="Total cost is large relative to deal budget."),
    )
    assert assessment.recommendation.strategy == "phased"
    assert len(assessment.employees) == 1


def _retention_line(**overrides) -> EmployeeRetentionLine:
    fields = dict(
        employee_id="NYX-011",
        level_code="L6",
        geo_code="IN-BLR",
        currency="INR",
        current_pay=5_000_000.0,
        range_mid=7_150_530.84,
        compa_ratio=0.70,
        underwater_threshold=0.85,
        underwater=True,
        unvested_equity_value=62_750.78,
        retention_award=2_150_530.84,
        retention_award_reporting_currency=24_718.75,
        award_phased_schedule=[PhaseAmount(phase=1, amount=1_075_265.42), PhaseAmount(phase=2, amount=1_075_265.42)],
    )
    fields.update(overrides)
    return EmployeeRetentionLine(**fields)


def test_employee_retention_line_accepts_valid_fields():
    line = _retention_line()
    assert line.underwater is True
    assert line.retention_award > 0


def test_retention_assessment_accepts_a_full_population():
    assessment = RetentionAssessment(
        as_of_date="2026-08-01",
        underwater_threshold=0.85,
        reporting_currency="USD",
        phase_splits=[0.5, 0.5],
        employees=[_retention_line()],
        total_award_day_one=24_718.75,
        total_award_phased_by_phase=[
            PhaseAmount(phase=1, amount=12_359.38), PhaseAmount(phase=2, amount=12_359.37)
        ],
        judgment=RetentionJudgment(critical_employee_ids=["NYX-011"], reasoning="Distinguished-level scope, significant unvested equity at risk."),
    )
    assert assessment.judgment.critical_employee_ids == ["NYX-011"]


def test_reconciliation_conflict_requires_at_least_one_affected_employee():
    with pytest.raises(ValidationError):
        ReconciliationConflict(
            description="test", cost_position="phase it", retention_position="don't", affected_employee_ids=[]
        )


def test_synthesis_result_accepts_no_conflicts():
    result = SynthesisResult(conflicts=[], recommended_plan="No tension; fund day-one.", requires_human_judgment=False)
    assert result.conflicts == []


def test_synthesis_result_accepts_a_real_conflict_requiring_judgment():
    result = SynthesisResult(
        conflicts=[
            ReconciliationConflict(
                description="Cost favors phasing; retention says phasing leaves a critical employee underwater "
                "for a year.",
                cost_position="Phase the $1.6M gap 50/50 over 2 years to manage budget impact.",
                retention_position="NYX-011 stays below the 0.85 compa-ratio threshold for a full year under "
                "that schedule -- unacceptable given their unvested equity is fully at risk.",
                affected_employee_ids=["NYX-011"],
            )
        ],
        recommended_plan="Fund NYX-011's award day-one; phase everyone else.",
        requires_human_judgment=True,
    )
    assert result.requires_human_judgment is True


def test_synthesis_result_rejects_requires_human_judgment_with_no_conflicts():
    with pytest.raises(ValidationError):
        SynthesisResult(conflicts=[], recommended_plan="test", requires_human_judgment=True)


def test_cost_recommendation_reasoning_strips_leaked_tags():
    rec = CostRecommendation(strategy="phased", reasoning='Total cost is large. </reasoning>')
    assert rec.reasoning == "Total cost is large."


def test_retention_judgment_reasoning_strips_leaked_tags():
    judgment = RetentionJudgment(critical_employee_ids=["NYX-011"], reasoning='Distinguished-level scope. <parameter name="x">')
    assert judgment.reasoning == "Distinguished-level scope."


def test_reconciliation_conflict_prose_fields_strip_leaked_tags():
    conflict = ReconciliationConflict(
        description='Cost favors phasing. </description>',
        cost_position="Phase it.",
        retention_position="Don't.",
        affected_employee_ids=["NYX-011"],
    )
    assert conflict.description == "Cost favors phasing."


def test_synthesis_result_recommended_plan_strips_leaked_tags():
    result = SynthesisResult(
        conflicts=[], recommended_plan='Fund day-one. </invoke>', requires_human_judgment=False
    )
    assert result.recommended_plan == "Fund day-one."
