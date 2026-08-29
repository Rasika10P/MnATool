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
from agents.synthesis import _cost_summary, _retention_summary, reconcile
from tests.fakes import FakeModel


def _cost_assessment(strategy="phased") -> CostAssessment:
    return CostAssessment(
        as_of_date="2026-08-01",
        annual_growth_rate=0.035,
        reporting_currency="USD",
        phase_splits=[0.5, 0.5],
        employees=[
            EmployeeCostLine(
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
        ],
        total_day_one_cost=18_390.80,
        total_phased_by_phase=[PhaseAmount(phase=1, amount=9_195.40), PhaseAmount(phase=2, amount=9_195.40)],
        recommendation=CostRecommendation(strategy=strategy, reasoning="Large total relative to population size."),
    )


def _retention_assessment(critical=True) -> RetentionAssessment:
    return RetentionAssessment(
        as_of_date="2026-08-01",
        underwater_threshold=0.85,
        reporting_currency="USD",
        phase_splits=[0.5, 0.5],
        employees=[
            EmployeeRetentionLine(
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
                award_phased_schedule=[
                    PhaseAmount(phase=1, amount=1_075_265.42), PhaseAmount(phase=2, amount=1_075_265.42)
                ],
            )
        ],
        total_award_day_one=24_718.75,
        total_award_phased_by_phase=[
            PhaseAmount(phase=1, amount=12_359.38), PhaseAmount(phase=2, amount=12_359.37)
        ],
        judgment=RetentionJudgment(
            critical_employee_ids=["NYX-011"] if critical else [],
            reasoning="Distinguished-level scope, significant unvested equity at risk." if critical else "No one critical.",
        ),
    )


def test_cost_summary_includes_strategy_and_figures():
    summary = _cost_summary(_cost_assessment())
    assert "phased" in summary
    assert "1,600,000.00" in summary
    assert "NYX-011" in summary


def test_retention_summary_includes_critical_ids_and_figures():
    summary = _retention_summary(_retention_assessment())
    assert "NYX-011" in summary
    assert "0.70" in summary


def test_reconcile_returns_the_conflict_the_model_surfaces():
    conflict_result = SynthesisResult(
        conflicts=[
            ReconciliationConflict(
                description="Phasing leaves NYX-011 underwater through year 1.",
                cost_position="Phase the $1.6M gap 50/50 over 2 years to manage budget impact.",
                retention_position="NYX-011 stays below 0.85 compa-ratio for a full year under that schedule.",
                affected_employee_ids=["NYX-011"],
            )
        ],
        recommended_plan="Fund NYX-011's retention award day-one; phase the rest of the cost gap.",
        requires_human_judgment=True,
    )
    fake = FakeModel(conflict_result, schema=SynthesisResult)

    result = reconcile(_cost_assessment(strategy="phased"), _retention_assessment(critical=True), model=fake)

    assert len(result.conflicts) == 1
    assert result.conflicts[0].affected_employee_ids == ["NYX-011"]
    assert result.requires_human_judgment is True


def test_reconcile_returns_no_conflict_when_the_model_finds_none():
    clean_result = SynthesisResult(
        conflicts=[], recommended_plan="Fund day-one; no critical underwater employees.", requires_human_judgment=False
    )
    fake = FakeModel(clean_result, schema=SynthesisResult)

    result = reconcile(_cost_assessment(strategy="day_one"), _retention_assessment(critical=False), model=fake)

    assert result.conflicts == []
    assert result.requires_human_judgment is False
