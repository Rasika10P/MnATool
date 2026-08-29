"""Pydantic schemas for the cost, retention and synthesis agents (build order item 6;
CLAUDE.md's M&A workflow step 7-8: "Cost and retention agents run in parallel... Synthesis
reconciles them").

CLAUDE.md non-negotiable 1: "Math in code, judgment in agents... An LLM must never compute a
pay figure." Every numeric field on CostAssessment and RetentionAssessment is populated by
agents/cost_model.py and agents/retention_model.py directly from tools.comp_math /
tools.data_access -- never parsed out of a model's structured-output call. The *only* schemas
an LLM actually produces are CostRecommendation and RetentionJudgment (this file) and
SynthesisResult: small, numberless, judgment-only objects. Everything else here is a plain
data container the agent code assembles and validates like any other typed value -- it just
happens to also be what gets shown to the comp manager and to the synthesis agent.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from agents.schemas import LevelCode


class PhaseAmount(BaseModel):
    """One phase of a phased schedule (tools.comp_math.phase_amount's output, typed)."""

    phase: int = Field(ge=1)
    amount: float = Field(ge=0.0)


class EmployeeCostLine(BaseModel):
    """One employee's cost-model figures. Deterministic top to bottom -- assembled by
    agents/cost_model.py from tools.data_access.lookup_market_percentile,
    tools.comp_math.compute_pay_gap and tools.comp_math.phase_amount. No field here is ever
    set from a model call."""

    employee_id: str
    job_id: str
    level_code: LevelCode
    geo_code: str
    currency: str
    current_pay: float = Field(ge=0.0)
    target_percentile: float = Field(description="e.g. 60.0, or 65.0 at L6+ per comp_philosophy.md's +5 rule")
    target_pay: float = Field(ge=0.0, description="Market value at target_percentile, aged to as_of_date")
    cost_gap: float = Field(
        ge=0.0, description="max(0, target_pay - current_pay) in this employee's own currency"
    )
    cost_gap_reporting_currency: float = Field(
        ge=0.0,
        description="cost_gap converted to CostAssessment.reporting_currency (tools.currency.convert_currency) "
        "-- what actually gets summed into the population totals, so a mixed-currency population's aggregate "
        "isn't a naive sum of different units",
    )
    phased_schedule: list[PhaseAmount] = Field(description="cost_gap (this employee's own currency) split by phase")


class CostRecommendation(BaseModel):
    """The only part of the cost agent's output a model produces: a funding-approach
    judgment given the already-computed totals, never a number of its own."""

    strategy: Literal["day_one", "phased"] = Field(description="Which funding approach the agent recommends")
    reasoning: str = Field(description="Why, given the total cost and its shape across the population")


class CostAssessment(BaseModel):
    """Full cost agent output for one crosswalked population. employees/totals are
    deterministic (see EmployeeCostLine); recommendation is the model's judgment call."""

    as_of_date: str = Field(description="The declared deal reference date this model runs against")
    annual_growth_rate: float
    reporting_currency: str = Field(
        description="Currency every population-level total is expressed in -- a crosswalked population can span "
        "multiple employee currencies, and summing raw cost_gap figures across currencies without converting to "
        "one common currency first would silently mix units"
    )
    phase_splits: list[float] = Field(description="e.g. [0.5, 0.5] for 50/50 over 2 years")
    employees: list[EmployeeCostLine]
    total_day_one_cost: float = Field(
        ge=0.0, description="Sum of every employee's cost_gap_reporting_currency, in reporting_currency"
    )
    total_phased_by_phase: list[PhaseAmount] = Field(description="total_day_one_cost split per phase_splits")
    recommendation: CostRecommendation


class EmployeeRetentionLine(BaseModel):
    """One employee's retention-model figures. Deterministic top to bottom -- assembled by
    agents/retention_model.py from tools.data_access.lookup_salary_structure,
    tools.comp_math.compute_pay_metrics, tools.comp_math.flag_underwater,
    tools.comp_math.compute_pay_gap and tools.comp_math.phase_amount."""

    employee_id: str
    level_code: LevelCode
    geo_code: str
    currency: str
    current_pay: float = Field(ge=0.0)
    range_mid: float = Field(ge=0.0, description="The negotiated level's Meridian range midpoint in this geo")
    compa_ratio: float
    underwater_threshold: float = Field(description="The compa-ratio cutoff applied, e.g. 0.85")
    underwater: bool
    unvested_equity_value: float | None = Field(
        default=None, description="Grant-value figure from the Nyx census, where the employee has any"
    )
    retention_award: float = Field(
        ge=0.0, description="max(0, range_mid - current_pay) if underwater, else 0.0, in this employee's own currency"
    )
    retention_award_reporting_currency: float = Field(
        ge=0.0,
        description="retention_award converted to RetentionAssessment.reporting_currency -- see "
        "EmployeeCostLine.cost_gap_reporting_currency for why this conversion has to happen per employee",
    )
    award_phased_schedule: list[PhaseAmount] = Field(description="retention_award (this employee's own currency) split by phase")


class RetentionJudgment(BaseModel):
    """The only part of the retention agent's output a model produces: which underwater
    employees are judged genuinely critical to retain (not every underwater person
    automatically qualifies -- that's a role-scope/seniority call, not a compa-ratio
    threshold, which is already deterministic), and the strategy recommendation."""

    critical_employee_ids: list[str] = Field(
        description="Subset of the underwater employee_ids the agent judges critical, given role scope/seniority"
    )
    reasoning: str


class RetentionAssessment(BaseModel):
    """Full retention agent output for one crosswalked population."""

    as_of_date: str
    underwater_threshold: float
    reporting_currency: str = Field(description="Same purpose as CostAssessment.reporting_currency")
    phase_splits: list[float]
    employees: list[EmployeeRetentionLine]
    total_award_day_one: float = Field(
        ge=0.0, description="Sum of every underwater employee's retention_award_reporting_currency"
    )
    total_award_phased_by_phase: list[PhaseAmount]
    judgment: RetentionJudgment


class ReconciliationConflict(BaseModel):
    """One place the cost and retention agents' recommendations pull in different
    directions -- surfaced explicitly, never averaged away. This is the whole point of the
    synthesis agent: a tension is a valid, complete output, not a failure to reach one number."""

    description: str = Field(description="What the tension actually is, in plain terms")
    cost_position: str
    retention_position: str
    affected_employee_ids: list[str] = Field(min_length=1)


class SynthesisResult(BaseModel):
    """The synthesis agent's output. No numeric fields of its own -- it reconciles the
    already-computed cost and retention figures in prose, it doesn't compute a new one
    (CLAUDE.md non-negotiable 1 applies here too, even though this schema's plain fields
    make that easy to satisfy without a validator: there's simply nothing numeric to set)."""

    conflicts: list[ReconciliationConflict] = Field(
        description="Empty list is a valid, meaningful result: no tension between cost and retention"
    )
    recommended_plan: str = Field(description="Prose synthesis of both agents' positions into one plan")
    requires_human_judgment: bool = Field(
        description="True when a conflict here can't be resolved by synthesis alone -- e.g. cost's phasing "
        "preference genuinely conflicts with retention's critical-underwater flag with no way to satisfy both"
    )

    @model_validator(mode="after")
    def _requires_human_judgment_implied_by_unresolved_conflicts(self) -> "SynthesisResult":
        # Not every conflict needs a human -- synthesis can resolve some on its own (e.g. a
        # low-dollar, non-critical tension the recommended_plan settles outright). But zero
        # conflicts can never require human judgment; there's nothing to adjudicate.
        if not self.conflicts and self.requires_human_judgment:
            raise ValueError("requires_human_judgment=True with no conflicts listed -- there's nothing to judge.")
        return self
