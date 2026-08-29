"""The retention agent (CLAUDE.md M&A workflow step 7; build order item 6). Flags employees
whose compa-ratio at their negotiated level falls below the retention-risk threshold, computes
a deterministic retention award for each, and judges which of those are genuinely critical to
retain.

level_framework.md section 7: "Retention risk is real, but it is a compensation remedy, not a
leveling argument. It is handled by the retention agent through retention awards or
red-circling -- never by moving a level." This module is that agent's award-sizing half.

Same discipline as agents/cost_model.py: every number on the returned RetentionAssessment is
computed here directly from tools.comp_math / tools.data_access, never asked of a model. The
one model call produces RetentionJudgment only -- which underwater employees are critical,
given their role scope/seniority (not recomputable from compa-ratio alone), and why.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from agents.model_router import get_model
from agents.modeling_schemas import EmployeeRetentionLine, PhaseAmount, RetentionAssessment, RetentionJudgment
from tools.comp_math import compute_pay_gap, compute_pay_metrics, flag_underwater, phase_amount
from tools.currency import convert_currency
from tools.data_access import DEFAULT_DATA_DIR, lookup_salary_structure

UNDERWATER_THRESHOLD = 0.85  # comp manager policy, confirmed for this build
PHASE_SPLITS = [0.5, 0.5]  # same phasing schedule as agents/cost_model.py
REPORTING_CURRENCY = "USD"  # same as agents/cost_model.py -- see that module for why this exists

_SYSTEM_PROMPT_TEMPLATE = """You are Meridian Silicon's retention agent. You do not compute \
any dollar figure or compa-ratio yourself -- every number below was already computed \
deterministically, before you were called, including which employees are "underwater" \
(compa-ratio below {threshold} at their negotiated level).

Your only job: of the underwater employees listed below, judge which are genuinely critical \
to retain. Not every underwater person automatically qualifies -- criticality is about role \
scope and seniority (would losing this person be a real loss to the organization, not just an \
uncomfortable pay gap), not the size of the compa-ratio gap itself. A junior employee \
compressed to 0.80 and a company-wide technical authority compressed to 0.83 are not equally \
urgent, even though the second gap is smaller.

Cite the specific evidence (role summary, level, scope) for each employee you flag as \
critical -- don't flag someone on compa-ratio alone, that's already reflected in them being \
in this list at all.

Underwater employees:
{underwater_summary}
"""


def _load_fx_rates(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "fx_rates.parquet")


def _line_for_employee(employee: dict, as_of_date: str, fx_rates: pd.DataFrame) -> EmployeeRetentionLine:
    """Deterministic: every field here comes from tools.data_access / tools.comp_math /
    tools.currency."""
    structure = lookup_salary_structure(employee["family_group"], employee["level_code"], employee["geo_code"])[
        "structure"
    ]
    range_mid = structure["range_mid"]

    metrics = compute_pay_metrics(
        employee["current_pay"], structure["range_min"], range_mid, structure["range_max"]
    )
    underwater = flag_underwater(metrics["compa_ratio"], UNDERWATER_THRESHOLD)["underwater"]

    if underwater:
        gap = compute_pay_gap(employee["current_pay"], range_mid)
        award = gap["cost"]
    else:
        award = 0.0
    phased = phase_amount(award, PHASE_SPLITS) if award > 0 else {"phases": [{"phase": i + 1, "amount": 0.0} for i in range(len(PHASE_SPLITS))]}
    # Same reasoning as agents/cost_model.py's cost_gap_reporting_currency: a crosswalked
    # population can span multiple employee currencies, so the figure that gets summed into
    # population totals has to be converted to one common currency first.
    award_reporting_currency = convert_currency(
        award, employee["currency"], REPORTING_CURRENCY, as_of_date, fx_rates
    )["converted_amount"]

    return EmployeeRetentionLine(
        employee_id=employee["employee_id"],
        level_code=employee["level_code"],
        geo_code=employee["geo_code"],
        currency=employee["currency"],
        current_pay=employee["current_pay"],
        range_mid=range_mid,
        compa_ratio=metrics["compa_ratio"],
        underwater_threshold=UNDERWATER_THRESHOLD,
        underwater=underwater,
        unvested_equity_value=employee.get("unvested_equity_value"),
        retention_award=award,
        retention_award_reporting_currency=award_reporting_currency,
        award_phased_schedule=[PhaseAmount(**p) for p in phased["phases"]],
    )


def _underwater_summary(employees: list[EmployeeRetentionLine], population: list[dict]) -> str:
    role_summary_by_id = {emp["employee_id"]: emp.get("role_summary", "") for emp in population}
    lines = []
    for e in employees:
        if not e.underwater:
            continue
        equity_note = f", unvested equity {e.unvested_equity_value:,.2f}" if e.unvested_equity_value else ""
        lines.append(
            f"- {e.employee_id} ({e.level_code}, {e.geo_code}): compa-ratio {e.compa_ratio:.2f}, "
            f"retention_award {e.retention_award:,.2f} {e.currency} "
            f"({e.retention_award_reporting_currency:,.2f} {REPORTING_CURRENCY}){equity_note}\n"
            f"  role summary: {role_summary_by_id.get(e.employee_id, '(not provided)')}"
        )
    return "\n".join(lines) if lines else "(none underwater)"


def assess_retention(population: list[dict], as_of_date: str, model=None) -> RetentionAssessment:
    """population: a crosswalked population, one dict per employee with employee_id,
    family_group (job_catalog's family_group column, e.g. "engineering" -- salary structures
    are genuinely defined at this grain, unlike survey market data; see agents/cost_model.py's
    docstring on why cost pricing can't use the same grain), level_code, geo_code, currency,
    current_pay (already in geo_code's local currency), and optionally unvested_equity_value
    and role_summary (used only as context for the criticality judgment, never as a number).

    as_of_date: the declared deal reference date -- recorded on the result for provenance,
    the same reference date agents.cost_model.assess_cost runs against, even though this
    module's own lookups (salary structures) aren't date-aged the way market survey data is.
    """
    fx_rates = _load_fx_rates()
    employees = [_line_for_employee(emp, as_of_date, fx_rates) for emp in population]

    total_award_day_one = sum(e.retention_award_reporting_currency for e in employees)
    total_phased = (
        phase_amount(total_award_day_one, PHASE_SPLITS)
        if total_award_day_one > 0
        else {"phases": [{"phase": i + 1, "amount": 0.0} for i in range(len(PHASE_SPLITS))]}
    )
    total_award_phased_by_phase = [PhaseAmount(**p) for p in total_phased["phases"]]

    underwater_employees = [e for e in employees if e.underwater]
    if underwater_employees:
        llm = model or get_model("judgment")
        structured_llm = llm.with_structured_output(RetentionJudgment)
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            threshold=UNDERWATER_THRESHOLD,
            underwater_summary=_underwater_summary(employees, population),
        )
        judgment = structured_llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Which underwater employees are critical to retain?"},
            ]
        )
    else:
        judgment = RetentionJudgment(critical_employee_ids=[], reasoning="No employees are underwater at their negotiated level -- nothing to judge.")

    return RetentionAssessment(
        as_of_date=as_of_date,
        underwater_threshold=UNDERWATER_THRESHOLD,
        reporting_currency=REPORTING_CURRENCY,
        phase_splits=PHASE_SPLITS,
        employees=employees,
        total_award_day_one=total_award_day_one,
        total_award_phased_by_phase=total_award_phased_by_phase,
        judgment=judgment,
    )
