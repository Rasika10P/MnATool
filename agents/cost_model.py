"""The cost agent (CLAUDE.md M&A workflow step 7; build order item 6). Prices every employee
in a crosswalked population to comp_philosophy.md's target market percentile and models the
cost gap two ways: fund it all at close ("day_one"), or phase it over comp_philosophy.md's
schedule.

CLAUDE.md non-negotiable 1 ("math in code, judgment in agents") applies literally here: every
number on the returned CostAssessment -- every employee's target_pay, cost_gap, phased
schedule, and the population totals -- is computed in this module directly from
tools.comp_math / tools.data_access, never asked of a model. The one model call this module
makes produces CostRecommendation only: given the already-computed totals, which funding
approach to recommend and why. See agents/modeling_schemas.py's module docstring for the
same discipline applied to the retention and synthesis agents.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from agents.model_router import get_model
from agents.modeling_schemas import CostAssessment, CostRecommendation, EmployeeCostLine, PhaseAmount
from tools.comp_math import compute_pay_gap, phase_amount
from tools.currency import convert_currency
from tools.data_access import DEFAULT_DATA_DIR, lookup_market_percentile

REPORTING_CURRENCY = "USD"  # Meridian is US-headquartered (docs/level_framework.md section 10)

# comp_philosophy.md's target-percentile table, keyed on job_catalog's `family` column --
# not the coarser `family_group` (only 3 values: engineering/corporate/gtm), which can't
# distinguish e.g. Digital Design from Application Software. See learnings.md.
TARGET_PERCENTILE_BY_FAMILY = {
    "Digital Design": 60.0,
    "Analog & Mixed-Signal": 60.0,
    "Physical Design": 60.0,
    "Design Verification": 60.0,
    "Silicon Validation & DFT": 60.0,
    "Systems & Architecture": 60.0,
    "Embedded Software": 50.0,
    "Platform Software": 50.0,
    "Application Software": 50.0,
    "Product & Program": 50.0,
    "Corporate": 50.0,
    "Go-to-Market": 50.0,  # on OTE (pay_element), per comp_philosophy.md's pay-mix rule
}
L6_PLUS_BONUS_POINTS = 5.0  # comp_philosophy.md: "L6 and above, any family: target + 5 points"
ANNUAL_MARKET_GROWTH_RATE = 0.035  # comp_philosophy.md
PHASE_SPLITS = [0.5, 0.5]  # comp_philosophy.md's phasing schedule: 50/50 over 2 years

_SYSTEM_PROMPT_TEMPLATE = """You are Meridian Silicon's cost agent, modeling the cost of an \
acquisition's compensation integration. You do not compute any dollar figure yourself -- \
every number below was already computed deterministically, before you were called. Your only \
job is to recommend a funding strategy for the total cost gap you're given.

Two funding approaches are available:
- day_one: fund the full cost gap immediately, at close.
- phased: fund it according to comp_philosophy.md's standard schedule ({phase_splits}) -- \
the first share now, the remainder at the one-year mark.

Recommend whichever approach the total cost and its shape across the population actually \
supports:
- A large total relative to the size of the population, or a cost concentrated in a few very \
large individual gaps, favors phasing to manage budget impact.
- A small total, evenly distributed, favors day_one -- phasing a small amount just adds \
administrative overhead (a second funding event to track and communicate) for no real budget \
benefit.

State your reasoning in terms of the actual figures you were given below -- cite them, don't \
restate them from memory or approximate them; if you need to reference a number, copy it \
exactly as given.

Population cost summary:
{summary}
"""


def _load_level_sort_order(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, int]:
    level_definitions = pd.read_parquet(data_dir / "level_definitions.parquet")
    return dict(zip(level_definitions.level_code, level_definitions.sort_order))


def _load_fx_rates(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "fx_rates.parquet")


def target_percentile_for(family: str, level_code: str, level_sort_order: dict[str, int]) -> float:
    """comp_philosophy.md's target percentile for one employee: the family's base target,
    plus 5 points if level_code is L6-and-above-equivalent (sort_order >= L6's, which also
    correctly covers M5/M6/M7 via level_definitions' own ordering -- see learnings.md-style
    reasoning in agents/negotiation_graph.py's sibling modules for why sort_order, not a
    hardcoded level set, is the robust way to express "L6 and above" across both tracks)."""
    base = TARGET_PERCENTILE_BY_FAMILY.get(family)
    if base is None:
        raise ValueError(f"No target percentile configured for family={family!r} -- comp_philosophy.md gap")
    if level_code not in level_sort_order:
        raise ValueError(f"Unknown level_code={level_code!r}")
    if level_sort_order[level_code] >= level_sort_order["L6"]:
        return base + L6_PLUS_BONUS_POINTS
    return base


def _line_for_employee(
    employee: dict, as_of_date: str, level_sort_order: dict[str, int], fx_rates: pd.DataFrame
) -> EmployeeCostLine:
    """Deterministic: every field here comes from tools.data_access / tools.comp_math /
    tools.currency."""
    target_percentile = target_percentile_for(employee["family"], employee["level_code"], level_sort_order)
    pay_element = "OTE" if employee.get("is_quota_carrying") else "base"

    market = lookup_market_percentile(
        job_id=employee["job_id"],
        geo_code=employee["geo_code"],
        target_percentile=target_percentile,
        as_of_date=as_of_date,
        annual_growth_rate=ANNUAL_MARKET_GROWTH_RATE,
        pay_element=pay_element,
    )
    gap = compute_pay_gap(employee["current_pay"], market["value"])
    phased = phase_amount(gap["cost"], PHASE_SPLITS)
    # A crosswalked population can span multiple employee currencies (USD/INR/EUR in this
    # dataset) -- cost_gap_reporting_currency is what actually gets summed into population
    # totals below, never the raw per-currency cost_gap. convert_currency is a no-op
    # (rate=1.0) when the employee is already in REPORTING_CURRENCY.
    gap_reporting_currency = convert_currency(
        gap["cost"], employee["currency"], REPORTING_CURRENCY, as_of_date, fx_rates
    )["converted_amount"]

    return EmployeeCostLine(
        employee_id=employee["employee_id"],
        job_id=employee["job_id"],
        level_code=employee["level_code"],
        geo_code=employee["geo_code"],
        currency=employee["currency"],
        current_pay=employee["current_pay"],
        target_percentile=target_percentile,
        target_pay=market["value"],
        cost_gap=gap["cost"],
        cost_gap_reporting_currency=gap_reporting_currency,
        phased_schedule=[PhaseAmount(**p) for p in phased["phases"]],
    )


def _population_summary(employees: list[EmployeeCostLine], total_day_one: float, total_phased: list[PhaseAmount]) -> str:
    lines = [
        f"- {e.employee_id} ({e.job_id}, {e.geo_code}): current {e.current_pay:,.2f} {e.currency}, "
        f"target (P{e.target_percentile:g}) {e.target_pay:,.2f} {e.currency}, cost_gap {e.cost_gap:,.2f} {e.currency} "
        f"({e.cost_gap_reporting_currency:,.2f} {REPORTING_CURRENCY})"
        for e in employees
    ]
    lines.append(f"Total day-one cost: {total_day_one:,.2f} {REPORTING_CURRENCY}")
    lines.append(
        f"Total phased ({REPORTING_CURRENCY}): "
        + "; ".join(f"phase {p.phase}: {p.amount:,.2f}" for p in total_phased)
    )
    return "\n".join(lines)


def assess_cost(population: list[dict], as_of_date: str, model=None) -> CostAssessment:
    """population: a crosswalked population, one dict per employee with employee_id, job_id
    (the negotiated job_id, e.g. "DD-UARCH-L6"), family (job_catalog's family column, e.g.
    "Digital Design"), level_code, geo_code, currency, current_pay (already in geo_code's
    local currency -- convert before calling this), and optionally is_quota_carrying.

    as_of_date: the declared deal reference date (data_model_spec.md section 4: "the
    acquisition cost model runs against a declared deal reference date so demo output is
    reproducible") -- not "today". Every employee is priced, and every currency conversion
    run, as of this same date.
    """
    level_sort_order = _load_level_sort_order()
    fx_rates = _load_fx_rates()
    employees = [_line_for_employee(emp, as_of_date, level_sort_order, fx_rates) for emp in population]

    total_day_one_cost = sum(e.cost_gap_reporting_currency for e in employees)
    total_phased = phase_amount(total_day_one_cost, PHASE_SPLITS)
    total_phased_by_phase = [PhaseAmount(**p) for p in total_phased["phases"]]

    llm = model or get_model("judgment")
    structured_llm = llm.with_structured_output(CostRecommendation)
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        phase_splits=PHASE_SPLITS,
        summary=_population_summary(employees, total_day_one_cost, total_phased_by_phase),
    )
    recommendation = structured_llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Recommend a funding strategy for this population."},
        ]
    )

    return CostAssessment(
        as_of_date=as_of_date,
        annual_growth_rate=ANNUAL_MARKET_GROWTH_RATE,
        reporting_currency=REPORTING_CURRENCY,
        phase_splits=PHASE_SPLITS,
        employees=employees,
        total_day_one_cost=total_day_one_cost,
        total_phased_by_phase=total_phased_by_phase,
        recommendation=recommendation,
    )
