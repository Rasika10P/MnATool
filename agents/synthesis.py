"""The synthesis agent (CLAUDE.md M&A workflow step 8; build order item 6). Reconciles the
cost agent's funding recommendation with the retention agent's criticality judgment.

The user's brief for this piece is the whole point of the module: "Synthesis must surface
conflicts between the two rather than averaging them away -- if cost favors phasing and
retention says phasing leaves critical people underwater, that tension is the output." This
agent's job is not to pick a winner or blend the two positions into a compromise number --
there's no number to blend; see agents/modeling_schemas.py's SynthesisResult, which has no
numeric fields of its own. It restates the two agents' already-computed positions, and where
they don't fit together, says so explicitly rather than picking one silently.
"""

from __future__ import annotations

from agents.model_router import get_model
from agents.modeling_schemas import CostAssessment, RetentionAssessment, SynthesisResult

_SYSTEM_PROMPT = """You are Meridian Silicon's synthesis agent, reconciling the cost agent's \
funding recommendation with the retention agent's retention judgment for the same crosswalked \
population.

Your job is not to pick a winner, split the difference, or average the two positions into a \
compromise. Cost and retention each reasoned correctly from their own deterministic figures \
(comp_philosophy.md pricing for cost, compa-ratio and award sizing for retention) and their \
own mandate -- when their recommendations don't fit together, that tension is real, and \
surfacing it clearly is the actual deliverable, not a defect to resolve away.

Specifically: if cost recommends phasing the funding, check whether any employee the \
retention agent flagged as critical would still be underwater during the phased-but-not-yet-
funded period. If so, that is a conflict -- name it, name the affected employee(s), state \
cost's position and retention's position in their own terms, and do not silently pick one. \
Set requires_human_judgment=True for any conflict that can't be resolved by the numbers \
alone (e.g. cost's phasing preference is about aggregate budget impact, not about any one \
person -- a genuine tradeoff between two legitimate concerns, not a data error either side \
got wrong).

A population with no such tension -- retention's critical employees are already funded \
day-one, or there's nobody critical/underwater at all -- is a valid, complete result: return \
an empty conflicts list and say so plainly in recommended_plan. Do not manufacture a conflict \
where the two positions actually agree.

Cost agent's assessment:
{cost_summary}

Retention agent's assessment:
{retention_summary}
"""


def _cost_summary(cost: CostAssessment) -> str:
    lines = [f"Recommended strategy: {cost.recommendation.strategy} -- {cost.recommendation.reasoning}"]
    lines.append(f"Total day-one cost: {cost.total_day_one_cost:,.2f} {cost.reporting_currency}")
    lines.append(
        f"Phased schedule ({cost.reporting_currency}): "
        + "; ".join(f"phase {p.phase}: {p.amount:,.2f}" for p in cost.total_phased_by_phase)
    )
    for e in cost.employees:
        lines.append(
            f"- {e.employee_id}: cost_gap {e.cost_gap:,.2f} {e.currency} "
            f"({e.cost_gap_reporting_currency:,.2f} {cost.reporting_currency})"
        )
    return "\n".join(lines)


def _retention_summary(retention: RetentionAssessment) -> str:
    lines = [f"Critical employees: {retention.judgment.critical_employee_ids} -- {retention.judgment.reasoning}"]
    lines.append(f"Underwater threshold: {retention.underwater_threshold}")
    lines.append(f"Total retention award (day-one): {retention.total_award_day_one:,.2f} {retention.reporting_currency}")
    lines.append(
        f"Award phased schedule ({retention.reporting_currency}): "
        + "; ".join(f"phase {p.phase}: {p.amount:,.2f}" for p in retention.total_award_phased_by_phase)
    )
    for e in retention.employees:
        if e.underwater:
            lines.append(
                f"- {e.employee_id}: compa-ratio {e.compa_ratio:.2f}, underwater=True, "
                f"retention_award {e.retention_award:,.2f} {e.currency} "
                f"({e.retention_award_reporting_currency:,.2f} {retention.reporting_currency}), "
                f"award phased: {[(p.phase, p.amount) for p in e.award_phased_schedule]}"
            )
    return "\n".join(lines)


def reconcile(cost: CostAssessment, retention: RetentionAssessment, model=None) -> SynthesisResult:
    llm = model or get_model("judgment")
    structured_llm = llm.with_structured_output(SynthesisResult)

    system_prompt = _SYSTEM_PROMPT.format(
        cost_summary=_cost_summary(cost),
        retention_summary=_retention_summary(retention),
    )
    return structured_llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Reconcile these two assessments."},
        ]
    )
