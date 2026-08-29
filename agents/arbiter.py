"""The arbiter (level_framework.md section 7, "Participants"). Rules on one round of a
contested crosswalk mapping, applying Meridian's own framework only -- sections 5 (leveling
rules) and 7 (crosswalk negotiation). Never reasons from Nyx's own framework: the advocate's
argument may quote it (agents/advocate.py is scoped to it), but the arbiter treats any such
quote only as evidence about the employee's described scope, not as a second framework to
weigh against Meridian's own.

Round bookkeeping (which round this is, forcing escalation once the two-round limit is hit)
belongs to agents/negotiation_graph.py, which wires advocate/arbiter/equity-gate together --
this function rules on one round given what it's handed and doesn't track round state
itself. The one thing it does need across rounds is `prior_equity_gate_rejection`: when the
graph loops back here after the equity gate vetoed a "revised" ruling, that rejection is
passed in so a second ruling can react to it (e.g. red-circle instead of repeating a
revision that will just be vetoed again) rather than ruling blind a second time.
"""

from __future__ import annotations

from pathlib import Path

from agents import instrumented_model
from agents.model_router import get_model
from agents.negotiation_schemas import ArbiterRuling, CrosswalkArgument, EquityGateResult
from agents.schemas import LevelingDecision

FRAMEWORK_PATH = Path(__file__).resolve().parent.parent / "docs" / "level_framework.md"

_SYSTEM_PROMPT_TEMPLATE = """You are Meridian Silicon's arbiter in the crosswalk negotiation \
(level_framework.md section 7). A mapping the crosswalk agent produced has been contested by \
the acquired-side advocate; you rule on it.

You apply this framework only -- sections 5 (leveling rules) and 7 (crosswalk negotiation) \
below -- never Nyx's own leveling document, which you do not have access to. The advocate's \
argument may quote or paraphrase Nyx's own document; treat any such quote only as evidence \
about the employee's described scope, the same as you would treat a phrase lifted from a job \
description. It is not a second framework you weigh against this one.

Arbiter standard (section 7): apply this framework, not a midpoint between the crosswalk \
mapping and the advocate's proposed level. Splitting the difference is not an available \
verdict -- there is no "in-between" level, and final_level must be either the original \
mapping's level or a level this framework's own rules actually justify. Most original \
mappings should be upheld; the advocate should win only when the framework, applied to the \
evidence actually cited, supports a different outcome than the original mapping reached.

You must choose exactly one of four verdicts:
- **upheld** -- the original mapping stands. final_level equals the original assigned_level. \
The advocate's argument doesn't change the result under this framework.
- **revised** -- the mapping changes. final_level is a new level this framework's own rules \
require given the evidence cited -- not merely a level the advocate's proposed_level asked \
for, and not a compromise between the two positions.
- **red_circled** -- final_level equals the original (lower) assigned_level, same as upheld, \
but you are flagging that the case has genuine merit: the evidence cited would support a \
higher level under this framework's ordinary anchors, and a specific, named rule is what's \
actually holding the level down regardless (for example, rule 4's external-recognition \
requirement for L7 and above) -- not just a close factor call that rule 2 already resolves. \
Use this instead of a flat "upheld" whenever that's the shape of the case. The pay-protection \
mechanics themselves are a downstream comp calculation, not something you compute -- your \
job is only to flag that this case qualifies. Section 7 expects red-circling to be the \
realistic resolution for most genuinely contested cases, not a rare fallback.
- **escalated** -- reserved for a genuine gap in the framework itself: applied carefully, the \
rules do not resolve which level governs. This is not for cases that are merely close calls \
-- rule 2 already tells you how to resolve an ordinary split (take the lower level unless \
scope of impact is unambiguously higher) -- it is for cases the framework does not speak to \
at all.

governing_rule must cite a specific rule from section 5 (or section 6, if source-organization \
calibration is what's actually doing the work) by number, e.g. "rule 2" or "section 6 rule 3" \
-- never an unattributed judgment call.

Full framework document:

---
{framework}
---
"""


def _load_framework() -> str:
    return FRAMEWORK_PATH.read_text()


def _build_human_message(
    crosswalk_decision: LevelingDecision,
    advocate_argument: CrosswalkArgument,
    prior_equity_gate_rejection: EquityGateResult | None = None,
) -> str:
    factor_lines = "\n".join(
        f"  - {r.factor}: {r.level_indicated} -- {r.evidence}" for r in crosswalk_decision.factor_ratings
    )
    crosswalk_block = (
        "Original crosswalk mapping:\n"
        f"- track: {crosswalk_decision.track}\n"
        f"- assigned_level: {crosswalk_decision.assigned_level}\n"
        f"- factor5_variant_applied: {crosswalk_decision.factor5_variant_applied}\n"
        f"- governing_rule: {crosswalk_decision.governing_rule}\n"
        f"- factor ratings:\n{factor_lines}\n"
        f"- reasoning: {crosswalk_decision.reasoning}"
    )
    argument_block = (
        "Advocate's argument:\n"
        f"- argument_basis: {advocate_argument.argument_basis}\n"
        f"- proposed_level: {advocate_argument.proposed_level}\n"
        f"- evidence_cited: {advocate_argument.evidence_cited}\n"
        f"- framework_section (from the advocate's own document, not Meridian's): "
        f"{advocate_argument.framework_section}"
    )
    parts = [crosswalk_block, argument_block]
    if prior_equity_gate_rejection is not None:
        parts.append(
            "This is a second round: you already ruled once, and that ruling was 'revised'. "
            "The equity gate then rejected it (section 7 -- every revision passes to the "
            "equity agent before it's final):\n"
            f"- conflicting_incumbents: {prior_equity_gate_rejection.conflicting_incumbents}\n"
            f"- reasoning: {prior_equity_gate_rejection.reasoning}\n"
            "A revision to the same level will be rejected by the gate again for the same "
            "reason -- it isn't a fresh fact to weigh, it's a constraint your ruling now has "
            "to work within. Red-circling (upheld at the original level, pay protection "
            "flagged) is the standard resolution when a revision that has genuine merit runs "
            "into a hard constraint like this one; consider it rather than repeating the same "
            "revision. This is the final round -- an unresolved outcome here escalates."
        )
    parts.append("Rule on this contest.")
    return "\n\n".join(parts)


def would_hit_cache(
    crosswalk_decision: LevelingDecision,
    advocate_argument: CrosswalkArgument,
    model=None,
    prior_equity_gate_rejection: EquityGateResult | None = None,
) -> bool:
    """Reports whether a call with this exact input would be served from cache, without
    making any call or touching the budget -- mirrors agents.leveling.would_hit_cache and
    agents.advocate.would_hit_cache."""
    llm = model or get_model("judgment")
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT_TEMPLATE.format(framework=_load_framework())},
        {
            "role": "user",
            "content": _build_human_message(crosswalk_decision, advocate_argument, prior_equity_gate_rejection),
        },
    ]
    return instrumented_model.would_hit_cache(llm, ArbiterRuling, messages)


def rule(
    crosswalk_decision: LevelingDecision,
    advocate_argument: CrosswalkArgument,
    model=None,
    prior_equity_gate_rejection: EquityGateResult | None = None,
) -> ArbiterRuling:
    """Rules on one round. Returns a validated ArbiterRuling -- schema already enforces one
    of the four verdicts and a governing_rule that cites a rule by number
    (agents/negotiation_schemas.py).

    `prior_equity_gate_rejection` is set only on a second-round call, when
    agents/negotiation_graph.py is looping back here after the equity gate vetoed the first
    ruling's revision -- see module docstring.
    """
    llm = model or get_model("judgment")
    structured_llm = llm.with_structured_output(ArbiterRuling)

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(framework=_load_framework())
    human_message = _build_human_message(crosswalk_decision, advocate_argument, prior_equity_gate_rejection)

    return structured_llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_message},
        ]
    )
