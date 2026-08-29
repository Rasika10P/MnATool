"""The acquired-side advocate (level_framework.md section 7, "Participants"). Contests a
proposed crosswalk mapping, or declines to.

Scoped deliberately to docs/nyx_level_framework.md alone -- Meridian's own framework
(docs/level_framework.md) is never loaded into this prompt and the system message tells the
model not to reason as if it had it. That constraint, not just admissibility, is why this
advocate can only ever argue "scope evidence not reflected in the mapping": the other three
admissible categories in section 7 (misapplied factor variant, misread factor anchor,
Meridian precedent) are all claims about Meridian's own framework or population, which this
agent structurally has no access to. _validate_admissible_for_advocate enforces that as a
guardrail rather than trusting the prompt alone.
"""

from __future__ import annotations

from pathlib import Path

from agents import instrumented_model
from agents.model_router import get_model
from agents.negotiation_schemas import AdvocateOutput

NYX_FRAMEWORK_PATH = Path(__file__).resolve().parent.parent / "docs" / "nyx_level_framework.md"

# The only basis this advocate can support without access to Meridian's own framework or
# population -- see module docstring.
_AVAILABLE_BASIS = "scope evidence not reflected in the mapping"

_SYSTEM_PROMPT_TEMPLATE = """You are the acquired-side advocate in Nyx Semiconductor's \
acquisition by Meridian Silicon. You represent Nyx employees during crosswalk negotiation \
(level_framework.md section 7) when a proposed mapping into Meridian's job architecture \
looks like it doesn't reflect their actual scope.

You have access to Nyx's own leveling document below, and to nothing else about Meridian's \
architecture. You do not know Meridian's factor definitions, its level anchors, its factor \
variants, or its population of leveling precedent -- do not invent or guess at any of them, \
and do not reason as if you had read Meridian's framework. Your only tool is a careful, \
literal reading of Nyx's own document against the employee's described scope.

Because of that restriction, the only argument available to you is: the scope evidence in \
this employee's role summary is not reflected in the proposed Meridian level. You cannot \
argue that Meridian misapplied a factor variant, misread a factor anchor, or ignored \
Meridian precedent -- those are claims about Meridian's own framework and population, and \
you have no basis to make them.

The following are never leveling arguments, however real they are as concerns:
- Retention risk or morale. Real, but a compensation remedy (a retention award or \
red-circling), never a reason to move a level.
- Title or seniority alone. Nyx's own document (section 3) treats title as the record of a \
past leveling decision, not evidence for a new one.
- Current pay, or what peers at Nyx were paid. Not scope evidence.
Do not raise any of these, even to acknowledge them as context.

Decide whether to contest:
- Contest only if you can point to specific scope evidence in the role summary, read \
against Nyx's own level anchors below, that plausibly supports a different Meridian level \
than the one proposed.
- If the proposed level already looks consistent with the scope described, decline to \
contest. Most mappings should stand -- section 7 expects the advocate to win only a \
minority of contested cases, and that starts with not contesting cases that don't warrant it.

Nyx's own leveling document, read literally:

---
{framework}
---
"""


def _load_framework() -> str:
    return NYX_FRAMEWORK_PATH.read_text()


def _build_human_message(role_summary: str, nyx_level: str, proposed_meridian_level: str) -> str:
    return (
        f"Employee's Nyx level: {nyx_level}\n\n"
        f"Role summary (from Nyx's HR records):\n{role_summary.strip()}\n\n"
        f"Proposed Meridian mapping: {proposed_meridian_level}\n\n"
        "Decide whether to contest this mapping. If you do, state the argument basis, the "
        "level you propose instead, the specific evidence, and which section of Nyx's "
        "document it comes from. If you don't, leave all of those fields null."
    )


def _validate_admissible_for_advocate(output: AdvocateOutput) -> AdvocateOutput:
    if output.contests and output.argument_basis != _AVAILABLE_BASIS:
        raise ValueError(
            f"Advocate produced argument_basis={output.argument_basis!r}, but this agent has "
            f"no access to Meridian's framework or population -- only {_AVAILABLE_BASIS!r} "
            "is available to it. This is a model error, not a valid argument; do not pass it "
            "through to the arbiter."
        )
    return output


def would_hit_cache(role_summary: str, nyx_level: str, proposed_meridian_level: str, model=None) -> bool:
    """Reports whether a call with this exact input would be served from cache, without
    making any call or touching the budget -- what --dry-run reports on. Mirrors
    agents.leveling.would_hit_cache."""
    llm = model or get_model("judgment")
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT_TEMPLATE.format(framework=_load_framework())},
        {"role": "user", "content": _build_human_message(role_summary, nyx_level, proposed_meridian_level)},
    ]
    return instrumented_model.would_hit_cache(llm, AdvocateOutput, messages)


def contest_mapping(
    role_summary: str,
    nyx_level: str,
    proposed_meridian_level: str,
    model=None,
) -> AdvocateOutput:
    """Runs the advocate on one employee. Returns a validated AdvocateOutput: either
    .contests is False (decline, all fields null), or True with argument_basis /
    proposed_level / evidence_cited / framework_section all populated -- call
    .as_crosswalk_argument() to get the typed CrosswalkArgument the arbiter and exception
    register expect.

    `proposed_meridian_level` is a LevelCode string produced upstream by the crosswalk agent
    (agents/leveling.py's level_role, applied to the employee's Nyx role with
    SourceOrgContext per section 6) -- this function takes it as given rather than computing
    it, so the advocate step stays a single, testable call.
    """
    llm = model or get_model("judgment")
    structured_llm = llm.with_structured_output(AdvocateOutput)

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(framework=_load_framework())
    human_message = _build_human_message(role_summary, nyx_level, proposed_meridian_level)

    output = structured_llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_message},
        ]
    )
    return _validate_admissible_for_advocate(output)
