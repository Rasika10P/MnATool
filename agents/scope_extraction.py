"""Structured scope extraction from free-text job descriptions -- CLAUDE.md's model routing
table: "Nebius -- job description parsing." Pulls what the text states about reports-to,
span of control, budget authority, decision scope and ownership scope into a ScopeProfile.

Extraction only. No leveling judgment happens here -- assigning a level from these signals
is agents.leveling's job. Nebius was tried on leveling itself first (agents.leveling.
level_role_routed, scripts/level_five_jobs_nebius_vs_claude.py, kept in the repo as evidence)
and dropped: it leveled one notch high on 4 of 5 test cases while reporting high confidence
on all five, including the one Claude flagged for escalation -- a calibration problem a
confidence threshold can't route around. Extraction is a narrower task (pull out what's
stated, don't judge it) that doesn't call for the same adjudication Claude provides.
"""

from __future__ import annotations

from agents import instrumented_model
from agents.model_router import get_model
from agents.schemas import ScopeProfile

_SYSTEM_PROMPT = """You extract a structured scope profile from a job description. This is \
extraction only -- you are not leveling this role, and you must not judge, infer, or guess \
beyond what the text actually states.

For reports_to, span_of_control and budget_authority: an explicit negative is a finding to \
record, not a blank to leave empty. If the text explicitly addresses one of these -- even \
only to say there is none, e.g. "no direct reports" or "no budget authority beyond \
headcount requisitions" -- set stated=true and put what the text says in value, close to \
its own wording, including the negative itself. Only set stated=false (and leave value \
null) when the text is completely silent on that dimension -- never mentions it either way. \
Do not treat "the text doesn't say anything positive" as a reason to leave a field \
unstated when the text actually states a negative.

For decision_scope and ownership_scope, report only what the text states, close to its own \
wording. Do not fill a gap with an assumption about what a role like this "usually" has. Do \
not assign a level, track, or any leveling-framework term, and do not draw on outside \
knowledge of how similar roles are typically scoped elsewhere -- extract only from the text \
given."""


def _build_messages(job_description: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Job description:\n{job_description.strip()}"},
    ]


def would_hit_cache(job_description: str, model=None) -> bool:
    """Reports whether a call with this exact input would be served from cache, without
    making any call or touching the budget -- what --dry-run reports on, same convention as
    agents.leveling.would_hit_cache."""
    llm = model or get_model("volume")
    return instrumented_model.would_hit_cache(llm, ScopeProfile, _build_messages(job_description))


def extract_scope_profile(job_description: str, model=None) -> ScopeProfile:
    """Extracts a ScopeProfile from one job description. `model` defaults to
    get_model("volume") (Nebius) -- the production routing for this task; pass an explicit
    model (e.g. get_model("judgment")) to run the same extraction on Claude for comparison,
    as scripts/parse_five_jobs_nebius_vs_claude.py does."""
    llm = model or get_model("volume")
    structured_llm = llm.with_structured_output(ScopeProfile)
    return structured_llm.invoke(_build_messages(job_description))
