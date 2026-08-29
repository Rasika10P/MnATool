"""
The leveling agent, as a plain function -- no LangGraph yet (build order item 2; the graph
wrapper comes in item 2's LangGraph step). Applies docs/level_framework.md literally: the
full document is loaded into the prompt rather than paraphrased, so "literal application"
isn't at the mercy of a summary drifting from the source.
"""

from __future__ import annotations

from pathlib import Path

from agents import instrumented_model
from agents.model_router import get_model
from agents.schemas import LevelingDecision, ScopeProfile, SourceOrgContext

FRAMEWORK_PATH = Path(__file__).resolve().parent.parent / "docs" / "level_framework.md"

_SYSTEM_PROMPT_TEMPLATE = """You are Meridian Silicon's leveling agent. You apply the level \
framework below literally. Every leveling decision must cite specific factors and anchors \
from it -- do not use outside knowledge of how other companies level roles.

Rules you must follow without exception (section 5 is binding):
- Rate every applicable factor: six for an IC role, seven for a manager role (span & budget \
included). Factor 5 is never skipped, and you must name which variant (5a/5b/5c) you applied \
based on the role's family group.
- Experience (years) is not a factor. If your reasoning leans on tenure, that reasoning is wrong.
- If a role rates at two adjacent levels, assign the LOWER one -- unless scope of impact is \
unambiguously at the higher level (rule 2).
- Deep-but-narrow technical work, without cross-domain breadth and organizational influence, \
caps at L5 -- it cannot reach L6 (rule 3).
- L7 and above require external recognition (publications, patents, standards bodies, external \
reputation) -- this cannot be inferred from internal reputation alone (rule 4).
- The job title is evidence, never the level itself. Level strictly from described scope \
(rule 6). A "Director" or "VP" title with no direct reports described is not automatically \
on the manager track.
- If source organization context is provided, treat it only as a prior to sanity-check your \
result against, per section 6 -- never as an input that shifts the level directly. State \
explicitly whether your result is consistent with or contradicts the prior, and if it \
contradicts, treat that as a signal to lower your confidence and consider escalating.
- If source organization context includes high platform_dependency, cap the technical \
depth/breadth factor at the level the person could sustain without that shared support \
(section 6, rule 3).
- If this is genuinely a close call, set escalation_factor to the specific factor whose \
resolution would settle it, and reflect the uncertainty in a lower confidence score.
- If an extracted scope profile is provided below, it is advisory -- a separate parsing \
step's summary of the job description, not a substitute for it. The job description text \
is authoritative; weigh the extracted profile as you would any other evidence, and if it \
seems to conflict with your own reading of the text, trust the text.

Full framework document:

---
{framework}
---
"""


def _load_framework() -> str:
    return FRAMEWORK_PATH.read_text()


def _render_scope_finding(field: str, finding) -> str:
    if not finding.stated:
        return f"- {field}: not mentioned in the text"
    return f"- {field}: explicitly stated -- {finding.value!r}"


def _render_scope_profile(scope_profile: ScopeProfile) -> str:
    lines = [
        _render_scope_finding("reports_to", scope_profile.reports_to),
        _render_scope_finding("span_of_control", scope_profile.span_of_control),
        _render_scope_finding("budget_authority", scope_profile.budget_authority),
        f"- decision_scope (extracted): {scope_profile.decision_scope}",
        f"- ownership_scope (extracted): {scope_profile.ownership_scope}",
    ]
    return "\n".join(lines)


def _build_human_message(
    job_description: str,
    source_org_context: SourceOrgContext | None,
    scope_profile: ScopeProfile | None = None,
) -> str:
    parts = [f"Job description:\n{job_description.strip()}"]
    if source_org_context is not None:
        context_lines = "\n".join(
            f"- {field}: {value}"
            for field, value in source_org_context.model_dump(exclude_none=True).items()
        )
        if context_lines:
            parts.append(f"Source organization context (section 6):\n{context_lines}")
        else:
            parts.append("Source organization context: none provided -- this is an internal Meridian role.")
    else:
        parts.append("Source organization context: none provided -- this is an internal Meridian role.")
    if scope_profile is not None:
        parts.append(
            "Extracted scope profile (advisory -- parsed from the job description above by a "
            "separate step; the job description text remains authoritative):\n"
            + _render_scope_profile(scope_profile)
        )
    return "\n\n".join(parts)


def would_hit_cache(
    job_description: str,
    source_org_context: SourceOrgContext | None,
    model=None,
    scope_profile: ScopeProfile | None = None,
) -> bool:
    """Reports whether a call with this exact input would be served from cache, without
    making any call or touching the budget -- what --dry-run reports on. Delegates to the
    generic checker in agents.instrumented_model, which is where the cache-key derivation
    actually lives now."""
    llm = model or get_model("judgment")
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT_TEMPLATE.format(framework=_load_framework())},
        {"role": "user", "content": _build_human_message(job_description, source_org_context, scope_profile)},
    ]
    return instrumented_model.would_hit_cache(llm, LevelingDecision, messages)


def _compute_escalate(confidence: float, escalation_factor: str | None, low: float, high: float) -> bool:
    """Rule 9's trigger, as a band rather than a single cutoff.

    A single threshold is unstable right at the boundary: the same case, run twice, came
    back with confidence 0.68 and 0.72 -- straddling any one cutoff placed near there and
    flipping escalate on nothing but sampling noise, not a real change in the decision.
    Below `low`, always escalate. Above `high`, never escalate. In between, escalate only if
    the model itself named a specific escalation_factor -- it did on both the 0.68 and 0.72
    runs, so this band escalates both instead of disagreeing with itself.
    """
    if confidence < low:
        return True
    if confidence > high:
        return False
    return escalation_factor is not None


def _run_leveling_call(
    job_description: str,
    source_org_context: SourceOrgContext | None,
    low_confidence_threshold: float,
    high_confidence_threshold: float,
    model,
    scope_profile: ScopeProfile | None = None,
) -> LevelingDecision:
    """The actual model call: build the prompt, get a validated decision, compute escalate.
    Both level_role (plain function) and the LangGraph level node (agents/leveling_graph.py)
    call this directly, so there is exactly one code path that can produce a decision --
    the graph conversion cannot introduce a behavior difference here even by accident.

    `scope_profile` is advisory (see the system prompt's rule on it): the graph's level node
    passes what its parse node extracted; level_role leaves it None unless a caller supplies
    one explicitly, so a bare level_role() call is unaffected by this parameter's existence.

    Caching, cost logging, session stats and the spend budget are not this function's
    concern -- get_model() returns a model with all four already applied
    (agents/instrumented_model.py), so this is plain with_structured_output(schema).invoke()
    like any other LangChain call.
    """
    llm = model or get_model("judgment")
    structured_llm = llm.with_structured_output(LevelingDecision)

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(framework=_load_framework())
    human_message = _build_human_message(job_description, source_org_context, scope_profile)

    decision = structured_llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_message},
        ]
    )

    escalate = _compute_escalate(
        decision.confidence, decision.escalation_factor, low_confidence_threshold, high_confidence_threshold
    )
    return decision.model_copy(update={"escalate": escalate})


def level_role(
    job_description: str,
    source_org_context: SourceOrgContext | None = None,
    low_confidence_threshold: float = 0.65,
    high_confidence_threshold: float = 0.75,
    model=None,
    scope_profile: ScopeProfile | None = None,
) -> LevelingDecision:
    """Level one role. Returns a validated LevelingDecision.

    `escalate` is computed by _compute_escalate from confidence (and, in the band between
    the two thresholds, escalation_factor), not left to the model to self-report -- rule 9's
    trigger is deterministic even though the confidence value itself is the model's judgment
    call.
    """
    return _run_leveling_call(
        job_description, source_org_context, low_confidence_threshold, high_confidence_threshold, model,
        scope_profile=scope_profile,
    )


def level_role_routed(
    job_description: str,
    source_org_context: SourceOrgContext | None = None,
    low_confidence_threshold: float = 0.65,
    high_confidence_threshold: float = 0.75,
    nebius_escalation_threshold: float = 0.75,
    scope_profile: ScopeProfile | None = None,
) -> dict:
    """Provider routing per CLAUDE.md's model routing table: first-pass leveling runs on
    Nebius (get_model("volume")). If Nebius's own confidence comes back below
    nebius_escalation_threshold, a second pass runs on Claude (get_model("judgment")) and
    its decision -- not Nebius's -- is what gets returned.

    This is which *provider* serves the decision -- distinct from LevelingDecision.escalate,
    which flags a decision for human review per rule 9 and can still be True on a
    Claude-served decision. Returns both passes (nebius_pass is None when Nebius's own
    confidence already cleared the bar) so the routing decision itself carries provenance,
    per CLAUDE.md non-negotiable 2 -- which provider produced a decision is exactly the kind
    of thing that must be attributable, not just the dollar figures.
    """
    nebius_pass = _run_leveling_call(
        job_description, source_org_context, low_confidence_threshold, high_confidence_threshold,
        model=get_model("volume"), scope_profile=scope_profile,
    )
    if nebius_pass.confidence >= nebius_escalation_threshold:
        return {"decision": nebius_pass, "served_by": "nebius", "nebius_pass": None}

    claude_pass = _run_leveling_call(
        job_description, source_org_context, low_confidence_threshold, high_confidence_threshold,
        model=get_model("judgment"), scope_profile=scope_profile,
    )
    return {"decision": claude_pass, "served_by": "anthropic", "nebius_pass": nebius_pass}
