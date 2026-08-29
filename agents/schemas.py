"""Pydantic schemas for agent output. The leveling agent's output is validated against
LevelingDecision -- no leveling decision reaches the rest of the system unvalidated."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from agents.text_sanitization import sanitize_prose_field

FACTOR_NAMES = (
    "scope_of_impact",
    "autonomy",
    "problem_complexity",
    "technical_depth_breadth",
    "ownership_scope",
    "influence",
    "span_and_budget",
)

# The 13 level codes (section 2). A single code only -- manager/IC equivalence (e.g. M3 -> L4)
# is a lookup against level_definitions.ic_equivalent, not something to encode here as
# combined notation like "M3/L4".
LevelCode = Literal[
    "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8",
    "M3", "M4", "M5", "M6", "M7",
]


class SourceOrgContext(BaseModel):
    """Section 6 of level_framework.md: context for leveling a role from outside Meridian
    (acquisitions, external hires). All fields optional -- omit entirely for an internal req."""

    source_headcount: int | None = None
    source_stage: Literal["seed", "growth", "late-stage private", "public"] | None = None
    source_type: Literal["whole company", "carve-out", "asset purchase"] | None = None
    parent_headcount: int | None = Field(
        default=None, description="Only set when source_type is carve-out"
    )
    org_depth: int | None = Field(
        default=None, description="Levels between IC and CEO in the source org"
    )
    platform_dependency: Literal["low", "medium", "high"] | None = None


class ScopeFinding(BaseModel):
    """One optional scope dimension, distinguishing "the text never brings this up" from
    "the text explicitly addresses it, including to say there is none." The two are
    different findings -- level_framework.md rule 6 needs to tell them apart: an explicit
    "no direct reports" on a Director title is affirmative evidence against the manager
    track, while a job description that simply never mentions reports is not evidence of
    anything either way.

    stated=True with value=None, or stated=False with a non-null value, are both invalid
    (see the validator below) -- the extraction model doesn't get to leave that ambiguous.
    """

    stated: bool = Field(
        description="True if the text addresses this dimension at all, even only to say "
        "there is none, e.g. 'no direct reports' or 'no budget authority'. False if the "
        "text is simply silent on it."
    )
    value: str | None = Field(
        default=None,
        description="What the text says, close to its own wording -- including an explicit "
        "negative like 'no direct reports'. Must be set (non-null) whenever stated is True, "
        "and must be null when stated is False.",
    )

    # error_handling_backlog.md entry 1 (agents/text_sanitization.py) -- runs before
    # _value_matches_stated below, so that validator sees the already-cleaned string.
    _sanitize_value = field_validator("value", mode="before")(staticmethod(sanitize_prose_field))

    @model_validator(mode="after")
    def _value_matches_stated(self) -> "ScopeFinding":
        if self.stated and self.value is None:
            raise ValueError(
                "stated=True requires a non-null value -- record what the text explicitly "
                "says, even when it's an explicit negative like 'no direct reports'."
            )
        if not self.stated and self.value is not None:
            raise ValueError("stated=False must have value=None -- if the text says something, stated must be True.")
        return self


class ScopeProfile(BaseModel):
    """Structured extraction of scope signals from a free-text job description -- CLAUDE.md's
    model routing table: Nebius handles job description parsing. This is extraction only --
    what the text states, close to its own wording. No leveling judgment: no level, no track,
    no factor rating belongs on this model."""

    reports_to: ScopeFinding = Field(
        description="Who this role reports to, e.g. 'VP of Engineering' or 'the two co-founders jointly'."
    )
    span_of_control: ScopeFinding = Field(
        description="The reporting structure this role manages -- headcount and team makeup, e.g. '6 direct reports, all individual contributors', or an explicit 'no direct reports'."
    )
    budget_authority: ScopeFinding = Field(
        description="What budget or spend authority is described, e.g. 'no budget authority beyond headcount requisitions'."
    )
    decision_scope: str = Field(
        description="What decisions this role is described as making independently vs. needing approval for, as stated in the text."
    )
    ownership_scope: str = Field(
        description="What this role owns or is accountable for, as described in the text."
    )

    # error_handling_backlog.md entry 1 (agents/text_sanitization.py).
    _sanitize_prose = field_validator("decision_scope", "ownership_scope", mode="before")(
        staticmethod(sanitize_prose_field)
    )


class FactorRating(BaseModel):
    """One rating against one of the seven leveling factors (section 3)."""

    factor: Literal[
        "scope_of_impact", "autonomy", "problem_complexity",
        "technical_depth_breadth", "ownership_scope", "influence", "span_and_budget",
    ]
    level_indicated: LevelCode = Field(
        description="The single level this factor's evidence points to alone, e.g. 'L4'. "
        "Never combined notation like 'M3/L4' -- if a manager level's IC equivalent matters, "
        "that's a separate lookup against level_definitions.ic_equivalent, not this field."
    )
    evidence: str = Field(description="The specific evidence from the job description supporting this rating -- absolute, not relative language (section 6)")

    # error_handling_backlog.md entry 1 (agents/text_sanitization.py).
    _sanitize_evidence = field_validator("evidence", mode="before")(staticmethod(sanitize_prose_field))


class LevelingDecision(BaseModel):
    """The leveling agent's output. Every field traces to a specific rule or factor in
    level_framework.md -- this is what non-negotiable 2 (provenance) means for a judgment
    call rather than a dollar figure."""

    track: Literal["IC", "MGR"] = Field(description="Individual contributor or manager track")
    assigned_level: LevelCode
    factor_ratings: list[FactorRating] = Field(
        description="One rating per applicable factor -- six for IC (factors 1-6), seven for MGR (1-7, span & budget included). Factor 5 is never omitted."
    )
    factor5_variant_applied: Literal["5a", "5b", "5c"] = Field(
        description="5a product lifecycle (engineering), 5b process/program (corporate), 5c revenue responsibility (GTM)"
    )
    confidence: float = Field(ge=0.0, le=1.0)
    governing_rule: str = Field(
        description="Which section-5 rule governed this decision, e.g. 'rule 2: lower level governs a split'"
    )
    alternative_level: LevelCode | None = Field(
        default=None, description="The other level weighed, if any -- a single code, same constraint as level_indicated"
    )
    alternative_reasoning: str | None = Field(
        default=None, description="Why alternative_level was not chosen. Required whenever alternative_level is set."
    )
    escalate: bool = Field(
        default=False,
        description="Set by the caller from confidence vs. threshold, not by the model -- do not set this yourself.",
    )
    escalation_factor: str | None = Field(
        default=None,
        description="Per rule 9: if this decision is a close call, name the specific factor whose resolution would settle it, even if confidence turns out to be above threshold",
    )
    reasoning: str = Field(description="Brief rationale tying the factor evidence to the assigned level")

    # error_handling_backlog.md entry 1 (agents/text_sanitization.py) -- the exact failure
    # this entry documents was observed on this schema's reasoning/alternative_level pair.
    _sanitize_prose = field_validator(
        "governing_rule", "alternative_reasoning", "escalation_factor", "reasoning", mode="before"
    )(staticmethod(sanitize_prose_field))
