"""Pydantic schemas for the crosswalk negotiation subgraph (level_framework.md section 7).
No contested mapping's outcome reaches the exception register unvalidated -- same discipline
as agents/schemas.py's LevelingDecision for the leveling agent.

Every schema here that is (or could become) the direct target of a with_structured_output
call is audited against the flag-plus-conditional-nesting antipattern: a bool field gating
whether a sibling Optional[BaseModel] field is populated. That shape is what produced
repeated malformed tool calls from Claude on AdvocateOutput (see its docstring and
docs/error_handling_backlog.md), so it's flattened there; EquityGateResult and
ExceptionRegisterEntry carry their own audit notes explaining why they were left as is."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from agents.schemas import LevelCode

# Section 7, "Admissible arguments": the advocate may argue only from evidence and from
# documented framework language. Title, retention risk, pay, tenure and peer pay are
# explicitly inadmissible -- CrosswalkArgument.argument_basis has no slot for them.
CrosswalkArgumentBasis = Literal[
    "scope evidence not reflected in the mapping",
    "misapplied factor variant",
    "misread factor anchor",
    "Meridian precedent",
]

Verdict = Literal["upheld", "revised", "red_circled", "escalated"]


class CrosswalkArgument(BaseModel):
    """One round of the acquired-side advocate's contest of a crosswalk mapping."""

    argument_basis: CrosswalkArgumentBasis
    proposed_level: LevelCode = Field(
        description="The Meridian level the advocate argues for -- a single code, same "
        "constraint as LevelingDecision.assigned_level."
    )
    evidence_cited: str = Field(description="The specific evidence supporting this argument")
    framework_section: str = Field(
        description="Which document and section the argument draws on, e.g. "
        "'nyx_level_framework.md section 3' or 'level_framework.md section 5 rule 3' for "
        "Meridian precedent"
    )


class AdvocateOutput(BaseModel):
    """The advocate agent's structured output for one round (agents/advocate.py).

    Flat by design -- no `contests: bool` paired with a conditionally-required nested
    `CrosswalkArgument` object. That shape (a flag gating an Optional[BaseModel]) is what
    repeatedly triggered malformed tool calls from Claude in practice: the model would drop
    the flag field, or collapse the nested object into a string, apparently because it was
    trying to express "whether" and "what" as one decision and failing to split it back into
    two separate places in the schema (see docs/error_handling_backlog.md's advocate-agent
    entry -- 6 consecutive failures on one case before a retry succeeded). Every field here
    is instead optional at the top level, all four moving together: not contesting means
    every field is null, contesting means every field is set. No partial state, and no
    nested object for a single structured-output call to get wrong.

    `contests` and `as_crosswalk_argument()` below are ordinary Python, not schema fields --
    computed from whichever state the flat fields are actually in, so there's nothing to
    validate for consistency because there's no second, redundant flag to drift from them.
    """

    argument_basis: CrosswalkArgumentBasis | None = Field(
        default=None,
        description="Null when not contesting. Kept as the same admissible-categories "
        "Literal as CrosswalkArgument.argument_basis when contesting -- flattening the "
        "shape doesn't relax which bases are admissible.",
    )
    proposed_level: LevelCode | None = Field(
        default=None, description="Null when not contesting; a single LevelCode when contesting."
    )
    evidence_cited: str | None = Field(default=None, description="Null when not contesting.")
    framework_section: str | None = Field(default=None, description="Null when not contesting.")

    @model_validator(mode="after")
    def _all_fields_move_together(self) -> "AdvocateOutput":
        populated = [
            self.argument_basis is not None,
            self.proposed_level is not None,
            self.evidence_cited is not None,
            self.framework_section is not None,
        ]
        if any(populated) and not all(populated):
            raise ValueError(
                "argument_basis, proposed_level, evidence_cited and framework_section must "
                "be either all null (not contesting) or all set (contesting) -- no partial "
                "state."
            )
        return self

    @property
    def contests(self) -> bool:
        return self.argument_basis is not None

    def as_crosswalk_argument(self) -> CrosswalkArgument | None:
        """Builds the typed CrosswalkArgument this advocate is arguing for, for callers
        (the arbiter, the exception register) that want the nested representation -- built
        here by plain Python from an already-validated flat result, never itself parsed out
        of a model's tool call. None when not contesting."""
        if not self.contests:
            return None
        return CrosswalkArgument(
            argument_basis=self.argument_basis,
            proposed_level=self.proposed_level,
            evidence_cited=self.evidence_cited,
            framework_section=self.framework_section,
        )


class ArbiterRuling(BaseModel):
    """The arbiter's ruling on one round of a contested mapping (section 7, "Arbiter
    standard"). The arbiter applies level_framework.md, not a midpoint between positions --
    governing_rule must cite a specific numbered rule, never a vibe."""

    verdict: Verdict
    governing_rule: str = Field(
        description="The specific numbered rule that governed this ruling, e.g. "
        "'rule 2: lower level governs a split' or 'section 6 rule 3: platform dependency'"
    )
    final_level: LevelCode
    reasoning: str = Field(description="Brief rationale tying the ruling to the governing rule")

    @model_validator(mode="after")
    def _governing_rule_cites_a_number(self) -> "ArbiterRuling":
        if not any(char.isdigit() for char in self.governing_rule):
            raise ValueError(
                "governing_rule must cite a rule by number (e.g. 'rule 2') -- the arbiter "
                "applies the framework, not an unattributed judgment call."
            )
        return self


class EquityGateResult(BaseModel):
    """The equity agent's check on a proposed revision (section 7, "Equity gate"). A
    revision that places an acquired employee above Meridian incumbents with demonstrably
    greater scope at the same level is rejected, with the conflicting incumbents named.

    Audited against the same flag-plus-conditional-nesting antipattern AdvocateOutput was
    flattened out of: `passed` gates `conflicting_incumbents` the same shape `contests` used
    to gate `argument`. Left as is deliberately -- `conflicting_incumbents` is a
    `list[str]`, not a nested `BaseModel`; the observed failures were specifically about an
    Optional[BaseModel] (its own required sub-fields) gated by a sibling flag, not about a
    flag paired with a conditionally-empty list, which has a much simpler JSON schema for a
    model to fill in correctly. No incident on this schema so far to justify flattening it
    pre-emptively; worth revisiting if one shows up."""

    passed: bool
    conflicting_incumbents: list[str] = Field(
        default_factory=list,
        description="Meridian incumbent IDs the revision would place below the acquired "
        "employee despite greater demonstrated scope. Required (non-empty) when passed is "
        "False; must be empty when passed is True.",
    )
    reasoning: str

    @model_validator(mode="after")
    def _conflicts_match_passed(self) -> "EquityGateResult":
        if self.passed and self.conflicting_incumbents:
            raise ValueError("passed=True must have no conflicting_incumbents.")
        if not self.passed and not self.conflicting_incumbents:
            raise ValueError(
                "passed=False requires at least one conflicting incumbent -- a rejection "
                "must name who it conflicts with."
            )
        return self


class ExceptionRegisterEntry(BaseModel):
    """One row of the exception register (data_model_spec.md section 2, `exception_register`).
    Every contested case is written here regardless of verdict, with both arguments, the
    ruling, and the governing rule -- section 7's provenance requirement for a judgment call.

    Audited for the same flag-plus-conditional-nesting antipattern AdvocateOutput was
    flattened out of: `advocate_argument` and `arbiter_ruling` are always-required nested
    models (no flag gates them, and no partial state is possible), and `equity_gate_result`
    is Optional but not gated by a sibling bool field on this model -- its presence tracks
    `verdict` instead (deliberately left unvalidated against it; see that field's own
    docstring). None of that is actually at risk from the failure mode this file's other
    schemas were rewritten for, because this model is never itself the target of a
    with_structured_output call -- it's assembled by plain Python code from
    already-validated CrosswalkArgument/ArbiterRuling/EquityGateResult results the advocate,
    arbiter and equity agents produced separately, not generated whole by a single model
    call. The nested-model fragility is specifically a structured-output-parsing problem;
    an object built directly in Python from already-parsed pieces never goes through that
    path."""

    case_id: str
    employee_id: str
    crosswalk_level: LevelCode = Field(
        description="The crosswalk agent's original mapping, before it was contested"
    )
    advocate_position: LevelCode = Field(
        description="The level the advocate argued for -- denormalized from "
        "advocate_argument.proposed_level for the register's own column"
    )
    advocate_argument: CrosswalkArgument
    arbiter_ruling: ArbiterRuling
    governing_rule_cited: str = Field(
        description="Denormalized from arbiter_ruling.governing_rule for the register's own column"
    )
    equity_gate_result: EquityGateResult | None = Field(
        default=None,
        description="Set whenever the ruling passed through the equity gate. Left "
        "unvalidated against verdict here -- whether a red-circled case is required to pass "
        "through the gate the same way a revised one is is a negotiation-rules question, "
        "not an engineering one (see CLAUDE.md); flagged for the comp manager rather than "
        "assumed.",
    )
    verdict: Verdict = Field(description="Denormalized from arbiter_ruling.verdict")
    round_count: int = Field(ge=1, le=2, description="Two rounds maximum (section 7, round limit)")

    @model_validator(mode="after")
    def _denormalized_fields_match_ruling(self) -> "ExceptionRegisterEntry":
        if self.advocate_position != self.advocate_argument.proposed_level:
            raise ValueError(
                "advocate_position must match advocate_argument.proposed_level -- they "
                "describe the same thing."
            )
        if self.verdict != self.arbiter_ruling.verdict:
            raise ValueError("verdict must match arbiter_ruling.verdict.")
        if self.governing_rule_cited != self.arbiter_ruling.governing_rule:
            raise ValueError("governing_rule_cited must match arbiter_ruling.governing_rule.")
        return self
