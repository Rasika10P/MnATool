import pytest
from pydantic import ValidationError

from agents.negotiation_schemas import (
    AdvocateOutput,
    ArbiterRuling,
    CrosswalkArgument,
    EquityGateResult,
    ExceptionRegisterEntry,
)


def _argument(**overrides):
    fields = dict(
        argument_basis="scope evidence not reflected in the mapping",
        proposed_level="L5",
        evidence_cited="owned PMIC subsystem across three tapeouts shipping 4M units",
        framework_section="nyx_level_framework.md section 3",
    )
    fields.update(overrides)
    return CrosswalkArgument(**fields)


def test_crosswalk_argument_accepts_admissible_basis():
    argument = _argument()
    assert argument.argument_basis == "scope evidence not reflected in the mapping"


@pytest.mark.parametrize(
    "inadmissible_basis",
    [
        "title",
        "retention risk",
        "morale",
        "current pay",
        "seniority",
        "tenure",
        "what peers at the acquired company received",
    ],
)
def test_crosswalk_argument_rejects_inadmissible_basis(inadmissible_basis):
    with pytest.raises(ValidationError):
        _argument(argument_basis=inadmissible_basis)


def test_crosswalk_argument_rejects_combined_notation_level():
    with pytest.raises(ValidationError):
        _argument(proposed_level="M3/L4")


def test_crosswalk_argument_rejects_unknown_level():
    with pytest.raises(ValidationError):
        _argument(proposed_level="L9")


def _contesting_advocate_output(**overrides):
    fields = dict(
        argument_basis="scope evidence not reflected in the mapping",
        proposed_level="L7",
        evidence_cited="company-wide final authority across the entire roadmap",
        framework_section="nyx_level_framework.md section 4",
    )
    fields.update(overrides)
    return AdvocateOutput(**fields)


def test_advocate_output_non_contesting_shape_validates():
    # The whole point of flattening: "not contesting" is just every field null, no separate
    # flag to keep in sync with that.
    output = AdvocateOutput()
    assert output.contests is False
    assert output.argument_basis is None
    assert output.proposed_level is None
    assert output.evidence_cited is None
    assert output.framework_section is None
    assert output.as_crosswalk_argument() is None


def test_advocate_output_contesting_shape_validates():
    output = _contesting_advocate_output()
    assert output.contests is True
    assert output.argument_basis == "scope evidence not reflected in the mapping"
    assert output.proposed_level == "L7"

    argument = output.as_crosswalk_argument()
    assert isinstance(argument, CrosswalkArgument)
    assert argument.argument_basis == "scope evidence not reflected in the mapping"
    assert argument.proposed_level == "L7"
    assert argument.evidence_cited == output.evidence_cited
    assert argument.framework_section == output.framework_section


@pytest.mark.parametrize(
    "overrides",
    [
        dict(argument_basis="scope evidence not reflected in the mapping"),
        dict(proposed_level="L7"),
        dict(evidence_cited="some evidence"),
        dict(framework_section="section 4"),
        dict(argument_basis="scope evidence not reflected in the mapping", proposed_level="L7"),
        dict(
            argument_basis="scope evidence not reflected in the mapping",
            proposed_level="L7",
            evidence_cited="some evidence",
        ),
    ],
)
def test_advocate_output_rejects_every_partial_state(overrides):
    with pytest.raises(ValidationError):
        AdvocateOutput(**overrides)


def test_advocate_output_rejects_inadmissible_basis():
    with pytest.raises(ValidationError):
        _contesting_advocate_output(argument_basis="retention risk")


def test_advocate_output_rejects_combined_notation_level():
    with pytest.raises(ValidationError):
        _contesting_advocate_output(proposed_level="M3/L4")


def _ruling(**overrides):
    fields = dict(
        verdict="upheld",
        governing_rule="rule 2: lower level governs a split",
        final_level="L4",
        reasoning="scope of impact was unambiguous at L4, not L5",
    )
    fields.update(overrides)
    return ArbiterRuling(**fields)


def test_arbiter_ruling_accepts_valid_verdict():
    assert _ruling().verdict == "upheld"


@pytest.mark.parametrize("verdict", ["upheld", "revised", "red_circled", "escalated"])
def test_arbiter_ruling_accepts_all_verdicts(verdict):
    assert _ruling(verdict=verdict).verdict == verdict


def test_arbiter_ruling_rejects_unknown_verdict():
    with pytest.raises(ValidationError):
        _ruling(verdict="split_the_difference")


def test_arbiter_ruling_requires_governing_rule_to_cite_a_number():
    with pytest.raises(ValidationError):
        _ruling(governing_rule="scope of impact was clearer at the lower level")


def test_arbiter_ruling_rejects_combined_notation_final_level():
    with pytest.raises(ValidationError):
        _ruling(final_level="M3/L4")


def test_equity_gate_result_accepts_pass_with_no_conflicts():
    result = EquityGateResult(passed=True, reasoning="no incumbents at L5 with greater scope")
    assert result.passed is True
    assert result.conflicting_incumbents == []


def test_equity_gate_result_accepts_fail_with_conflicts():
    result = EquityGateResult(
        passed=False,
        conflicting_incumbents=["MER-0142", "MER-0198"],
        reasoning="both incumbents have greater demonstrated scope at L5",
    )
    assert result.passed is False
    assert result.conflicting_incumbents == ["MER-0142", "MER-0198"]


def test_equity_gate_result_rejects_pass_with_conflicts_listed():
    with pytest.raises(ValidationError):
        EquityGateResult(passed=True, conflicting_incumbents=["MER-0142"], reasoning="test")


def test_equity_gate_result_rejects_fail_with_no_conflicts_named():
    with pytest.raises(ValidationError):
        EquityGateResult(passed=False, conflicting_incumbents=[], reasoning="test")


def _entry(**overrides):
    fields = dict(
        case_id="CASE-001",
        employee_id="NYX-009",
        crosswalk_level="L4",
        advocate_position="L5",
        advocate_argument=_argument(proposed_level="L5"),
        arbiter_ruling=_ruling(verdict="revised", final_level="L5"),
        governing_rule_cited="rule 2: lower level governs a split",
        equity_gate_result=EquityGateResult(passed=True, reasoning="no conflicting incumbents"),
        verdict="revised",
        round_count=1,
    )
    fields.update(overrides)
    return ExceptionRegisterEntry(**fields)


def test_exception_register_entry_accepts_consistent_record():
    entry = _entry()
    assert entry.verdict == "revised"
    assert entry.round_count == 1


def test_exception_register_entry_rejects_advocate_position_mismatch():
    with pytest.raises(ValidationError):
        _entry(advocate_position="L6")


def test_exception_register_entry_rejects_verdict_mismatch_with_ruling():
    with pytest.raises(ValidationError):
        _entry(verdict="upheld")


def test_exception_register_entry_rejects_governing_rule_mismatch():
    with pytest.raises(ValidationError):
        _entry(governing_rule_cited="rule 3: deep-but-narrow caps at L5")


def test_exception_register_entry_rejects_round_count_above_two():
    with pytest.raises(ValidationError):
        _entry(round_count=3)


def test_exception_register_entry_rejects_round_count_below_one():
    with pytest.raises(ValidationError):
        _entry(round_count=0)


def test_exception_register_entry_allows_no_equity_gate_result():
    entry = _entry(
        arbiter_ruling=_ruling(verdict="upheld", final_level="L4"),
        equity_gate_result=None,
        verdict="upheld",
    )
    assert entry.equity_gate_result is None
