import pytest
from pydantic import ValidationError

from agents.schemas import FactorRating, LevelingDecision, ScopeFinding


def test_scope_finding_accepts_not_mentioned():
    finding = ScopeFinding(stated=False, value=None)
    assert finding.stated is False
    assert finding.value is None


def test_scope_finding_accepts_explicit_negative():
    # The whole point of this type: "no direct reports" is stated=True, not left null just
    # because the content itself is a negative.
    finding = ScopeFinding(stated=True, value="no direct reports")
    assert finding.stated is True
    assert finding.value == "no direct reports"


def test_scope_finding_rejects_stated_true_without_value():
    with pytest.raises(ValidationError):
        ScopeFinding(stated=True, value=None)


def test_scope_finding_rejects_stated_false_with_value():
    with pytest.raises(ValidationError):
        ScopeFinding(stated=False, value="6 direct reports")


def test_level_indicated_accepts_valid_code():
    rating = FactorRating(factor="scope_of_impact", level_indicated="M3", evidence="6-10 reports")
    assert rating.level_indicated == "M3"


def test_level_indicated_rejects_combined_notation():
    with pytest.raises(ValidationError):
        FactorRating(factor="scope_of_impact", level_indicated="M3/L4", evidence="6-10 reports")


def test_level_indicated_rejects_unknown_code():
    with pytest.raises(ValidationError):
        FactorRating(factor="scope_of_impact", level_indicated="L9", evidence="nonsense")


def _decision(**overrides):
    fields = dict(
        track="IC",
        assigned_level="L4",
        factor_ratings=[FactorRating(factor="scope_of_impact", level_indicated="L4", evidence="owns a subsystem")],
        factor5_variant_applied="5a",
        confidence=0.8,
        governing_rule="rule 1",
        reasoning="test",
    )
    fields.update(overrides)
    return LevelingDecision(**fields)


def test_assigned_level_accepts_valid_code():
    assert _decision(assigned_level="M4").assigned_level == "M4"


def test_assigned_level_rejects_combined_notation():
    with pytest.raises(ValidationError):
        _decision(assigned_level="M3/L4")


def test_assigned_level_rejects_unknown_code():
    with pytest.raises(ValidationError):
        _decision(assigned_level="L9")


def test_alternative_level_accepts_valid_code_and_none():
    assert _decision(alternative_level="L5", alternative_reasoning="close call").alternative_level == "L5"
    assert _decision().alternative_level is None


def test_alternative_level_rejects_combined_notation():
    with pytest.raises(ValidationError):
        _decision(alternative_level="L5 -- considered because of scope")


def test_scope_finding_value_strips_leaked_tags():
    finding = ScopeFinding(stated=True, value="no direct reports </value>")
    assert finding.value == "no direct reports"


def test_factor_rating_evidence_strips_leaked_tags():
    rating = FactorRating(factor="scope_of_impact", level_indicated="L4", evidence='owns a subsystem <parameter name="x">')
    assert rating.evidence == "owns a subsystem"


def test_leveling_decision_reasoning_strips_leaked_tags():
    # error_handling_backlog.md entry 1's exact observed shape.
    dirty = "Senior Staff Engineer). </reasoning>\n<parameter name=\"alternative_level\">L6"
    assert _decision(reasoning=dirty).reasoning == "Senior Staff Engineer). L6"


def test_leveling_decision_governing_rule_strips_leaked_tags():
    assert _decision(governing_rule="rule 2 </governing_rule>").governing_rule == "rule 2"
