"""Tests for _compute_escalate's band, replacing the single-cutoff comparison that let the
same case (confidence 0.68 vs 0.72 across two live runs) flip escalate on sampling noise.
"""

import pytest

from agents.instrumented_model import InstrumentedModel
from agents.leveling import _build_human_message, _compute_escalate, level_role, would_hit_cache
from agents.schemas import FactorRating, LevelingDecision, ScopeFinding, ScopeProfile
from tests.fakes import FakeModel

LOW, HIGH = 0.65, 0.75

_SCOPE_PROFILE = ScopeProfile(
    reports_to=ScopeFinding(stated=True, value="VP of Engineering"),
    span_of_control=ScopeFinding(stated=True, value="No direct reports"),
    budget_authority=ScopeFinding(stated=False, value=None),
    decision_scope="Worked independently with minimal oversight.",
    ownership_scope="Owned the RF transceiver block design end-to-end.",
)


def _decision() -> LevelingDecision:
    return LevelingDecision(
        track="IC",
        assigned_level="L4",
        factor_ratings=[FactorRating(factor="scope_of_impact", level_indicated="L4", evidence="owns a subsystem")],
        factor5_variant_applied="5a",
        confidence=0.8,
        governing_rule="rule 1",
        reasoning="test",
    )


def test_human_message_without_scope_profile_has_no_advisory_section():
    message = _build_human_message("Some job description.", None)
    assert "Extracted scope profile" not in message


def test_human_message_with_scope_profile_states_explicit_negative():
    message = _build_human_message("Some job description.", None, scope_profile=_SCOPE_PROFILE)
    assert "Extracted scope profile" in message
    assert "advisory" in message
    # The whole point of the fix: an explicit "no direct reports" must show up as a stated
    # finding, not be indistinguishable from a field the text never addressed.
    assert "span_of_control: explicitly stated -- 'No direct reports'" in message
    assert "budget_authority: not mentioned in the text" in message


def test_would_hit_cache_is_sensitive_to_scope_profile():
    # would_hit_cache must build the exact same messages _run_leveling_call sends, including
    # the advisory section -- otherwise --dry-run's cache prediction silently drifts from
    # what a real run would actually hit. Calling level_role once (with scope_profile=None)
    # populates the cache for that exact prompt; a would_hit_cache check with a scope_profile
    # attached must miss, because it's genuinely a different prompt to the model.
    fake = FakeModel(_decision(), model_name="scope-profile-cache-test")
    fake_model = InstrumentedModel(fake)
    job_description = "A role that gets called once, no scope profile."

    level_role(job_description, model=fake_model)

    assert would_hit_cache(job_description, None, model=fake_model) is True
    assert would_hit_cache(job_description, None, model=fake_model, scope_profile=_SCOPE_PROFILE) is False


@pytest.mark.parametrize("confidence", [0.0, 0.3, 0.64, 0.649])
def test_below_low_always_escalates_regardless_of_factor(confidence):
    assert _compute_escalate(confidence, escalation_factor=None, low=LOW, high=HIGH) is True
    assert _compute_escalate(confidence, escalation_factor="scope_of_impact", low=LOW, high=HIGH) is True


@pytest.mark.parametrize("confidence", [0.751, 0.8, 0.95, 1.0])
def test_above_high_never_escalates_regardless_of_factor(confidence):
    assert _compute_escalate(confidence, escalation_factor=None, low=LOW, high=HIGH) is False
    assert _compute_escalate(confidence, escalation_factor="scope_of_impact", low=LOW, high=HIGH) is False


def test_band_escalates_when_factor_present():
    # The two observed real values that motivated this band -- both must escalate.
    assert _compute_escalate(0.68, escalation_factor="technical_depth_breadth", low=LOW, high=HIGH) is True
    assert _compute_escalate(0.72, escalation_factor="ownership_scope", low=LOW, high=HIGH) is True


def test_band_does_not_escalate_without_factor():
    assert _compute_escalate(0.68, escalation_factor=None, low=LOW, high=HIGH) is False
    assert _compute_escalate(0.72, escalation_factor=None, low=LOW, high=HIGH) is False


def test_band_boundaries_are_inclusive():
    # Exactly 0.65 and exactly 0.75 both fall inside the band (checked via factor presence),
    # not into the always/never zones.
    assert _compute_escalate(0.65, escalation_factor="x", low=LOW, high=HIGH) is True
    assert _compute_escalate(0.65, escalation_factor=None, low=LOW, high=HIGH) is False
    assert _compute_escalate(0.75, escalation_factor="x", low=LOW, high=HIGH) is True
    assert _compute_escalate(0.75, escalation_factor=None, low=LOW, high=HIGH) is False
