"""Tests for level_role_routed: Nebius first pass, Claude second pass only when Nebius's
own confidence falls below the escalation threshold. Uses two distinct FakeModels so a test
can tell, from which decision comes back, which provider actually served it -- monkeypatches
agents.leveling.get_model directly rather than agents.model_router.get_model, since leveling
imports the name into its own module scope.
"""

import agents.leveling as leveling
from agents.schemas import FactorRating, LevelingDecision
from tests.fakes import FakeModel

NEBIUS_DECISION = LevelingDecision(
    track="IC", assigned_level="L4",
    factor_ratings=[FactorRating(factor="scope_of_impact", level_indicated="L4", evidence="nebius evidence")],
    factor5_variant_applied="5a", confidence=0.9, governing_rule="rule 1", reasoning="nebius reasoning",
)

CLAUDE_DECISION = LevelingDecision(
    track="IC", assigned_level="L5",
    factor_ratings=[FactorRating(factor="scope_of_impact", level_indicated="L5", evidence="claude evidence")],
    factor5_variant_applied="5a", confidence=0.9, governing_rule="rule 1", reasoning="claude reasoning",
)


def _patch_get_model(monkeypatch, nebius_decision, claude_decision):
    nebius_fake = FakeModel(nebius_decision, model_name="Qwen/Qwen3-30B-A3B-Instruct-2507")
    claude_fake = FakeModel(claude_decision, model_name="claude-sonnet-5")

    def fake_get_model(tier):
        return {"volume": nebius_fake, "judgment": claude_fake}[tier]

    monkeypatch.setattr(leveling, "get_model", fake_get_model)
    return nebius_fake, claude_fake


def test_high_nebius_confidence_stays_on_nebius_no_second_pass(monkeypatch):
    high_confidence = NEBIUS_DECISION.model_copy(update={"confidence": 0.9})
    nebius_fake, claude_fake = _patch_get_model(monkeypatch, high_confidence, CLAUDE_DECISION)

    result = leveling.level_role_routed("some job description")

    assert result["served_by"] == "nebius"
    assert result["decision"].assigned_level == "L4"
    assert result["nebius_pass"] is None
    assert nebius_fake.structured_model.call_count == 1
    assert claude_fake.structured_model.call_count == 0


def test_low_nebius_confidence_escalates_to_claude(monkeypatch):
    low_confidence = NEBIUS_DECISION.model_copy(update={"confidence": 0.5})
    nebius_fake, claude_fake = _patch_get_model(monkeypatch, low_confidence, CLAUDE_DECISION)

    result = leveling.level_role_routed("some job description")

    assert result["served_by"] == "anthropic"
    assert result["decision"].assigned_level == "L5"  # Claude's decision, not Nebius's
    assert result["nebius_pass"] is not None
    assert result["nebius_pass"].confidence == 0.5
    assert nebius_fake.structured_model.call_count == 1
    assert claude_fake.structured_model.call_count == 1


def test_nebius_escalation_threshold_is_configurable(monkeypatch):
    mid_confidence = NEBIUS_DECISION.model_copy(update={"confidence": 0.8})
    nebius_fake, claude_fake = _patch_get_model(monkeypatch, mid_confidence, CLAUDE_DECISION)

    result = leveling.level_role_routed("some job description", nebius_escalation_threshold=0.85)

    assert result["served_by"] == "anthropic"
    assert claude_fake.structured_model.call_count == 1
