"""Tests for extract_scope_profile: correct schema requested, default routes to
get_model("volume") (Nebius), explicit override bypasses routing entirely. Also
extract_scope_profile_with_claude_fallback: Nebius-succeeds, Nebius-exhausted-falls-back-to-
Claude, and both-exhausted-still-raises."""

import pytest
from pydantic import ValidationError

import agents.scope_extraction as scope_extraction
from agents.instrumented_model import StructuredOutputError
from agents.schemas import ScopeFinding, ScopeProfile
from tests.fakes import FakeModel

FIXED_PROFILE = ScopeProfile(
    reports_to=ScopeFinding(stated=True, value="VP of Engineering"),
    span_of_control=ScopeFinding(stated=True, value="8 direct reports, all individual contributors"),
    budget_authority=ScopeFinding(stated=False, value=None),
    decision_scope="Sets sprint priorities for the team.",
    ownership_scope="Owns firmware and hardware bring-up for the product line.",
)


def test_extract_scope_profile_returns_the_parsed_profile():
    fake_model = FakeModel(FIXED_PROFILE, schema=ScopeProfile)
    result = scope_extraction.extract_scope_profile("some job description", model=fake_model)
    assert result == FIXED_PROFILE
    assert fake_model.structured_model.call_count == 1


def test_extract_scope_profile_requests_scope_profile_schema_not_leveling_decision():
    fake_model = FakeModel(FIXED_PROFILE, schema=ScopeProfile)
    # FakeModel.with_structured_output asserts the exact schema passed in -- if
    # extract_scope_profile ever requested LevelingDecision by mistake this would fail loudly
    # rather than silently returning the wrong shape.
    scope_extraction.extract_scope_profile("some job description", model=fake_model)


def test_no_explicit_model_routes_through_get_model_volume(monkeypatch):
    fake_model = FakeModel(FIXED_PROFILE, schema=ScopeProfile)
    calls = []

    def fake_get_model(tier):
        calls.append(tier)
        return fake_model

    monkeypatch.setattr(scope_extraction, "get_model", fake_get_model)
    result = scope_extraction.extract_scope_profile("some job description")

    assert calls == ["volume"]
    assert result == FIXED_PROFILE


def _routed_fakes(monkeypatch, volume_fake, judgment_fake):
    def fake_get_model(tier):
        return {"volume": volume_fake, "judgment": judgment_fake}[tier]

    monkeypatch.setattr(scope_extraction, "get_model", fake_get_model)


def test_fallback_returns_nebius_result_without_touching_claude(monkeypatch):
    nebius_fake = FakeModel(FIXED_PROFILE, schema=ScopeProfile)
    claude_fake = FakeModel(FIXED_PROFILE, schema=ScopeProfile)
    _routed_fakes(monkeypatch, nebius_fake, claude_fake)

    result = scope_extraction.extract_scope_profile_with_claude_fallback("some job description")

    assert result == FIXED_PROFILE
    assert nebius_fake.structured_model.call_count == 1
    assert claude_fake.structured_model.call_count == 0, "must not fall back when Nebius succeeds"


def test_fallback_uses_claude_when_nebius_exhausts_retries(monkeypatch):
    nebius_error = StructuredOutputError(
        "ScopeProfile", "fake-nebius-model", [ValueError("stated=True requires non-null value")]
    )
    nebius_fake = FakeModel(None, schema=ScopeProfile, parsing_error=nebius_error)
    claude_fake = FakeModel(FIXED_PROFILE, schema=ScopeProfile)
    _routed_fakes(monkeypatch, nebius_fake, claude_fake)

    result = scope_extraction.extract_scope_profile_with_claude_fallback("some job description")

    assert result == FIXED_PROFILE
    assert claude_fake.structured_model.call_count == 1


def test_fallback_propagates_if_claude_also_exhausts_retries(monkeypatch):
    nebius_error = StructuredOutputError("ScopeProfile", "fake-nebius-model", [ValueError("bad")])
    claude_error = StructuredOutputError("ScopeProfile", "fake-claude-model", [ValueError("also bad")])
    nebius_fake = FakeModel(None, schema=ScopeProfile, parsing_error=nebius_error)
    claude_fake = FakeModel(None, schema=ScopeProfile, parsing_error=claude_error)
    _routed_fakes(monkeypatch, nebius_fake, claude_fake)

    with pytest.raises(StructuredOutputError) as exc_info:
        scope_extraction.extract_scope_profile_with_claude_fallback("some job description")

    assert exc_info.value.model_name == "fake-claude-model"
