"""Tests for extract_scope_profile: correct schema requested, default routes to
get_model("volume") (Nebius), explicit override bypasses routing entirely."""

import pytest
from pydantic import ValidationError

import agents.scope_extraction as scope_extraction
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
