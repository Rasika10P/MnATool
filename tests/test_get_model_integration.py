"""The central claim of this architecture: an agent that gets its model via get_model() --
not an explicit override -- inherits caching, cost logging, session stats, and the spend
budget automatically. Monkeypatches the ChatAnthropic constructor itself so this is provable
without real credentials or a network call, while still exercising get_model()'s own code
path end to end (unlike tests/test_instrumented_model.py, which wraps a fake directly)."""

from agents.leveling import level_role
from agents.schemas import FactorRating, LevelingDecision
from tests.fakes import FakeModel


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


def test_get_model_default_path_still_caches(monkeypatch):
    import agents.model_router as model_router

    fake = FakeModel(_decision(), model_name="claude-sonnet-5")
    monkeypatch.setattr(model_router, "ChatAnthropic", lambda **kwargs: fake)

    job_description = "Owns a subsystem across a full development cycle, default-path test."
    first = level_role(job_description)  # no model= override -- must go through get_model()
    second = level_role(job_description)

    assert fake.raw_structured_model.call_count == 1, "second call via get_model() should have hit the cache"
    assert first.model_dump() == second.model_dump()
