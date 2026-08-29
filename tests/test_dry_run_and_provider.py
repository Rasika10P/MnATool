from agents.cost_logging import get_session_stats, log_call
from agents.instrumented_model import InstrumentedModel
from agents.leveling import level_role, would_hit_cache
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


def test_would_hit_cache_false_before_any_call():
    fake_model = FakeModel(_decision(), model_name="dry-run-test-model")
    assert would_hit_cache("An uncalled role description.", None, model=fake_model) is False


def test_would_hit_cache_true_after_a_call_and_makes_no_new_call():
    fake = FakeModel(_decision(), model_name="dry-run-test-model-2")
    fake_model = InstrumentedModel(fake)  # matches how a real agent gets its model, via get_model()
    job_description = "A role that gets called once."

    level_role(job_description, model=fake_model)
    assert would_hit_cache(job_description, None, model=fake_model) is True
    assert fake.raw_structured_model.call_count == 1, "would_hit_cache must not itself trigger a call"


def test_log_call_records_provider_and_session_stats():
    entry = log_call("claude-sonnet-5", 100, 50, cached=False, context="test", provider="anthropic")
    assert entry["provider"] == "anthropic"

    stats = get_session_stats().summary()
    assert stats["calls"] >= 1
    assert "anthropic" in stats["cost_by_provider"]


def test_session_stats_separates_cost_by_provider():
    log_call("claude-sonnet-5", 1_000_000, 0, cached=False, context="a", provider="anthropic")
    log_call("some-nebius-model", 1_000_000, 0, cached=False, context="b", provider="nebius")

    stats = get_session_stats().summary()
    assert stats["cost_by_provider"]["anthropic"] == 2.00
    assert stats["cost_by_provider"]["nebius"] == 0.0  # unpriced model logs cost_usd=None -> counted as 0
