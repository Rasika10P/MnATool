"""Tests for InstrumentedModel: caching, cost logging, session stats, and the spend budget
applied at the model layer, independent of any specific agent. Any future agent that gets
its model via get_model() inherits these same guarantees without its own tests for them --
these tests are what makes that claim true, not agents/leveling.py's own test suite.
"""

import json

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

import agents.cost_logging as cost_logging
from agents.cost_logging import get_session_stats
from agents.instrumented_model import (
    MAX_ATTEMPTS,
    InstrumentedModel,
    StructuredOutputError,
    _cache_key_parts,
    _detect_provider,
    would_hit_cache,
)
from agents.schemas import FactorRating, LevelingDecision, SourceOrgContext
from agents.spend_guard import BudgetExceededError, reset_default_budget
from tests.fakes import FakeModel

MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "human"}]


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


def test_identical_call_hits_cache_not_the_model():
    fake = FakeModel(_decision(), model_name="itest-1")
    structured = InstrumentedModel(fake).with_structured_output(LevelingDecision)

    first = structured.invoke(MESSAGES)
    second = structured.invoke(MESSAGES)

    assert fake.raw_structured_model.call_count == 1, "second call should have been served from cache"
    assert first.model_dump() == second.model_dump()


def test_different_messages_do_not_share_cache_entry():
    fake = FakeModel(_decision(), model_name="itest-2")
    structured = InstrumentedModel(fake).with_structured_output(LevelingDecision)

    structured.invoke(MESSAGES)
    structured.invoke([{"role": "system", "content": "sys"}, {"role": "user", "content": "different"}])

    assert fake.raw_structured_model.call_count == 2


def test_different_schema_does_not_share_cache_entry():
    # Same messages, different schema name -> different cache key, even against identical text.
    assert _cache_key_parts(LevelingDecision, MESSAGES) != _cache_key_parts(SourceOrgContext, MESSAGES)


def test_would_hit_cache_false_then_true_after_a_call():
    fake = FakeModel(_decision(), model_name="itest-4")
    wrapped = InstrumentedModel(fake)

    assert would_hit_cache(wrapped, LevelingDecision, MESSAGES) is False
    wrapped.with_structured_output(LevelingDecision).invoke(MESSAGES)
    assert would_hit_cache(wrapped, LevelingDecision, MESSAGES) is True


def test_would_hit_cache_never_triggers_a_call():
    fake = FakeModel(_decision(), model_name="itest-4b")
    wrapped = InstrumentedModel(fake)
    would_hit_cache(wrapped, LevelingDecision, MESSAGES)
    assert fake.raw_structured_model.call_count == 0


def test_cache_hit_logs_zero_cost_and_records_session_stats():
    fake = FakeModel(_decision(), model_name="itest-5")
    structured = InstrumentedModel(fake).with_structured_output(LevelingDecision)

    structured.invoke(MESSAGES)
    structured.invoke(MESSAGES)

    stats = get_session_stats().summary()
    assert stats["calls"] == 2
    assert stats["cache_hits"] == 1


def test_budget_blocks_a_priced_model_before_the_call():
    reset_default_budget(cap_usd=0.0000001)
    fake = FakeModel(_decision(), model_name="claude-sonnet-5")  # a name present in PRICING
    structured = InstrumentedModel(fake).with_structured_output(LevelingDecision)

    with pytest.raises(BudgetExceededError):
        structured.invoke(MESSAGES)

    assert fake.raw_structured_model.call_count == 0, "the model must not be called once the budget check fails"


def test_cache_hit_is_never_blocked_by_an_exhausted_budget():
    fake = FakeModel(_decision(), model_name="claude-sonnet-5")
    structured = InstrumentedModel(fake).with_structured_output(LevelingDecision)

    reset_default_budget(cap_usd=2.0)
    structured.invoke(MESSAGES)  # populate the cache under a healthy budget

    reset_default_budget(cap_usd=0.0000001)
    result = structured.invoke(MESSAGES)  # must still succeed -- cache hit, no call, no cost
    assert result.assigned_level == "L4"
    assert fake.raw_structured_model.call_count == 1


def test_parsing_error_retries_then_raises_structured_output_error_and_is_not_cached():
    # A parsing failure that never clears retries MAX_ATTEMPTS times, then raises the
    # wrapper's own exception (not the raw parsing error) so callers get a single, clear
    # failure mode regardless of which underlying shape of malformed output caused it.
    error = ValueError("malformed output")
    fake = FakeModel(None, model_name="itest-6", parsing_error=error)
    wrapped = InstrumentedModel(fake)

    with pytest.raises(StructuredOutputError) as exc_info:
        wrapped.with_structured_output(LevelingDecision).invoke(MESSAGES)

    assert fake.raw_structured_model.call_count == MAX_ATTEMPTS
    assert exc_info.value.schema_name == "LevelingDecision"
    assert exc_info.value.attempts == [error, error, error]
    assert would_hit_cache(wrapped, LevelingDecision, MESSAGES) is False


def test_parsing_error_retries_then_succeeds_and_is_cached():
    error = ValueError("malformed output")
    good_decision = _decision()
    fake = FakeModel(
        None, model_name="itest-8",
        sequence=[(None, error), (None, error), (good_decision, None)],
    )
    wrapped = InstrumentedModel(fake)

    result = wrapped.with_structured_output(LevelingDecision).invoke(MESSAGES)

    assert result.model_dump() == good_decision.model_dump()
    assert fake.raw_structured_model.call_count == 3
    assert would_hit_cache(wrapped, LevelingDecision, MESSAGES) is True


def test_each_attempt_is_logged_with_its_own_attempt_number():
    error = ValueError("malformed output")
    good_decision = _decision()
    fake = FakeModel(
        None, model_name="itest-9",
        sequence=[(None, error), (good_decision, None)],
    )
    wrapped = InstrumentedModel(fake)

    wrapped.with_structured_output(LevelingDecision).invoke(MESSAGES)

    log_path = cost_logging.DEFAULT_LOG_PATH
    entries = [json.loads(line) for line in log_path.read_text().splitlines()]
    itest9_entries = [e for e in entries if e["model"] == "itest-9"]
    assert [e["attempt"] for e in itest9_entries] == [1, 2]

    stats = get_session_stats().summary()
    assert stats["retries"] == 1


def test_exhausting_all_attempts_logs_every_attempt():
    error = ValueError("malformed output")
    fake = FakeModel(None, model_name="itest-10", parsing_error=error)
    wrapped = InstrumentedModel(fake)

    with pytest.raises(StructuredOutputError):
        wrapped.with_structured_output(LevelingDecision).invoke(MESSAGES)

    log_path = cost_logging.DEFAULT_LOG_PATH
    entries = [json.loads(line) for line in log_path.read_text().splitlines()]
    itest10_entries = [e for e in entries if e["model"] == "itest-10"]
    assert [e["attempt"] for e in itest10_entries] == [1, 2, 3]

    stats = get_session_stats().summary()
    assert stats["retries"] == 2  # attempts 2 and 3; attempt 1 is not itself a retry


def test_budget_is_checked_before_every_retry_attempt():
    # A retry is a real, separately billed call -- an exhausted budget must stop a retry
    # loop partway through, not just before the first attempt. Cap is calibrated so attempt
    # 1's projected cost clears it but attempt 2's (projected cost, plus attempt 1's now-
    # actual logged spend) does not -- see Budget.project/check_before_call in
    # agents/spend_guard.py: projection uses max_output_tokens (2048, FakeModel's default)
    # as a worst case, while what actually gets recorded after a call is FakeRawMessage's
    # real usage (100 input / 50 output tokens), which is much smaller.
    error = ValueError("malformed output")
    fake = FakeModel(None, model_name="claude-sonnet-5", parsing_error=error)  # a name present in PRICING
    wrapped = InstrumentedModel(fake)

    reset_default_budget(cap_usd=0.0208)
    with pytest.raises(BudgetExceededError):
        wrapped.with_structured_output(LevelingDecision).invoke(MESSAGES)

    assert fake.raw_structured_model.call_count == 1, "attempt 1 should run; attempt 2 should be blocked before calling the model"


def test_provider_detected_from_model_class():
    assert _detect_provider(ChatAnthropic(model="claude-sonnet-5", api_key="x")) == "anthropic"
    assert _detect_provider(ChatOpenAI(model="x", api_key="x", base_url="http://example.invalid")) == "nebius"
    assert _detect_provider(FakeModel(_decision())) == "unknown"


def test_other_attributes_fall_through_to_wrapped_model():
    fake = FakeModel(_decision(), model_name="itest-7")
    wrapped = InstrumentedModel(fake)
    assert wrapped.model == "itest-7"
    assert wrapped.max_tokens == 2048
