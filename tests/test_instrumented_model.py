"""Tests for InstrumentedModel: caching, cost logging, session stats, and the spend budget
applied at the model layer, independent of any specific agent. Any future agent that gets
its model via get_model() inherits these same guarantees without its own tests for them --
these tests are what makes that claim true, not agents/leveling.py's own test suite.
"""

import json
import time

import anthropic
import httpx
import openai
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

import agents.cost_logging as cost_logging
import agents.instrumented_model as instrumented_model
from agents.cost_logging import get_session_stats
from agents.instrumented_model import (
    CACHE_MODE_DEMO,
    CACHE_MODE_FILL,
    CACHE_MODE_LIVE,
    MAX_ATTEMPTS,
    DemoModeCacheMissError,
    InstrumentedModel,
    ModelCallError,
    ModelTimeoutError,
    StructuredOutputError,
    _backoff_seconds,
    _cache_key_parts,
    _detect_provider,
    get_cache_mode,
    set_cache_mode,
    would_hit_cache,
)
from agents.schemas import FactorRating, LevelingDecision, SourceOrgContext
from agents.spend_guard import BudgetExceededError, reset_default_budget
from tests.fakes import FakeModel, FakeNetworkFlakyModel

_FAKE_REQUEST = httpx.Request("POST", "https://example.invalid/v1/messages")


def _timeout_error() -> anthropic.APITimeoutError:
    return anthropic.APITimeoutError(request=_FAKE_REQUEST)


def _connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(message="connection reset", request=_FAKE_REQUEST)


@pytest.fixture(autouse=True)
def no_real_sleeping(monkeypatch):
    """Every retry-with-backoff test below fakes the failure instantly -- there is no real
    network delay to wait out, so the exponential backoff's actual sleep would only slow the
    suite down for no reason. Recording calls (instead of just no-op-ing them) lets the
    backoff-schedule test below assert on what *would* have been slept without the test
    suite actually pausing for it."""
    calls = []
    monkeypatch.setattr(instrumented_model.time, "sleep", lambda seconds: calls.append(seconds))
    return calls

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


def test_leaked_tag_in_raw_tool_call_retries_then_succeeds_clean():
    # error_handling_backlog.md entry 1: Pydantic validates cleanly both times (no
    # parsing_error either attempt) -- the leak only shows up in the raw tool-call args, the
    # same shape a real leaked-tag response has once field validators have already sanitized
    # the parsed object. attempt 1's raw args leak; attempt 2's don't.
    good_decision = _decision()
    fake = FakeModel(
        good_decision, model_name="itest-leak-1",
        sequence=[(good_decision, None), (good_decision, None)],
        tool_call_args_sequence=[{"reasoning": "leaked </reasoning>"}, {"reasoning": "clean"}],
    )
    wrapped = InstrumentedModel(fake)

    result = wrapped.with_structured_output(LevelingDecision).invoke(MESSAGES)

    assert result.model_dump() == good_decision.model_dump()
    assert fake.raw_structured_model.call_count == 2, "a clean-but-leaked attempt must trigger a retry"
    assert would_hit_cache(wrapped, LevelingDecision, MESSAGES) is True


def test_leaked_tag_on_every_attempt_still_returns_decision_not_raise():
    # Exhausting every retry on leaked-but-parseable output must not drop the decision --
    # the field validators already guarantee the returned object is sanitized, so accepting
    # it beats raising StructuredOutputError and losing the row entirely.
    good_decision = _decision()
    fake = FakeModel(
        good_decision, model_name="itest-leak-2",
        tool_call_args_sequence=[{"reasoning": "leaked </reasoning>"}],  # same leak every attempt
    )
    wrapped = InstrumentedModel(fake)

    result = wrapped.with_structured_output(LevelingDecision).invoke(MESSAGES)

    assert result.model_dump() == good_decision.model_dump()
    assert fake.raw_structured_model.call_count == MAX_ATTEMPTS
    assert would_hit_cache(wrapped, LevelingDecision, MESSAGES) is True


def test_leaked_tag_retry_counts_as_a_retry_in_session_stats():
    good_decision = _decision()
    fake = FakeModel(
        good_decision, model_name="itest-leak-3",
        sequence=[(good_decision, None), (good_decision, None)],
        tool_call_args_sequence=[{"reasoning": "leaked </reasoning>"}, {"reasoning": "clean"}],
    )
    wrapped = InstrumentedModel(fake)

    wrapped.with_structured_output(LevelingDecision).invoke(MESSAGES)

    stats = get_session_stats().summary()
    assert stats["retries"] == 1


def test_no_leak_when_raw_message_has_no_tool_calls():
    # Every pre-existing FakeRawMessage (no tool_call_args passed) has no .tool_calls
    # attribute at all -- confirms the leak check treats that as "nothing to scan," not a
    # false positive, so every test written before this feature existed stays valid.
    fake = FakeModel(_decision(), model_name="itest-leak-4")
    wrapped = InstrumentedModel(fake)

    result = wrapped.with_structured_output(LevelingDecision).invoke(MESSAGES)

    assert fake.raw_structured_model.call_count == 1


def test_provider_detected_from_model_class():
    assert _detect_provider(ChatAnthropic(model="claude-sonnet-5", api_key="x")) == "anthropic"
    assert _detect_provider(ChatOpenAI(model="x", api_key="x", base_url="http://example.invalid")) == "nebius"
    assert _detect_provider(FakeModel(_decision())) == "unknown"


def test_other_attributes_fall_through_to_wrapped_model():
    fake = FakeModel(_decision(), model_name="itest-7")
    wrapped = InstrumentedModel(fake)
    assert wrapped.model == "itest-7"
    assert wrapped.max_tokens == 2048


def test_cache_mode_defaults_to_fill():
    # conftest.py's autouse fixture already resets this before every test, but the claim
    # itself -- existing scripts/agents/tests are unaffected unless something opts in -- is
    # worth asserting directly, not just relying on every other test passing to imply it.
    assert get_cache_mode() == CACHE_MODE_FILL


def test_live_mode_bypasses_a_warm_cache_hit_and_overwrites_it():
    fake = FakeModel(_decision(), model_name="itest-live-1")
    structured = InstrumentedModel(fake).with_structured_output(LevelingDecision)

    structured.invoke(MESSAGES)  # warms the cache under fill mode
    set_cache_mode(CACHE_MODE_LIVE)
    try:
        structured.invoke(MESSAGES)
    finally:
        set_cache_mode(CACHE_MODE_FILL)

    assert fake.raw_structured_model.call_count == 2, "live mode must make a real call even though a cache entry exists"

    # fill mode afterward should see the freshly-written entry, not need another real call.
    structured.invoke(MESSAGES)
    assert fake.raw_structured_model.call_count == 2, "the live call's result should have been written back to cache"


def test_demo_mode_blocks_a_cache_miss_without_calling_the_model():
    fake = FakeModel(_decision(), model_name="itest-demo-1")
    structured = InstrumentedModel(fake).with_structured_output(LevelingDecision)

    set_cache_mode(CACHE_MODE_DEMO)
    try:
        with pytest.raises(DemoModeCacheMissError):
            structured.invoke(MESSAGES)
    finally:
        set_cache_mode(CACHE_MODE_FILL)

    assert fake.raw_structured_model.call_count == 0, "demo mode must never reach the real model on a cache miss"


def test_demo_mode_still_serves_a_warm_cache_hit():
    fake = FakeModel(_decision(), model_name="itest-demo-2")
    structured = InstrumentedModel(fake).with_structured_output(LevelingDecision)

    structured.invoke(MESSAGES)  # warms the cache while cache-only is off
    set_cache_mode(CACHE_MODE_DEMO)
    try:
        result = structured.invoke(MESSAGES)
    finally:
        set_cache_mode(CACHE_MODE_FILL)

    assert result.model_dump() == _decision().model_dump()
    assert fake.raw_structured_model.call_count == 1, "the second call should be the cache hit, not a second real call"


class _RawToolCallingFake:
    """A raw (unwrapped) fake with just enough surface for InstrumentedModel.bind_tools to
    wrap it: .model/.max_tokens (read directly) and .bind_tools(tools) -> something with
    .invoke(messages) returning an AIMessage-shaped response. tests.fakes.FakeBoundTools/
    FakeAIMessage already have exactly that shape (built for agents/pricing_agent.py's own
    tests, which fake at the InstrumentedModel boundary rather than below it) -- reused here
    one layer lower, to test _InstrumentedToolCallingRunnable itself."""

    def __init__(self, responses):
        self.model = "itest-tool-calling"
        self.max_tokens = 2048
        self._responses = responses

    def bind_tools(self, tools):
        from tests.fakes import FakeBoundTools

        return FakeBoundTools(self._responses)


def test_demo_mode_blocks_a_tool_calling_cache_miss_without_calling_the_model():
    from tests.fakes import FakeAIMessage

    raw = _RawToolCallingFake([FakeAIMessage(content="done")])
    bound = InstrumentedModel(raw).bind_tools([], context="itest-tool-demo-1")

    set_cache_mode(CACHE_MODE_DEMO)
    try:
        with pytest.raises(DemoModeCacheMissError):
            bound.invoke(MESSAGES)
    finally:
        set_cache_mode(CACHE_MODE_FILL)


def test_demo_mode_still_serves_a_warm_tool_calling_cache_hit():
    from tests.fakes import FakeAIMessage

    raw = _RawToolCallingFake([FakeAIMessage(content="done")])
    bound = InstrumentedModel(raw).bind_tools([], context="itest-tool-demo-2")

    bound.invoke(MESSAGES)  # warms the cache while cache-only is off
    set_cache_mode(CACHE_MODE_DEMO)
    try:
        result = bound.invoke(MESSAGES)
    finally:
        set_cache_mode(CACHE_MODE_FILL)

    assert result.content == "done"


def test_live_mode_bypasses_a_warm_tool_calling_cache_hit():
    from tests.fakes import FakeAIMessage, FakeBoundTools

    raw = _RawToolCallingFake([FakeAIMessage(content="first"), FakeAIMessage(content="second")])
    bound = InstrumentedModel(raw).bind_tools([], context="itest-tool-live-1")

    first = bound.invoke(MESSAGES)  # warms the cache under fill mode
    assert first.content == "first"

    set_cache_mode(CACHE_MODE_LIVE)
    try:
        second = bound.invoke(MESSAGES)
    finally:
        set_cache_mode(CACHE_MODE_FILL)

    assert second.content == "second", "live mode must make a real (fresh) call, not return the cached first response"


class _RawEmbeddingsFake:
    """Minimal double for OpenAIEmbeddings: .model (read directly) and .embed_documents(texts)
    -> list[list[float]]. call_count lets a test confirm a cache hit skipped the real call."""

    def __init__(self, vectors_by_text: dict):
        self.model = "itest-embed-model"
        self._vectors_by_text = vectors_by_text
        self.call_count = 0

    def embed_documents(self, texts):
        self.call_count += 1
        return [self._vectors_by_text[t] for t in texts]


def test_embed_documents_hits_cache_not_the_model():
    raw = _RawEmbeddingsFake({"a": [0.1, 0.2], "b": [0.3, 0.4]})
    wrapped = InstrumentedModel(raw)

    first = wrapped.embed_documents(["a", "b"])
    second = wrapped.embed_documents(["a", "b"])

    assert raw.call_count == 1, "second call should have been served from cache"
    assert first == second == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_query_is_embed_documents_of_one():
    raw = _RawEmbeddingsFake({"solo": [1.0, 2.0, 3.0]})
    wrapped = InstrumentedModel(raw)
    assert wrapped.embed_query("solo") == [1.0, 2.0, 3.0]


def test_demo_mode_blocks_an_embedding_cache_miss_without_calling_the_model():
    raw = _RawEmbeddingsFake({"never cached": [0.0]})
    wrapped = InstrumentedModel(raw)

    set_cache_mode(CACHE_MODE_DEMO)
    try:
        with pytest.raises(DemoModeCacheMissError):
            wrapped.embed_documents(["never cached"])
    finally:
        set_cache_mode(CACHE_MODE_FILL)

    assert raw.call_count == 0, "demo mode must never reach the real embeddings client on a miss"


def test_demo_mode_still_serves_a_warm_embedding_cache_hit():
    raw = _RawEmbeddingsFake({"warm": [9.0, 9.0]})
    wrapped = InstrumentedModel(raw)

    wrapped.embed_documents(["warm"])  # warms the cache while cache-only is off
    set_cache_mode(CACHE_MODE_DEMO)
    try:
        result = wrapped.embed_documents(["warm"])
    finally:
        set_cache_mode(CACHE_MODE_FILL)

    assert result == [[9.0, 9.0]]
    assert raw.call_count == 1


def test_live_mode_bypasses_a_warm_embedding_cache_hit_and_overwrites_it():
    raw = _RawEmbeddingsFake({"text": [1.0, 1.0]})
    wrapped = InstrumentedModel(raw)

    wrapped.embed_documents(["text"])  # warms the cache under fill mode
    raw._vectors_by_text["text"] = [2.0, 2.0]  # a "changed" embedding for the same input

    set_cache_mode(CACHE_MODE_LIVE)
    try:
        live_result = wrapped.embed_documents(["text"])
    finally:
        set_cache_mode(CACHE_MODE_FILL)

    assert live_result == [[2.0, 2.0]], "live mode must call the real client, not return the stale cached vector"
    assert raw.call_count == 2

    # fill mode afterward should see the freshly-overwritten entry, not the original.
    filled_result = wrapped.embed_documents(["text"])
    assert filled_result == [[2.0, 2.0]]
    assert raw.call_count == 2


# ---------------------------------------------------------------------------
# Network timeouts + retry bounds
# ---------------------------------------------------------------------------
#
# A real timeout surfaces as anthropic.APITimeoutError / openai.APITimeoutError -- both
# are-a APIConnectionError (confirmed directly against both SDKs), which is what
# agents/instrumented_model.py actually catches. FakeNetworkFlakyModel raises whatever
# exception it's given from .invoke() itself, standing in for "the real client's
# request never came back" -- a mocked slow response, without a test that's actually slow.


def test_timeout_raises_a_typed_exception_not_a_hang():
    # "Don't hang" is the point being tested here, not just asserted -- a real bug (no
    # exception handling at all around a slow client) would make this test itself hang
    # instead of failing cleanly, so the wall-clock assertion below is the actual proof,
    # not decoration. time.sleep is faked (see no_real_sleeping), so a passing run should
    # take milliseconds regardless of MAX_ATTEMPTS.
    fake = FakeNetworkFlakyModel(decision=None, errors=[_timeout_error()] * MAX_ATTEMPTS)
    wrapped = InstrumentedModel(fake)

    started = time.perf_counter()
    with pytest.raises(ModelTimeoutError) as exc_info:
        wrapped.with_structured_output(LevelingDecision).invoke(MESSAGES)
    elapsed = time.perf_counter() - started

    assert elapsed < 2.0, "a timeout must raise promptly, never hang waiting on a stuck call"
    assert fake.call_count == MAX_ATTEMPTS
    assert len(exc_info.value.attempts) == MAX_ATTEMPTS
    assert all(isinstance(e, anthropic.APITimeoutError) for e in exc_info.value.attempts)


def test_network_error_retries_up_to_max_attempts_then_raises_model_call_error():
    fake = FakeNetworkFlakyModel(decision=None, errors=[_connection_error()] * MAX_ATTEMPTS)
    wrapped = InstrumentedModel(fake)

    with pytest.raises(ModelCallError) as exc_info:
        wrapped.with_structured_output(LevelingDecision).invoke(MESSAGES)

    assert fake.call_count == MAX_ATTEMPTS
    assert not isinstance(exc_info.value, ModelTimeoutError), "a plain connection error is not a timeout"
    assert len(exc_info.value.attempts) == MAX_ATTEMPTS


def test_the_fourth_attempt_never_happens():
    """The exact "never retry indefinitely" guarantee: MAX_ATTEMPTS=3, so a model that
    fails on every single call must be called exactly 3 times -- never a 4th, no matter how
    tempting one more try might seem."""
    fake = FakeNetworkFlakyModel(decision=None, errors=[_connection_error()] * 10)  # far more failures available than attempts allowed
    wrapped = InstrumentedModel(fake)

    with pytest.raises(ModelCallError):
        wrapped.with_structured_output(LevelingDecision).invoke(MESSAGES)

    assert fake.call_count == 3
    assert fake.call_count == MAX_ATTEMPTS
    assert fake.call_count != 4


def test_network_retry_never_fails_silently_it_always_raises_or_returns():
    """Exhaustion is never swallowed -- either a decision comes back, or ModelCallError is
    raised. There is no third outcome (returning None, logging and moving on, ...)."""
    fake = FakeNetworkFlakyModel(decision=None, errors=[_connection_error()] * MAX_ATTEMPTS)
    wrapped = InstrumentedModel(fake)
    try:
        result = wrapped.with_structured_output(LevelingDecision).invoke(MESSAGES)
    except ModelCallError:
        return  # the one acceptable outcome besides a real decision
    assert result is not None, "must not silently return a falsy/empty result instead of raising"


def test_network_error_that_clears_within_max_attempts_succeeds():
    # Proves the policy is a real retry, not just a fast-fail dressed up as one: two
    # failures, then a decision on the third (and final) allowed attempt.
    fake = FakeNetworkFlakyModel(decision=_decision(), errors=[_connection_error(), _timeout_error()])
    wrapped = InstrumentedModel(fake)

    result = wrapped.with_structured_output(LevelingDecision).invoke(MESSAGES)

    assert result.assigned_level == "L4"
    assert fake.call_count == 3


def test_network_retry_uses_exponential_backoff_between_attempts_only(no_real_sleeping):
    fake = FakeNetworkFlakyModel(decision=None, errors=[_connection_error()] * MAX_ATTEMPTS)
    wrapped = InstrumentedModel(fake)

    with pytest.raises(ModelCallError):
        wrapped.with_structured_output(LevelingDecision).invoke(MESSAGES)

    # MAX_ATTEMPTS=3 failures -> 2 backoff sleeps (before retries 2 and 3), never a 3rd
    # sleep after the final attempt, since nothing follows it but raising.
    assert no_real_sleeping == [_backoff_seconds(1), _backoff_seconds(2)]
    assert no_real_sleeping[1] > no_real_sleeping[0], "backoff must actually grow, not stay flat"


def test_validation_error_retry_is_unaffected_by_the_network_retry_policy():
    # Regression guard: a plain parsing failure (the call succeeded; the response just
    # didn't validate) must keep raising StructuredOutputError, completely untouched by the
    # network-error path added alongside it -- "distinguishing network errors from
    # validation errors" means both keep their own, correct exception type.
    error = ValueError("malformed output")
    fake = FakeModel(None, model_name="itest-network-distinct", parsing_error=error)
    wrapped = InstrumentedModel(fake)

    with pytest.raises(StructuredOutputError):
        wrapped.with_structured_output(LevelingDecision).invoke(MESSAGES)

    assert fake.raw_structured_model.call_count == MAX_ATTEMPTS
