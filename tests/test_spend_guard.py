import pytest

from agents.spend_guard import Budget, BudgetExceededError


def test_project_unpriced_model_is_zero():
    b = Budget(cap_usd=2.0)
    assert b.project("some-unpriced-model", 100_000, 2048) == 0.0


def test_project_known_model_uses_pricing():
    b = Budget(cap_usd=2.0)
    # claude-sonnet-5: $2/$10 per 1M input/output
    cost = b.project("claude-sonnet-5", 1_000_000, 1_000_000)
    assert cost == 12.00


def test_check_before_call_passes_when_under_cap():
    b = Budget(cap_usd=2.0)
    b.check_before_call("claude-sonnet-5", ["short prompt"], max_output_tokens=2048)  # should not raise


def test_check_before_call_raises_when_projection_exceeds_cap():
    b = Budget(cap_usd=0.0001)  # tiny cap, any real call blows it
    with pytest.raises(BudgetExceededError):
        b.check_before_call("claude-sonnet-5", ["a prompt with some real length to it"] * 20, max_output_tokens=2048)


def test_record_accumulates_spend_and_later_call_can_trip_cap():
    b = Budget(cap_usd=0.01)
    b.record(0.009)
    # a further call projected at more than the remaining $0.001 headroom should now raise
    with pytest.raises(BudgetExceededError):
        b.check_before_call("claude-sonnet-5", ["x" * 4000], max_output_tokens=2048)


def test_record_none_cost_is_a_noop():
    b = Budget(cap_usd=1.0)
    b.record(None)
    assert b.spent_usd == 0.0


def test_error_message_is_clear_not_a_raw_traceback():
    b = Budget(cap_usd=0.0001)
    with pytest.raises(BudgetExceededError) as exc_info:
        b.check_before_call("claude-sonnet-5", ["x" * 4000], max_output_tokens=2048)
    message = str(exc_info.value)
    assert "cap" in message.lower()
    assert "$" in message
