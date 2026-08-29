"""Tests for agents/pricing_agent.py -- the one agent that exercises real bind_tools
tool-calling. The model is faked (FakeToolCallingModel, tests/fakes.py); every tool call it
requests is executed for real against tools/agent_tools.py and the committed data, since the
whole point of this agent is that the *numbers* always come from a real deterministic call,
never from the model. DD-UARCH-L5 / US-SJC is the same real (job_id, geo_code) pair
tests/test_data_access.py already establishes has both market data and, via engineering/L5,
committed structure/incumbent rows.
"""

from agents.pricing_agent import PricingJudgment, price_role
from tests.fakes import FakeAIMessage, FakeToolCallingModel

JOB_ID, GEO_CODE = "DD-UARCH-L5", "US-SJC"


def _judgment(**overrides) -> PricingJudgment:
    base = dict(
        is_offer_defensible=True,
        primary_concern=None,
        reasoning="Market and internal equity both support this placement.",
        recommended_next_step="proceed",
    )
    base.update(overrides)
    return PricingJudgment(**base)


def test_model_chooses_which_tools_to_call_and_real_data_flows_through():
    """Two real tool calls (read_job_architecture, then lookup_market_data using the
    family_group/level_code it just learned), then the model declares itself done."""
    responses = [
        FakeAIMessage(tool_calls=[{"name": "read_job_architecture", "args": {"job_id": JOB_ID}, "id": "call-1"}]),
        FakeAIMessage(
            tool_calls=[
                {
                    "name": "lookup_market_data",
                    "args": {
                        "family_group": "engineering", "level_code": "L5", "geo_code": GEO_CODE,
                        "as_of_date": "2026-08-01", "annual_growth_rate": 0.035,
                    },
                    "id": "call-2",
                }
            ]
        ),
        FakeAIMessage(content="Found job architecture and market data; USD matches, no conversion needed."),
    ]
    model = FakeToolCallingModel(responses, _judgment())

    result = price_role(
        job_id=JOB_ID, geo_code=GEO_CODE, candidate_salary=180_000, candidate_currency="USD",
        as_of_date="2026-08-01", model=model,
    )

    assert [c.tool_name for c in result.tool_calls] == ["read_job_architecture", "lookup_market_data"]
    assert result.tool_calls[0].error is None
    assert result.tool_calls[0].result["job"]["family_group"] == "engineering"
    assert result.tool_calls[1].error is None
    assert result.tool_calls[1].result["p50"] > 0
    assert result.judgment.is_offer_defensible is True


def test_uses_no_tools_when_the_model_declares_itself_done_immediately():
    model = FakeToolCallingModel([FakeAIMessage(content="Nothing to check.")], _judgment())
    result = price_role(
        job_id=JOB_ID, geo_code=GEO_CODE, candidate_salary=180_000, candidate_currency="USD",
        as_of_date="2026-08-01", model=model,
    )
    assert result.tool_calls == []


def test_failed_tool_call_is_recorded_not_raised():
    """A tool call against a nonexistent job_id raises inside the real tool -- the agent
    records the error on that ToolCallRecord and keeps going, matching ASSIGNMENT.md's
    error-handling contract (a tool error surfaces, it doesn't crash the run)."""
    responses = [
        FakeAIMessage(tool_calls=[{"name": "read_job_architecture", "args": {"job_id": "NOT-A-REAL-JOB"}, "id": "call-1"}]),
        FakeAIMessage(content="That job_id doesn't exist; declining to guess."),
    ]
    model = FakeToolCallingModel(responses, _judgment(is_offer_defensible=False, recommended_next_step="escalate"))

    result = price_role(
        job_id="NOT-A-REAL-JOB", geo_code=GEO_CODE, candidate_salary=180_000, candidate_currency="USD",
        as_of_date="2026-08-01", model=model,
    )

    assert result.tool_calls[0].result is None
    assert "NOT-A-REAL-JOB" in result.tool_calls[0].error
    assert result.judgment.is_offer_defensible is False


def test_write_mapping_decision_is_not_among_the_bound_tools():
    from agents.pricing_agent import TOOLS

    assert "write_mapping_decision" not in {t.name for t in TOOLS}
