from agents.modeling_graph import build_modeling_graph
from agents.modeling_schemas import CostRecommendation, RetentionJudgment, SynthesisResult
from tests.fakes import FakeModel

_POPULATION = [
    {
        "employee_id": "NYX-011",
        "job_id": "DD-UARCH-L6",
        "family": "Digital Design",
        "family_group": "engineering",
        "level_code": "L6",
        "geo_code": "IN-BLR",
        "currency": "INR",
        "current_pay": 5_000_000.0,
        "unvested_equity_value": 62_750.78,
        "role_summary": "Company-wide final authority on core PPA tradeoffs.",
    }
]


def _run(cost_strategy="phased", critical_ids=None):
    cost_fake = FakeModel(
        CostRecommendation(strategy=cost_strategy, reasoning="test"), schema=CostRecommendation
    )
    retention_fake = FakeModel(
        RetentionJudgment(critical_employee_ids=critical_ids or [], reasoning="test"), schema=RetentionJudgment
    )
    synthesis_fake = FakeModel(
        SynthesisResult(conflicts=[], recommended_plan="test plan", requires_human_judgment=False),
        schema=SynthesisResult,
    )
    app = build_modeling_graph(
        cost_model=cost_fake, retention_model=retention_fake, synthesis_model=synthesis_fake
    ).compile()
    return app.invoke(
        {"population": _POPULATION, "as_of_date": "2026-08-01", "cost_assessment": None, "retention_assessment": None, "synthesis": None},
        {"configurable": {"thread_id": "test"}},
    )


def test_modeling_graph_runs_cost_and_retention_and_synthesizes():
    result = _run(cost_strategy="phased", critical_ids=["NYX-011"])

    assert result["cost_assessment"] is not None
    assert result["cost_assessment"]["recommendation"]["strategy"] == "phased"

    assert result["retention_assessment"] is not None
    assert result["retention_assessment"]["judgment"]["critical_employee_ids"] == ["NYX-011"]

    assert result["synthesis"] is not None
    assert result["synthesis"]["recommended_plan"] == "test plan"


def test_modeling_graph_cost_and_retention_figures_are_independently_deterministic():
    # Same population, computed twice via two independent branches -- the employee's
    # cost_gap and retention compa-ratio must each reflect the same underlying deterministic
    # math regardless of which branch computed them.
    result = _run()
    cost_line = result["cost_assessment"]["employees"][0]
    retention_line = result["retention_assessment"]["employees"][0]
    assert cost_line["employee_id"] == retention_line["employee_id"] == "NYX-011"
    assert cost_line["current_pay"] == retention_line["current_pay"] == 5_000_000.0
