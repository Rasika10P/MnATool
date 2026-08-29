"""Regression test for error_handling_backlog.md entry 2: one employee's leveling call
failing must not take the rest of the 25-employee fan-out down with it. Forces employee 12
of 25 to fail (its job_description carries a marker string a fault-injecting fake model
raises on) and confirms all 25 Send tasks still complete -- 24 real decisions plus one
{"employee_id": ..., "error": ...} entry for the forced failure, not a raised exception that
would have taken the other 24 with it under the pre-fix behavior.
"""

import uuid

from agents.leveling_batch_graph import build_batch_graph, get_checkpointer
from agents.schemas import FactorRating, LevelingDecision
from tests.fakes import FakeFaultInjectingModel

FAILURE_TRIGGER = "TRIGGER_FAILURE_MARKER"


def _valid_decision() -> LevelingDecision:
    return LevelingDecision(
        track="IC",
        assigned_level="L4",
        factor_ratings=[FactorRating(factor="scope_of_impact", level_indicated="L4", evidence="test")],
        factor5_variant_applied="5a",
        confidence=0.8,
        governing_rule="rule 1",
        reasoning="test decision",
    )


def _employees(n: int = 25, fail_index: int = 11) -> list[dict]:
    """25 employees, 1-indexed employee_ids EMP-001..EMP-025. `fail_index` is 0-indexed, so
    fail_index=11 is EMP-012 -- "employee 12 of 25" in 1-indexed terms."""
    employees = []
    for i in range(n):
        description = f"Employee {i + 1}: does real engineering work."
        if i == fail_index:
            description += f" {FAILURE_TRIGGER}"
        employees.append({"employee_id": f"EMP-{i + 1:03d}", "job_description": description, "source_org_context": None})
    return employees


def test_one_forced_failure_among_25_does_not_take_down_the_batch(tmp_path):
    model = FakeFaultInjectingModel(
        decision=_valid_decision(),
        failure=RuntimeError("simulated structured-output failure for this employee only"),
        fail_when_content_contains=FAILURE_TRIGGER,
    )
    employees = _employees(n=25, fail_index=11)  # employee 12 of 25

    checkpointer = get_checkpointer(tmp_path / "batch.sqlite")
    app = build_batch_graph(model=model).compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}}
    result = app.invoke({"employees": employees, "decisions": []}, config, durability="sync")
    decisions = result["decisions"]

    assert len(decisions) == 25, "every employee must checkpoint a result, failed or not"

    by_id = {d["employee_id"]: d for d in decisions}
    failed = {emp_id: d for emp_id, d in by_id.items() if "error" in d}
    succeeded = {emp_id: d for emp_id, d in by_id.items() if "error" not in d}

    assert list(failed.keys()) == ["EMP-012"]
    assert "simulated structured-output failure" in failed["EMP-012"]["error"]

    assert len(succeeded) == 24
    for emp_id, d in succeeded.items():
        assert d["assigned_level"] == "L4", f"{emp_id} should have a real decision, not a partial/missing one"
