"""Tests for the advocate/arbiter/equity-gate subgraph (agents/negotiation_graph.py).
Advocate and arbiter are faked (schema-typed FakeModel, per tests/fakes.py); the equity gate
is real and deterministic (agents/equity_gate.py makes no model call), so these tests pick
real (family_group, level_code, geo_code) combinations from the committed data with known
behavior -- the same L7/engineering veto and the no-incumbent corporate/L3 pass already
established in tests/test_equity_gate.py.
"""

import json

import agents.negotiation_graph as negotiation_graph
from agents.negotiation_graph import build_negotiation_graph
from agents.negotiation_schemas import AdvocateOutput, ArbiterRuling
from agents.schemas import FactorRating, LevelingDecision
from tests.fakes import FakeModel

# engineering/L7: 3 real incumbents (MER-0234 US-SJC compa ~0.99, MER-0235 EU-MUC compa
# ~0.72, MER-0236 IN-BLR compa ~1.08) -- a candidate in Austin at $280k (compa ~1.18) vetoes
# against all three (tests/test_equity_gate.py). corporate/L3 has a salary structure but
# zero incumbents anywhere -- the gate passes vacuously.
VETO_FAMILY, VETO_LEVEL, VETO_GEO, VETO_SALARY = "engineering", "L7", "US-AUS", 280_000.0
PASS_FAMILY, PASS_LEVEL, PASS_GEO, PASS_SALARY = "corporate", "L3", "US-SJC", 100_000.0


def _crosswalk_decision(assigned_level="L6") -> LevelingDecision:
    return LevelingDecision(
        track="IC",
        assigned_level=assigned_level,
        factor_ratings=[FactorRating(factor="scope_of_impact", level_indicated=assigned_level, evidence="test")],
        factor5_variant_applied="5a",
        confidence=0.72,
        governing_rule="rule 4",
        reasoning="test crosswalk decision",
    )


def _decline_output() -> AdvocateOutput:
    return AdvocateOutput()


def _contest_output(proposed_level="L7") -> AdvocateOutput:
    return AdvocateOutput(
        argument_basis="scope evidence not reflected in the mapping",
        proposed_level=proposed_level,
        evidence_cited="test evidence",
        framework_section="nyx_level_framework.md section 4",
    )


def _ruling(verdict="upheld", final_level="L6") -> ArbiterRuling:
    return ArbiterRuling(
        verdict=verdict,
        governing_rule="rule 2: lower level governs a split",
        final_level=final_level,
        reasoning="test ruling",
    )


def _initial_state(
    crosswalk_decision, family_group=VETO_FAMILY, candidate_geo_code=VETO_GEO, candidate_salary=VETO_SALARY
):
    return {
        "case_id": "CASE-TEST-001",
        "employee_id": "NYX-TEST",
        "role_summary": "test role summary",
        "nyx_level": "Distinguished MTS",
        "crosswalk_decision": crosswalk_decision.model_dump(),
        "family_group": family_group,
        "candidate_geo_code": candidate_geo_code,
        "candidate_salary": candidate_salary,
        "round_count": 0,
        "contested": None,
        "advocate_output": None,
        "arbiter_ruling": None,
        "equity_gate_result": None,
        "rounds": [],
        "gate_checks": [],
        "final_verdict": None,
        "final_level": None,
        "exception_register_entry": None,
    }


def _run(advocate_output, arbiter_ruling_sequence, family_group=VETO_FAMILY, candidate_geo_code=VETO_GEO, candidate_salary=VETO_SALARY, crosswalk_level="L6"):
    advocate_fake = FakeModel(advocate_output, model_name="fake-advocate", schema=AdvocateOutput)
    arbiter_fake = FakeModel(
        None, model_name="fake-arbiter", schema=ArbiterRuling,
        sequence=[(r, None) for r in arbiter_ruling_sequence] if arbiter_ruling_sequence else None,
    )
    app = build_negotiation_graph(advocate_model=advocate_fake, arbiter_model=arbiter_fake).compile()
    initial_state = _initial_state(
        _crosswalk_decision(crosswalk_level), family_group, candidate_geo_code, candidate_salary
    )
    return app.invoke(initial_state, {"configurable": {"thread_id": "test"}})


def test_advocate_declines_short_circuits_with_no_exception_register():
    result = _run(_decline_output(), arbiter_ruling_sequence=None)
    assert result["contested"] is False
    assert result["round_count"] == 0
    assert result["final_verdict"] == "upheld"
    assert result["final_level"] == "L6"
    assert result["exception_register_entry"] is None
    assert result["rounds"] == []


def test_arbiter_upholds_finalizes_without_equity_gate():
    result = _run(_contest_output(), arbiter_ruling_sequence=[_ruling(verdict="upheld", final_level="L6")])
    assert result["contested"] is True
    assert result["round_count"] == 1
    assert result["final_verdict"] == "upheld"
    assert result["final_level"] == "L6"
    assert result["gate_checks"] == []  # equity gate never runs for a non-revised verdict
    assert result["exception_register_entry"] is not None
    assert result["exception_register_entry"]["verdict"] == "upheld"
    assert result["exception_register_entry"]["round_count"] == 1


def test_revised_verdict_gate_passes_finalizes_as_revised():
    result = _run(
        _contest_output(proposed_level=PASS_LEVEL),
        arbiter_ruling_sequence=[_ruling(verdict="revised", final_level=PASS_LEVEL)],
        family_group=PASS_FAMILY, candidate_geo_code=PASS_GEO, candidate_salary=PASS_SALARY,
        crosswalk_level="L2",
    )
    assert result["round_count"] == 1
    assert len(result["gate_checks"]) == 1
    assert result["equity_gate_result"]["passed"] is True
    assert result["final_verdict"] == "revised"
    assert result["final_level"] == PASS_LEVEL
    assert result["exception_register_entry"]["equity_gate_result"]["passed"] is True


def test_revised_verdict_gate_fails_then_round_two_red_circles():
    result = _run(
        _contest_output(proposed_level=VETO_LEVEL),
        arbiter_ruling_sequence=[
            _ruling(verdict="revised", final_level=VETO_LEVEL),
            _ruling(verdict="red_circled", final_level="L6"),
        ],
    )
    assert result["round_count"] == 2
    assert len(result["rounds"]) == 2
    assert len(result["gate_checks"]) == 1  # only round 1 reached the gate; red_circled skips it
    assert result["gate_checks"][0]["result"]["passed"] is False
    assert result["final_verdict"] == "red_circled"
    assert result["final_level"] == "L6"
    entry = result["exception_register_entry"]
    assert entry["verdict"] == "red_circled"
    assert entry["round_count"] == 2
    assert entry["arbiter_ruling"]["verdict"] == "red_circled"  # the round-2 ruling, not round 1's


def test_forced_escalation_after_two_vetoed_rounds():
    revised_ruling = _ruling(verdict="revised", final_level=VETO_LEVEL)
    result = _run(
        _contest_output(proposed_level=VETO_LEVEL),
        arbiter_ruling_sequence=[revised_ruling, revised_ruling],
    )
    assert result["round_count"] == 2
    assert len(result["gate_checks"]) == 2
    assert all(gc["result"]["passed"] is False for gc in result["gate_checks"])
    assert result["final_verdict"] == "escalated"
    assert result["final_level"] == "L6"  # the original crosswalk level, since nothing was actually revised
    entry = result["exception_register_entry"]
    assert entry["verdict"] == "escalated"
    assert entry["arbiter_ruling"]["verdict"] == "escalated"  # synthetic escalation ruling, matches piece 1's
    assert entry["arbiter_ruling"]["final_level"] == "L6"
    assert entry["round_count"] == 2


def test_exception_register_actually_persists_to_the_isolated_path():
    _run(_contest_output(), arbiter_ruling_sequence=[_ruling(verdict="upheld", final_level="L6")])
    register_path = negotiation_graph.DEFAULT_EXCEPTION_REGISTER_PATH
    assert register_path.exists()
    lines = register_path.read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["case_id"] == "CASE-TEST-001"
    assert entry["verdict"] == "upheld"
