"""Build order item 4, step 5: wires the advocate (agents/advocate.py), arbiter
(agents/arbiter.py) and equity gate (agents/equity_gate.py) into one LangGraph subgraph, per
level_framework.md section 7.

Flow:
    advocate -> [declines] -> finalize
             -> [contests] -> arbiter -> [upheld/red_circled/escalated] -> finalize
                                      -> [revised] -> equity_gate -> [passed] -> finalize
                                                                   -> [failed, round < 2] -> arbiter (round 2)
                                                                   -> [failed, round == 2] -> finalize (forced escalate)

The crosswalk mapping being contested is NOT a node in this graph. Section 7 lists the
crosswalk agent as a participant distinct from advocate/arbiter/equity, and it isn't part of
the round loop -- the mapping under contest doesn't change between rounds, only the
negotiation over it does. It's produced once, upstream (agents.leveling.level_role, the same
stand-in used by scripts/arbiter_nyx_011.py), and handed to this graph as input. run_negotiation
below does that one-shot call before invoking the graph, so the observable flow is still
"crosswalk proposal -> advocate -> arbiter -> ..." end to end.

Round counting: one round = one arbiter ruling. MAX_ROUNDS = 2 (section 7, "round limit").
The round that gets vetoed by the equity gate loops back to the *arbiter*, not the advocate
-- the advocate's argument doesn't change between rounds, only the arbiter's response to a
gate rejection does (see agents/arbiter.py's `prior_equity_gate_rejection` parameter). If
round 2 is also a "revised" verdict the gate rejects, the graph forces the outcome to
"escalated" itself rather than looping a third time -- section 7 caps rounds at two.

Every contested case (the advocate did not decline) writes an ExceptionRegisterEntry to the
persistent register regardless of verdict, per section 7: "All contested cases -- regardless
of verdict -- are written to an exception register." A case the advocate declined to contest
never reaches an arbiter verdict at all, so there's nothing to register -- the original
mapping simply stands.
"""

from __future__ import annotations

import json
import operator
import sqlite3
from pathlib import Path
from typing import Annotated, Optional, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from agents.advocate import contest_mapping
from agents.arbiter import rule as arbiter_rule
from agents.equity_gate import check_equity
from agents.leveling import level_role
from agents.negotiation_schemas import AdvocateOutput, ArbiterRuling, EquityGateResult, ExceptionRegisterEntry
from agents.schemas import LevelingDecision, SourceOrgContext

MAX_ROUNDS = 2

DEFAULT_CHECKPOINT_DB = Path(__file__).resolve().parent.parent / "data" / "negotiation_checkpoints.sqlite"
DEFAULT_EXCEPTION_REGISTER_PATH = Path(__file__).resolve().parent.parent / "data" / "exception_register.jsonl"


class NegotiationState(TypedDict):
    # Input -- set once, before the graph runs, and never changed by any node.
    case_id: str
    employee_id: str
    role_summary: str
    nyx_level: str
    crosswalk_decision: dict  # LevelingDecision.model_dump()
    family_group: str
    candidate_geo_code: str
    candidate_salary: float  # what the revision would pay -- see agents/equity_gate.py

    # Working state.
    round_count: int
    contested: Optional[bool]
    advocate_output: Optional[dict]  # AdvocateOutput.model_dump(), set once
    arbiter_ruling: Optional[dict]  # ArbiterRuling.model_dump(), latest round
    equity_gate_result: Optional[dict]  # EquityGateResult.model_dump(), latest round or None
    rounds: Annotated[list[dict], operator.add]  # transcript: [{"round": n, "ruling": {...}}, ...]
    gate_checks: Annotated[list[dict], operator.add]  # transcript: [{"round": n, "result": {...}}, ...]

    # Output.
    final_verdict: Optional[str]
    final_level: Optional[str]
    exception_register_entry: Optional[dict]


def _append_exception_register(entry: ExceptionRegisterEntry, path: Path | None = None) -> None:
    # Resolved at call time (a None default, not DEFAULT_EXCEPTION_REGISTER_PATH bound
    # directly) so tests can redirect by monkeypatching the module attribute -- matches
    # agents/cost_logging.py's log_call, which has the identical requirement.
    path = path if path is not None else DEFAULT_EXCEPTION_REGISTER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry.model_dump(mode="json")) + "\n")


def build_negotiation_graph(advocate_model=None, arbiter_model=None):
    """`advocate_model`/`arbiter_model` are overrides for tests (fakes with a
    .with_structured_output method), same convention as agents.leveling_graph.build_graph.
    The equity gate takes no model -- it's fully deterministic (agents/equity_gate.py)."""

    def advocate_node(state: NegotiationState) -> dict:
        print("[advocate] executing", flush=True)
        crosswalk_decision = LevelingDecision(**state["crosswalk_decision"])
        output = contest_mapping(
            state["role_summary"], state["nyx_level"], crosswalk_decision.assigned_level, model=advocate_model
        )
        return {"advocate_output": output.model_dump(), "contested": output.contests}

    def arbiter_node(state: NegotiationState) -> dict:
        round_number = state.get("round_count", 0) + 1
        print(f"[arbiter] executing (round {round_number})", flush=True)
        crosswalk_decision = LevelingDecision(**state["crosswalk_decision"])
        advocate_output = AdvocateOutput(**state["advocate_output"])
        argument = advocate_output.as_crosswalk_argument()
        prior_rejection = None
        if round_number > 1 and state.get("equity_gate_result") is not None:
            prior_rejection = EquityGateResult(**state["equity_gate_result"])

        ruling = arbiter_rule(
            crosswalk_decision, argument, model=arbiter_model, prior_equity_gate_rejection=prior_rejection
        )
        return {
            "round_count": round_number,
            "arbiter_ruling": ruling.model_dump(),
            "rounds": [{"round": round_number, "ruling": ruling.model_dump()}],
        }

    def equity_gate_node(state: NegotiationState) -> dict:
        ruling = state["arbiter_ruling"]
        print(f"[equity_gate] executing for proposed level {ruling['final_level']}", flush=True)
        result = check_equity(
            family_group=state["family_group"],
            level_code=ruling["final_level"],
            candidate_geo_code=state["candidate_geo_code"],
            candidate_salary=state["candidate_salary"],
        )
        return {
            "equity_gate_result": result.model_dump(),
            "gate_checks": [{"round": state["round_count"], "result": result.model_dump()}],
        }

    def finalize_node(state: NegotiationState) -> dict:
        print("[finalize] executing", flush=True)
        if not state.get("contested"):
            return {
                "final_verdict": "upheld",
                "final_level": state["crosswalk_decision"]["assigned_level"],
                "exception_register_entry": None,
            }

        last_ruling_dict = state["arbiter_ruling"]
        last_gate_dict = state.get("equity_gate_result")
        forced_escalation = (
            last_ruling_dict["verdict"] == "revised"
            and last_gate_dict is not None
            and not last_gate_dict["passed"]
            and state["round_count"] >= MAX_ROUNDS
        )

        if forced_escalation:
            effective_ruling = ArbiterRuling(
                verdict="escalated",
                governing_rule="section 7 round limit: two rounds maximum, unresolved after the second ruling",
                final_level=state["crosswalk_decision"]["assigned_level"],
                reasoning=(
                    f"Arbiter ruled 'revised' to {last_ruling_dict['final_level']} across "
                    f"{state['round_count']} round(s); the equity gate rejected it each time it "
                    f"was checked (most recent: {last_gate_dict['reasoning']}). Unresolved after "
                    "the round limit -- routed to a human with both positions stated (section 7)."
                ),
            )
        else:
            effective_ruling = ArbiterRuling(**last_ruling_dict)

        advocate_output = AdvocateOutput(**state["advocate_output"])
        entry = ExceptionRegisterEntry(
            case_id=state["case_id"],
            employee_id=state["employee_id"],
            crosswalk_level=state["crosswalk_decision"]["assigned_level"],
            advocate_position=advocate_output.proposed_level,
            advocate_argument=advocate_output.as_crosswalk_argument(),
            arbiter_ruling=effective_ruling,
            governing_rule_cited=effective_ruling.governing_rule,
            equity_gate_result=EquityGateResult(**last_gate_dict) if last_gate_dict is not None else None,
            verdict=effective_ruling.verdict,
            round_count=state["round_count"],
        )
        _append_exception_register(entry)

        return {
            "final_verdict": effective_ruling.verdict,
            "final_level": effective_ruling.final_level,
            "exception_register_entry": entry.model_dump(mode="json"),
        }

    def _route_after_advocate(state: NegotiationState) -> str:
        return "arbiter" if state["contested"] else "finalize"

    def _route_after_arbiter(state: NegotiationState) -> str:
        return "equity_gate" if state["arbiter_ruling"]["verdict"] == "revised" else "finalize"

    def _route_after_equity_gate(state: NegotiationState) -> str:
        if state["equity_gate_result"]["passed"]:
            return "finalize"
        if state["round_count"] >= MAX_ROUNDS:
            return "finalize"  # finalize_node detects the forced-escalation case itself
        return "arbiter"

    graph = StateGraph(NegotiationState)
    graph.add_node("advocate", advocate_node)
    graph.add_node("arbiter", arbiter_node)
    graph.add_node("equity_gate", equity_gate_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "advocate")
    graph.add_conditional_edges("advocate", _route_after_advocate, ["arbiter", "finalize"])
    graph.add_conditional_edges("arbiter", _route_after_arbiter, ["equity_gate", "finalize"])
    graph.add_conditional_edges("equity_gate", _route_after_equity_gate, ["arbiter", "finalize"])
    graph.add_edge("finalize", END)
    return graph


def get_checkpointer(db_path: Path = DEFAULT_CHECKPOINT_DB) -> SqliteSaver:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)


def run_negotiation(
    case_id: str,
    employee_id: str,
    role_summary: str,
    nyx_level: str,
    job_description: str,
    family_group: str,
    candidate_geo_code: str,
    candidate_salary: float,
    source_org_context: SourceOrgContext | None = None,
    thread_id: str = "default-negotiation",
    db_path: Path = DEFAULT_CHECKPOINT_DB,
) -> dict:
    """Convenience entry point. Runs the crosswalk agent once (agents.leveling.level_role,
    standing in for section 7's crosswalk-agent participant) to produce the mapping under
    contest, then the advocate/arbiter/equity-gate subgraph against it. Returns the final
    state dict -- final_verdict, final_level, exception_register_entry, and the full
    rounds/gate_checks transcript.
    """
    crosswalk_decision = level_role(job_description, source_org_context=source_org_context)

    initial_state: NegotiationState = {
        "case_id": case_id,
        "employee_id": employee_id,
        "role_summary": role_summary,
        "nyx_level": nyx_level,
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

    checkpointer = get_checkpointer(db_path)
    app = build_negotiation_graph().compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke(initial_state, config, durability="sync")
