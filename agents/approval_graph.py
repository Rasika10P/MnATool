"""Build order item 5, gate 4: human approval before any write to leveling_decisions.

ASSIGNMENT.md's framework table: "Human-in-the-loop ... final approval before any write."
tools/decisions.py's write_mapping_decision is the deterministic write action; this graph is
the gate in front of it -- nothing upstream of `gate_node` decides whether a write happens.

Two nodes:
    gate  -- interrupt()s with the full context a reviewer needs (employee, assigned level,
             evidence, both positions if this decision came out of a negotiation), and waits
             for a Command(resume=...) carrying the reviewer's verdict.
    write -- calls write_mapping_decision only if the verdict wasn't "rejected". A rejection
             leaves leveling_decisions untouched; written_record stays None.

One employee, one thread_id, one pending decision: this graph is invoked per employee, not
batched, because interrupt() pauses the whole run at the single first interrupt it hits and
a human reviews one case at a time (ASSIGNMENT.md: "Humans intervene by overriding the level
or accepting the escalation").
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from tools.decisions import DEFAULT_DB_PATH, write_mapping_decision

DEFAULT_CHECKPOINT_DB = Path(__file__).resolve().parent.parent / "data" / "approval_checkpoints.sqlite"

# Resolved at call time, not bound directly, so tests can redirect by monkeypatching this
# module attribute -- same convention as agents/negotiation_graph.py's exception-register path.
DECISIONS_DB_PATH = DEFAULT_DB_PATH


class ApprovalState(TypedDict):
    # Input -- the decision being proposed for a write, plus optional negotiation context.
    job_or_employee_ref: str
    assigned_level: str
    confidence: float
    factor_ratings: list[dict]
    factor5_variant_applied: Optional[str]
    alternative_considered: Optional[str]
    governing_rule: Optional[str]
    source_document_hash: Optional[str]
    negotiation_context: Optional[dict]  # {"nyx_level": ..., "advocate_proposed_level": ..., "final_verdict": ...} when this decision came out of a contest; None otherwise

    # Set by gate_node from the reviewer's Command(resume=...).
    reviewer_verdict: Optional[str]  # "approved" | "approved_with_override" | "rejected"
    final_level: Optional[str]

    # Set by write_node.
    written_record: Optional[dict]


def build_approval_graph():
    def gate_node(state: ApprovalState) -> dict:
        decision = interrupt(
            {
                "employee_id": state["job_or_employee_ref"],
                "assigned_level": state["assigned_level"],
                "confidence": state["confidence"],
                "factor_ratings": state["factor_ratings"],
                "governing_rule": state["governing_rule"],
                "alternative_considered": state.get("alternative_considered"),
                "negotiation_context": state.get("negotiation_context"),
            }
        )
        return {
            "reviewer_verdict": decision["verdict"],
            "final_level": decision.get("override_level") or state["assigned_level"],
        }

    def write_node(state: ApprovalState) -> dict:
        if state["reviewer_verdict"] == "rejected":
            return {"written_record": None}
        record = write_mapping_decision(
            job_or_employee_ref=state["job_or_employee_ref"],
            assigned_level=state["final_level"],
            confidence=state["confidence"],
            factor_ratings=state["factor_ratings"],
            factor5_variant_applied=state.get("factor5_variant_applied"),
            alternative_considered=state.get("alternative_considered"),
            governing_rule=state["governing_rule"],
            reviewer_verdict=state["reviewer_verdict"],
            source_document_hash=state.get("source_document_hash"),
            db_path=DECISIONS_DB_PATH,
        )
        return {"written_record": record}

    graph = StateGraph(ApprovalState)
    graph.add_node("gate", gate_node)
    graph.add_node("write", write_node)
    graph.add_edge(START, "gate")
    graph.add_edge("gate", "write")
    graph.add_edge("write", END)
    return graph


def get_checkpointer(db_path: Path = DEFAULT_CHECKPOINT_DB) -> SqliteSaver:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)


def start_approval(initial_state: ApprovalState, thread_id: str, db_path: Path = DEFAULT_CHECKPOINT_DB) -> dict:
    """Runs the graph up to the gate's interrupt(). Returns the raw invoke() result -- check
    result["__interrupt__"] for the payload to show the reviewer; it's truthy iff the graph is
    paused waiting on this thread_id."""
    checkpointer = get_checkpointer(db_path)
    app = build_approval_graph().compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke(initial_state, config, durability="sync")


def resume_approval(thread_id: str, verdict: str, override_level: str | None = None, db_path: Path = DEFAULT_CHECKPOINT_DB) -> dict:
    """Resumes a paused thread_id with the reviewer's decision. verdict is one of "approved",
    "approved_with_override", "rejected"; override_level is required (and only meaningful)
    for "approved_with_override"."""
    checkpointer = get_checkpointer(db_path)
    app = build_approval_graph().compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke(Command(resume={"verdict": verdict, "override_level": override_level}), config, durability="sync")
