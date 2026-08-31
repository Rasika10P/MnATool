"""Build order item 5, gate 2: forced human review when a role has no equivalent in
Meridian's architecture.

ASSIGNMENT.md's framework table: "forced escalation when a role has no equivalent or
negotiation hits the round limit ... Humans intervene by overriding the level or accepting
the escalation." app/pipeline.py's resolve_mapping already detects this deterministically
(Nyx's Photonics dept and "Engineering Manager" sub-family have no Meridian family_group /
job mapping established anywhere in this codebase -- see that module's docstring) and
returns mapped=False with every unmet reason. Until now that result only drove a passive
"No equivalent role" status label in the UI (app/Home.py's _employee_status) -- nothing
actually paused the run or asked a human to decide anything. This graph is the real gate in
front of that dead end.

One node pair:
    gate -- interrupt()s with the employee's identity and every reason resolve_mapping
            couldn't complete a mapping, and waits for a Command(resume=...) carrying the
            reviewer's verdict: "escalated" (confirm there really is no equivalent -- this
            employee is handed off entirely, the same as CLAUDE.md's Photonics/Fellow
            planted problems) or "manually_mapped" (the reviewer supplies the missing
            family_group/geo_code/job_prefix pieces themselves, and the employee can re-enter
            the normal negotiation/modeling pipeline with that mapping).
    log  -- appends the review to data/no_equivalent_reviews.jsonl regardless of verdict, the
            same "every case gets a provenance record no matter the outcome" convention
            agents/negotiation_graph.py already applies to the exception register -- a human
            declining to map a role is exactly the kind of judgment call CLAUDE.md's
            non-negotiable 2 means to keep attributable, not just the dollar figures
            downstream of it.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

DEFAULT_CHECKPOINT_DB = Path(__file__).resolve().parent.parent / "data" / "no_equivalent_checkpoints.sqlite"
DEFAULT_REVIEW_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "no_equivalent_reviews.jsonl"

# Resolved at call time, not bound directly -- same convention as
# agents/negotiation_graph.py's DEFAULT_EXCEPTION_REGISTER_PATH and
# agents/approval_graph.py's DECISIONS_DB_PATH, so tests can redirect by monkeypatching this
# module attribute.
REVIEW_LOG_PATH = DEFAULT_REVIEW_LOG_PATH


class NoEquivalentState(TypedDict):
    # Input -- set once, from app/pipeline.py's resolve_mapping result for a mapped=False employee.
    employee_id: str
    job_title: str
    dept: str
    sub_family: str
    reason: str  # resolve_mapping's semicolon-joined list of unmet mapping pieces

    # Set by gate_node from the reviewer's Command(resume=...).
    reviewer_verdict: Optional[str]  # "escalated" | "manually_mapped"
    manual_mapping: Optional[dict]  # {"family_group", "geo_code", "job_prefix"} when manually_mapped; None otherwise

    # Set by log_node.
    review_entry: Optional[dict]


def build_no_equivalent_graph():
    def gate_node(state: NoEquivalentState) -> dict:
        decision = interrupt(
            {
                "employee_id": state["employee_id"],
                "job_title": state["job_title"],
                "dept": state["dept"],
                "sub_family": state["sub_family"],
                "reason": state["reason"],
            }
        )
        return {
            "reviewer_verdict": decision["verdict"],
            "manual_mapping": decision.get("manual_mapping"),
        }

    def log_node(state: NoEquivalentState) -> dict:
        entry = {
            "employee_id": state["employee_id"],
            "job_title": state["job_title"],
            "dept": state["dept"],
            "sub_family": state["sub_family"],
            "reason": state["reason"],
            "verdict": state["reviewer_verdict"],
            "manual_mapping": state.get("manual_mapping"),
        }
        path = REVIEW_LOG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return {"review_entry": entry}

    graph = StateGraph(NoEquivalentState)
    graph.add_node("gate", gate_node)
    graph.add_node("log", log_node)
    graph.add_edge(START, "gate")
    graph.add_edge("gate", "log")
    graph.add_edge("log", END)
    return graph


def get_checkpointer(db_path: Path = DEFAULT_CHECKPOINT_DB) -> SqliteSaver:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)


def start_no_equivalent_review(
    initial_state: NoEquivalentState, thread_id: str, db_path: Path = DEFAULT_CHECKPOINT_DB
) -> dict:
    """Runs the graph up to the gate's interrupt(). Returns the raw invoke() result -- check
    result["__interrupt__"] for the payload to show the reviewer; it's truthy iff the graph
    is paused waiting on this thread_id."""
    checkpointer = get_checkpointer(db_path)
    app = build_no_equivalent_graph().compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke(initial_state, config, durability="sync")


def resume_no_equivalent_review(
    thread_id: str, verdict: str, manual_mapping: dict | None = None, db_path: Path = DEFAULT_CHECKPOINT_DB
) -> dict:
    """Resumes a paused thread_id with the reviewer's decision. verdict is "escalated" or
    "manually_mapped"; manual_mapping is required (and only meaningful) for "manually_mapped"
    -- {"family_group": ..., "geo_code": ..., "job_prefix": ...}, the exact three pieces
    app/pipeline.py's resolve_mapping already knows how to look up when they exist."""
    checkpointer = get_checkpointer(db_path)
    app = build_no_equivalent_graph().compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke(
        Command(resume={"verdict": verdict, "manual_mapping": manual_mapping}), config, durability="sync"
    )
