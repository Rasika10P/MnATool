"""
Fans the leveling agent out over a population using LangGraph's Send API (build order
item 3; ASSIGNMENT.md Friday gate: "25 employees leveled in parallel, run resumable").

Each employee is dispatched as an independent Send task to level_employee, which runs the
same parse-validation + _run_leveling_call as the single-role graph (agents/leveling_graph.py)
-- one employee's failure or slowness doesn't block the others. Results collect into
`decisions` via an additive reducer, so partial completion (e.g. after a kill) is exactly
the list of employees who finished, no more and no less.
"""

from __future__ import annotations

import operator
import os
import sqlite3
import time
from pathlib import Path
from typing import Annotated, Optional, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.leveling import _run_leveling_call
from agents.schemas import SourceOrgContext

DEFAULT_BATCH_CHECKPOINT_DB = Path(__file__).resolve().parent.parent / "data" / "leveling_batch_checkpoints.sqlite"

LOW_CONFIDENCE_THRESHOLD = 0.65
HIGH_CONFIDENCE_THRESHOLD = 0.75


class BatchState(TypedDict):
    employees: list[dict]  # input: [{"employee_id", "job_description", "source_org_context"}, ...]
    employee_id: str  # per-task field, set by each Send
    job_description: str  # per-task field, set by each Send
    source_org_context: Optional[dict]  # per-task field, set by each Send
    stagger_index: int  # per-task field, only used by the demo delay hook
    decisions: Annotated[list[dict], operator.add]


def _dispatch(state: BatchState) -> list[Send]:
    return [
        Send(
            "level_employee",
            {
                "employee_id": emp["employee_id"],
                "job_description": emp["job_description"],
                "source_org_context": emp.get("source_org_context"),
                "stagger_index": i,
            },
        )
        for i, emp in enumerate(state["employees"])
    ]


def build_batch_graph(model=None):
    """`model` is a test override, same convention as agents.leveling_graph.build_graph."""

    def level_employee(state: BatchState) -> dict:
        print(f"[level_employee] executing for {state['employee_id']}", flush=True)

        # Demo-only hook, same pattern as agents/leveling_graph.py's single-role delay --
        # staggers each employee's start so scripts/batch_kill_demo.py can kill partway
        # through a real batch at a predictable point (some done, some not), instead of
        # racing real API latency (which this session has seen vary from <1s to 12s+ on
        # identical calls). Production runs never set this.
        stagger_seconds = float(os.environ.get("LEVELING_DEMO_STAGGER_SECONDS", "0"))
        if stagger_seconds:
            time.sleep(stagger_seconds * state["stagger_index"])

        if not state["job_description"].strip():
            raise ValueError(f"{state['employee_id']}: job_description must not be empty")

        context = SourceOrgContext(**state["source_org_context"]) if state.get("source_org_context") else None
        decision = _run_leveling_call(
            state["job_description"], context,
            LOW_CONFIDENCE_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD,
            model=model,
        )
        print(f"[level_employee] done for {state['employee_id']}", flush=True)
        return {"decisions": [{"employee_id": state["employee_id"], **decision.model_dump()}]}

    graph = StateGraph(BatchState)
    graph.add_node("level_employee", level_employee)
    graph.add_conditional_edges(START, _dispatch, ["level_employee"])
    graph.add_edge("level_employee", END)
    return graph


def get_checkpointer(db_path: Path = DEFAULT_BATCH_CHECKPOINT_DB) -> SqliteSaver:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)


def run_batch(
    employees: list[dict],
    thread_id: str = "default-batch",
    db_path: Path = DEFAULT_BATCH_CHECKPOINT_DB,
) -> list[dict]:
    checkpointer = get_checkpointer(db_path)
    app = build_batch_graph().compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    # durability="sync": LangGraph's default ("async") dispatches checkpoint writes in the
    # background, which a hard kill can lose before it lands -- confirmed by direct
    # inspection of the checkpointer's writes table (individual Send-task completions
    # never appeared there under a real mid-batch kill, no matter how long the wait before
    # killing, until this was set explicitly). Required for the resumability this fan-out
    # is built to provide; see scripts/batch_kill_demo.py.
    result = app.invoke({"employees": employees, "decisions": []}, config, durability="sync")
    return result["decisions"]
