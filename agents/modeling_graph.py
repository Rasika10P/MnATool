"""Wires the cost, retention and synthesis agents into one LangGraph subgraph (build order
item 6; CLAUDE.md M&A workflow steps 7-8): "Cost and retention agents run in parallel...
Synthesis reconciles them." Cost and retention both depend only on the same input population
and don't depend on each other, so they run in the same LangGraph superstep -- no Send fan-out
needed here (that's for per-employee dispatch, e.g. agents/leveling_batch_graph.py; this is
two independent whole-population branches). Synthesis waits for both to finish, per LangGraph's
own join semantics on a node with edges from multiple predecessors.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from agents.cost_model import assess_cost
from agents.modeling_schemas import CostAssessment, RetentionAssessment
from agents.retention_model import assess_retention
from agents.synthesis import reconcile

DEFAULT_CHECKPOINT_DB = Path(__file__).resolve().parent.parent / "data" / "modeling_checkpoints.sqlite"


class ModelingState(TypedDict):
    population: list[dict]
    as_of_date: str
    cost_assessment: Optional[dict]
    retention_assessment: Optional[dict]
    synthesis: Optional[dict]


def build_modeling_graph(cost_model=None, retention_model=None, synthesis_model=None):
    """`cost_model`/`retention_model`/`synthesis_model` are overrides for tests (fakes with a
    .with_structured_output method), same convention as every other build_graph in this repo.
    """

    def cost_node(state: ModelingState) -> dict:
        print("[cost] executing", flush=True)
        result = assess_cost(state["population"], state["as_of_date"], model=cost_model)
        return {"cost_assessment": result.model_dump()}

    def retention_node(state: ModelingState) -> dict:
        print("[retention] executing", flush=True)
        result = assess_retention(state["population"], state["as_of_date"], model=retention_model)
        return {"retention_assessment": result.model_dump()}

    def synthesis_node(state: ModelingState) -> dict:
        print("[synthesis] executing", flush=True)
        cost = CostAssessment(**state["cost_assessment"])
        retention = RetentionAssessment(**state["retention_assessment"])
        result = reconcile(cost, retention, model=synthesis_model)
        return {"synthesis": result.model_dump()}

    graph = StateGraph(ModelingState)
    graph.add_node("cost", cost_node)
    graph.add_node("retention", retention_node)
    graph.add_node("synthesis", synthesis_node)

    graph.add_edge(START, "cost")
    graph.add_edge(START, "retention")
    graph.add_edge("cost", "synthesis")
    graph.add_edge("retention", "synthesis")
    graph.add_edge("synthesis", END)
    return graph


def get_checkpointer(db_path: Path = DEFAULT_CHECKPOINT_DB) -> SqliteSaver:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)


def run_modeling(
    population: list[dict],
    as_of_date: str,
    thread_id: str = "default-modeling",
    db_path: Path = DEFAULT_CHECKPOINT_DB,
) -> dict:
    checkpointer = get_checkpointer(db_path)
    app = build_modeling_graph().compile(checkpointer=checkpointer)
    initial_state: ModelingState = {
        "population": population,
        "as_of_date": as_of_date,
        "cost_assessment": None,
        "retention_assessment": None,
        "synthesis": None,
    }
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke(initial_state, config, durability="sync")
