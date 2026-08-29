"""
LangGraph version of the leveling agent -- two nodes (parse, level) plus a SqliteSaver
checkpointer (build order item 2, second half; SETUP.md step B).

The level node calls agents.leveling._run_leveling_call directly -- the exact same
prompt-building and model call as the plain-function version, not a re-implementation of it.

The parse node runs agents.scope_extraction.extract_scope_profile on Nebius (CLAUDE.md model
routing: "Nebius -- job description parsing"): structured extraction of reports-to, span of
control, budget authority, decision scope and ownership scope, stored in state as
scope_profile. The level node passes that extraction into _run_leveling_call as advisory
context (agents/leveling.py's system prompt: the job description text stays authoritative,
the extraction is evidence to weigh, not a substitute) -- so this graph's leveling decisions
are no longer byte-identical to a bare level_role(job_description) call with no scope_profile
argument; they differ by exactly that one additional piece of advisory evidence. Nebius was
tried on leveling itself first and dropped (see agents.leveling.level_role_routed's
docstring); parsing is the task CLAUDE.md actually routes to it.
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Optional, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from agents.leveling import _run_leveling_call
from agents.schemas import ScopeProfile, SourceOrgContext
from agents.scope_extraction import extract_scope_profile

DEFAULT_CHECKPOINT_DB = Path(__file__).resolve().parent.parent / "data" / "leveling_checkpoints.sqlite"


class LevelingState(TypedDict):
    job_description: str
    source_org_context: Optional[dict]
    low_confidence_threshold: float
    high_confidence_threshold: float
    parsed: bool
    scope_profile: Optional[dict]
    decision: Optional[dict]


def build_graph(level_model=None, parse_model=None):
    """`level_model`/`parse_model` are overrides for tests (fakes with a
    .with_structured_output method) -- production callers leave both None and get the
    normal get_model("judgment") / get_model("volume") routing.
    """

    def parse_node(state: LevelingState) -> dict:
        print("[parse] executing", flush=True)
        if not state["job_description"].strip():
            raise ValueError("job_description must not be empty")
        profile = extract_scope_profile(state["job_description"], model=parse_model)
        return {"parsed": True, "scope_profile": profile.model_dump()}

    def level_node(state: LevelingState) -> dict:
        print("[level] executing", flush=True)
        # Demo-only hook: a deliberate, controllable pause before the real model call, so
        # scripts/checkpoint_kill_demo.py has a wide, reliable kill window instead of racing
        # real API latency (observed anywhere from <1s to 12s+ across identical calls this
        # session -- too variable to time a kill against). Production runs never set this.
        demo_delay = float(os.environ.get("LEVELING_DEMO_DELAY_SECONDS", "0"))
        if demo_delay:
            time.sleep(demo_delay)
        context = SourceOrgContext(**state["source_org_context"]) if state.get("source_org_context") else None
        scope_profile = ScopeProfile(**state["scope_profile"]) if state.get("scope_profile") else None
        decision = _run_leveling_call(
            state["job_description"], context,
            state["low_confidence_threshold"], state["high_confidence_threshold"],
            model=level_model, scope_profile=scope_profile,
        )
        return {"decision": decision.model_dump()}

    graph = StateGraph(LevelingState)
    graph.add_node("parse", parse_node)
    graph.add_node("level", level_node)
    graph.add_edge(START, "parse")
    graph.add_edge("parse", "level")
    graph.add_edge("level", END)
    return graph


def get_checkpointer(db_path: Path = DEFAULT_CHECKPOINT_DB) -> SqliteSaver:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)


def run_leveling(
    job_description: str,
    source_org_context: SourceOrgContext | None = None,
    low_confidence_threshold: float = 0.65,
    high_confidence_threshold: float = 0.75,
    thread_id: str = "default",
    db_path: Path = DEFAULT_CHECKPOINT_DB,
) -> dict:
    """Convenience entry point: build the graph, run it to completion, return the decision
    dict. For the checkpoint-kill-resume demo, use build_graph()/get_checkpointer() and
    .invoke() directly instead -- see scripts/checkpoint_start.py and checkpoint_resume.py.
    """
    checkpointer = get_checkpointer(db_path)
    app = build_graph().compile(checkpointer=checkpointer)
    initial_state: LevelingState = {
        "job_description": job_description,
        "source_org_context": source_org_context.model_dump(exclude_none=True) if source_org_context else None,
        "low_confidence_threshold": low_confidence_threshold,
        "high_confidence_threshold": high_confidence_threshold,
        "parsed": False,
        "scope_profile": None,
        "decision": None,
    }
    config = {"configurable": {"thread_id": thread_id}}
    # durability="sync" -- see agents/leveling_batch_graph.py's run_batch for why the
    # default ("async") isn't safe for a checkpoint a kill/resume demo depends on.
    result = app.invoke(initial_state, config, durability="sync")
    return result["decision"]
