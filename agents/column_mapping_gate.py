"""Build order item 5, gate 1: column mapping confirmation before ingest.

ASSIGNMENT.md's framework table: "Three gates: column mapping confirmation before ingest,
[...]". app/pipeline.py's validate_uploaded_census currently requires an exact match against
CENSUS_COLUMNS and rejects anything else outright (its own docstring: "Wiring a validated
upload into the actual run is future work, not something to silently half-implement by
pointing the real pipeline at unvetted data") -- a real acquired-company census will not use
Meridian's own column names, so an exact-match gate would reject every real upload. This
graph is the actual gate: propose a mapping, then require a human to confirm or correct it
before a single row is ingested.

Two nodes:
    suggest -- tools.column_mapping.suggest_column_mapping (deterministic, no model call)
               proposes {target_column: raw_column_or_None} from the uploaded workbook's
               actual headers.
    gate    -- interrupt()s with the raw headers and the suggested mapping (plus which
               required target columns have no match yet) and waits for a
               Command(resume=...) carrying the reviewer's confirmed_mapping -- a complete
               {target_column: raw_column} covering every entry in TARGET_COLUMNS, whether
               that means accepting the suggestion or overriding it by hand.

Every review is appended to data/column_mapping_reviews.jsonl regardless of how much the
human changed, the same "every case gets a provenance record" convention
agents/no_equivalent_gate.py and agents/negotiation_graph.py's exception register already
apply -- which raw column got treated as which Meridian field is exactly the kind of thing
CLAUDE.md's non-negotiable 2 means to keep attributable, since every downstream number for
this employee traces back through whichever column the human said was "Base".
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from tools.column_mapping import TARGET_COLUMNS, suggest_column_mapping

DEFAULT_CHECKPOINT_DB = Path(__file__).resolve().parent.parent / "data" / "column_mapping_checkpoints.sqlite"
DEFAULT_REVIEW_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "column_mapping_reviews.jsonl"

# Resolved at call time, not bound directly -- same convention as
# agents/no_equivalent_gate.py's REVIEW_LOG_PATH, so tests can redirect by monkeypatching
# this module attribute.
REVIEW_LOG_PATH = DEFAULT_REVIEW_LOG_PATH


class ColumnMappingState(TypedDict):
    # Input -- set once, from the uploaded workbook's actual header row.
    source_name: str  # filename or other identifier for the upload, for the review log
    raw_columns: list[str]

    # Set by suggest_node.
    suggested_mapping: Optional[dict]  # {target_column: raw_column_or_None}

    # Set by gate_node from the reviewer's Command(resume=...).
    confirmed_mapping: Optional[dict]  # {target_column: raw_column}, every TARGET_COLUMNS entry present

    # Set by log_node.
    review_entry: Optional[dict]


def build_column_mapping_graph():
    def suggest_node(state: ColumnMappingState) -> dict:
        return {"suggested_mapping": suggest_column_mapping(state["raw_columns"])}

    def gate_node(state: ColumnMappingState) -> dict:
        suggested = state["suggested_mapping"]
        unmatched = [target for target in TARGET_COLUMNS if suggested.get(target) is None]
        decision = interrupt(
            {
                "source_name": state["source_name"],
                "raw_columns": state["raw_columns"],
                "suggested_mapping": suggested,
                "unmatched_targets": unmatched,
            }
        )
        return {"confirmed_mapping": decision["confirmed_mapping"]}

    def log_node(state: ColumnMappingState) -> dict:
        entry = {
            "source_name": state["source_name"],
            "raw_columns": state["raw_columns"],
            "suggested_mapping": state["suggested_mapping"],
            "confirmed_mapping": state["confirmed_mapping"],
        }
        path = REVIEW_LOG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return {"review_entry": entry}

    graph = StateGraph(ColumnMappingState)
    graph.add_node("suggest", suggest_node)
    graph.add_node("gate", gate_node)
    graph.add_node("log", log_node)
    graph.add_edge(START, "suggest")
    graph.add_edge("suggest", "gate")
    graph.add_edge("gate", "log")
    graph.add_edge("log", END)
    return graph


def get_checkpointer(db_path: Path = DEFAULT_CHECKPOINT_DB) -> SqliteSaver:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)


def start_column_mapping_review(
    initial_state: ColumnMappingState, thread_id: str, db_path: Path = DEFAULT_CHECKPOINT_DB
) -> dict:
    """Runs the graph up to the gate's interrupt(). Returns the raw invoke() result -- check
    result["__interrupt__"] for the payload to show the reviewer; it's truthy iff the graph
    is paused waiting on this thread_id. The payload's suggested_mapping is pre-computed by
    suggest_node, so the reviewer sees a proposal, never a blank form."""
    checkpointer = get_checkpointer(db_path)
    app = build_column_mapping_graph().compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke(initial_state, config, durability="sync")


def resume_column_mapping_review(
    thread_id: str, confirmed_mapping: dict, db_path: Path = DEFAULT_CHECKPOINT_DB
) -> dict:
    """Resumes a paused thread_id with the reviewer's confirmed mapping -- a complete
    {target_column: raw_column} covering every entry in tools.column_mapping.TARGET_COLUMNS,
    whether that means accepting the suggestion verbatim or overriding some of it by hand."""
    checkpointer = get_checkpointer(db_path)
    app = build_column_mapping_graph().compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke(Command(resume={"confirmed_mapping": confirmed_mapping}), config, durability="sync")
