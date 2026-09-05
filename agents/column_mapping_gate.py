"""Build order item 5, gate 1: column mapping confirmation before ingest.

ASSIGNMENT.md's framework table: "Three gates: column mapping confirmation before ingest,
[...]". app/pipeline.py's validate_uploaded_census currently requires an exact match against
CENSUS_COLUMNS and rejects anything else outright (its own docstring: "Wiring a validated
upload into the actual run is future work, not something to silently half-implement by
pointing the real pipeline at unvetted data") -- a real acquired-company census will not use
Meridian's own column names, so an exact-match gate would reject every real upload. This
graph is the actual gate: propose a mapping, then require a human to confirm or correct it
before a single row is ingested.

Four nodes:
    suggest  -- tools.column_mapping.suggest_column_mapping (deterministic, no model call)
                proposes {target_column: raw_column_or_None} from the uploaded workbook's
                actual headers. Runs once, at START.
    gate     -- interrupt()s with the raw headers, the mapping to show as the starting point
                (the fresh suggestion on the first pass; whatever the reviewer last
                submitted on a re-prompt, so a correction never has to be redone from
                scratch), and which target columns -- required or not -- have no match yet.
                Waits for a Command(resume=...) carrying the reviewer's confirmed_mapping.
    validate -- deterministic, no model call: checks confirmed_mapping against
                tools.column_mapping.REQUIRED_COLUMNS. Loops back to gate (a fresh
                interrupt(), with the specific missing required columns named in the
                payload) if any required column is still unmapped; a target column the
                pipeline merely reads-if-present (Bonus, Unvested Options, Start, Role
                Summary) never blocks. Nothing reaches log until this passes.
    log      -- appended to data/column_mapping_reviews.jsonl, the same "every case gets a
                provenance record" convention agents/no_equivalent_gate.py and
                agents/negotiation_graph.py's exception register already apply -- which raw
                column got treated as which Meridian field is exactly the kind of thing
                CLAUDE.md's non-negotiable 2 means to keep attributable, since every
                downstream number for this employee traces back through whichever column
                the human said was "Base". Only a mapping that cleared validate is ever
                logged -- an abandoned, still-incomplete attempt leaves no record, the same
                as never having started.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from tools.column_mapping import REQUIRED_COLUMNS, TARGET_COLUMNS, suggest_column_mapping

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

    # Set by gate_node from the reviewer's Command(resume=...) -- the reviewer's latest
    # submission, whether or not it actually clears validate_node yet.
    confirmed_mapping: Optional[dict]

    # Set by validate_node. None until a confirm attempt has been checked; [] once a
    # confirmed_mapping covers every REQUIRED_COLUMNS entry.
    missing_required: Optional[list]

    # Set by log_node.
    review_entry: Optional[dict]


def build_column_mapping_graph():
    def suggest_node(state: ColumnMappingState) -> dict:
        return {"suggested_mapping": suggest_column_mapping(state["raw_columns"])}

    def gate_node(state: ColumnMappingState) -> dict:
        # The reviewer's own last submission is the starting point on a re-prompt (a
        # required column was still missing), not the original suggestion -- correcting one
        # field shouldn't undo every other field the reviewer already got right.
        baseline = state.get("confirmed_mapping") or state["suggested_mapping"]
        unmatched = [target for target in TARGET_COLUMNS if baseline.get(target) is None]
        decision = interrupt(
            {
                "source_name": state["source_name"],
                "raw_columns": state["raw_columns"],
                "suggested_mapping": baseline,
                "unmatched_targets": unmatched,
                "missing_required": state.get("missing_required"),
            }
        )
        return {"confirmed_mapping": decision["confirmed_mapping"]}

    def validate_node(state: ColumnMappingState) -> dict:
        missing = [target for target in REQUIRED_COLUMNS if state["confirmed_mapping"].get(target) is None]
        return {"missing_required": missing}

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

    def _route_after_validate(state: ColumnMappingState) -> str:
        return "gate" if state["missing_required"] else "log"

    graph = StateGraph(ColumnMappingState)
    graph.add_node("suggest", suggest_node)
    graph.add_node("gate", gate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("log", log_node)
    graph.add_edge(START, "suggest")
    graph.add_edge("suggest", "gate")
    graph.add_edge("gate", "validate")
    graph.add_conditional_edges("validate", _route_after_validate, ["gate", "log"])
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
    """Resumes a paused thread_id with the reviewer's confirmed mapping -- a
    {target_column: raw_column_or_None} entry for every target in
    tools.column_mapping.TARGET_COLUMNS, whether that means accepting the suggestion
    verbatim or overriding some of it by hand. A required column (REQUIRED_COLUMNS) left
    None fails validate_node and returns *another* paused result at a fresh interrupt() --
    check result["__interrupt__"] again rather than assuming this call always finalizes;
    result["__interrupt__"][0].value["missing_required"] names what's still missing. Only a
    mapping covering every required column reaches log_node and returns a final state with
    "review_entry" set."""
    checkpointer = get_checkpointer(db_path)
    app = build_column_mapping_graph().compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke(Command(resume={"confirmed_mapping": confirmed_mapping}), config, durability="sync")
