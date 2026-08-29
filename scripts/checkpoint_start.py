"""Starts a leveling run under a given thread_id. Meant to be killed mid-run by
scripts/checkpoint_kill_demo.py -- see checkpoint_resume.py for the other half."""

import sys

from dotenv import load_dotenv

load_dotenv()

from agents.leveling_graph import DEFAULT_CHECKPOINT_DB, build_graph, get_checkpointer

thread_id = sys.argv[1]

checkpointer = get_checkpointer(DEFAULT_CHECKPOINT_DB)
app = build_graph().compile(checkpointer=checkpointer)

initial_state = {
    "job_description": """Physical Design Engineer. Owns place-and-route and timing closure
    for a subsystem within our next-generation SoC, working independently across the full
    development cycle from RTL handoff through tapeout. Sets their own methodology for the
    hardest blocks and is consulted by the architecture team on physical-implementability
    tradeoffs rather than being directed.""",
    "source_org_context": None,
    "low_confidence_threshold": 0.65,
    "high_confidence_threshold": 0.75,
    "parsed": False,
    "decision": None,
}
config = {"configurable": {"thread_id": thread_id}}
result = app.invoke(initial_state, config, durability="sync")
print("COMPLETED (should not print if the controller killed this process in time)")
print(result["decision"])
