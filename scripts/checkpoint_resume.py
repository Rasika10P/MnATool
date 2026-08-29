"""Resumes a leveling run from the last checkpoint for a given thread_id, in a fresh
process. Passing None as the input (rather than a fresh initial_state) is what tells
LangGraph to continue from wherever this thread_id last left off instead of starting over."""

import sys

from dotenv import load_dotenv

load_dotenv()

from agents.leveling_graph import DEFAULT_CHECKPOINT_DB, build_graph, get_checkpointer

thread_id = sys.argv[1]

checkpointer = get_checkpointer(DEFAULT_CHECKPOINT_DB)
app = build_graph().compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": thread_id}}

result = app.invoke(None, config, durability="sync")
print("RESUMED RUN COMPLETED")
print(result["decision"])
