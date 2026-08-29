"""Resumes a batch fan-out run from the last checkpoint for a given thread_id, in a fresh
process. Passing None as input tells LangGraph to continue from wherever this thread_id
left off, re-scheduling only the Send tasks that never completed."""

import sys

from dotenv import load_dotenv

load_dotenv()

from agents.leveling_batch_graph import DEFAULT_BATCH_CHECKPOINT_DB, build_batch_graph, get_checkpointer

thread_id = sys.argv[1]

checkpointer = get_checkpointer(DEFAULT_BATCH_CHECKPOINT_DB)
app = build_batch_graph().compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": thread_id}}

result = app.invoke(None, config, durability="sync")
print("RESUMED BATCH COMPLETED")
print(f"decisions: {len(result['decisions'])}")
print(sorted(d["employee_id"] for d in result["decisions"]))
