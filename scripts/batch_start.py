"""Starts a real fan-out run over the Nyx employees under a given thread_id. Meant to be
killed mid-batch by scripts/batch_kill_demo.py -- see batch_resume.py for the other half."""

import argparse

from dotenv import load_dotenv

load_dotenv()

from agents.leveling_batch_graph import DEFAULT_BATCH_CHECKPOINT_DB, build_batch_graph, get_checkpointer
from agents.spend_guard import BudgetExceededError, reset_default_budget
from scripts._cli_common import dry_run_report, print_session_summary
from scripts.level_nyx_batch import load_nyx_employees

parser = argparse.ArgumentParser()
parser.add_argument("thread_id")
parser.add_argument("--limit", type=int, default=3, help="max number of Nyx employees to run (default: 3)")
parser.add_argument("--budget", type=float, default=2.0, help="run cost cap in USD (default: 2.0)")
parser.add_argument("--dry-run", action="store_true", help="report projected API calls without making them")
args = parser.parse_args()
thread_id = args.thread_id

employees = load_nyx_employees()[: args.limit]

if args.dry_run:
    items = [(emp["employee_id"], emp["job_description"], emp["source_org_context"]) for emp in employees]
    dry_run_report(items)
    raise SystemExit(0)

reset_default_budget(args.budget)
checkpointer = get_checkpointer(DEFAULT_BATCH_CHECKPOINT_DB)
app = build_batch_graph().compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": thread_id}}

try:
    result = app.invoke({"employees": employees, "decisions": []}, config, durability="sync")
    print("BATCH COMPLETED (should not print if the controller killed this process in time)")
    print(f"decisions: {len(result['decisions'])}")
except BudgetExceededError as e:
    print(f"\n{'=' * 70}\nRUN ABORTED -- BUDGET EXCEEDED (cap: ${args.budget:.2f})\n{'=' * 70}")
    print(str(e))
finally:
    print_session_summary()
