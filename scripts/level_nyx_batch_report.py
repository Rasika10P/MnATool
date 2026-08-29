"""Reports per-employee leveling results for the Nyx census fan-out: assigned level,
confidence, escalate flag, plus the level distribution and escalation count.

scripts/level_nyx_batch.py already proves the fan-out mechanics (Send parallelism vs. a
sequential loop, wall-clock comparison) -- this script doesn't repeat that (which would
re-level all 25 employees twice for no reason here) and only runs the fan-out once, to
report on decision quality against the Role Summary data now in the census.
"""

import argparse

from dotenv import load_dotenv

load_dotenv()

from agents.leveling_batch_graph import run_batch
from scripts._cli_common import dry_run_report, run_with_budget_guard
from scripts.level_nyx_batch import load_nyx_employees


def _run(employees):
    decisions = run_batch(employees, thread_id="nyx-batch-report")
    by_employee_id = {d["employee_id"]: d for d in decisions}
    titles = {emp["employee_id"]: emp["job_description"].split(".", 1)[0].removeprefix("Job title: ") for emp in employees}

    print(f"\n{'employee_id':12} {'title':30} {'level':6} {'conf':6} {'escalate'}")
    print("-" * 78)
    for employee_id in sorted(by_employee_id):
        d = by_employee_id[employee_id]
        print(f"{employee_id:12} {titles[employee_id]:30} {d['assigned_level']:6} {d['confidence']:<6.2f} {d['escalate']}")

    level_counts: dict[str, int] = {}
    for d in decisions:
        level_counts[d["assigned_level"]] = level_counts.get(d["assigned_level"], 0) + 1
    escalated = sum(1 for d in decisions if d["escalate"])
    avg_conf = sum(d["confidence"] for d in decisions) / len(decisions)

    print(f"\n{'=' * 78}")
    print(f"level distribution: {dict(sorted(level_counts.items()))}")
    print(f"escalated: {escalated}/{len(decisions)}   avg confidence: {avg_conf:.2f}")
    print(f"{'=' * 78}")


def main(limit: int, budget: float, dry_run: bool):
    employees = load_nyx_employees()[:limit]
    print(f"loaded {len(employees)} Nyx employees (limit={limit})")

    if dry_run:
        items = [(emp["employee_id"], emp["job_description"], emp["source_org_context"]) for emp in employees]
        dry_run_report(items)
        return

    run_with_budget_guard(budget, lambda: _run(employees))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3, help="max number of Nyx employees to run (default: 3)")
    parser.add_argument("--budget", type=float, default=2.0, help="run cost cap in USD (default: 2.0)")
    parser.add_argument("--dry-run", action="store_true", help="report projected API calls without making them")
    args = parser.parse_args()
    main(args.limit, args.budget, args.dry_run)
