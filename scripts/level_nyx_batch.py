"""Build order item 3: fan the leveling graph out over the 25-employee Nyx census with
Send, and compare wall-clock time against a sequential loop over the same 25 employees.

Data note: the Nyx census (data/parquet/nyx_census.xlsx) carries a Role Summary column --
2-3 sentences per employee generated once with Claude and committed (see NYX_ROSTER in
data/generate.py), deliberately inconsistent in quality like a real data-room export, and
titled against Nyx's own five-level ladder (docs/nyx_level_framework.md) rather than
Meridian's. Within each Nyx level, evidence is deliberately spread so some rows read toward
the lower plausible Meridian level and some toward the higher one -- the ambiguity the
crosswalk negotiation is built to resolve, not a defect in the data.
"""

import argparse
import time

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from agents.leveling import level_role
from agents.leveling_batch_graph import run_batch
from agents.schemas import SourceOrgContext
from scripts._cli_common import add_cache_mode_arg, dry_run_report, run_with_budget_guard

CENSUS_PATH = "data/parquet/nyx_census.xlsx"
ACQUISITION_CONTEXT_PATH = "data/parquet/acquisition_context.parquet"


def load_nyx_employees() -> list[dict]:
    census = pd.read_excel(CENSUS_PATH)
    ctx_row = pd.read_parquet(ACQUISITION_CONTEXT_PATH).iloc[0]

    # parent_headcount only applies to a carve-out (see SourceOrgContext's own field
    # description); Nyx is "whole company", so it's omitted rather than passed through.
    source_org_context = SourceOrgContext(
        source_headcount=int(ctx_row.source_headcount),
        source_stage=ctx_row.source_stage,
        source_type=ctx_row.source_type,
        org_depth=int(ctx_row.org_depth),
        platform_dependency=ctx_row.platform_dependency,
    ).model_dump(exclude_none=True)

    return [
        {
            "employee_id": row["Emp ID"],
            "job_description": f"Job title: {row['Job Title']}. Department: {row['Dept']}. {row['Role Summary']}",
            "source_org_context": source_org_context,
        }
        for _, row in census.iterrows()
    ]


def run_sequential(employees: list[dict]) -> tuple[list[dict], float]:
    t0 = time.time()
    decisions = []
    for emp in employees:
        context = SourceOrgContext(**emp["source_org_context"]) if emp["source_org_context"] else None
        decision = level_role(emp["job_description"], source_org_context=context)
        decisions.append({"employee_id": emp["employee_id"], **decision.model_dump()})
    return decisions, time.time() - t0


def summarize(label: str, decisions: list[dict], elapsed: float) -> None:
    escalated = sum(1 for d in decisions if d["escalate"])
    avg_conf = sum(d["confidence"] for d in decisions) / len(decisions)
    print(f"\n{label}")
    print(f"  wall clock: {elapsed:.1f}s for {len(decisions)} employees")
    print(f"  escalated: {escalated}/{len(decisions)}  avg confidence: {avg_conf:.2f}")
    levels = sorted({d["assigned_level"] for d in decisions})
    print(f"  levels assigned: {levels}")


def _run(employees):
    print("\n=== running SEQUENTIAL loop ===")
    seq_decisions, seq_elapsed = run_sequential(employees)
    summarize("SEQUENTIAL", seq_decisions, seq_elapsed)

    print("\n=== running FAN-OUT (Send) ===")
    t0 = time.time()
    batch_decisions = run_batch(employees, thread_id="nyx-batch-comparison")
    batch_elapsed = time.time() - t0
    summarize("FAN-OUT", batch_decisions, batch_elapsed)

    print(f"\n{'=' * 70}")
    print(f"sequential: {seq_elapsed:.1f}s   fan-out: {batch_elapsed:.1f}s   speedup: {seq_elapsed / batch_elapsed:.1f}x")
    print(f"both produced {len(seq_decisions)} and {len(batch_decisions)} decisions (expect {len(employees)}, {len(employees)})")
    print(f"{'=' * 70}")


def main(limit: int, budget: float, dry_run: bool, cache_mode: str):
    employees = load_nyx_employees()[:limit]
    print(f"loaded {len(employees)} Nyx employees (limit={limit})")

    if dry_run:
        # The fan-out pass re-levels the exact same prompts as the sequential pass, so it's
        # entirely cache hits once the sequential pass has run -- report the unique
        # population once rather than double-counting the same calls twice.
        items = [(emp["employee_id"], emp["job_description"], emp["source_org_context"]) for emp in employees]
        dry_run_report(items)
        print("(the fan-out pass re-uses the same prompts and would be cache hits after the sequential pass runs)")
        return

    run_with_budget_guard(budget, lambda: _run(employees), cache_mode=cache_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3, help="max number of Nyx employees to run (default: 3)")
    parser.add_argument("--budget", type=float, default=2.0, help="run cost cap in USD (default: 2.0)")
    parser.add_argument("--dry-run", action="store_true", help="report projected API calls without making them")
    add_cache_mode_arg(parser)
    args = parser.parse_args()
    main(args.limit, args.budget, args.dry_run, args.cache_mode)
