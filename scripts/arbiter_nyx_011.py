"""Negotiation piece 3 demo: extends scripts/advocate_nyx_011.py one step further -- runs
one Nyx employee through the crosswalk agent, then the advocate, then (when the advocate
contests) the arbiter, and prints all three outputs.

Two cases exercised so far, both against the same Digital Design / Microarchitecture
sub-family so the crosswalk anchors line up for comparison:

- NYX-011 (default) -- Distinguished MTS, company-wide authority, no external recognition.
  The crosswalk agent's own reasoning already names rule 4 (external recognition required
  for L7+) as the sole thing capping an otherwise L7-caliber scope at L6, and the advocate
  draws on Nyx's Distinguished MTS anchor for company-wide domain authority. Real merit,
  blocked by a named rule -- the shape section 7 describes for red-circling.
- NYX-009 -- Principal MTS title carried since a reorg, but "day to day the work looks like
  ordinary core-team microarchitecture spec work..., shared with two peers at the Senior MTS
  level doing comparable scope" -- titled above the described scope. Rule 6 (title is
  evidence, not input) should govern: the crosswalk agent levels from the described scope,
  not the Principal MTS title, so there's little for the advocate's own document-scoped
  argument to work with -- expected to be a weak contest and a plain upheld.
"""

import argparse

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from agents.advocate import contest_mapping
from agents.arbiter import rule as arbiter_rule
from agents.leveling import level_role
from agents.schemas import SourceOrgContext
from scripts._cli_common import run_with_budget_guard

CENSUS_PATH = "data/parquet/nyx_census.xlsx"
ACQUISITION_CONTEXT_PATH = "data/parquet/acquisition_context.parquet"


def load_employee(employee_id: str) -> dict:
    census = pd.read_excel(CENSUS_PATH)
    row = census.loc[census["Emp ID"] == employee_id].iloc[0]
    ctx_row = pd.read_parquet(ACQUISITION_CONTEXT_PATH).iloc[0]

    source_org_context = SourceOrgContext(
        source_headcount=int(ctx_row.source_headcount),
        source_stage=ctx_row.source_stage,
        source_type=ctx_row.source_type,
        org_depth=int(ctx_row.org_depth),
        platform_dependency=ctx_row.platform_dependency,
    )
    return {
        "employee_id": row["Emp ID"],
        "nyx_level": row["Job Title"].split(" - ")[0],
        "role_summary": row["Role Summary"],
        "job_description": f"Job title: {row['Job Title']}. Department: {row['Dept']}. {row['Role Summary']}",
        "source_org_context": source_org_context,
    }


def run(employee_id: str) -> None:
    employee = load_employee(employee_id)

    print(f"{'=' * 70}\n{employee['employee_id']} -- crosswalk mapping\n{'=' * 70}")
    crosswalk_decision = level_role(
        employee["job_description"], source_org_context=employee["source_org_context"]
    )
    print(crosswalk_decision.model_dump_json(indent=2))

    print(f"\n{'=' * 70}\n{employee['employee_id']} -- advocate\n{'=' * 70}")
    advocate_output = contest_mapping(
        employee["role_summary"],
        employee["nyx_level"],
        crosswalk_decision.assigned_level,
    )
    print(advocate_output.model_dump_json(indent=2))

    if not advocate_output.contests:
        print("\nAdvocate declined to contest -- nothing for the arbiter to rule on.")
        return

    print(f"\n{'=' * 70}\n{employee['employee_id']} -- arbiter\n{'=' * 70}")
    ruling = arbiter_rule(crosswalk_decision, advocate_output.as_crosswalk_argument())
    print(ruling.model_dump_json(indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--employee-id", default="NYX-011", help="Nyx census Emp ID (default: NYX-011)")
    parser.add_argument("--budget", type=float, default=1.0, help="run cost cap in USD (default: 1.0)")
    args = parser.parse_args()
    run_with_budget_guard(args.budget, lambda: run(args.employee_id))
