"""Negotiation piece 2 demo: runs NYX-011 (Distinguished MTS -- Microarchitecture, "company-
wide final authority... has never published, patented, or presented outside Nyx") through
the crosswalk agent (agents.leveling.level_role, standing in for the crosswalk-agent
participant per level_framework.md section 7) to get a proposed Meridian mapping, then hands
that mapping to the advocate (agents.advocate.contest_mapping) to see whether it contests.

Exercises the case the advocate is for: strong internal authority, explicitly no external
recognition -- Meridian's rule 4 (external recognition required for L7+) likely caps this
below what the scope otherwise reads as, and the advocate has to decide whether Nyx's own
document (which never requires external recognition for Distinguished MTS -- see section 4,
"often, but not always, paired with recognition outside Nyx") gives it grounds to contest.
"""

import argparse

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from agents.advocate import contest_mapping
from agents.leveling import level_role
from agents.schemas import SourceOrgContext
from scripts._cli_common import run_with_budget_guard

CENSUS_PATH = "data/parquet/nyx_census.xlsx"
ACQUISITION_CONTEXT_PATH = "data/parquet/acquisition_context.parquet"
EMPLOYEE_ID = "NYX-011"


def load_employee() -> dict:
    census = pd.read_excel(CENSUS_PATH)
    row = census.loc[census["Emp ID"] == EMPLOYEE_ID].iloc[0]
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


def run() -> None:
    employee = load_employee()

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=1.0, help="run cost cap in USD (default: 1.0)")
    args = parser.parse_args()
    run_with_budget_guard(args.budget, run)
