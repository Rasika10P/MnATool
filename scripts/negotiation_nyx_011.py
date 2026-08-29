"""Negotiation piece 5 demo: runs the full advocate/arbiter/equity-gate subgraph
(agents/negotiation_graph.py) end to end on one Nyx employee (default NYX-011) and prints
the full transcript -- crosswalk mapping, every round's argument/ruling/gate result, and the
final verdict.

candidate_salary limitation, worth being explicit about: the equity gate needs a dollar
figure for "what would this person be paid if revised" (agents/equity_gate.py -- caller-
supplied on purpose, since no pricing agent exists yet to compute it). But the *level* being
revised to isn't known until the arbiter actually rules, inside the graph -- there's no
clean way to pass a single fixed dollar figure that's correctly scaled for whatever level
the arbiter ends up choosing. This script anchors candidate_salary to the *advocate's*
proposed level's range midpoint in the candidate's geo (looked up once, upfront, via
tools.data_access.lookup_salary_structure) -- reasonable here because the advocate can only
coherently propose L7 for this employee (Nyx's Distinguished MTS anchor points there, and
agents/advocate.py has no other level to argue for), so an L7-anchored figure should stay
correctly scaled even if the arbiter's own final_level ends up matching it, which it almost
always will on a "revised" verdict. This is a real limitation of a single fixed
candidate_salary input, not a solved problem -- a proper fix would price the candidate
dynamically once the arbiter's final_level is known, which needs a real pricing agent
(build order: market pricing desk workflow, not built yet).
"""

import argparse
import json

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from agents.negotiation_graph import run_negotiation
from agents.schemas import SourceOrgContext
from scripts._cli_common import run_with_budget_guard
from tools.data_access import lookup_salary_structure

CENSUS_PATH = "data/parquet/nyx_census.xlsx"
ACQUISITION_CONTEXT_PATH = "data/parquet/acquisition_context.parquet"

# Nyx sub_family -> Meridian family_group/job sub-family, and Nyx city -> Meridian geo_code.
# Small, explicit maps rather than a lookup table -- this script only ever runs one family
# (Digital Design / Microarchitecture) and the handful of Nyx census locations.
NYX_FAMILY_TO_MERIDIAN = {"Digital Design": "engineering"}
NYX_CITY_TO_GEO_CODE = {"Bangalore": "IN-BLR", "San Jose": "US-SJC", "Eindhoven": "EU-EIN"}

# What level the advocate can coherently propose for this employee -- see module docstring
# on why this anchors candidate_salary. Update if this script is pointed at a different
# employee whose advocate argument would target a different level.
ANCHOR_LEVEL_FOR_CANDIDATE_SALARY = "L7"


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
    family_group = NYX_FAMILY_TO_MERIDIAN[row["Dept"]]
    geo_code = NYX_CITY_TO_GEO_CODE[row["Location"]]
    anchor_structure = lookup_salary_structure(family_group, ANCHOR_LEVEL_FOR_CANDIDATE_SALARY, geo_code)["structure"]

    return {
        "employee_id": row["Emp ID"],
        "nyx_level": row["Job Title"].split(" - ")[0],
        "role_summary": row["Role Summary"],
        "job_description": f"Job title: {row['Job Title']}. Department: {row['Dept']}. {row['Role Summary']}",
        "source_org_context": source_org_context,
        "family_group": family_group,
        "candidate_geo_code": geo_code,
        "candidate_salary": anchor_structure["range_mid"],
    }


def run(employee_id: str) -> None:
    employee = load_employee(employee_id)
    print(
        f"Candidate salary anchor: {ANCHOR_LEVEL_FOR_CANDIDATE_SALARY} range midpoint in "
        f"{employee['candidate_geo_code']} = {employee['candidate_salary']:,.2f} "
        f"(see script docstring on this limitation)\n"
    )

    result = run_negotiation(
        case_id=f"CASE-{employee['employee_id']}",
        employee_id=employee["employee_id"],
        role_summary=employee["role_summary"],
        nyx_level=employee["nyx_level"],
        job_description=employee["job_description"],
        family_group=employee["family_group"],
        candidate_geo_code=employee["candidate_geo_code"],
        candidate_salary=employee["candidate_salary"],
        source_org_context=employee["source_org_context"],
        thread_id=f"negotiation-{employee['employee_id']}",
    )

    print(f"{'=' * 70}\n{employee['employee_id']} -- crosswalk mapping\n{'=' * 70}")
    print(
        f"assigned_level: {result['crosswalk_decision']['assigned_level']}  "
        f"governing_rule: {result['crosswalk_decision']['governing_rule']}"
    )

    print(f"\n{'=' * 70}\n{employee['employee_id']} -- advocate\n{'=' * 70}")
    print(json.dumps(result["advocate_output"], indent=2))

    if not result["contested"]:
        print("\nAdvocate declined to contest -- negotiation ends here, no exception register entry.")
    else:
        for round_entry in result["rounds"]:
            print(f"\n{'=' * 70}\n{employee['employee_id']} -- arbiter, round {round_entry['round']}\n{'=' * 70}")
            print(json.dumps(round_entry["ruling"], indent=2))

            gate_entry = next((g for g in result["gate_checks"] if g["round"] == round_entry["round"]), None)
            if gate_entry is not None:
                print(f"\n{'-' * 70}\n{employee['employee_id']} -- equity gate, round {round_entry['round']}\n{'-' * 70}")
                print(json.dumps(gate_entry["result"], indent=2))

        print(f"\n{'=' * 70}\n{employee['employee_id']} -- FINAL VERDICT\n{'=' * 70}")
        print(f"verdict: {result['final_verdict']}   final_level: {result['final_level']}   rounds: {result['round_count']}")

        print(f"\n{'=' * 70}\n{employee['employee_id']} -- exception register entry\n{'=' * 70}")
        print(json.dumps(result["exception_register_entry"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--employee-id", default="NYX-011", help="Nyx census Emp ID (default: NYX-011)")
    parser.add_argument("--budget", type=float, default=2.0, help="run cost cap in USD (default: 2.0)")
    args = parser.parse_args()
    run_with_budget_guard(args.budget, lambda: run(args.employee_id))
