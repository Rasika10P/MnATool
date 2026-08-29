"""Build order item 6 demo: crosswalks a small population of real Nyx employees, then runs
the cost/retention/synthesis subgraph (agents/modeling_graph.py) on the result end to end.

Spans all three of the Nyx census's currencies (USD/INR/EUR) on purpose -- this used to be
restricted to USD-only rows because the generator's non-USD "Base" figures weren't real
currency conversions (see learnings.md); that's fixed now (data/generate.py's
build_nyx_census uses tools.currency.convert_currency against dated FX), and
agents/cost_model.py / agents/retention_model.py now convert every employee's figures to a
single reporting_currency before summing population totals, so a mixed-currency population
like this one rolls up correctly instead of silently mixing units.

Crosswalk only (agents.leveling.level_role), not the full negotiation subgraph -- for cost/
retention modeling, what matters is each employee's final negotiated level, and running the
full advocate/arbiter/equity-gate loop for every employee is a separate concern already
proven in build order item 4 (agents/negotiation_graph.py). Treating the crosswalk agent's
top-line level as final is a reasonable stand-in for a demo population where most cases are
uncontested anyway.
"""

import argparse
import json

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from agents.leveling import level_role
from agents.modeling_graph import run_modeling
from agents.schemas import SourceOrgContext
from scripts._cli_common import add_cache_mode_arg, run_with_budget_guard

CENSUS_PATH = "data/parquet/nyx_census.xlsx"
ACQUISITION_CONTEXT_PATH = "data/parquet/acquisition_context.parquet"
AS_OF_DATE = "2026-08-01"  # declared deal reference date

EMPLOYEE_IDS = ["NYX-001", "NYX-002", "NYX-003"]  # MTS I/II/Sr MTS, RTL Design -- USD/INR/EUR respectively

SUB_FAMILY_TO_JOB_PREFIX = {"RTL Design": "DD-RTL", "Microarchitecture": "DD-UARCH", "Analog Design": "ANA-AD", "RF": "ANA-RF"}
FAMILY_TO_MERIDIAN_FAMILY_GROUP = {"Digital Design": "engineering", "Analog & Mixed-Signal": "engineering"}
LOCATION_TO_GEO_CODE = {"San Jose, CA": "US-SJC", "San Jose": "US-SJC", "Bangalore": "IN-BLR", "Eindhoven": "EU-EIN"}


def _sub_family_from_title(title: str) -> str:
    # e.g. "MTS I - RTL Design" -> "RTL Design"; "Distinguished MTS, RF" -> "RF"
    tail = title.split(" - ")[-1] if " - " in title else title.split(", ")[-1]
    return tail.strip()


def crosswalk_population(employee_ids: list[str]) -> list[dict]:
    census = pd.read_excel(CENSUS_PATH)
    ctx_row = pd.read_parquet(ACQUISITION_CONTEXT_PATH).iloc[0]
    source_org_context = SourceOrgContext(
        source_headcount=int(ctx_row.source_headcount),
        source_stage=ctx_row.source_stage,
        source_type=ctx_row.source_type,
        org_depth=int(ctx_row.org_depth),
        platform_dependency=ctx_row.platform_dependency,
    )

    population = []
    for employee_id in employee_ids:
        row = census.loc[census["Emp ID"] == employee_id].iloc[0]
        job_description = f"Job title: {row['Job Title']}. Department: {row['Dept']}. {row['Role Summary']}"
        decision = level_role(job_description, source_org_context=source_org_context)

        sub_family = _sub_family_from_title(row["Job Title"])
        job_id = f"{SUB_FAMILY_TO_JOB_PREFIX[sub_family]}-{decision.assigned_level}"
        family_group = FAMILY_TO_MERIDIAN_FAMILY_GROUP[row["Dept"]]

        print(
            f"{employee_id} ({row['Job Title']}) -> crosswalked to {decision.assigned_level} "
            f"({decision.governing_rule}) -> job_id {job_id}"
        )

        population.append(
            {
                "employee_id": employee_id,
                "job_id": job_id,
                "family": row["Dept"],
                "family_group": family_group,
                "level_code": decision.assigned_level,
                "geo_code": LOCATION_TO_GEO_CODE[row["Location"]],
                "currency": row["Curr"],
                "current_pay": float(row["Base"]),
                "unvested_equity_value": float(row["Unvested Options"]) if pd.notna(row["Unvested Options"]) else None,
                "role_summary": row["Role Summary"],
            }
        )
    return population


def _print_rollup_check(local_field: str, reporting_field: str, total_field: str, assessment: dict) -> None:
    """Visible confirmation that the population total is the sum of each employee's
    already-converted figure, not a naive sum across mismatched currencies."""
    naive_sum = sum(e[local_field] for e in assessment["employees"])
    correct_sum = sum(e[reporting_field] for e in assessment["employees"])
    print(
        f"\nRollup check ({total_field}): naive mixed-currency sum = {naive_sum:,.2f} (meaningless -- "
        f"mixes {assessment['reporting_currency']} with raw INR/EUR figures); "
        f"correct total = {assessment[total_field]:,.2f} {assessment['reporting_currency']} "
        f"(matches sum of converted figures: {correct_sum == assessment[total_field]})"
    )


def run() -> None:
    population = crosswalk_population(EMPLOYEE_IDS)

    print(f"\n{'=' * 70}\nCost/retention/synthesis, {len(population)}-employee population\n{'=' * 70}")
    result = run_modeling(population, as_of_date=AS_OF_DATE, thread_id="modeling-demo")

    print(f"\n{'=' * 70}\nCOST ASSESSMENT\n{'=' * 70}")
    print(json.dumps(result["cost_assessment"], indent=2))
    _print_rollup_check("cost_gap", "cost_gap_reporting_currency", "total_day_one_cost", result["cost_assessment"])

    print(f"\n{'=' * 70}\nRETENTION ASSESSMENT\n{'=' * 70}")
    print(json.dumps(result["retention_assessment"], indent=2))
    _print_rollup_check(
        "retention_award", "retention_award_reporting_currency", "total_award_day_one", result["retention_assessment"]
    )

    print(f"\n{'=' * 70}\nSYNTHESIS\n{'=' * 70}")
    print(json.dumps(result["synthesis"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=2.0, help="run cost cap in USD (default: 2.0)")
    add_cache_mode_arg(parser)
    args = parser.parse_args()
    run_with_budget_guard(args.budget, run, cache_mode=args.cache_mode)
