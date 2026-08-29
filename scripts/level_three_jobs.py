"""Runs the leveling agent on three job descriptions and prints the validated decisions.
Exercises: rule 1 (experience isn't a factor), the manager span & budget anchor, and
section 6 source-org calibration combined with rule 6 (title is evidence, not input).
"""

import argparse

from dotenv import load_dotenv

load_dotenv()

from agents.leveling import level_role
from agents.schemas import SourceOrgContext
from scripts._cli_common import dry_run_report, run_with_budget_guard

JOBS = [
    (
        "Internal IC role -- Physical Design",
        """Physical Design Engineer. Owns place-and-route and timing closure for a subsystem
        within our next-generation SoC, working independently across the full development
        cycle from RTL handoff through tapeout. Sets their own methodology for the hardest
        blocks and is consulted by the architecture team on physical-implementability
        tradeoffs rather than being directed. Informally mentors two junior engineers.
        Influences physical design and timing decisions within the Physical Design function;
        not yet driving strategy beyond it. Six years of related experience.""",
        None,
    ),
    (
        "Internal manager role -- Embedded Firmware",
        """Engineering Manager, Firmware. Leads a team of six embedded software engineers
        building device driver and RTOS integration work for our sensor platform. Reviews
        team output, sets sprint priorities, and represents the team in cross-functional
        program reviews. Owns the team's near-term roadmap and works with product management
        on scope tradeoffs. No budget authority beyond headcount requisitions. All six direct
        reports are individual contributors.""",
        None,
    ),
    (
        "Acquired-company role -- \"Director of Analog Design\"",
        """Director of Analog Design. Owned the RF transceiver block design end-to-end across
        two tapeouts for our flagship product, from architecture through characterization.
        Worked independently with minimal oversight from the VP of Engineering, who reviewed
        outcomes rather than approach. Consulted by the layout and test teams on
        design-for-test tradeoffs. Represents analog design in customer technical reviews.
        Relies heavily on a shared central CAD and methodology team for tooling and flow
        support. No direct reports.""",
        SourceOrgContext(
            source_headcount=45,
            source_stage="growth",
            source_type="whole company",
            org_depth=3,
            platform_dependency="high",
        ),
    ),
]


def _run(jobs):
    for label, description, context in jobs:
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        decision = level_role(description, source_org_context=context)
        print(decision.model_dump_json(indent=2))


def main(limit: int, budget: float, dry_run: bool):
    jobs = JOBS[:limit]
    if dry_run:
        dry_run_report(jobs)
        return
    run_with_budget_guard(budget, lambda: _run(jobs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3, help="max number of jobs to run (default: 3)")
    parser.add_argument("--budget", type=float, default=2.0, help="run cost cap in USD (default: 2.0)")
    parser.add_argument("--dry-run", action="store_true", help="report projected API calls without making them")
    args = parser.parse_args()
    main(args.limit, args.budget, args.dry_run)
