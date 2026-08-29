"""Two adversarial leveling cases, designed to fail if the agent leans on title or tenure
instead of described scope:

1. An inflated title at a small company -- section 6 expected drift plus rule 6 (title is
   evidence, not input). Scope described is textbook M3; title says VP.
2. A deep-but-narrow senior IC -- rule 3 (deep-but-narrow caps at L5, cannot reach L6) and
   rule 4 (no L7+ without external recognition), stacked against strong scope/autonomy/
   influence language that would otherwise read L6.
"""

import argparse

from dotenv import load_dotenv

load_dotenv()

from agents.leveling import level_role
from agents.schemas import SourceOrgContext
from scripts._cli_common import add_cache_mode_arg, dry_run_report, run_with_budget_guard

CASES = [
    (
        "Adversarial 1 -- inflated title, thin scope (\"VP of Engineering\")",
        """VP of Engineering. Leads all engineering at a 40-person company, with 8 direct
        reports, all individual contributors across firmware and hardware bring-up. Sets
        sprint priorities and reviews team output on the company's single embedded product
        line. Represents engineering in weekly leadership standups but does not set overall
        company technical strategy -- that is set jointly by the two co-founders. No budget
        authority beyond headcount requisitions; equipment and vendor spend is approved by
        the CFO. Problems are scoped within the current product's firmware and integration
        issues; the broader roadmap is set elsewhere.""",
        SourceOrgContext(
            source_headcount=40,
            source_stage="growth",
            source_type="whole company",
            org_depth=1,
        ),
    ),
    (
        "Adversarial 2 -- deep-but-narrow senior IC",
        """Individual contributor with fifteen years focused exclusively on static timing
        analysis and timing closure methodology for signoff. Regarded internally as the final
        authority on timing closure across the company -- every product team escalates
        unresolved timing issues here, and this person sets the internal timing closure
        methodology and sign-off criteria for all tapeouts company-wide, influencing the
        timing budget across every business unit. Works with full autonomy, setting direction
        for the timing closure domain without oversight. Has never worked outside static
        timing analysis -- no experience in place & route, verification, or any adjacent
        discipline. No patents, publications, standards body participation, or external
        industry recognition.""",
        None,
    ),
]


def _run(cases):
    for label, description, context in cases:
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        decision = level_role(description, source_org_context=context)
        print(decision.model_dump_json(indent=2))


def main(limit: int, budget: float, dry_run: bool, cache_mode: str):
    cases = CASES[:limit]
    if dry_run:
        dry_run_report(cases)
        return
    run_with_budget_guard(budget, lambda: _run(cases), cache_mode=cache_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3, help="max number of cases to run (default: 3)")
    parser.add_argument("--budget", type=float, default=2.0, help="run cost cap in USD (default: 2.0)")
    parser.add_argument("--dry-run", action="store_true", help="report projected API calls without making them")
    add_cache_mode_arg(parser)
    args = parser.parse_args()
    main(args.limit, args.budget, args.dry_run, args.cache_mode)
