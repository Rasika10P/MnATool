"""Runs the same five job descriptions from earlier this session through the LangGraph
version and diffs the structural fields (track, assigned_level, factor5_variant_applied,
escalate) against what the plain-function version produced live, recorded below.

Free-text fields (reasoning, evidence, alternative_reasoning) are expected to differ in
wording between runs -- Claude isn't seeded (claude-sonnet-5 rejects an explicit
temperature), so prose varies even through byte-identical code. What must not vary is the
decision itself. tests/test_leveling_graph.py is the rigorous, deterministic version of
this check (fixed fake model, byte-for-byte dict comparison); this script is the live
qualitative confirmation on the real five cases.
"""

import argparse

from dotenv import load_dotenv

load_dotenv()

from agents.leveling_graph import run_leveling
from agents.schemas import SourceOrgContext
from scripts._cli_common import add_cache_mode_arg, dry_run_report, run_with_budget_guard

# (label, job_description, source_org_context, previously recorded structural result)
CASES = [
    (
        "1. Internal IC -- Physical Design",
        """Physical Design Engineer. Owns place-and-route and timing closure for a subsystem
        within our next-generation SoC, working independently across the full development
        cycle from RTL handoff through tapeout. Sets their own methodology for the hardest
        blocks and is consulted by the architecture team on physical-implementability
        tradeoffs rather than being directed. Informally mentors two junior engineers.
        Influences physical design and timing decisions within the Physical Design function;
        not yet driving strategy beyond it. Six years of related experience.""",
        None,
        {"track": "IC", "assigned_level": "L4", "factor5_variant_applied": "5a", "escalate": False},
    ),
    (
        "2. Internal manager -- Embedded Firmware",
        """Engineering Manager, Firmware. Leads a team of six embedded software engineers
        building device driver and RTOS integration work for our sensor platform. Reviews
        team output, sets sprint priorities, and represents the team in cross-functional
        program reviews. Owns the team's near-term roadmap and works with product management
        on scope tradeoffs. No budget authority beyond headcount requisitions. All six direct
        reports are individual contributors.""",
        None,
        {"track": "MGR", "assigned_level": "M3", "factor5_variant_applied": "5a", "escalate": False},
    ),
    (
        "3. Acquired -- \"Director of Analog Design\"",
        """Director of Analog Design. Owned the RF transceiver block design end-to-end across
        two tapeouts for our flagship product, from architecture through characterization.
        Worked independently with minimal oversight from the VP of Engineering, who reviewed
        outcomes rather than approach. Consulted by the layout and test teams on
        design-for-test tradeoffs. Represents analog design in customer technical reviews.
        Relies heavily on a shared central CAD and methodology team for tooling and flow
        support. No direct reports.""",
        SourceOrgContext(
            source_headcount=45, source_stage="growth", source_type="whole company",
            org_depth=3, platform_dependency="high",
        ),
        {"track": "IC", "assigned_level": "L4", "factor5_variant_applied": "5a", "escalate": True},
    ),
    (
        "4. Adversarial -- inflated title (\"VP of Engineering\")",
        """VP of Engineering. Leads all engineering at a 40-person company, with 8 direct
        reports, all individual contributors across firmware and hardware bring-up. Sets
        sprint priorities and reviews team output on the company's single embedded product
        line. Represents engineering in weekly leadership standups but does not set overall
        company technical strategy -- that is set jointly by the two co-founders. No budget
        authority beyond headcount requisitions; equipment and vendor spend is approved by
        the CFO. Problems are scoped within the current product's firmware and integration
        issues; the broader roadmap is set elsewhere.""",
        SourceOrgContext(source_headcount=40, source_stage="growth", source_type="whole company", org_depth=1),
        {"track": "MGR", "assigned_level": "M3", "factor5_variant_applied": "5a", "escalate": False},
    ),
    (
        "5. Adversarial -- deep-but-narrow senior IC",
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
        {"track": "IC", "assigned_level": "L5", "factor5_variant_applied": "5a", "escalate": False},
    ),
]

STRUCTURAL_FIELDS = ["track", "assigned_level", "factor5_variant_applied", "escalate"]


def _run(cases):
    all_match = True
    for i, (label, description, context, previous) in enumerate(cases):
        decision = run_leveling(description, source_org_context=context, thread_id=f"diff-check-{i}")
        current = {field: decision[field] for field in STRUCTURAL_FIELDS}
        match = current == previous
        all_match &= match
        status = "MATCH" if match else "DIFFERS"
        print(f"\n{label}\n  status: {status}")
        print(f"  previous: {previous}")
        print(f"  now:      {current}")

    print(f"\n{'=' * 70}\nALL STRUCTURAL FIELDS MATCH: {all_match}\n{'=' * 70}")


def main(limit: int, budget: float, dry_run: bool, cache_mode: str):
    cases = CASES[:limit]
    if dry_run:
        dry_run_report([(label, description, context) for label, description, context, _ in cases])
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
