"""Runs the same five job descriptions from scripts/level_five_jobs_via_graph.py through
extract_scope_profile on both providers -- Nebius (the actual parse-node routing) and Claude
(for comparison only; Claude never runs this in production) -- to see how similar the
extraction is across providers. Extraction is a narrower task than leveling judgment, so
provider agreement here doesn't have to (and isn't expected to) resemble the leveling
divergence in scripts/level_five_jobs_nebius_vs_claude.py.
"""

import argparse

from dotenv import load_dotenv

load_dotenv()

from agents.instrumented_model import set_cache_mode
from agents.model_router import get_model
from agents.schemas import ScopeProfile
from agents.scope_extraction import extract_scope_profile, would_hit_cache
from scripts._cli_common import add_cache_mode_arg, print_session_summary
from scripts.level_five_jobs_via_graph import CASES


def _dry_run_report(cases, nebius, claude) -> None:
    hits = 0
    total = 0
    for label, description, _context, _baseline in cases:
        for provider_label, model in (("nebius", nebius), ("claude", claude)):
            total += 1
            hit = would_hit_cache(description, model=model)
            hits += hit
            marker = "[cached]" if hit else "[would call API]"
            print(f"  {marker:18} {label} ({provider_label})")
    print(f"\nDRY RUN: {total} items -> {hits} cache hits, {total - hits} new API calls. No calls made.")


def _format_finding(finding) -> str:
    if not finding.stated:
        return "(not mentioned)"
    return f"explicit: {finding.value!r}"


def _print_profile(label: str, profile: ScopeProfile) -> None:
    print(f"  {label}:")
    print(f"    reports_to:        {_format_finding(profile.reports_to)}")
    print(f"    span_of_control:   {_format_finding(profile.span_of_control)}")
    print(f"    budget_authority:  {_format_finding(profile.budget_authority)}")
    print(f"    decision_scope:    {profile.decision_scope}")
    print(f"    ownership_scope:   {profile.ownership_scope}")


def _extract_or_report_failure(label: str, description: str, model, provider_label: str) -> ScopeProfile | None:
    # A malformed-structured-output glitch on one case/provider (docs/error_handling_backlog.md
    # entries 1-4) shouldn't take down the other 4 cases' comparison -- this is a demo script
    # gathering evidence, not item 8's real per-task catch/retry/degrade design, so "report and
    # move on" is enough here.
    try:
        return extract_scope_profile(description, model=model)
    except Exception as e:
        print(f"  {provider_label} FAILED on {label}: {type(e).__name__}: {e}")
        return None


def _run(cases, budget: float, cache_mode: str):
    from agents.cost_logging import reset_session_stats
    from agents.spend_guard import BudgetExceededError, reset_default_budget

    set_cache_mode(cache_mode)
    reset_session_stats()
    reset_default_budget(budget)
    nebius = get_model("volume")
    claude = get_model("judgment")

    try:
        for label, description, _context, _baseline in cases:
            print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
            nebius_profile = _extract_or_report_failure(label, description, nebius, "Nebius")
            claude_profile = _extract_or_report_failure(label, description, claude, "Claude")
            if nebius_profile is not None:
                _print_profile("Nebius", nebius_profile)
            if claude_profile is not None:
                _print_profile("Claude", claude_profile)
    except BudgetExceededError as e:
        print(f"\n{'=' * 78}\nRUN ABORTED -- BUDGET EXCEEDED (cap: ${budget:.2f})\n{'=' * 78}")
        print(str(e))
    finally:
        print_session_summary()


def main(limit: int, budget: float, dry_run: bool, cache_mode: str):
    cases = CASES[:limit]
    if dry_run:
        _dry_run_report(cases, get_model("volume"), get_model("judgment"))
        return
    _run(cases, budget, cache_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3, help="max number of cases to run (default: 3)")
    parser.add_argument("--budget", type=float, default=2.0, help="run cost cap in USD (default: 2.0)")
    parser.add_argument("--dry-run", action="store_true", help="report projected API calls without making them")
    add_cache_mode_arg(parser)
    args = parser.parse_args()
    main(args.limit, args.budget, args.dry_run, args.cache_mode)
