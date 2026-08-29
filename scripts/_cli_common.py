"""Shared CLI helpers for population-running scripts: --dry-run reporting, --budget wiring,
and the end-of-run session summary print. Kept in one place so the pattern (and the abort
message) is identical everywhere instead of five slightly-different copies.
"""

from agents.cost_logging import get_session_stats, reset_session_stats
from agents.leveling import would_hit_cache
from agents.schemas import SourceOrgContext
from agents.spend_guard import BudgetExceededError, reset_default_budget


def dry_run_report(items: list[tuple[str, str, SourceOrgContext | dict | None]], model=None) -> None:
    """items: (label, job_description, source_org_context) triples. source_org_context may
    be a SourceOrgContext, a plain dict (as the batch scripts carry it), or None.

    `model` defaults to None, which checks the cache under get_model("judgment") (Claude) --
    pass an explicit model (e.g. get_model("volume")) so a Nebius-routed script's dry run
    reports against the cache it will actually hit, not Claude's."""
    hits = 0
    for label, job_description, context in items:
        if isinstance(context, dict):
            context = SourceOrgContext(**context)
        hit = would_hit_cache(job_description, context, model=model)
        hits += hit
        marker = "[cached]" if hit else "[would call API]"
        print(f"  {marker:18} {label}")
    misses = len(items) - hits
    print(f"\nDRY RUN: {len(items)} items -> {hits} cache hits, {misses} new API calls. No calls made.")


def print_session_summary() -> None:
    summary = get_session_stats().summary()
    print(f"\n{'=' * 70}\nSESSION SUMMARY")
    print(f"  calls made:  {summary['calls']}")
    print(f"  cache hits:  {summary['cache_hits']}")
    print(f"  retries:     {summary['retries']}")
    print(f"  cost by provider: {summary['cost_by_provider']}")
    print(f"  total cost:  ${summary['total_cost_usd']:.4f}")
    print(f"{'=' * 70}")


def run_with_budget_guard(cap_usd: float, fn) -> None:
    """Resets the session counter and budget for this run, then calls fn(). Aborts cleanly
    (a short message, no traceback) if the cap is hit partway through; always prints the
    session summary afterward, whether the run finished or was aborted."""
    reset_session_stats()
    reset_default_budget(cap_usd)
    try:
        fn()
    except BudgetExceededError as e:
        print(f"\n{'=' * 70}\nRUN ABORTED -- BUDGET EXCEEDED (cap: ${cap_usd:.2f})\n{'=' * 70}")
        print(str(e))
    finally:
        print_session_summary()
