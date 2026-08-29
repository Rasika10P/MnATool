"""Shared CLI helpers for population-running scripts: --dry-run reporting, --budget wiring,
--cache-mode wiring, and the end-of-run session summary print. Kept in one place so the
pattern (and the abort message) is identical everywhere instead of a dozen slightly-different
copies.
"""

from agents.cost_logging import get_session_stats, reset_session_stats
from agents.instrumented_model import CACHE_MODE_FILL, VALID_CACHE_MODES, DemoModeCacheMissError, set_cache_mode
from agents.leveling import would_hit_cache
from agents.schemas import SourceOrgContext
from agents.spend_guard import BudgetExceededError, reset_default_budget


def add_cache_mode_arg(parser) -> None:
    """--cache-mode demo|live|fill, defaulting to fill -- the same default agents/
    instrumented_model.py itself falls back to when nothing sets a mode at all, so a script
    that never calls this (there shouldn't be any left) behaves identically to one that does
    and gets the default explicitly."""
    parser.add_argument(
        "--cache-mode",
        choices=VALID_CACHE_MODES,
        default=CACHE_MODE_FILL,
        help=(
            "demo (cache only, never calls the API -- a miss shows as an error, not a real "
            "call), live (bypass the cache entirely, always call, write results back), "
            f"fill (cache + call the API only on a miss -- warms the cache; default: {CACHE_MODE_FILL})"
        ),
    )


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


def run_with_budget_guard(cap_usd: float, fn, cache_mode: str = CACHE_MODE_FILL) -> None:
    """Applies cache_mode (fill by default, matching add_cache_mode_arg's own default),
    resets the session counter and budget for this run, then calls fn(). Aborts cleanly (a
    short message, no traceback) on a budget overrun or a demo-mode cache miss; always
    prints the session summary afterward, whether the run finished or was aborted."""
    set_cache_mode(cache_mode)
    reset_session_stats()
    reset_default_budget(cap_usd)
    try:
        fn()
    except DemoModeCacheMissError as e:
        print(f"\n{'=' * 70}\nRUN ABORTED -- DEMO MODE CACHE MISS\n{'=' * 70}")
        print(str(e))
    except BudgetExceededError as e:
        print(f"\n{'=' * 70}\nRUN ABORTED -- BUDGET EXCEEDED (cap: ${cap_usd:.2f})\n{'=' * 70}")
        print(str(e))
    finally:
        print_session_summary()
