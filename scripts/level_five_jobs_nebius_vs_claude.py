"""Runs the same five job descriptions from scripts/level_five_jobs_via_graph.py through
Nebius (get_model("volume")), single pass, no escalation to Claude -- the point here is to
see where the two providers agree and diverge, so silently swapping a low-confidence Nebius
result for a Claude one (as agents.leveling.level_role_routed does in production) would
defeat the comparison. Claude's side of the table is the structural decisions already
recorded live in level_five_jobs_via_graph.py's CASES, not re-run here.
"""

import argparse

from dotenv import load_dotenv

load_dotenv()

from agents.leveling import level_role
from agents.model_router import get_model
from agents.schemas import LevelingDecision
from scripts._cli_common import dry_run_report, print_session_summary
from scripts.level_five_jobs_via_graph import CASES

# CASES entries are (label, job_description, source_org_context, previous_claude_structural).
# previous_claude_structural only has track/assigned_level/factor5_variant_applied/escalate --
# it doesn't carry confidence or factor_ratings (those weren't captured at the time), so the
# Claude column below shows what's available and marks the rest "n/a (not captured)".

CLAUDE_CONFIDENCE_NOTE = "n/a (not captured when baseline was recorded)"


def _format_factor_ratings(decision: LevelingDecision) -> str:
    return "; ".join(f"{r.factor}={r.level_indicated}" for r in decision.factor_ratings)


def _run(cases, budget: float):
    from agents.cost_logging import reset_session_stats
    from agents.spend_guard import BudgetExceededError, reset_default_budget

    reset_session_stats()
    reset_default_budget(budget)
    nebius = get_model("volume")

    try:
        for label, description, context, claude_baseline in cases:
            print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")

            nebius_decision = level_role(description, source_org_context=context, model=nebius)

            print("  Claude (recorded baseline):")
            print(f"    assigned_level:     {claude_baseline['assigned_level']} (track {claude_baseline['track']})")
            print(f"    factor5_variant:    {claude_baseline['factor5_variant_applied']}")
            print(f"    escalate:           {claude_baseline['escalate']}")
            print(f"    confidence:         {CLAUDE_CONFIDENCE_NOTE}")
            print(f"    factor_ratings:     {CLAUDE_CONFIDENCE_NOTE}")

            print("  Nebius (this run):")
            print(f"    assigned_level:     {nebius_decision.assigned_level} (track {nebius_decision.track})")
            print(f"    factor5_variant:    {nebius_decision.factor5_variant_applied}")
            print(f"    confidence:         {nebius_decision.confidence:.2f}")
            print(f"    escalation_factor:  {nebius_decision.escalation_factor}")
            print(f"    escalate:           {nebius_decision.escalate}")
            print(f"    factor_ratings:     {_format_factor_ratings(nebius_decision)}")

            level_match = nebius_decision.assigned_level == claude_baseline["assigned_level"]
            print(f"  --> assigned_level {'MATCHES' if level_match else 'DIVERGES'} the Claude baseline")
    except BudgetExceededError as e:
        print(f"\n{'=' * 78}\nRUN ABORTED -- BUDGET EXCEEDED (cap: ${budget:.2f})\n{'=' * 78}")
        print(str(e))
    finally:
        print_session_summary()


def main(limit: int, budget: float, dry_run: bool):
    cases = CASES[:limit]
    if dry_run:
        dry_run_report(
            [(label, description, context) for label, description, context, _ in cases],
            model=get_model("volume"),
        )
        return
    _run(cases, budget)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3, help="max number of cases to run (default: 3)")
    parser.add_argument("--budget", type=float, default=2.0, help="run cost cap in USD (default: 2.0)")
    parser.add_argument("--dry-run", action="store_true", help="report projected API calls without making them")
    args = parser.parse_args()
    main(args.limit, args.budget, args.dry_run)
