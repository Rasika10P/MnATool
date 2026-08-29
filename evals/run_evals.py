"""Eval harness for the leveling agent (ASSIGNMENT.md build order item 9). Loads
evals/labeled_cases.jsonl, runs agents.leveling.level_role over every case, and writes
evals/results.md: exact-level match rate, within-one-level rate, escalation precision and
recall, and a per-case table (expected vs. assigned, with the governing rule cited).

Real leveling calls always go to Claude (agents.model_router's "judgment" tier -- CLAUDE.md's
model routing: leveling adjudication is Claude-only, Nebius was tried and dropped, see
learnings.md). --fake swaps in FixedFakeModel below instead: no API call, no cost, and
because it always returns the exact same decision, the resulting metrics are hand-verifiable
against labeled_cases.jsonl by eye -- that's the point of running it before the real one.

Cases with expected_level=None (not yet hand-labeled by a comp professional -- CLAUDE.md:
level mapping is a domain decision, not something this harness invents) are still run and
shown in the per-case table, but excluded from every metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agents.leveling import level_role
from agents.schemas import FactorRating, LevelingDecision, SourceOrgContext
from app.pipeline import load_level_titles
from evals.scoring import CaseResult, build_results_markdown, level_rank

CASES_PATH = Path(__file__).resolve().parent / "labeled_cases.jsonl"
RESULTS_PATH = Path(__file__).resolve().parent / "results.md"


class FixedFakeModel:
    """Always returns the same LevelingDecision (FIXED_LEVEL, confidence above the
    escalate-high threshold so escalate is always False), regardless of what's asked.
    Proves the harness's own plumbing -- JSONL parsing, the level_role call site, scoring
    arithmetic, markdown rendering -- end to end without an API call and without the fake's
    own "accuracy" needing to mean anything. Check evals/results.md against
    labeled_cases.jsonl by eye: exact-match rate should equal (labeled cases whose
    expected_level == FIXED_LEVEL) / (labeled cases), and escalation recall should be 0%
    (this fake never escalates) whenever at least one labeled case expects escalation.
    """

    FIXED_LEVEL = "L4"
    FIXED_CONFIDENCE = 0.9

    def __init__(self):
        self.model = "fixed-fake-model"
        self.max_tokens = 2048

    def with_structured_output(self, schema, **kwargs):
        assert schema is LevelingDecision
        return self

    def invoke(self, messages):
        return LevelingDecision(
            track="IC",
            assigned_level=self.FIXED_LEVEL,
            factor_ratings=[
                FactorRating(factor="scope_of_impact", level_indicated=self.FIXED_LEVEL, evidence="fixed fake output")
            ],
            factor5_variant_applied="5a",
            confidence=self.FIXED_CONFIDENCE,
            governing_rule="fake model -- not a real rule citation",
            reasoning="Fixed output from evals.run_evals.FixedFakeModel, used to smoke-test the harness.",
        )


def load_cases(limit: int | None = None) -> list[dict]:
    cases = []
    with open(CASES_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases[:limit] if limit else cases


def run_cases(cases: list[dict], model) -> list[CaseResult]:
    results = []
    for case in cases:
        raw_context = case.get("source_org_context")
        context = SourceOrgContext(**raw_context) if raw_context else None
        try:
            decision = level_role(case["job_description"], source_org_context=context, model=model)
            results.append(
                CaseResult(
                    case_id=case["case_id"],
                    label=case["label"],
                    expected_level=case.get("expected_level"),
                    assigned_level=decision.assigned_level,
                    expected_escalate=case.get("expected_escalate"),
                    escalate=decision.escalate,
                    governing_rule=decision.governing_rule,
                )
            )
        except Exception as e:
            results.append(
                CaseResult(
                    case_id=case["case_id"],
                    label=case["label"],
                    expected_level=case.get("expected_level"),
                    assigned_level=None,
                    expected_escalate=case.get("expected_escalate"),
                    escalate=None,
                    governing_rule=None,
                    error=str(e),
                )
            )
    return results


def main(fake: bool, limit: int | None, budget: float) -> None:
    cases = load_cases(limit)

    if fake:
        results = run_cases(cases, model=FixedFakeModel())
    else:
        from agents.cost_logging import get_session_stats, reset_session_stats
        from agents.spend_guard import BudgetExceededError, reset_default_budget

        reset_session_stats()
        reset_default_budget(budget)
        try:
            results = run_cases(cases, model=None)  # None -> level_role's own get_model("judgment")
        except BudgetExceededError as e:
            print(f"Run stopped -- budget exceeded: {e}")
            return
        print("Session cost:", get_session_stats().summary())

    rank = level_rank({code: v["sort_order"] for code, v in load_level_titles().items()})
    markdown = build_results_markdown(results, rank)
    RESULTS_PATH.write_text(markdown)
    print(f"\nWrote {RESULTS_PATH}\n")
    print(markdown)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", action="store_true", help="use FixedFakeModel instead of a real Claude call")
    parser.add_argument("--limit", type=int, default=None, help="max number of cases to run (default: all)")
    parser.add_argument("--budget", type=float, default=2.0, help="run cost cap in USD, ignored with --fake (default: 2.0)")
    args = parser.parse_args()
    main(args.fake, args.limit, args.budget)
