"""Pure scoring functions for the leveling eval harness (evals/run_evals.py). No model call,
no file I/O -- kept separate so the arithmetic itself is unit-testable directly
(tests/test_evals_scoring.py), the same "math in code, tested directly" discipline
CLAUDE.md's non-negotiable 1 asks of every comp calculation, applied here to the harness's
own scoring instead of a pay figure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CaseResult:
    case_id: str
    label: str
    expected_level: str | None
    assigned_level: str | None
    expected_escalate: bool | None
    escalate: bool | None
    governing_rule: str | None
    error: str | None = None


def level_rank(level_sort_order: dict[str, int]) -> dict[str, int]:
    """level_sort_order: {level_code: sort_order}, e.g. from
    {code: v["sort_order"] for code, v in app.pipeline.load_level_titles().items()}.

    Returns a dense rank {level_code: 0..12} by sort_order -- "within one level" means "one
    rung apart on the unified 13-rung IC+MGR ladder," not a specific numeric sort_order gap.
    The raw sort_order values aren't evenly spaced (L1->L2 is a gap of 10; M3 sits only 5
    above L4, interleaved between L4 and L5) precisely so ties resolve correctly, but that
    makes the raw numbers unsuitable for an "is this within N" comparison directly.
    """
    ordered = sorted(level_sort_order.items(), key=lambda kv: kv[1])
    return {code: i for i, (code, _) in enumerate(ordered)}


def _labeled_for_level(results: list[CaseResult]) -> list[CaseResult]:
    return [r for r in results if r.expected_level is not None and r.assigned_level is not None]


def exact_match_rate(results: list[CaseResult]) -> float | None:
    labeled = _labeled_for_level(results)
    if not labeled:
        return None
    return sum(1 for r in labeled if r.assigned_level == r.expected_level) / len(labeled)


def within_one_level_rate(results: list[CaseResult], rank: dict[str, int]) -> float | None:
    labeled = _labeled_for_level(results)
    if not labeled:
        return None
    hits = sum(1 for r in labeled if abs(rank[r.assigned_level] - rank[r.expected_level]) <= 1)
    return hits / len(labeled)


def escalation_precision_recall(results: list[CaseResult]) -> tuple[float | None, float | None]:
    """None (reported as "n/a", never 0.0) when there's nothing to divide by -- a run with no
    predicted escalations has undefined precision, not zero precision, and a run with no
    actual-escalation cases has undefined recall. Conflating "undefined" with "zero" would
    make a harness with too few escalation cases look like it's failing when it's actually
    just under-tested for that metric specifically.
    """
    labeled = [r for r in results if r.expected_escalate is not None and r.escalate is not None]
    predicted_positive = [r for r in labeled if r.escalate]
    actual_positive = [r for r in labeled if r.expected_escalate]
    true_positive = sum(1 for r in labeled if r.escalate and r.expected_escalate)
    precision = true_positive / len(predicted_positive) if predicted_positive else None
    recall = true_positive / len(actual_positive) if actual_positive else None
    return precision, recall


def match_label(result: CaseResult, rank: dict[str, int]) -> str:
    if result.error:
        return "error"
    if result.expected_level is None:
        return "not yet labeled"
    if result.assigned_level == result.expected_level:
        return "exact"
    if abs(rank[result.assigned_level] - rank[result.expected_level]) <= 1:
        return "within one"
    return "miss"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def build_results_markdown(results: list[CaseResult], rank: dict[str, int]) -> str:
    exact = exact_match_rate(results)
    within_one = within_one_level_rate(results, rank)
    precision, recall = escalation_precision_recall(results)
    n_labeled = sum(1 for r in results if r.expected_level is not None)

    lines = [
        "# Leveling eval results",
        "",
        f"{n_labeled} of {len(results)} cases labeled.",
        "",
        f"- **Exact-level match rate:** {_pct(exact)}",
        f"- **Within-one-level rate:** {_pct(within_one)}",
        f"- **Escalation precision:** {_pct(precision)}",
        f"- **Escalation recall:** {_pct(recall)}",
        "",
        "| Case | Label | Expected | Assigned | Result | Escalate (expected/actual) | Governing rule |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        result_label = match_label(r, rank)
        expected_level = r.expected_level or "—"
        assigned_level = r.assigned_level or ("ERROR" if r.error else "—")
        expected_escalate = "—" if r.expected_escalate is None else str(r.expected_escalate)
        actual_escalate = "—" if r.escalate is None else str(r.escalate)
        rule = r.error if r.error else (r.governing_rule or "—")
        lines.append(
            f"| {r.case_id} | {r.label} | {expected_level} | {assigned_level} | {result_label} "
            f"| {expected_escalate} / {actual_escalate} | {rule} |"
        )
    return "\n".join(lines) + "\n"
