"""Unit tests for the eval harness's own scoring arithmetic (evals/scoring.py) -- no model
call, no JSONL file, just the math. This is the rigorous proof the harness's metrics are
computed correctly; evals/run_evals.py --fake is the live, human-eyeballable version of the
same claim against the real labeled_cases.jsonl file.
"""

from evals.scoring import (
    CaseResult,
    build_results_markdown,
    escalation_precision_recall,
    exact_match_rate,
    level_rank,
    match_label,
    within_one_level_rate,
)

# Same 13-code ladder as level_definitions.parquet (app.pipeline.load_level_titles), inlined
# here so this test doesn't depend on reading the committed data file for pure-math checks.
LEVEL_SORT_ORDER = {
    "L1": 10, "L2": 20, "L3": 30, "L4": 40, "M3": 45, "L5": 50, "M4": 55,
    "L6": 60, "M5": 65, "L7": 70, "M6": 75, "L8": 80, "M7": 85,
}
RANK = level_rank(LEVEL_SORT_ORDER)


def _case(case_id="c1", expected_level="L4", assigned_level="L4", expected_escalate=None, escalate=None, error=None):
    return CaseResult(
        case_id=case_id, label=case_id, expected_level=expected_level, assigned_level=assigned_level,
        expected_escalate=expected_escalate, escalate=escalate, governing_rule="rule 1", error=error,
    )


def test_level_rank_is_dense_and_ordered():
    assert RANK["L1"] == 0
    assert RANK["L4"] == 3
    assert RANK["M3"] == 4  # sits between L4 and L5 by sort_order
    assert RANK["M7"] == 12


def test_exact_match_rate_counts_only_labeled_cases():
    results = [
        _case("c1", "L4", "L4"),
        _case("c2", "L5", "L4"),
        _case("c3", expected_level=None, assigned_level="L4"),  # not yet labeled -- excluded
    ]
    assert exact_match_rate(results) == 0.5


def test_exact_match_rate_none_when_nothing_labeled():
    assert exact_match_rate([_case("c1", expected_level=None, assigned_level="L4")]) is None


def test_within_one_level_rate():
    results = [
        _case("c1", "L4", "L4"),  # exact -> within one
        _case("c2", "L4", "M3"),  # rank 3 vs 4 -> within one
        _case("c3", "L4", "L6"),  # rank 3 vs 5 -> miss
    ]
    assert within_one_level_rate(results, RANK) == 2 / 3


def test_escalation_precision_recall_basic():
    results = [
        _case("c1", expected_escalate=True, escalate=True),   # TP
        _case("c2", expected_escalate=False, escalate=True),  # FP
        _case("c3", expected_escalate=True, escalate=False),  # FN
        _case("c4", expected_escalate=False, escalate=False),  # TN
    ]
    precision, recall = escalation_precision_recall(results)
    assert precision == 0.5  # 1 TP / 2 predicted positive
    assert recall == 0.5     # 1 TP / 2 actual positive


def test_escalation_precision_is_none_when_nothing_predicted_positive():
    results = [_case("c1", expected_escalate=True, escalate=False)]
    precision, recall = escalation_precision_recall(results)
    assert precision is None  # undefined, not 0.0 -- see docstring
    assert recall == 0.0


def test_escalation_recall_is_none_when_no_actual_positives():
    results = [_case("c1", expected_escalate=False, escalate=False)]
    precision, recall = escalation_precision_recall(results)
    assert recall is None
    assert precision is None  # also nothing predicted positive here


def test_match_label_categories():
    assert match_label(_case("c1", "L4", "L4"), RANK) == "exact"
    assert match_label(_case("c2", "L4", "M3"), RANK) == "within one"
    assert match_label(_case("c3", "L4", "L7"), RANK) == "miss"
    assert match_label(_case("c4", expected_level=None, assigned_level="L4"), RANK) == "not yet labeled"
    assert match_label(_case("c5", "L4", None, error="boom"), RANK) == "error"


def test_build_results_markdown_includes_metrics_and_every_case():
    results = [
        _case("case-01", "L4", "L4", expected_escalate=False, escalate=False),
        _case("case-02", "L5", "M3", expected_escalate=True, escalate=True),
    ]
    md = build_results_markdown(results, RANK)
    assert "Exact-level match rate:** 50%" in md
    assert "case-01" in md and "case-02" in md
    assert "Escalation precision:** 100%" in md
    assert "Escalation recall:** 100%" in md


def test_build_results_markdown_handles_unlabeled_and_error_cases():
    results = [
        _case("case-01", expected_level=None, assigned_level="L4"),
        _case("case-02", "L4", None, error="StructuredOutputError: exhausted retries"),
    ]
    md = build_results_markdown(results, RANK)
    assert "Exact-level match rate:** n/a" in md
    assert "not yet labeled" in md
    assert "ERROR" in md
    assert "StructuredOutputError" in md
