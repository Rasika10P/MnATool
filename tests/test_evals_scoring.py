"""Unit tests for the eval harness's own scoring arithmetic (evals/scoring.py) -- no model
call, no spreadsheet, just the math. This is the rigorous proof the harness's metrics are
computed correctly; evals/run_evals.py --dry-run is the live, cache-aware version of the same
claim against the real evals/labeled_cases.xlsx file.
"""

from evals.scoring import (
    CaseResult,
    build_results_markdown,
    escalation_precision_recall,
    exact_match_rate,
    level_rank,
    outcome,
    rule_citation_key,
    rule_compliance,
    slice_exact_match,
    within_one_level_rate,
)

# Same 13-code ladder as level_definitions.parquet (app.pipeline.load_level_titles), inlined
# here so this test doesn't depend on reading the committed data file for pure-math checks.
LEVEL_SORT_ORDER = {
    "L1": 10, "L2": 20, "L3": 30, "L4": 40, "M3": 45, "L5": 50, "M4": 55,
    "L6": 60, "M5": 65, "L7": 70, "M6": 75, "L8": 80, "M7": 85,
}
RANK = level_rank(LEVEL_SORT_ORDER)


def _case(
    case_id="c1",
    expected_level="L4",
    assigned_level="L4",
    expected_escalate=None,
    escalate=None,
    error=None,
    expected_track=None,
    source=None,
    source_type=None,
    rule_under_test=None,
    governing_rule="rule 1: scope of impact is primary",
    factor5_variant_applied="5a",
):
    return CaseResult(
        case_id=case_id,
        source=source,
        role_summary="test role summary",
        source_headcount=None,
        source_stage=None,
        source_type=source_type,
        expected_track=expected_track,
        expected_level=expected_level,
        expected_escalate=expected_escalate,
        rule_under_test=rule_under_test,
        label_notes=None,
        assigned_level=assigned_level,
        escalate=escalate,
        governing_rule=governing_rule,
        error=error,
        factor5_variant_applied=factor5_variant_applied if not error else None,
    )


def test_level_rank_is_dense_and_ordered():
    assert RANK["L1"] == 0
    assert RANK["L4"] == 3
    assert RANK["M3"] == 4  # sits between L4 and L5 by sort_order
    assert RANK["M7"] == 12


def test_rule_citation_key_normalizes_section_5_rules():
    assert rule_citation_key("rule 3: deep-but-narrow does not reach L6") == "rule3"
    assert rule_citation_key("Rule 3") == "rule3"
    assert rule_citation_key("rule 2: lower level governs a split") == "rule2"


def test_rule_citation_key_normalizes_section_6_rules():
    assert rule_citation_key("section 6 rule 3: platform dependency") == "section6rule3"
    assert rule_citation_key("SECTION 6 RULE 3") == "section6rule3"
    # A bare "rule 3" and a "section 6 rule 3" are different citations -- one is section 5,
    # the other section 6 -- and must not collide just because both mention "3".
    assert rule_citation_key("rule 3") != rule_citation_key("section 6 rule 3")


def test_rule_citation_key_none_when_no_citation_present():
    assert rule_citation_key(None) is None
    assert rule_citation_key("") is None
    assert rule_citation_key("the agent gave no reasoning at all") is None


def test_family_group_from_variant():
    assert _case(factor5_variant_applied="5a").family_group == "engineering"
    assert _case(factor5_variant_applied="5b").family_group == "corporate"
    assert _case(factor5_variant_applied="5c").family_group == "go-to-market"


def test_family_group_none_when_no_decision_was_produced():
    assert _case(error="boom").family_group is None


def test_rule_matched_true_when_citations_agree():
    r = _case(rule_under_test="rule 3: deep-but-narrow does not reach L6", governing_rule="rule 3: capped at L5")
    assert r.rule_matched is True


def test_rule_matched_false_when_citations_disagree():
    r = _case(rule_under_test="rule 3: deep-but-narrow does not reach L6", governing_rule="rule 2: lower level governs")
    assert r.rule_matched is False


def test_rule_matched_none_when_not_under_test():
    assert _case(rule_under_test=None).rule_matched is None


def test_rule_matched_none_on_error():
    r = _case(rule_under_test="rule 3", error="boom", governing_rule=None)
    assert r.rule_matched is None


def test_rule_matched_credits_a_non_leading_citation():
    # A real response cited "rule 6" first, then "section 6 rule 3" later in the same
    # sentence -- the case was testing section 6 rule 3, and it must not lose credit just
    # because rule 6 happened to come first in the text.
    r = _case(
        rule_under_test="section 6 rule 3: platform dependency must be assessed for carve-outs",
        governing_rule=(
            "rule 6: title in source document is evidence, not input; level from described "
            "scope. Also section 6 rule 3: platform dependency caps technical depth/breadth."
        ),
    )
    assert r.rule_matched is True


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


def test_outcome_categories():
    assert outcome(_case("c1", "L4", "L4"), RANK) == "exact"
    assert outcome(_case("c2", "L4", "M3"), RANK) == "within-one"
    assert outcome(_case("c3", "L4", "L7"), RANK) == "miss"
    assert outcome(_case("c4", expected_level=None, assigned_level="L4"), RANK) == "not yet labeled"
    assert outcome(_case("c5", "L4", None, error="boom"), RANK) == "error"


def test_slice_exact_match_by_family_group():
    results = [
        _case("c1", "L4", "L4", factor5_variant_applied="5a"),  # engineering, exact
        _case("c2", "L4", "L5", factor5_variant_applied="5a"),  # engineering, miss
        _case("c3", "M3", "M3", factor5_variant_applied="5b"),  # corporate, exact
    ]
    sliced = slice_exact_match(results, lambda r: r.family_group)
    assert sliced["engineering"] == (0.5, 2)
    assert sliced["corporate"] == (1.0, 1)


def test_slice_exact_match_drops_none_keys():
    results = [_case("c1", "L4", "L4", factor5_variant_applied="5a"), _case("c2", "L4", None, error="boom")]
    sliced = slice_exact_match(results, lambda r: r.family_group)
    assert list(sliced.keys()) == ["engineering"]


def test_rule_compliance_reports_citation_and_accuracy_separately():
    results = [
        # Cited the right rule and landed on the expected level.
        _case("c1", "L5", "L5", rule_under_test="rule 3: deep-but-narrow does not reach L6",
              governing_rule="rule 3: capped at L5"),
        # Cited a different rule, but still landed on the expected level (right answer, wrong reasoning).
        _case("c2", "L5", "L5", rule_under_test="rule 3: deep-but-narrow does not reach L6",
              governing_rule="rule 1: scope of impact is primary"),
        # Not testing any rule -- excluded from this table entirely.
        _case("c3", "L4", "L4", rule_under_test=None),
    ]
    compliance = rule_compliance(results)
    assert list(compliance.keys()) == ["rule 3: deep-but-narrow does not reach L6"]
    stats = compliance["rule 3: deep-but-narrow does not reach L6"]
    assert stats["n"] == 2
    assert stats["citation_rate"] == 0.5  # 1 of 2 actually cited rule 3
    assert stats["accuracy_rate"] == 1.0  # both landed on the expected level


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
    assert "Accuracy by family group" in md
    assert "Per-rule compliance" in md


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
