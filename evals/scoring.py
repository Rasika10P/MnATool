"""Pure scoring functions for the leveling eval harness (evals/run_evals.py). No model call,
no file I/O -- kept separate so the arithmetic itself is unit-testable directly
(tests/test_evals_scoring.py), the same "math in code, tested directly" discipline
CLAUDE.md's non-negotiable 1 asks of every comp calculation, applied here to the harness's
own scoring instead of a pay figure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 5a/5b/5c aren't a family_group value themselves (level_framework.md section 3: the variant
# is *selected by* family group) -- this is the coarse 3-way family bucket the variant the
# agent actually named implies, the only family-group signal a LevelingDecision carries at
# all. Slicing accuracy "by family group" in the summary sheet means slicing by this.
FAMILY_GROUP_BY_VARIANT = {
    "5a": "engineering",
    "5b": "corporate",
    "5c": "go-to-market",
}

# "rule N: ..." (section 5) or "section N rule M: ..." (section 6) -- the exact two citation
# shapes both the model's governing_rule and a hand-typed rule_under_test use elsewhere in
# this codebase (agents/arbiter.py, app/Home.py's _short_rule_citation). Matched
# case-insensitively and only on the leading number(s); the prose after the colon is free
# text on both sides and is never required to agree.
_RULE_PATTERN = re.compile(r"(?:section\s*(\d+)\s*)?rule\s*(\d+)", re.IGNORECASE)


def rule_citation_key(text: str | None) -> str | None:
    """Normalizes the *first* rule citation in text down to e.g. "rule3" or "section6rule3"
    -- None if the text doesn't contain a recognizable citation at all (an empty
    rule_under_test cell, or a governing_rule string that for some reason cites nothing by
    number). rule_under_test is expected to name exactly one rule, so "first" is also "only"
    there; for a governing_rule that may cite more than one (see rule_citation_keys below),
    this is "the rule it leads with," not "every rule it mentions."""
    if not text:
        return None
    match = _RULE_PATTERN.search(text)
    if not match:
        return None
    section, rule = match.groups()
    return f"section{section}rule{rule}" if section else f"rule{rule}"


def rule_citation_keys(text: str | None) -> set[str]:
    """Every rule citation in text, normalized the same way as rule_citation_key -- a
    governing_rule frequently cites more than one rule in the same sentence (e.g. "rule 6:
    title is not input ... Also section 6 rule 3: platform dependency caps ..."), and
    rule_matched below must credit any of them, not just whichever happens to come first."""
    if not text:
        return set()
    keys = set()
    for match in _RULE_PATTERN.finditer(text):
        section, rule = match.groups()
        keys.add(f"section{section}rule{rule}" if section else f"rule{rule}")
    return keys


@dataclass
class CaseResult:
    # Echoed straight from the input row (evals/labeled_cases.xlsx) -- carried through to
    # results.xlsx so a reader never has to cross-reference the two workbooks by case_id.
    case_id: str
    source: str | None
    role_summary: str
    source_headcount: int | None
    source_stage: str | None
    source_type: str | None
    expected_track: str | None
    expected_level: str | None
    expected_escalate: bool | None
    rule_under_test: str | None
    label_notes: str | None

    # What the agent actually did.
    track: str | None = None
    assigned_level: str | None = None
    confidence: float | None = None
    escalate: bool | None = None
    governing_rule: str | None = None
    alternative_level: str | None = None
    factor5_variant_applied: str | None = None
    error: str | None = None

    @property
    def family_group(self) -> str | None:
        if self.factor5_variant_applied is None:
            return None
        return FAMILY_GROUP_BY_VARIANT.get(self.factor5_variant_applied)

    @property
    def rule_matched(self) -> bool | None:
        """None (not "no match") when there's nothing to check -- either this case isn't
        testing a specific rule (rule_under_test blank) or it errored before producing a
        governing_rule to compare. Distinguishing "not applicable" from "checked and
        disagreed" is exactly why every other tri-state field in this module (expected_escalate,
        the precision/recall pair) already refuses to collapse into a bool.

        Checked against *every* citation in governing_rule, not just the first: a real
        response cited "rule 6 ... Also section 6 rule 3 ..." in one sentence, and the case
        was testing section 6 rule 3 specifically -- crediting only the leading citation
        would have called that a miss when the agent did, in fact, name the right rule.
        """
        if not self.rule_under_test or self.governing_rule is None:
            return None
        target = rule_citation_key(self.rule_under_test)
        return target in rule_citation_keys(self.governing_rule)


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


def outcome(result: CaseResult, rank: dict[str, int]) -> str:
    """One of "exact" / "within-one" / "miss" -- the three the Results sheet's outcome
    column is specified to carry -- plus two states that aren't a scoring outcome at all but
    still need a truthful cell value: "error" (the leveling call itself failed) and "not yet
    labeled" (expected_level is blank -- this row was run so its output could be reviewed,
    but excluded from every rate above, per this module's own _labeled_for_level)."""
    if result.error:
        return "error"
    if result.expected_level is None:
        return "not yet labeled"
    if result.assigned_level == result.expected_level:
        return "exact"
    if abs(rank[result.assigned_level] - rank[result.expected_level]) <= 1:
        return "within-one"
    return "miss"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def slice_exact_match(results: list[CaseResult], key_fn) -> dict[str, tuple[float | None, int]]:
    """Exact-match rate and case count per bucket of key_fn(result) -- the "accuracy sliced
    by X" summary tables. A result whose key_fn(result) is None is dropped from every bucket
    (e.g. family_group is None when the agent's own call errored before producing a
    factor5_variant_applied) rather than pooled into a misleading "None" bucket."""
    labeled = _labeled_for_level(results)
    buckets: dict[str, list[CaseResult]] = {}
    for r in labeled:
        key = key_fn(r)
        if key is None:
            continue
        buckets.setdefault(key, []).append(r)
    return {key: (exact_match_rate(rs), len(rs)) for key, rs in sorted(buckets.items())}


def rule_compliance(results: list[CaseResult]) -> dict[str, dict]:
    """Per rule_under_test (blank excluded -- a case not testing a specific rule has nothing
    to report here): how many cases tested it, how often the agent's governing_rule actually
    cited it (citation_rate), and how often the case landed on the expected level
    (accuracy_rate, exact match only -- level_framework.md's own bar, not within-one). A rule
    can be cited correctly and still miss the level (the agent named the right rule but
    misapplied it) or land on the right level while citing a different rule (right answer,
    wrong reasoning) -- reporting both numbers separately is the point; collapsing them into
    one score would hide exactly that distinction.
    """
    buckets: dict[str, list[CaseResult]] = {}
    for r in results:
        if not r.rule_under_test:
            continue
        buckets.setdefault(r.rule_under_test, []).append(r)

    compliance = {}
    for rule, rs in sorted(buckets.items()):
        checkable = [r for r in rs if r.rule_matched is not None]
        citation_rate = (sum(1 for r in checkable if r.rule_matched) / len(checkable)) if checkable else None
        compliance[rule] = {
            "n": len(rs),
            "citation_rate": citation_rate,
            "accuracy_rate": exact_match_rate(rs),
        }
    return compliance


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
    ]

    def _slice_table(title: str, slice_result: dict[str, tuple[float | None, int]]) -> list[str]:
        rows = [f"### {title}", "", "| Bucket | Exact-match rate | n |", "|---|---|---|"]
        for key, (rate, n) in slice_result.items():
            rows.append(f"| {key} | {_pct(rate)} | {n} |")
        rows.append("")
        return rows

    lines += _slice_table("Accuracy by family group", slice_exact_match(results, lambda r: r.family_group))
    lines += _slice_table("Accuracy by track", slice_exact_match(results, lambda r: r.expected_track))
    lines += _slice_table("Accuracy by source", slice_exact_match(results, lambda r: r.source))
    lines += _slice_table(
        "Accuracy by source_type", slice_exact_match(results, lambda r: r.source_type or "internal (no source org)")
    )

    compliance = rule_compliance(results)
    lines += [
        "### Per-rule compliance",
        "",
        "| Rule under test | n | Cited correctly | Landed on expected level |",
        "|---|---|---|---|",
    ]
    for rule, stats in compliance.items():
        lines.append(f"| {rule} | {stats['n']} | {_pct(stats['citation_rate'])} | {_pct(stats['accuracy_rate'])} |")
    lines.append("")

    lines += [
        "### Per-case detail",
        "",
        "| Case | Source | Expected | Assigned | Outcome | Escalate (expected/actual) | Governing rule | Rule matched |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        result_label = outcome(r, rank)
        expected_level = r.expected_level or "—"
        assigned_level = r.assigned_level or ("ERROR" if r.error else "—")
        expected_escalate = "—" if r.expected_escalate is None else str(r.expected_escalate)
        actual_escalate = "—" if r.escalate is None else str(r.escalate)
        rule = r.error if r.error else (r.governing_rule or "—")
        matched = "—" if r.rule_matched is None else ("yes" if r.rule_matched else "no")
        lines.append(
            f"| {r.case_id} | {r.source or '—'} | {expected_level} | {assigned_level} | {result_label} "
            f"| {expected_escalate} / {actual_escalate} | {rule} | {matched} |"
        )
    return "\n".join(lines) + "\n"
