"""Eval harness for the leveling agent (build order item 9), spreadsheet-driven. Reads
evals/labeled_cases.xlsx (hand-labeled by a comp professional -- see
evals/generate_labeled_cases.py for how that sheet is built and why expected_level/
expected_track/expected_escalate are never invented here), runs agents.leveling.level_role
over every case, and writes:

  evals/results.xlsx -- a *separate* workbook (Results + Summary sheets). This script only
      ever reads labeled_cases.xlsx, never opens it for writing -- a hand-labeled row is
      never at risk from a re-run, no matter how the run goes.
  evals/results.md -- the same Summary-sheet numbers plus a per-case table, for the README.

Real leveling calls go through agents.model_router's "judgment" tier (Claude -- CLAUDE.md's
model routing: leveling adjudication is Claude-only). Every call is cached by
agents/instrumented_model.py under the default "fill" cache mode (scripts/_cli_common.py's
add_cache_mode_arg/run_with_budget_guard, the same convention every other population-running
script in this repo already uses): a case already run with identical inputs is served from
cache at zero cost, so re-running after labeling a few more rows only pays for the new ones.
--dry-run reports cache hits/misses without spending anything, matching every other script
here that touches the API.

Cases with expected_level blank (not yet hand-labeled) are still run and shown in
results.xlsx/results.md -- so a comp professional can see what the agent would say before
deciding the label -- but excluded from every rate, exactly like the retired
evals/labeled_cases.jsonl harness this one replaces.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from agents.leveling import level_role
from agents.schemas import SourceOrgContext
from app.pipeline import load_level_titles
from evals.generate_labeled_cases import COLUMNS as INPUT_COLUMNS
from evals.scoring import (
    CaseResult,
    build_results_markdown,
    escalation_precision_recall,
    exact_match_rate,
    level_rank,
    outcome,
    rule_compliance,
    slice_exact_match,
    within_one_level_rate,
)
from scripts._cli_common import add_cache_mode_arg, dry_run_report, run_with_budget_guard

CASES_PATH = Path(__file__).resolve().parent / "labeled_cases.xlsx"
RESULTS_XLSX_PATH = Path(__file__).resolve().parent / "results.xlsx"
RESULTS_MD_PATH = Path(__file__).resolve().parent / "results.md"

RESULT_COLUMNS = [
    "assigned_level", "confidence", "escalate", "governing_rule", "alternative_level",
    "outcome", "rule_matched", "factor5_variant_applied", "family_group",
]


def _clean(value):
    """pandas reads a blank Excel cell as float NaN regardless of the column's intended
    type -- None everywhere downstream expects "not provided", the same contract
    SourceOrgContext's own fields already use."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _parse_escalate(value) -> bool | None:
    value = _clean(value)
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("y", "yes", "true"):
        return True
    if text in ("n", "no", "false"):
        return False
    raise ValueError(f"expected_escalate must be Y/N (or blank), got {value!r}")


def load_cases(limit: int | None = None) -> list[dict]:
    df = pd.read_excel(CASES_PATH)
    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{CASES_PATH} is missing column(s) {missing} -- was it hand-edited into a "
            "different shape than evals/generate_labeled_cases.py produces?"
        )
    cases = []
    for _, row in df.iterrows():
        case = {col: _clean(row[col]) for col in INPUT_COLUMNS}
        case["source_headcount"] = int(case["source_headcount"]) if case["source_headcount"] is not None else None
        case["expected_escalate"] = _parse_escalate(row["expected_escalate"])
        cases.append(case)
    return cases[:limit] if limit else cases


def _source_org_context(case: dict) -> SourceOrgContext | None:
    fields = {
        k: case[k] for k in ("source_headcount", "source_stage", "source_type") if case[k] is not None
    }
    return SourceOrgContext(**fields) if fields else None


def _base_result_kwargs(case: dict) -> dict:
    return {
        "case_id": case["case_id"],
        "source": case["source"],
        "role_summary": case["role_summary"],
        "source_headcount": case["source_headcount"],
        "source_stage": case["source_stage"],
        "source_type": case["source_type"],
        "expected_track": case["expected_track"],
        "expected_level": case["expected_level"],
        "expected_escalate": case["expected_escalate"],
        "rule_under_test": case["rule_under_test"],
        "label_notes": case["label_notes"],
    }


def run_cases(cases: list[dict], model) -> list[CaseResult]:
    results = []
    for case in cases:
        context = _source_org_context(case)
        try:
            decision = level_role(case["role_summary"], source_org_context=context, model=model)
            results.append(
                CaseResult(
                    **_base_result_kwargs(case),
                    track=decision.track,
                    assigned_level=decision.assigned_level,
                    confidence=decision.confidence,
                    escalate=decision.escalate,
                    governing_rule=decision.governing_rule,
                    alternative_level=decision.alternative_level,
                    factor5_variant_applied=decision.factor5_variant_applied,
                )
            )
        except Exception as e:
            results.append(CaseResult(**_base_result_kwargs(case), error=str(e)))
    return results


def write_results_xlsx(results: list[CaseResult], rank: dict[str, int], path: Path) -> None:
    rows = []
    for r in results:
        row = {col: getattr(r, col) for col in INPUT_COLUMNS}
        row.update(
            {
                "assigned_level": r.assigned_level,
                "confidence": r.confidence,
                "escalate": r.escalate,
                "governing_rule": r.error if r.error else r.governing_rule,
                "alternative_level": r.alternative_level,
                "outcome": outcome(r, rank),
                "rule_matched": r.rule_matched,
                "factor5_variant_applied": r.factor5_variant_applied,
                "family_group": r.family_group,
            }
        )
        rows.append(row)
    results_df = pd.DataFrame(rows, columns=INPUT_COLUMNS + RESULT_COLUMNS)

    exact = exact_match_rate(results)
    within_one = within_one_level_rate(results, rank)
    precision, recall = escalation_precision_recall(results)
    n_labeled = sum(1 for r in results if r.expected_level is not None)

    headline_df = pd.DataFrame(
        [
            {"metric": "cases labeled", "value": f"{n_labeled} of {len(results)}"},
            {"metric": "exact-level match rate", "value": exact},
            {"metric": "within-one-level rate", "value": within_one},
            {"metric": "escalation precision", "value": precision},
            {"metric": "escalation recall", "value": recall},
        ]
    )

    def _slice_df(slice_result: dict[str, tuple[float | None, int]]) -> pd.DataFrame:
        return pd.DataFrame(
            [{"bucket": key, "exact_match_rate": rate, "n": n} for key, (rate, n) in slice_result.items()]
        )

    family_df = _slice_df(slice_exact_match(results, lambda r: r.family_group))
    track_df = _slice_df(slice_exact_match(results, lambda r: r.expected_track))
    source_df = _slice_df(slice_exact_match(results, lambda r: r.source))
    source_type_df = _slice_df(
        slice_exact_match(results, lambda r: r.source_type or "internal (no source org)")
    )
    compliance_df = pd.DataFrame(
        [
            {"rule_under_test": rule, "n": stats["n"], "citation_rate": stats["citation_rate"],
             "accuracy_rate": stats["accuracy_rate"]}
            for rule, stats in rule_compliance(results).items()
        ]
    )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="Results", index=False)

        row = 0
        headline_df.to_excel(writer, sheet_name="Summary", index=False, startrow=row)
        row += len(headline_df) + 3

        for title, df in [
            ("Accuracy by family group", family_df),
            ("Accuracy by track", track_df),
            ("Accuracy by source", source_df),
            ("Accuracy by source_type", source_type_df),
            ("Per-rule compliance", compliance_df),
        ]:
            sheet = writer.sheets["Summary"]
            sheet.cell(row=row + 1, column=1, value=title)
            row += 1
            if df.empty:
                sheet.cell(row=row + 1, column=1, value="(no cases)")
                row += 2
                continue
            df.to_excel(writer, sheet_name="Summary", index=False, startrow=row)
            row += len(df) + 3

        for sheet_name in ("Results", "Summary"):
            ws = writer.sheets[sheet_name]
            for col_cells in ws.columns:
                length = max((len(str(c.value)) for c in col_cells if c.value is not None), default=10)
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(length + 2, 10), 60)


def main(limit: int | None, dry_run: bool, budget: float, cache_mode: str) -> None:
    cases = load_cases(limit)

    if dry_run:
        items = [
            (case["case_id"], case["role_summary"], _source_org_context(case))
            for case in cases
        ]
        dry_run_report(items)
        return

    def _run() -> None:
        results = run_cases(cases, model=None)  # None -> level_role's own get_model("judgment")
        rank = level_rank({code: v["sort_order"] for code, v in load_level_titles().items()})

        write_results_xlsx(results, rank, RESULTS_XLSX_PATH)
        print(f"Wrote {RESULTS_XLSX_PATH}")

        markdown = build_results_markdown(results, rank)
        RESULTS_MD_PATH.write_text(markdown)
        print(f"Wrote {RESULTS_MD_PATH}\n")
        print(markdown)

    run_with_budget_guard(budget, _run, cache_mode=cache_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="max number of cases to run (default: all)")
    parser.add_argument("--budget", type=float, default=2.0, help="run cost cap in USD (default: 2.0)")
    parser.add_argument("--dry-run", action="store_true", help="report cache hits/misses without calling or writing anything")
    add_cache_mode_arg(parser)
    args = parser.parse_args()
    main(args.limit, args.dry_run, args.budget, args.cache_mode)
