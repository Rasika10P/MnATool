"""Deterministic heuristic for guessing which raw column in an uploaded census workbook
corresponds to each of the fields app/pipeline.py's pipeline actually reads (CENSUS_COLUMNS).
A normalized-name matcher, not a model call -- matching "Employee ID" to "Emp ID" is exact
string logic, not judgment, and CLAUDE.md non-negotiable 1 keeps this out of an LLM's hands
the same way it keeps arithmetic out.

This never decides the mapping on its own. It only proposes one; the proposal is what
build order item 5's column-mapping-confirmation gate (agents/column_mapping_gate.py)
interrupt()s with, and a human always confirms or corrects it before any row is ingested.
"""

from __future__ import annotations

import re

# app/pipeline.py's CENSUS_COLUMNS, in the order the Nyx census generator produces them.
# Not imported from there directly to avoid tools/ (deterministic, no Streamlit) depending on
# app/ (Streamlit) -- app/ already depends on tools/, never the reverse, same layering
# agents/secrets.py's docstring establishes for agents/ vs. app/.
TARGET_COLUMNS = [
    "Emp ID", "Job Title", "Dept", "Location", "Curr", "Base", "Bonus",
    "Unvested Options", "Start", "Role Summary",
]

# Plausible header spellings seen in the wild for each target field, beyond the target
# column's own name (which is always tried first). Purely additive to the fallback identity
# match -- an exact (normalized) match to the target name itself always wins first.
FIELD_SYNONYMS: dict[str, list[str]] = {
    "Emp ID": ["employee id", "employee_id", "empid", "emp no", "employee number", "id"],
    "Job Title": ["title", "position", "job", "role title", "position title"],
    "Dept": ["department", "group", "team", "division"],
    "Location": ["site", "office", "city", "work location"],
    "Curr": ["currency", "ccy", "pay currency"],
    "Base": ["base pay", "base salary", "salary", "annual base", "base compensation"],
    "Bonus": ["target bonus", "annual bonus", "bonus target"],
    "Unvested Options": ["unvested equity", "equity", "options", "unvested grant"],
    "Start": ["start date", "hire date", "date of hire"],
    "Role Summary": ["summary", "description", "job description", "notes", "role description"],
}


def _normalize(name: str) -> str:
    return re.sub(r"[\s_-]+", " ", name.strip().lower())


def suggest_column_mapping(raw_columns: list[str]) -> dict[str, str | None]:
    """Best-guess {target_column: raw_column_or_None} for every entry in TARGET_COLUMNS.

    Each raw column can be claimed by at most one target (first target in TARGET_COLUMNS
    order wins a tie), so this never proposes mapping two Meridian fields to the same
    uploaded column. A target with no plausible match maps to None -- left for a human to
    fill in by hand at the confirmation gate, never guessed past what actually matched.
    """
    normalized_raw = {raw: _normalize(raw) for raw in raw_columns}
    mapping: dict[str, str | None] = {}
    claimed: set[str] = set()

    for target in TARGET_COLUMNS:
        candidates = [_normalize(target)] + FIELD_SYNONYMS.get(target, [])
        match = None
        for raw, normalized in normalized_raw.items():
            if raw in claimed:
                continue
            if normalized in candidates:
                match = raw
                break
        mapping[target] = match
        if match is not None:
            claimed.add(match)

    return mapping
