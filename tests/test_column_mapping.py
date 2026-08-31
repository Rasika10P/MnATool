"""Tests for tools/column_mapping.py's suggest_column_mapping -- a plain deterministic
matcher, no model call, no LangGraph."""

from tools.column_mapping import TARGET_COLUMNS, suggest_column_mapping


def test_exact_header_names_all_match():
    mapping = suggest_column_mapping(list(TARGET_COLUMNS))
    assert mapping == {c: c for c in TARGET_COLUMNS}


def test_synonym_headers_match_via_normalization():
    raw = [
        "Employee ID", "Job Title", "Department", "Office", "Currency", "Base Salary",
        "Target Bonus", "Unvested Equity", "Hire Date", "Notes",
    ]
    mapping = suggest_column_mapping(raw)
    assert mapping["Emp ID"] == "Employee ID"
    assert mapping["Job Title"] == "Job Title"
    assert mapping["Dept"] == "Department"
    assert mapping["Location"] == "Office"
    assert mapping["Curr"] == "Currency"
    assert mapping["Base"] == "Base Salary"
    assert mapping["Bonus"] == "Target Bonus"
    assert mapping["Unvested Options"] == "Unvested Equity"
    assert mapping["Start"] == "Hire Date"
    assert mapping["Role Summary"] == "Notes"


def test_unrecognized_column_maps_to_none_never_guessed():
    mapping = suggest_column_mapping(["Emp ID", "Some Totally Unrelated Column"])
    assert mapping["Emp ID"] == "Emp ID"
    assert mapping["Job Title"] is None
    assert mapping["Role Summary"] is None


def test_no_raw_column_is_claimed_by_two_targets():
    # "ID" alone is a synonym only of Emp ID -- confirms one raw column can't double as a
    # match for a second target even if it loosely resembles more than one.
    mapping = suggest_column_mapping(["ID"])
    claimed = [v for v in mapping.values() if v is not None]
    assert claimed == ["ID"]
    assert mapping["Emp ID"] == "ID"
