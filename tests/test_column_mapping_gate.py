"""Tests for the column-mapping-confirmation gate (agents/column_mapping_gate.py) -- build
order item 5, gate 1: a human confirms or corrects the proposed column mapping before any
row of an uploaded census is ingested. Isolates both the LangGraph checkpoint db and the
review log to tmp_path, same convention as tests/test_no_equivalent_gate.py's
isolated_review_log fixture.
"""

import json
import uuid

import pytest

import agents.column_mapping_gate as column_mapping_gate
from agents.column_mapping_gate import resume_column_mapping_review, start_column_mapping_review
from tools.column_mapping import REQUIRED_COLUMNS, TARGET_COLUMNS


@pytest.fixture(autouse=True)
def isolated_review_log(tmp_path, monkeypatch):
    path = tmp_path / "column_mapping_reviews.jsonl"
    monkeypatch.setattr(column_mapping_gate, "REVIEW_LOG_PATH", path)
    return path


@pytest.fixture
def checkpoint_db(tmp_path):
    return tmp_path / "column_mapping_checkpoints.sqlite"


def _state(**overrides) -> dict:
    base = {
        "source_name": "acme_census.xlsx",
        "raw_columns": ["Employee ID", "Job Title", "Department", "Office", "Currency", "Base Salary"],
        "suggested_mapping": None,
        "confirmed_mapping": None,
        "missing_required": None,
        "review_entry": None,
    }
    base.update(overrides)
    return base


def test_start_pauses_with_a_precomputed_suggestion(checkpoint_db):
    thread_id = f"test-{uuid.uuid4()}"
    result = start_column_mapping_review(_state(), thread_id, db_path=checkpoint_db)

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["source_name"] == "acme_census.xlsx"
    assert payload["suggested_mapping"]["Emp ID"] == "Employee ID"
    assert payload["suggested_mapping"]["Dept"] == "Department"
    # Bonus/Unvested Options/Start/Role Summary have no header in this upload -- flagged, not guessed.
    assert set(payload["unmatched_targets"]) == {"Bonus", "Unvested Options", "Start", "Role Summary"}


def test_confirming_the_suggestion_verbatim_logs_it(checkpoint_db, isolated_review_log):
    thread_id = f"test-{uuid.uuid4()}"
    start_result = start_column_mapping_review(_state(), thread_id, db_path=checkpoint_db)
    suggested = start_result["__interrupt__"][0].value["suggested_mapping"]
    confirmed = {**suggested, "Bonus": None, "Unvested Options": None, "Start": None, "Role Summary": None}

    result = resume_column_mapping_review(thread_id, confirmed_mapping=confirmed, db_path=checkpoint_db)

    assert result["confirmed_mapping"] == confirmed
    logged = json.loads(isolated_review_log.read_text().strip())
    assert logged["confirmed_mapping"] == confirmed
    assert logged["suggested_mapping"] == suggested


def test_human_correction_overrides_a_wrong_suggestion(checkpoint_db, isolated_review_log):
    # "Office" would otherwise suggest-match Location -- the human instead points it at
    # nothing and manually maps a differently-named column to Location.
    thread_id = f"test-{uuid.uuid4()}"
    state = _state(raw_columns=["Employee ID", "Job Title", "Department", "Office", "Work Site", "Currency", "Base Salary"])
    start_column_mapping_review(state, thread_id, db_path=checkpoint_db)

    confirmed = {c: None for c in TARGET_COLUMNS}
    confirmed.update({"Emp ID": "Employee ID", "Job Title": "Job Title", "Dept": "Department", "Location": "Work Site", "Curr": "Currency", "Base": "Base Salary"})
    result = resume_column_mapping_review(thread_id, confirmed_mapping=confirmed, db_path=checkpoint_db)

    assert result["confirmed_mapping"]["Location"] == "Work Site"
    logged = json.loads(isolated_review_log.read_text().strip())
    assert logged["suggested_mapping"]["Location"] == "Office"  # what was proposed
    assert logged["confirmed_mapping"]["Location"] == "Work Site"  # what the human actually chose


def test_every_completed_review_is_logged(checkpoint_db, isolated_review_log):
    for i in range(2):
        thread_id = f"test-{uuid.uuid4()}"
        start_column_mapping_review(_state(source_name=f"upload-{i}.xlsx"), thread_id, db_path=checkpoint_db)
        confirmed = {c: None for c in TARGET_COLUMNS}
        confirmed.update({c: c for c in REQUIRED_COLUMNS})  # the default state's raw_columns == TARGET_COLUMNS
        resume_column_mapping_review(thread_id, confirmed_mapping=confirmed, db_path=checkpoint_db)

    lines = isolated_review_log.read_text().strip().splitlines()
    assert len(lines) == 2


def test_missing_required_column_blocks_and_reprompts_instead_of_logging(checkpoint_db, isolated_review_log):
    # Every required column left unmapped -- validate_node must refuse to log this and pause
    # again rather than silently accepting an unusable mapping.
    thread_id = f"test-{uuid.uuid4()}"
    start_column_mapping_review(_state(), thread_id, db_path=checkpoint_db)

    confirmed = {c: None for c in TARGET_COLUMNS}
    result = resume_column_mapping_review(thread_id, confirmed_mapping=confirmed, db_path=checkpoint_db)

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert set(payload["missing_required"]) == set(REQUIRED_COLUMNS)
    # The reprompt shows the reviewer's own (incomplete) submission back, not the original
    # suggestion -- nothing they already got right should need re-entering.
    assert payload["suggested_mapping"] == confirmed
    assert not isolated_review_log.exists() or isolated_review_log.read_text().strip() == ""


def test_fixing_the_missing_required_column_after_a_reprompt_then_logs(checkpoint_db, isolated_review_log):
    thread_id = f"test-{uuid.uuid4()}"
    start_column_mapping_review(_state(), thread_id, db_path=checkpoint_db)

    incomplete = {c: None for c in TARGET_COLUMNS}
    blocked = resume_column_mapping_review(thread_id, confirmed_mapping=incomplete, db_path=checkpoint_db)
    assert "__interrupt__" in blocked

    complete = {**incomplete, **{c: c for c in REQUIRED_COLUMNS}}
    result = resume_column_mapping_review(thread_id, confirmed_mapping=complete, db_path=checkpoint_db)

    assert "__interrupt__" not in result
    assert result["confirmed_mapping"] == complete
    logged = json.loads(isolated_review_log.read_text().strip())
    assert logged["confirmed_mapping"] == complete


def test_resume_is_the_same_thread_the_pause_started(checkpoint_db):
    with pytest.raises(Exception):
        resume_column_mapping_review(f"never-started-{uuid.uuid4()}", confirmed_mapping={}, db_path=checkpoint_db)
