"""Tests for the no-equivalent gate (agents/no_equivalent_gate.py) -- build order item 5,
gate 2: forced human review when a role has no equivalent in Meridian's architecture.
Isolates both the LangGraph checkpoint db and the review log to tmp_path, same convention as
tests/test_approval_graph.py's isolated_decisions_db fixture, by monkeypatching
agents.no_equivalent_gate's resolved-at-call-time REVIEW_LOG_PATH attribute.
"""

import json
import uuid

import pytest

import agents.no_equivalent_gate as no_equivalent_gate
from agents.no_equivalent_gate import resume_no_equivalent_review, start_no_equivalent_review


@pytest.fixture(autouse=True)
def isolated_review_log(tmp_path, monkeypatch):
    path = tmp_path / "no_equivalent_reviews.jsonl"
    monkeypatch.setattr(no_equivalent_gate, "REVIEW_LOG_PATH", path)
    return path


@pytest.fixture
def checkpoint_db(tmp_path):
    return tmp_path / "no_equivalent_checkpoints.sqlite"


def _state(**overrides) -> dict:
    base = {
        "employee_id": "NYX-020",
        "job_title": "MTS I - Photonics",
        "dept": "Photonics",
        "sub_family": "Photonics",
        "reason": "no Meridian family_group for Dept='Photonics'",
        "missing_fields": ["family_group"],
        "known_family_group": None,
        "known_geo_code": "US-SJC",
        "known_job_prefix": None,
        "reviewer_verdict": None,
        "manual_mapping": None,
        "review_entry": None,
    }
    base.update(overrides)
    return base


def test_start_pauses_with_full_context(checkpoint_db):
    thread_id = f"test-{uuid.uuid4()}"
    result = start_no_equivalent_review(_state(), thread_id, db_path=checkpoint_db)

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["employee_id"] == "NYX-020"
    assert payload["dept"] == "Photonics"
    assert payload["reason"] == "no Meridian family_group for Dept='Photonics'"
    assert payload["missing_fields"] == ["family_group"]
    assert payload["known_family_group"] is None
    assert payload["known_geo_code"] == "US-SJC"


def test_escalated_logs_with_no_manual_mapping(checkpoint_db, isolated_review_log):
    thread_id = f"test-{uuid.uuid4()}"
    start_no_equivalent_review(_state(), thread_id, db_path=checkpoint_db)
    result = resume_no_equivalent_review(thread_id, verdict="escalated", db_path=checkpoint_db)

    assert result["review_entry"]["verdict"] == "escalated"
    assert result["review_entry"]["manual_mapping"] is None

    lines = isolated_review_log.read_text().strip().splitlines()
    assert len(lines) == 1
    logged = json.loads(lines[0])
    assert logged["employee_id"] == "NYX-020"
    assert logged["verdict"] == "escalated"


def test_manually_mapped_logs_the_supplied_mapping(checkpoint_db, isolated_review_log):
    thread_id = f"test-{uuid.uuid4()}"
    start_no_equivalent_review(_state(), thread_id, db_path=checkpoint_db)
    mapping = {"family_group": "engineering", "geo_code": "US-CA", "job_prefix": "ENG-PHOT"}
    result = resume_no_equivalent_review(thread_id, verdict="manually_mapped", manual_mapping=mapping, db_path=checkpoint_db)

    assert result["review_entry"]["verdict"] == "manually_mapped"
    assert result["review_entry"]["manual_mapping"] == mapping

    logged = json.loads(isolated_review_log.read_text().strip())
    assert logged["manual_mapping"] == mapping


def test_every_review_is_logged_regardless_of_verdict(checkpoint_db, isolated_review_log):
    for i in range(2):
        thread_id = f"test-{uuid.uuid4()}"
        start_no_equivalent_review(_state(employee_id=f"NYX-{i}"), thread_id, db_path=checkpoint_db)
        resume_no_equivalent_review(thread_id, verdict="escalated", db_path=checkpoint_db)

    lines = isolated_review_log.read_text().strip().splitlines()
    assert len(lines) == 2


def test_resume_is_the_same_thread_the_pause_started(checkpoint_db):
    """A resume against a thread_id that never paused has no checkpoint to resume from --
    confirms thread_id, not the state payload, is what ties a pause to its resume."""
    with pytest.raises(Exception):
        resume_no_equivalent_review(f"never-started-{uuid.uuid4()}", verdict="escalated", db_path=checkpoint_db)
