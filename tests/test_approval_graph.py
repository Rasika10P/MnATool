"""Tests for the write-approval gate (agents/approval_graph.py) -- build order item 5, gate
4: "final approval before any write" to leveling_decisions. Isolates both the LangGraph
checkpoint db and the DuckDB decisions db to tmp_path, same convention as
tests/test_decisions.py's db_path fixture, by monkeypatching agents.approval_graph's
resolved-at-call-time DECISIONS_DB_PATH attribute.
"""

import uuid

import duckdb
import pytest

import agents.approval_graph as approval_graph
from agents.approval_graph import resume_approval, start_approval


@pytest.fixture(autouse=True)
def isolated_decisions_db(tmp_path, monkeypatch):
    monkeypatch.setattr(approval_graph, "DECISIONS_DB_PATH", tmp_path / "comp.duckdb")
    return tmp_path / "comp.duckdb"


@pytest.fixture
def checkpoint_db(tmp_path):
    return tmp_path / "approval_checkpoints.sqlite"


def _state(**overrides) -> dict:
    base = {
        "job_or_employee_ref": "NYX-001",
        "assigned_level": "L5",
        "confidence": 0.82,
        "factor_ratings": [{"factor": "scope_of_impact", "level_indicated": "L5"}],
        "factor5_variant_applied": "5a",
        "alternative_considered": None,
        "governing_rule": "rule 2: lower level governs a split",
        "source_document_hash": "abc123",
        "negotiation_context": None,
        "reviewer_verdict": None,
        "final_level": None,
        "written_record": None,
    }
    base.update(overrides)
    return base


def test_start_approval_pauses_with_full_context(checkpoint_db):
    thread_id = f"test-{uuid.uuid4()}"
    result = start_approval(_state(), thread_id, db_path=checkpoint_db)

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["employee_id"] == "NYX-001"
    assert payload["assigned_level"] == "L5"
    assert payload["governing_rule"] == "rule 2: lower level governs a split"


def test_approved_writes_to_leveling_decisions(checkpoint_db, isolated_decisions_db):
    thread_id = f"test-{uuid.uuid4()}"
    start_approval(_state(), thread_id, db_path=checkpoint_db)
    result = resume_approval(thread_id, verdict="approved", db_path=checkpoint_db)

    assert result["written_record"] is not None
    assert result["final_level"] == "L5"

    con = duckdb.connect(str(isolated_decisions_db))
    rows = con.execute(
        "SELECT assigned_level, reviewer_verdict FROM leveling_decisions WHERE job_or_employee_ref = ?",
        ["NYX-001"],
    ).fetchall()
    con.close()
    assert rows == [("L5", "approved")]


def test_approved_with_override_writes_overridden_level(checkpoint_db, isolated_decisions_db):
    thread_id = f"test-{uuid.uuid4()}"
    start_approval(_state(), thread_id, db_path=checkpoint_db)
    result = resume_approval(thread_id, verdict="approved_with_override", override_level="L6", db_path=checkpoint_db)

    assert result["final_level"] == "L6"
    con = duckdb.connect(str(isolated_decisions_db))
    rows = con.execute("SELECT assigned_level FROM leveling_decisions WHERE job_or_employee_ref = ?", ["NYX-001"]).fetchall()
    con.close()
    assert rows == [("L6",)]


def test_rejected_never_touches_leveling_decisions(checkpoint_db, isolated_decisions_db):
    thread_id = f"test-{uuid.uuid4()}"
    start_approval(_state(), thread_id, db_path=checkpoint_db)
    result = resume_approval(thread_id, verdict="rejected", db_path=checkpoint_db)

    assert result["written_record"] is None
    assert not isolated_decisions_db.exists()


def test_resume_is_the_same_thread_the_pause_started(checkpoint_db, isolated_decisions_db):
    """A resume against a thread_id that never paused has no checkpoint to resume from --
    confirms thread_id, not the state payload, is what ties a pause to its resume."""
    with pytest.raises(Exception):
        resume_approval(f"never-started-{uuid.uuid4()}", verdict="approved", db_path=checkpoint_db)
