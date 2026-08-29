"""
write_mapping_decision -- the one write tool among the six (ASSIGNMENT.md). Persists a
leveling decision to data/comp.duckdb, per the schema in docs/data_model_spec.md section 2.

This function is the deterministic write action itself, not the approval gate in front of
it. The gate is a LangGraph interrupt() (build order item 5) that calls this only after a
human approves -- nothing here decides whether a write is allowed.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "comp.duckdb"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS leveling_decisions (
    decision_id TEXT PRIMARY KEY,
    job_or_employee_ref TEXT NOT NULL,
    assigned_level TEXT NOT NULL,
    confidence DOUBLE NOT NULL,
    factor_ratings TEXT NOT NULL,
    factor5_variant_applied TEXT,
    alternative_considered TEXT,
    governing_rule TEXT,
    reviewer_verdict TEXT,
    source_document_hash TEXT,
    created_at TIMESTAMP NOT NULL
)
"""


def write_mapping_decision(
    job_or_employee_ref: str,
    assigned_level: str,
    confidence: float,
    factor_ratings: dict,
    factor5_variant_applied: str | None = None,
    alternative_considered: str | None = None,
    governing_rule: str | None = None,
    reviewer_verdict: str | None = None,
    source_document_hash: str | None = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict:
    """Insert one leveling decision. Returns the full persisted record (including the
    generated decision_id and created_at) as confirmation of what was actually written."""
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be in [0.0, 1.0]; got {confidence}")

    decision_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute(_SCHEMA)
        con.execute(
            """
            INSERT INTO leveling_decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                decision_id,
                job_or_employee_ref,
                assigned_level,
                confidence,
                json.dumps(factor_ratings),
                factor5_variant_applied,
                alternative_considered,
                governing_rule,
                reviewer_verdict,
                source_document_hash,
                created_at,
            ],
        )
    finally:
        con.close()

    return {
        "inputs": {
            "job_or_employee_ref": job_or_employee_ref,
            "assigned_level": assigned_level,
            "confidence": confidence,
            "factor_ratings": factor_ratings,
        },
        "decision_id": decision_id,
        "created_at": created_at,
        "db_path": str(db_path),
    }
