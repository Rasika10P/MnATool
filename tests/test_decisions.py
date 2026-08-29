import duckdb
import pytest

from tools.decisions import write_mapping_decision


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "comp.duckdb"


def test_write_mapping_decision_persists(db_path):
    result = write_mapping_decision(
        job_or_employee_ref="NYX-001",
        assigned_level="L5",
        confidence=0.82,
        factor_ratings={"scope": 4, "autonomy": 4, "complexity": 3},
        governing_rule="rule_2",
        db_path=db_path,
    )
    assert result["decision_id"]
    assert result["inputs"]["assigned_level"] == "L5"

    con = duckdb.connect(str(db_path))
    rows = con.execute("SELECT * FROM leveling_decisions WHERE decision_id = ?", [result["decision_id"]]).fetchdf()
    con.close()
    assert len(rows) == 1
    assert rows.iloc[0].job_or_employee_ref == "NYX-001"


def test_write_mapping_decision_bad_confidence_raises(db_path):
    with pytest.raises(ValueError, match="confidence"):
        write_mapping_decision(
            job_or_employee_ref="NYX-002", assigned_level="L4", confidence=1.5,
            factor_ratings={}, db_path=db_path,
        )


def test_write_mapping_decision_appends_not_overwrites(db_path):
    write_mapping_decision("NYX-003", "L4", 0.7, {}, db_path=db_path)
    write_mapping_decision("NYX-004", "L5", 0.9, {}, db_path=db_path)
    con = duckdb.connect(str(db_path))
    count = con.execute("SELECT COUNT(*) FROM leveling_decisions").fetchone()[0]
    con.close()
    assert count == 2
