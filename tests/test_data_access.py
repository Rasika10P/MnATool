import pandas as pd
import pytest

from tools.data_access import (
    NoMarketDataError,
    check_internal_equity,
    list_family_level_incumbent_locations,
    lookup_market_data,
    lookup_market_percentile,
    lookup_salary_structure,
    read_job_architecture,
)


def test_read_job_architecture_known_job():
    result = read_job_architecture("PD-SI-L5")
    assert result["inputs"] == {"job_id": "PD-SI-L5"}
    assert result["job"]["job_id"] == "PD-SI-L5"
    assert result["job"]["family"] == "Physical Design"
    assert result["job"]["level_code"] == "L5"


def test_read_job_architecture_unknown_job_raises():
    with pytest.raises(ValueError, match="No job_catalog row"):
        read_job_architecture("NOT-A-REAL-JOB")


def test_lookup_market_data_happy_path():
    result = lookup_market_data(
        family_group="engineering", level_code="L5", geo_code="US-SJC",
        as_of_date="2026-08-01", annual_growth_rate=0.035,
    )
    assert result["p50"] > 0
    assert len(result["source_rows"]) >= 1
    assert result["inputs"]["family_group"] == "engineering"


def test_lookup_market_data_empty_raises():
    # gtm family has no M7-equivalent survey coverage at every geo; corporate has no L1 at all --
    # pick a combination guaranteed absent from job_catalog entirely.
    with pytest.raises(NoMarketDataError):
        lookup_market_data(
            family_group="corporate", level_code="L1", geo_code="US-SJC",
            as_of_date="2026-08-01", annual_growth_rate=0.035,
        )


def test_check_internal_equity_known_job():
    # Use a job/geo combo that should have at least one incumbent in the submission population.
    from pathlib import Path
    incumbents = pd.read_parquet(Path("data/parquet/incumbents.parquet"))
    row = incumbents.iloc[0]
    result = check_internal_equity(job_id=row.job_id, geo_code=row.geo_code, candidate_salary=row.base_salary)
    assert result["peer_count"] >= 1
    assert result["candidate_compa_ratio"] is not None


def test_check_internal_equity_no_peers_returns_empty_not_error():
    result = check_internal_equity(job_id="PD-SI-L5", geo_code="US-SJC", candidate_salary=999_999_999)
    # candidate_salary is nonsense but the job/geo combo may just have zero incumbents --
    # either way this must not raise.
    assert "peer_count" in result


def test_lookup_salary_structure_known_combo():
    result = lookup_salary_structure(family_group="engineering", level_code="L7", geo_code="US-SJC")
    assert result["inputs"] == {"family_group": "engineering", "level_code": "L7", "geo_code": "US-SJC"}
    assert result["structure"]["range_min"] < result["structure"]["range_mid"] < result["structure"]["range_max"]


def test_lookup_salary_structure_unknown_combo_raises():
    with pytest.raises(ValueError, match="No salary_structures row"):
        lookup_salary_structure(family_group="not-a-real-family", level_code="L7", geo_code="US-SJC")


def test_list_family_level_incumbent_locations_l7_engineering():
    # This dataset's real L7 population: 3 incumbents company-wide, each in a different
    # sub-family (job_id) and a different geo -- exactly the sparsity that motivates
    # agents/equity_gate.py aggregating across the whole family_group rather than relying on
    # a single job_id+geo match.
    pairs = list_family_level_incumbent_locations(family_group="engineering", level_code="L7")
    assert set(pairs) == {("SA-ARCH-L7", "IN-BLR"), ("PD-STA-L7", "US-SJC"), ("SV-TE-L7", "EU-MUC")}


def test_list_family_level_incumbent_locations_no_incumbents_returns_empty():
    pairs = list_family_level_incumbent_locations(family_group="corporate", level_code="L1")
    assert pairs == []


def test_lookup_market_percentile_returns_a_positive_value_for_a_known_job():
    result = lookup_market_percentile(
        job_id="DD-UARCH-L5", geo_code="US-SJC", target_percentile=50,
        as_of_date="2026-08-01", annual_growth_rate=0.035,
    )
    assert result["value"] > 0
    assert result["inputs"]["job_id"] == "DD-UARCH-L5"


def test_lookup_market_percentile_is_scoped_to_job_id_not_blended_across_family_group():
    # engineering/L5/US-SJC spans 9 distinct job_ids (DD-RTL-L5, DD-UARCH-L5, ANA-AD-L5, ...),
    # each with its own survey_code_primary -- lookup_market_data's family_group+level_code
    # join blends all 9 together, which is wrong for one specific employee's specific job.
    # lookup_market_percentile must return exactly DD-UARCH-L5's own row, not the 9-way blend.
    # (This synthetic dataset's P50 happens to be identical across all 9 sub-families at this
    # particular level+geo, so the *value* can't distinguish scoped from blended here -- the
    # source_rows count is the real, reliable proof that scoping is actually job_id-exact and
    # not silently falling back to the family-wide join.)
    scoped = lookup_market_percentile(
        job_id="DD-UARCH-L5", geo_code="US-SJC", target_percentile=50,
        as_of_date="2026-08-01", annual_growth_rate=0.035,
    )
    blended = lookup_market_data(
        family_group="engineering", level_code="L5", geo_code="US-SJC",
        as_of_date="2026-08-01", annual_growth_rate=0.035,
    )
    assert len(scoped["source_rows"]) == 1
    assert scoped["source_rows"][0]["survey_code"] == "SYN-044-DD-UARCH-L5"
    assert len(blended["source_rows"]) == 9


def test_lookup_market_percentile_no_data_raises():
    with pytest.raises(NoMarketDataError):
        lookup_market_percentile(
            job_id="NOT-A-REAL-JOB", geo_code="US-SJC", target_percentile=60,
            as_of_date="2026-08-01", annual_growth_rate=0.035,
        )
