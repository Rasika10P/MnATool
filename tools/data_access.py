"""
Read-only lookups against the parquet data layer, via DuckDB (locked stack: DuckDB + Parquet).

Every function returns its inputs and its source rows alongside the answer -- the provenance
rule applies to lookups as much as to math (CLAUDE.md non-negotiable 2).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from tools.comp_math import age_market_value, interpolate_percentile

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "parquet"


class NoMarketDataError(LookupError):
    """No survey_data rows exist for the requested (family_group, level_code, geo_code) slice.

    Per ASSIGNMENT.md's error-handling table: empty market data means the pricing agent
    declines rather than guessing, and the case escalates. This exception is what a caller
    catches to trigger that -- lookup_market_data never returns a fabricated or zero figure.
    """


def _connect(data_dir: Path):
    return duckdb.connect(database=":memory:", read_only=False), data_dir


def read_job_architecture(job_id: str, data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    """Look up one job's canonical definition: its job_catalog row joined with its
    level_definitions row (title, track, ic_equivalent, bonus target, etc).
    """
    con, data_dir = _connect(data_dir)
    job_path = data_dir / "job_catalog.parquet"
    level_path = data_dir / "level_definitions.parquet"

    rows = con.execute(
        """
        SELECT j.*, l.track, l.level_title, l.ic_equivalent, l.sort_order,
               l.target_bonus_pct, l.equity_tier
        FROM read_parquet(?) j
        JOIN read_parquet(?) l ON j.level_code = l.level_code
        WHERE j.job_id = ?
        """,
        [str(job_path), str(level_path), job_id],
    ).fetchdf()

    if rows.empty:
        raise ValueError(f"No job_catalog row for job_id={job_id!r}")

    return {
        "inputs": {"job_id": job_id},
        "job": rows.iloc[0].to_dict(),
        "source": {"job_catalog": str(job_path), "level_definitions": str(level_path)},
    }


def lookup_market_data(
    family_group: str,
    level_code: str,
    geo_code: str,
    as_of_date,
    annual_growth_rate: float,
    pay_element: str = "base",
    data_dir: Path = DEFAULT_DATA_DIR,
) -> dict:
    """Market P50 for (family_group, level_code, geo_code), aged to `as_of_date` at the
    caller-supplied `annual_growth_rate` (a comp-philosophy assumption, not something this
    function invents). Averages across every matching survey source, ages each independently
    from its own effective_date, then averages the aged values -- so a stale source and a
    current one don't get blended before aging accounts for the gap between them.

    Raises NoMarketDataError if nothing matches -- see that class's docstring.
    """
    con, data_dir = _connect(data_dir)
    job_path = data_dir / "job_catalog.parquet"
    survey_data_path = data_dir / "survey_data.parquet"

    matches = con.execute(
        """
        SELECT DISTINCT sd.*
        FROM read_parquet(?) j
        JOIN read_parquet(?) sd ON sd.survey_code = j.survey_code_primary
        WHERE j.family_group = ? AND j.level_code = ? AND sd.geo_code = ? AND sd.pay_element = ?
        """,
        [str(job_path), str(survey_data_path), family_group, level_code, geo_code, pay_element],
    ).fetchdf()

    if matches.empty:
        raise NoMarketDataError(
            f"No survey_data for family_group={family_group!r} level_code={level_code!r} "
            f"geo_code={geo_code!r} pay_element={pay_element!r}"
        )

    as_of = pd.Timestamp(as_of_date)
    aged_p50s = []
    for _, row in matches.iterrows():
        months = (as_of.year - row.effective_date.year) * 12 + (as_of.month - row.effective_date.month)
        aged_p50s.append(age_market_value(row.p50, months, annual_growth_rate)["aged_value"])

    return {
        "inputs": {
            "family_group": family_group,
            "level_code": level_code,
            "geo_code": geo_code,
            "as_of_date": as_of,
            "annual_growth_rate": annual_growth_rate,
            "pay_element": pay_element,
        },
        "p50": sum(aged_p50s) / len(aged_p50s),
        "source_rows": matches.to_dict("records"),
        "source": str(survey_data_path),
    }


def lookup_market_percentile(
    job_id: str,
    geo_code: str,
    target_percentile: float,
    as_of_date,
    annual_growth_rate: float,
    pay_element: str = "base",
    data_dir: Path = DEFAULT_DATA_DIR,
) -> dict:
    """Market value at an arbitrary target percentile (e.g. P60, or P65 for comp_philosophy.
    md's "target + 5 points" at L6+) for one specific job_id -- not survey_data's own fixed
    P50, and deliberately *not* lookup_market_data's family_group+level_code join. Every
    job_id has its own distinct survey_code_primary (confirmed against job_catalog: even
    within one family_group+level_code, e.g. engineering/L5, DD-RTL-L5 and DD-UARCH-L5 point
    at different survey codes) -- joining on the coarser family_group blends market data
    across sub-families that don't belong together, e.g. Digital Design with Analog Design.
    Scoped to job_id precisely instead, so that never happens here. See learnings.md.

    Interpolates each matching source row's own p25/p50/p75/p90 to target_percentile
    (tools.comp_math.interpolate_percentile), ages the interpolated value from that row's own
    effective_date (tools.comp_math.age_market_value), then averages across sources -- same
    order of operations as lookup_market_data.

    Raises NoMarketDataError if nothing matches -- same contract as lookup_market_data.
    """
    con, data_dir = _connect(data_dir)
    job_path = data_dir / "job_catalog.parquet"
    survey_data_path = data_dir / "survey_data.parquet"

    matches = con.execute(
        """
        SELECT DISTINCT sd.*
        FROM read_parquet(?) j
        JOIN read_parquet(?) sd ON sd.survey_code = j.survey_code_primary
        WHERE j.job_id = ? AND sd.geo_code = ? AND sd.pay_element = ?
        """,
        [str(job_path), str(survey_data_path), job_id, geo_code, pay_element],
    ).fetchdf()

    if matches.empty:
        raise NoMarketDataError(
            f"No survey_data for job_id={job_id!r} geo_code={geo_code!r} pay_element={pay_element!r}"
        )

    as_of = pd.Timestamp(as_of_date)
    aged_values = []
    for _, row in matches.iterrows():
        interpolated = interpolate_percentile(row.p25, row.p50, row.p75, row.p90, target_percentile)["value"]
        months = (as_of.year - row.effective_date.year) * 12 + (as_of.month - row.effective_date.month)
        aged_values.append(age_market_value(interpolated, months, annual_growth_rate)["aged_value"])

    return {
        "inputs": {
            "job_id": job_id,
            "geo_code": geo_code,
            "target_percentile": target_percentile,
            "as_of_date": as_of,
            "annual_growth_rate": annual_growth_rate,
            "pay_element": pay_element,
        },
        "value": sum(aged_values) / len(aged_values),
        "source_rows": matches.to_dict("records"),
        "source": str(survey_data_path),
    }


def check_internal_equity(
    job_id: str,
    geo_code: str,
    candidate_salary: float,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> dict:
    """Compare `candidate_salary` against current incumbents in the same job_id and geo_code.
    Returns the peer compa-ratio distribution and where the candidate would land in it --
    the leveling/pricing agent decides what to do with a compression or inversion signal;
    this only reports the peer facts.
    """
    con, data_dir = _connect(data_dir)
    incumbents_path = data_dir / "incumbents.parquet"
    structures_path = data_dir / "salary_structures.parquet"
    job_path = data_dir / "job_catalog.parquet"

    peers = con.execute(
        """
        SELECT i.employee_id, i.base_salary, s.range_min, s.range_mid, s.range_max
        FROM read_parquet(?) i
        JOIN read_parquet(?) j ON j.job_id = i.job_id
        JOIN read_parquet(?) s
          ON s.geo_code = i.geo_code AND s.level_code = i.level_code AND s.family_group = j.family_group
        WHERE i.job_id = ? AND i.geo_code = ?
        """,
        [str(incumbents_path), str(job_path), str(structures_path), job_id, geo_code],
    ).fetchdf()

    inputs = {"job_id": job_id, "geo_code": geo_code, "candidate_salary": candidate_salary}

    if peers.empty:
        return {
            "inputs": inputs,
            "peer_count": 0,
            "peer_compa_ratios": [],
            "candidate_compa_ratio": None,
            "source_rows": [],
        }

    peer_compa_ratios = (peers.base_salary / peers.range_mid).tolist()
    candidate_compa_ratio = candidate_salary / peers.iloc[0].range_mid

    return {
        "inputs": inputs,
        "peer_count": len(peers),
        "peer_compa_ratios": peer_compa_ratios,
        "peer_compa_ratio_min": min(peer_compa_ratios),
        "peer_compa_ratio_max": max(peer_compa_ratios),
        "candidate_compa_ratio": candidate_compa_ratio,
        "below_all_peers": candidate_salary < peers.base_salary.min(),
        "above_all_peers": candidate_salary > peers.base_salary.max(),
        "source_rows": peers.to_dict("records"),
    }


def lookup_salary_structure(
    family_group: str,
    level_code: str,
    geo_code: str,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> dict:
    """One salary_structures row for (family_group, level_code, geo_code) -- looked up
    directly, independent of whether any incumbent currently holds that exact job.
    check_internal_equity can't compute a compa-ratio unless it already has a peer row to
    join range_mid through; this is what the equity gate uses instead to price a candidate
    who may be the first person proposed at that level in their geo (agents/equity_gate.py).
    """
    con, data_dir = _connect(data_dir)
    structures_path = data_dir / "salary_structures.parquet"

    rows = con.execute(
        "SELECT * FROM read_parquet(?) WHERE family_group = ? AND level_code = ? AND geo_code = ?",
        [str(structures_path), family_group, level_code, geo_code],
    ).fetchdf()

    if rows.empty:
        raise ValueError(
            f"No salary_structures row for family_group={family_group!r} "
            f"level_code={level_code!r} geo_code={geo_code!r}"
        )

    return {
        "inputs": {"family_group": family_group, "level_code": level_code, "geo_code": geo_code},
        "structure": rows.iloc[0].to_dict(),
        "source": str(structures_path),
    }


def list_family_level_incumbent_locations(
    family_group: str,
    level_code: str,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> list[tuple[str, str]]:
    """Every distinct (job_id, geo_code) pair with at least one Meridian incumbent at this
    family_group and level -- the set of check_internal_equity calls needed to see every
    peer in the family, not just whichever single sub-family+geo a caller happens to name.
    Senior levels are sparse enough (L7 is 3 people company-wide in this dataset, each in a
    different sub-family *and* a different geo) that a single job_id+geo call routinely
    finds 0-1 peers even when the family genuinely has several people at that level --
    agents/equity_gate.py aggregates across every pair this returns rather than relying on
    one exact match.
    """
    con, data_dir = _connect(data_dir)
    incumbents_path = data_dir / "incumbents.parquet"
    job_path = data_dir / "job_catalog.parquet"

    rows = con.execute(
        """
        SELECT DISTINCT i.job_id, i.geo_code
        FROM read_parquet(?) i
        JOIN read_parquet(?) j ON j.job_id = i.job_id
        WHERE j.family_group = ? AND j.level_code = ?
        """,
        [str(incumbents_path), str(job_path), family_group, level_code],
    ).fetchdf()

    return list(rows.itertuples(index=False, name=None))
