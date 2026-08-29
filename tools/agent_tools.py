"""LangChain @tool wrappers around the deterministic functions in tools/ -- the six tools
named in ASSIGNMENT.md's framework table (read_job_architecture, lookup_market_data,
convert_currency, compute_pay_metrics, check_internal_equity, write_mapping_decision).

The plain functions in comp_math.py, currency.py, data_access.py and decisions.py remain the
actual implementation, and every existing call site in this codebase (agents/, app/, tests/)
keeps calling them directly, unwrapped. These wrappers exist only so an LLM tool-calling loop
(a model bound via .bind_tools([...])) has something with a JSON-primitive signature and a
docstring written for a model rather than for a caller who already knows the data model --
e.g. convert_currency's real implementation takes an already-loaded fx_rates DataFrame, which
an LLM tool call can't supply; the wrapper loads it from disk instead.

write_mapping_decision is wrapped here for completeness (ASSIGNMENT.md names it as one of the
six) but is never bound to a model anywhere in this codebase. CLAUDE.md's non-negotiable
"write actions get a human" means the only path to this write is agents/approval_graph.py's
interrupt() gate -- an autonomous tool-calling loop must not be handed this tool.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from tools import comp_math, currency, data_access, decisions

DEFAULT_DATA_DIR = data_access.DEFAULT_DATA_DIR
DEFAULT_FX_RATES_PATH = DEFAULT_DATA_DIR / "fx_rates.parquet"


@tool
def read_job_architecture(job_id: str) -> dict:
    """Look up one Meridian job's canonical definition -- job_catalog joined with
    level_definitions (title, track, IC-equivalent, target bonus, equity tier) -- by job_id,
    e.g. 'DD-RTL-L5'. Raises if job_id doesn't exist."""
    return data_access.read_job_architecture(job_id)


@tool
def lookup_market_data(
    family_group: str,
    level_code: str,
    geo_code: str,
    as_of_date: str,
    annual_growth_rate: float,
    pay_element: str = "base",
) -> dict:
    """Market P50 pay for a (family_group, level_code, geo_code) slice, aged from each
    matching survey source's own effective date to as_of_date (YYYY-MM-DD) at
    annual_growth_rate. Raises if no survey data matches this exact slice -- never guesses or
    substitutes a nearby one."""
    return data_access.lookup_market_data(
        family_group, level_code, geo_code, as_of_date, annual_growth_rate, pay_element
    )


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str, as_of_date: str) -> dict:
    """Convert an amount between currencies using the dated FX rate for the calendar month
    containing as_of_date (YYYY-MM-DD). Raises if that exact month has no rate on file --
    never interpolates or falls back to a different month."""
    fx_rates = pd.read_parquet(DEFAULT_FX_RATES_PATH)
    return currency.convert_currency(amount, from_currency, to_currency, as_of_date, fx_rates)


@tool
def compute_pay_metrics(salary: float, range_min: float, range_mid: float, range_max: float) -> dict:
    """Compa-ratio, range penetration, and cost-to-minimum for one salary against one salary
    range (range_min/range_mid/range_max from a salary structure)."""
    return comp_math.compute_pay_metrics(salary, range_min, range_mid, range_max)


@tool
def check_internal_equity(job_id: str, geo_code: str, candidate_salary: float) -> dict:
    """Compare candidate_salary against current Meridian incumbents in the same job_id and
    geo_code. Returns the peer compa-ratio distribution and where the candidate would land in
    it -- reports the facts, does not itself decide whether a gap is acceptable."""
    return data_access.check_internal_equity(job_id, geo_code, candidate_salary)


@tool
def write_mapping_decision(
    job_or_employee_ref: str,
    assigned_level: str,
    confidence: float,
    factor_ratings: dict,
    governing_rule: str | None = None,
    reviewer_verdict: str | None = None,
) -> dict:
    """Persist one leveling decision to leveling_decisions. Do not call this from an
    autonomous tool-calling loop -- every write must go through the human approval gate
    (agents.approval_graph) first."""
    return decisions.write_mapping_decision(
        job_or_employee_ref=job_or_employee_ref,
        assigned_level=assigned_level,
        confidence=confidence,
        factor_ratings=factor_ratings,
        governing_rule=governing_rule,
        reviewer_verdict=reviewer_verdict,
    )
