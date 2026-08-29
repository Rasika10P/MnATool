"""Orchestration for the Streamlit crosswalk page: loads the Nyx census, runs leveling
across the population, runs the full negotiation subgraph on contested mappings, and runs
cost/retention/synthesis modeling. Reuses the existing agents and tools untouched -- no new
agent or comp logic lives here, only the glue to call them over a real population and hold
the results for the UI.

Mapping tables below (geo codes, Meridian family_group, job sub-family prefixes) are the
same ones already committed in scripts/negotiation_nyx_011.py and scripts/modeling_demo.py,
reused verbatim rather than re-derived. Photonics (Dept) and "Engineering Manager" (sub-
family) have no entry, on purpose: CLAUDE.md's build log already documents the Photonics
group as having no Meridian equivalent, and no sub-family/job mapping for the Engineering
Manager titles has been established anywhere in this codebase. Employees in either bucket
still get leveled (leveling needs no mapping, only the job description text), but are
excluded from negotiation and cost/retention modeling and surfaced as needing human review
instead -- the one-liner's "hands off to a human when a role has no equivalent in our
architecture," not a mapping invented here to force a number through.

Per-employee failures (a structured-output call that exhausts its retries, a missing
salary_structures row, a negotiated level with no matching job code, missing census
currency) are caught individually so one bad row can't take down the whole run --
error_handling_backlog.md documents this as the intended contract, not yet implemented at
the agent layer, so this module does it at the orchestration layer instead.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from agents.advocate import contest_mapping
from agents.cost_logging import PRICING
from agents.leveling_batch_graph import run_batch_streaming
from agents.model_router import get_model
from agents.modeling_graph import run_modeling
from agents.negotiation_graph import run_negotiation
from agents.schemas import SourceOrgContext
from agents.scope_extraction import extract_scope_profile_with_claude_fallback
from tools.data_access import lookup_salary_structure

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "parquet"
CENSUS_PATH = DATA_DIR / "nyx_census.xlsx"
ACQUISITION_CONTEXT_PATH = DATA_DIR / "acquisition_context.parquet"

# Deal reference date -- same constant scripts/modeling_demo.py already established, reused
# rather than substituted with "today" so every run prices against the same point in time.
AS_OF_DATE = "2026-08-01"

# scripts/modeling_demo.py's LOCATION_TO_GEO_CODE, unioned with negotiation_nyx_011.py's
# NYX_CITY_TO_GEO_CODE (a strict subset of it).
GEO_CODE_BY_LOCATION = {
    "San Jose, CA": "US-SJC",
    "San Jose": "US-SJC",
    "Bangalore": "IN-BLR",
    "Eindhoven": "EU-EIN",
}
# scripts/modeling_demo.py's FAMILY_TO_MERIDIAN_FAMILY_GROUP. Photonics has no entry.
FAMILY_GROUP_BY_DEPT = {
    "Digital Design": "engineering",
    "Analog & Mixed-Signal": "engineering",
}
# scripts/modeling_demo.py's SUB_FAMILY_TO_JOB_PREFIX. "Engineering Manager" has no entry.
JOB_PREFIX_BY_SUBFAMILY = {
    "RTL Design": "DD-RTL",
    "Microarchitecture": "DD-UARCH",
    "Analog Design": "ANA-AD",
    "RF": "ANA-RF",
}

# The Nyx census's own column order (data/generate.py's build_nyx_census) -- what an upload
# is validated against and what the downloadable template offers.
CENSUS_COLUMNS = [
    "Emp ID", "Job Title", "Dept", "Location", "Curr", "Base", "Bonus",
    "Unvested Options", "Start", "Role Summary",
]

# docs/nyx_level_framework.md section 2: one ladder, five levels, individual-contributor in
# name and in practice, no separate manager track. Fellow (section 2) is an honorific held
# atop Principal or Distinguished MTS, not a sixth level -- listed separately so it isn't
# mistaken for a level the way it deliberately isn't one in the source document.
NYX_LADDER = [
    "MTS I", "MTS II", "Senior MTS", "Principal MTS", "Distinguished MTS",
]
NYX_LADDER_NOTE = (
    "No manager track -- a manager keeps their MTS level and adds a functional label "
    "(e.g. \"Senior MTS, Engineering Manager\"). \"Fellow\" is an honorific atop Principal "
    "or Distinguished MTS, not a sixth level."
)

ProgressCallback = Optional[Callable[[int, int, str], None]]


def _split_title(title: str) -> tuple[str, str]:
    """(nyx_level, sub_family) from a Nyx census title. The census's title format is
    deliberately inconsistent (data/generate.py's NYX_ROSTER docstring) -- some rows use
    " - " ("MTS I - RTL Design"), others use ", " ("Distinguished MTS, RF"). Handles both,
    combining what the two existing scripts each did separately (negotiation_nyx_011.py's
    dash-only level split, modeling_demo.py's dual-delimiter tail split) into one function
    that gets both halves right for every row in the census, including the comma-delimited
    "Distinguished MTS, RF" that a dash-only split would return whole and unsplit.
    """
    delimiter = " - " if " - " in title else ", "
    head, tail = title.split(delimiter, 1)
    return head.strip(), tail.strip()


def load_census() -> tuple[list[dict], SourceOrgContext]:
    """Every row of the Nyx census, plus the one shared source_org_context (section 6) that
    applies to the whole population -- Nyx is a single acquisition, not per-employee."""
    census = pd.read_excel(CENSUS_PATH)
    ctx_row = pd.read_parquet(ACQUISITION_CONTEXT_PATH).iloc[0]

    source_org_context = SourceOrgContext(
        source_headcount=int(ctx_row.source_headcount),
        source_stage=ctx_row.source_stage,
        source_type=ctx_row.source_type,
        org_depth=int(ctx_row.org_depth),
        platform_dependency=ctx_row.platform_dependency,
    )

    employees = []
    for _, row in census.iterrows():
        nyx_level, sub_family = _split_title(row["Job Title"])
        employees.append(
            {
                "employee_id": row["Emp ID"],
                "job_title": row["Job Title"],
                "nyx_level": nyx_level,
                "sub_family": sub_family,
                "dept": row["Dept"],
                "location": row["Location"],
                "currency": row["Curr"] if pd.notna(row["Curr"]) else None,
                "current_pay": float(row["Base"]) if pd.notna(row["Base"]) else None,
                "unvested_equity_value": (
                    float(row["Unvested Options"]) if pd.notna(row["Unvested Options"]) else None
                ),
                "role_summary": row["Role Summary"],
                "job_description": (
                    f"Job title: {row['Job Title']}. Department: {row['Dept']}. {row['Role Summary']}"
                ),
            }
        )
    return employees, source_org_context


def resolve_mapping(employee: dict) -> dict:
    """Whether this employee can proceed past leveling into negotiation/modeling. Never
    guesses a mapping that isn't already established in this codebase (see module
    docstring) -- returns mapped=False with every unmet reason listed, rather than a partial
    mapping, whenever any piece is missing."""
    reasons = []
    family_group = FAMILY_GROUP_BY_DEPT.get(employee["dept"])
    if family_group is None:
        reasons.append(f"no Meridian family_group for Dept={employee['dept']!r}")
    geo_code = GEO_CODE_BY_LOCATION.get(employee["location"])
    if geo_code is None:
        reasons.append(f"no geo_code for Location={employee['location']!r}")
    job_prefix = JOB_PREFIX_BY_SUBFAMILY.get(employee["sub_family"])
    if job_prefix is None:
        reasons.append(f"no Meridian job mapping for sub-family={employee['sub_family']!r}")
    if employee["currency"] is None:
        reasons.append("missing currency in census")
    if employee["current_pay"] is None:
        reasons.append("missing base pay in census")

    if reasons:
        return {"mapped": False, "reason": "; ".join(reasons)}
    return {
        "mapped": True,
        "family_group": family_group,
        "family": employee["dept"],
        "geo_code": geo_code,
        "job_prefix": job_prefix,
    }


def estimate_live_run_cost(employees: list[dict]) -> float:
    """Rough pre-run estimate for a full live crosswalk, shown at the Live-mode confirm step
    (app/Home.py) before a run that bypasses the cache entirely. Uses the same chars/4 token
    heuristic agents/spend_guard.py already applies for its own pre-call budget projection,
    and the same PRICING table agents/cost_logging.py logs actual spend against -- not a
    number invented for this UI, the same approximation the rest of this codebase already
    accepts for "how much will this roughly cost before we know the real usage."

    Covers the two stages every employee always goes through (leveling on Claude, scope
    extraction on Nebius). Negotiation (only contested mappings) and modeling (3 calls total,
    independent of headcount) are data-dependent -- rather than guess a contested-rate, a
    flat 1.4x multiplier is applied to the leveling+extraction subtotal, which is what a real
    live run of this exact 25-employee census actually cost relative to those two stages
    alone in this project's own development history. This is a heads-up for the confirm
    step, not a guarantee -- the spend limit is what actually caps a live run, not this
    number.
    """
    claude_model = get_model("judgment").model
    nebius_model = get_model("volume").model
    claude_rates = PRICING.get(claude_model, {"input": 0.0, "output": 0.0})
    nebius_rates = PRICING.get(nebius_model, {"input": 0.0, "output": 0.0})

    subtotal = 0.0
    for emp in employees:
        input_tokens = len(emp["job_description"]) // 4
        # Leveling: Claude, ~400 output tokens is typical for a LevelingDecision's reasoning.
        subtotal += (input_tokens / 1_000_000) * claude_rates["input"] + (400 / 1_000_000) * claude_rates["output"]
        # Scope extraction: Nebius, ~300 output tokens is typical for a ScopeProfile.
        subtotal += (input_tokens / 1_000_000) * nebius_rates["input"] + (300 / 1_000_000) * nebius_rates["output"]

    return subtotal * 1.4


def run_leveling_stage(
    employees: list[dict], source_org_context: SourceOrgContext, progress_cb: ProgressCallback = None
) -> dict[str, dict]:
    """Crosswalk pass over the population via agents.leveling_batch_graph's real Send
    fan-out (build order item 3), not a sequential loop -- every employee is dispatched as an
    independent task and runs concurrently, the same mechanics scripts/level_nyx_batch.py
    measures at ~53s for 25 employees cold. run_batch_streaming reports each employee's
    decision as soon as its task completes, which is what makes per-employee progress
    possible without giving up that parallelism (a single blocking .invoke() only returns
    once every task is done).

    Returns {employee_id: LevelingDecision.model_dump()}. A row whose structured-output call
    failed all its retries is marked {"error": str} -- either caught locally (once the
    fan-out itself is exhausted) or, if it took the whole in-flight batch down with it
    (error_handling_backlog.md entry 2, not yet fixed at the graph layer), every employee
    that never got a turn is marked the same way rather than silently missing from the
    results table.
    """
    total = len(employees)
    context_dict = source_org_context.model_dump(exclude_none=True) if source_org_context else None
    batch_employees = [
        {"employee_id": e["employee_id"], "job_description": e["job_description"], "source_org_context": context_dict}
        for e in employees
    ]

    decisions: dict[str, dict] = {}
    completed = 0

    def _on_done(entry: dict) -> None:
        nonlocal completed
        completed += 1
        decisions[entry["employee_id"]] = {k: v for k, v in entry.items() if k != "employee_id"}
        if progress_cb:
            progress_cb(completed, total, entry["employee_id"])

    error: Exception | None = None
    try:
        run_batch_streaming(batch_employees, on_employee_done=_on_done)
    except Exception as e:
        error = e

    if error is not None:
        for emp in employees:
            if emp["employee_id"] not in decisions:
                decisions[emp["employee_id"]] = {"error": f"batch run aborted before this employee completed: {error}"}

    return decisions


def _dedupe_by_round(entries: list[dict]) -> list[dict]:
    """Collapses a rounds/gate_checks list to one entry per round number, keeping the last --
    cheap insurance against the exact class of bug this was written to fix (see
    _run_negotiation_for_employee's thread_id comment), applied the same way
    error_handling_backlog.md entry 3 recommends deduping the leveling batch's decisions:
    correct regardless of whether the underlying cause is this one or something else."""
    by_round = {e["round"]: e for e in entries}
    return [by_round[n] for n in sorted(by_round)]


def _run_negotiation_for_employee(
    employee: dict, decision: dict, mapping: dict, source_org_context: SourceOrgContext
) -> dict:
    """Runs the advocate first, standalone, to learn the level it would propose -- the same
    technique scripts/negotiation_nyx_011.py used (anchor candidate_salary to the advocate's
    proposed level's range midpoint), generalized here to whichever level the advocate
    actually proposes for this employee instead of a single hardcoded constant, since this
    runs across many employees rather than one fixed case. Passing the same
    source_org_context used in the leveling stage keeps this call's internal re-crosswalk
    (inside run_negotiation) a cache hit against stage 1's decision rather than a second,
    differently-prompted call that could disagree with it.
    """
    crosswalk_level = decision["assigned_level"]
    advocate_output = contest_mapping(employee["role_summary"], employee["nyx_level"], crosswalk_level)

    if advocate_output.contests:
        structure = lookup_salary_structure(
            mapping["family_group"], advocate_output.proposed_level, mapping["geo_code"]
        )["structure"]
        candidate_salary = structure["range_mid"]
    else:
        candidate_salary = 0.0  # never reaches the equity gate on an uncontested case

    result = run_negotiation(
        case_id=f"CASE-{employee['employee_id']}",
        employee_id=employee["employee_id"],
        role_summary=employee["role_summary"],
        nyx_level=employee["nyx_level"],
        job_description=employee["job_description"],
        family_group=mapping["family_group"],
        candidate_geo_code=mapping["geo_code"],
        candidate_salary=candidate_salary,
        source_org_context=source_org_context,
        # A fresh thread_id every call, not one fixed per employee_id: NegotiationState.rounds
        # and .gate_checks use an additive (operator.add) reducer, and LangGraph's SqliteSaver
        # checkpoint for a thread_id persists on disk across process runs. A fixed thread_id
        # here meant every repeated "Run crosswalk" click -- this session, or a fresh
        # `streamlit run` days later -- kept appending to whatever that employee's rounds list
        # already held from every earlier run, rather than starting clean (confirmed directly:
        # one employee's rounds list held 10 duplicate round-1 entries after ~10 pipeline runs
        # against the same fixed thread_id). agents/leveling_batch_graph.py's
        # run_batch_streaming already uses uuid4() for the same reason.
        thread_id=f"streamlit-negotiation-{employee['employee_id']}-{uuid.uuid4()}",
    )
    result["rounds"] = _dedupe_by_round(result["rounds"])
    result["gate_checks"] = _dedupe_by_round(result["gate_checks"])
    return result


def run_negotiation_stage(
    employees: list[dict],
    decisions: dict[str, dict],
    mappings: dict[str, dict],
    source_org_context: SourceOrgContext,
    progress_cb: ProgressCallback = None,
) -> dict[str, dict]:
    """Runs the full advocate/arbiter/equity-gate subgraph for every mapped employee whose
    leveling call succeeded. Returns {employee_id: run_negotiation's result dict}, or
    {employee_id: {"error": str}} for a row that failed (caught individually, same
    continue-the-batch discipline as run_leveling_stage)."""
    eligible = [
        e
        for e in employees
        if mappings[e["employee_id"]]["mapped"] and "error" not in decisions[e["employee_id"]]
    ]
    results: dict[str, dict] = {}
    total = len(eligible)
    for i, emp in enumerate(eligible, start=1):
        if progress_cb:
            progress_cb(i, total, emp["employee_id"])
        try:
            results[emp["employee_id"]] = _run_negotiation_for_employee(
                emp, decisions[emp["employee_id"]], mappings[emp["employee_id"]], source_org_context
            )
        except Exception as e:
            results[emp["employee_id"]] = {"error": str(e)}
    return results


def _load_job_catalog_ids() -> set[str]:
    df = pd.read_parquet(DATA_DIR / "job_catalog.parquet")
    return set(df["job_id"])


def _load_salary_structure_keys() -> set[tuple[str, str, str]]:
    df = pd.read_parquet(DATA_DIR / "salary_structures.parquet")
    return set(zip(df["family_group"], df["level_code"], df["geo_code"]))


def build_modeling_population(
    employees: list[dict],
    decisions: dict[str, dict],
    mappings: dict[str, dict],
    negotiation_results: dict[str, dict],
) -> tuple[list[dict], dict[str, str]]:
    """The negotiated population handed to agents.modeling_graph.run_modeling: one dict per
    employee at their *final negotiated level* (modeling_demo.py's own docstring: "what
    matters is each employee's final negotiated level"), not the raw crosswalk level.

    Validates job_id and (family_group, level_code, geo_code) against the actual data before
    including an employee -- a negotiated level with no corresponding Meridian job code or
    salary structure (e.g. a manager-track level revised onto a job prefix that only defines
    IC codes) would otherwise crash agents.cost_model/retention_model's per-employee loop for
    the *entire* population, not just that one row (error_handling_backlog.md entry 2's exact
    failure mode, here at the modeling layer). Excluded employees are returned separately
    with a reason, for the UI to surface as needing human review rather than silently
    dropped.
    """
    job_ids = _load_job_catalog_ids()
    structure_keys = _load_salary_structure_keys()

    population = []
    excluded: dict[str, str] = {}
    for emp in employees:
        emp_id = emp["employee_id"]
        mapping = mappings[emp_id]
        if not mapping["mapped"]:
            continue  # already surfaced via mapping["reason"] -- not a modeling-stage exclusion
        decision = decisions.get(emp_id, {})
        if "error" in decision:
            continue  # already surfaced via the leveling stage's own error
        neg_result = negotiation_results.get(emp_id)
        if neg_result is None or "error" in neg_result:
            continue  # already surfaced via the negotiation stage's own error

        final_level = neg_result["final_level"]
        job_id = f"{mapping['job_prefix']}-{final_level}"

        if job_id not in job_ids:
            excluded[emp_id] = f"negotiated level {final_level} has no Meridian job code ({job_id})"
            continue
        if (mapping["family_group"], final_level, mapping["geo_code"]) not in structure_keys:
            excluded[emp_id] = (
                f"no salary structure for {mapping['family_group']}/{final_level}/{mapping['geo_code']}"
            )
            continue

        population.append(
            {
                "employee_id": emp_id,
                "job_id": job_id,
                "family": mapping["family"],
                "family_group": mapping["family_group"],
                "level_code": final_level,
                "geo_code": mapping["geo_code"],
                "currency": emp["currency"],
                "current_pay": emp["current_pay"],
                "unvested_equity_value": emp["unvested_equity_value"],
                "role_summary": emp["role_summary"],
            }
        )
    return population, excluded


def run_modeling_stage(population: list[dict]) -> dict | None:
    """None when the population is empty (nothing eligible reached modeling) -- a valid,
    meaningful result the UI should render as "no employees eligible," not an error."""
    if not population:
        return None
    return run_modeling(population, as_of_date=AS_OF_DATE, thread_id="streamlit-modeling")


def run_scope_extraction_stage(employees: list[dict], progress_cb: ProgressCallback = None) -> dict[str, dict]:
    """Runs agents.scope_extraction.extract_scope_profile_with_claude_fallback (Nebius,
    falling back to Claude if Nebius exhausts its own retries -- CLAUDE.md's model routing:
    "Nebius handles job description parsing") for every employee, independent of leveling
    itself.

    This is the same extraction agents/leveling_graph.py's parse node runs, but
    agents/leveling_batch_graph.py's Send fan-out (what run_leveling_stage above actually
    calls, for real per-employee parallelism) deliberately does not wire a parse step in --
    its BatchState has no scope_profile field, by design, for the build-order-item-3 fan-out
    demo. So the ScopeProfile shown here is genuine extracted evidence about what the job
    description states, but it is informational -- displayed alongside the leveling decision
    as the evidence a human would want to see, not proof that this specific decision was
    computed from it. Sequential, not fan-out: no Send-based batch graph exists for this step
    yet (only agents/leveling_batch_graph.py's leveling fan-out does), and every call here
    caches exactly like every other agents.instrumented_model call, so a warm cache makes
    this as fast as the leveling stage on any re-run.
    """
    total = len(employees)
    profiles: dict[str, dict] = {}
    for i, emp in enumerate(employees, start=1):
        if progress_cb:
            progress_cb(i, total, emp["employee_id"])
        try:
            profile = extract_scope_profile_with_claude_fallback(emp["job_description"])
            profiles[emp["employee_id"]] = profile.model_dump()
        except Exception as e:
            profiles[emp["employee_id"]] = {"error": str(e)}
    return profiles


def load_level_titles() -> dict[str, dict]:
    """Meridian's own ladder (level_definitions.parquet) as {level_code: {title, track,
    sort_order}} -- used both for the per-employee "level title" display and for the
    Meridian side of the framework reference panel. Read directly from the committed data,
    not re-typed by hand, so it can never drift from what leveling decisions are actually
    validated against (agents/schemas.py's LevelCode literal)."""
    df = pd.read_parquet(DATA_DIR / "level_definitions.parquet")
    return {
        row["level_code"]: {"title": row["level_title"], "track": row["track"], "sort_order": row["sort_order"]}
        for _, row in df.sort_values("sort_order").iterrows()
    }


def build_census_template() -> bytes:
    """An empty census workbook with the exact columns the pipeline reads (CENSUS_COLUMNS),
    plus one illustrative example row, as .xlsx bytes for a download button. Matches the
    shape data/generate.py's build_nyx_census produces -- not a new schema invented for the
    upload flow."""
    import io

    example_row = {
        "Emp ID": "ACME-001",
        "Job Title": "Senior Engineer II - Widget Design",
        "Dept": "Widget Engineering",
        "Location": "Austin",
        "Curr": "USD",
        "Base": 150000,
        "Bonus": 15000,
        "Unvested Options": 40000,
        "Start": "2022-03-01",
        "Role Summary": (
            "Owns the widget subsystem end to end across two product lines; makes routine "
            "technical calls without sign-off; no direct reports."
        ),
    }
    df = pd.DataFrame([example_row], columns=CENSUS_COLUMNS)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Census")
    return buffer.getvalue()


def validate_uploaded_census(file) -> tuple[bool, str, pd.DataFrame | None]:
    """Structural validation only -- confirms an uploaded workbook has the expected columns
    and returns the parsed rows for a preview. Does not feed the upload into the pipeline:
    every stage above still reads the committed Nyx census (CENSUS_PATH) regardless of what
    validates here. Wiring a validated upload into the actual run is future work, not
    something to silently half-implement by pointing the real pipeline at unvetted data.
    """
    try:
        df = pd.read_excel(file)
    except Exception as e:
        return False, f"Could not read this file as an Excel workbook: {e}", None

    missing = [c for c in CENSUS_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in CENSUS_COLUMNS]
    if missing:
        reason = f"Missing required column(s): {', '.join(missing)}."
        if extra:
            reason += f" Unexpected column(s) present: {', '.join(extra)}."
        return False, reason, None

    return True, f"{len(df)} employee(s) found, all required columns present.", df
