"""Streamlit page for the M&A crosswalk war room (CLAUDE.md's first workflow). Loads the
committed Nyx census, crosswalks all 25 employees, negotiates every contested mapping, and
models cost/retention/synthesis for the population that clears mapping + negotiation --
reusing agents/leveling_batch_graph.py, agents/negotiation_graph.py and
agents/modeling_graph.py exactly as built for build order items 2-6. All orchestration
(mapping tables, scope extraction, per-employee error handling, stage sequencing) lives in
app/pipeline.py; this file is rendering only.

Upload is structural-validation only, not a second pipeline input yet -- every stage below
still reads the committed data/parquet/nyx_census.xlsx regardless of what's uploaded (see
app/pipeline.py's validate_uploaded_census docstring). Every model call
agents/instrumented_model.py makes is disk-cached (agents/llm_cache.py): a first run against
a cold cache makes real API calls (and is billed, hence the budget cap below); every later
run of the same population is served entirely from cache, no live calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agents.cost_logging import get_session_stats, reset_session_stats
from agents.instrumented_model import CACHE_MODE_LIVE, DemoModeCacheMissError
from agents.secrets import sync_secrets_to_env
from agents.spend_guard import BudgetExceededError, reset_default_budget
from app.demo_mode import render_and_apply_mode_control, render_mode_badge
from app.pipeline import (
    NYX_LADDER,
    NYX_LADDER_NOTE,
    build_census_template,
    build_modeling_population,
    estimate_live_run_cost,
    load_census,
    load_level_titles,
    resolve_mapping,
    run_leveling_stage,
    run_modeling_stage,
    run_negotiation_stage,
    run_scope_extraction_stage,
    validate_uploaded_census,
)

sync_secrets_to_env()

st.set_page_config(page_title="Meridian Crosswalk", layout="wide")

# Validated categorical palette (dataviz skill, references/palette.md) -- fixed slot order,
# never cycled. Only the first two slots are used anywhere on this page.
COLOR_BLUE = "#2a78d6"
COLOR_ORANGE = "#eb6834"
CHART_CHROME = {
    "surface": "#fcfcfb",
    "primary_ink": "#0b0b0b",
    "secondary_ink": "#52514e",
    "muted": "#898781",
    "gridline": "#e1e0d9",
}


# ---------------------------------------------------------------------------
# Framework reference (static -- no pipeline run required)
# ---------------------------------------------------------------------------


def _render_framework_reference() -> None:
    st.subheader("The two frameworks")
    st.caption(
        "Two ladders of different shape, laid side by side rather than mapped one-to-one — "
        "the mismatch is what the crosswalk negotiation exists to resolve, not a detail to "
        "paper over here."
    )
    col_nyx, col_meridian = st.columns(2)

    with col_nyx:
        st.markdown("**Nyx Semiconductor — 1 ladder, 5 levels**")
        for level in NYX_LADDER:
            st.markdown(f"- {level}")
        st.caption(NYX_LADDER_NOTE)

    with col_meridian:
        st.markdown("**Meridian Silicon — 2 tracks, 13 levels**")
        titles = load_level_titles()
        ic_levels = [lc for lc, v in titles.items() if v["track"] == "IC"]
        mgr_levels = [lc for lc, v in titles.items() if v["track"] == "MGR"]
        ic_line = " → ".join(f"{lc} ({titles[lc]['title']})" for lc in ic_levels)
        mgr_line = " → ".join(f"{lc} ({titles[lc]['title']})" for lc in mgr_levels)
        st.markdown(f"- **IC track:** {ic_line}")
        st.markdown(f"- **Manager track:** {mgr_line}")
        st.caption("13 levels across two tracks, vs. Nyx's single 5-level IC-only ladder.")


# ---------------------------------------------------------------------------
# Sidebar: census source + run controls
# ---------------------------------------------------------------------------


def _render_sidebar(mode: str, employees: list[dict]) -> tuple[float, bool]:
    with st.sidebar:
        st.header("Census source")
        st.caption("Currently running against the committed Nyx census (25 employees).")

        st.download_button(
            "Download census template (.xlsx)",
            data=build_census_template(),
            file_name="census_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        uploaded = st.file_uploader("Upload a census (.xlsx)", type=["xlsx"])
        if uploaded is not None:
            ok, message, df = validate_uploaded_census(uploaded)
            if ok:
                st.success(message)
                st.caption("Not yet wired to the pipeline — the run below still uses the committed Nyx census.")
                with st.expander("Preview"):
                    st.dataframe(df, hide_index=True, width="stretch")
            else:
                st.error(message)

        st.divider()
        st.header("Run")
        # Always visible next to the run button, not just in the sidebar's mode control
        # above -- someone watching a demo should see at a glance whether a run was live
        # or cached without hunting for it.
        render_mode_badge(mode)
        if mode == CACHE_MODE_LIVE:
            st.caption(f"Estimated cost if run now: ~\\${estimate_live_run_cost(employees):.2f} (approximate)")
        budget_cap = st.number_input("Spend limit (USD)", min_value=0.5, max_value=50.0, value=5.0, step=0.5)
        run_clicked = st.button("Map all 25 employees", type="primary")
        cost_metrics_slot = st.container()

        return budget_cap, run_clicked, cost_metrics_slot


def _render_cost_metrics() -> None:
    summary = get_session_stats().summary()
    st.caption("This run")
    st.metric("Model calls", summary["calls"])
    hit_rate = (summary["cache_hits"] / summary["calls"] * 100) if summary["calls"] else 0.0
    st.metric("Served from cache", f"{summary['cache_hits']} ({hit_rate:.0f}%)")
    st.metric("Cost", f"${summary['total_cost_usd']:.4f}")
    if summary["retries"]:
        st.caption(f"Retries: {summary['retries']}")


# ---------------------------------------------------------------------------
# Pipeline run
# ---------------------------------------------------------------------------


def _run_pipeline(budget_cap: float) -> None:
    reset_session_stats()
    reset_default_budget(budget_cap)

    employees, source_org_context = load_census()
    mappings = {e["employee_id"]: resolve_mapping(e) for e in employees}
    n_mapped = sum(1 for m in mappings.values() if m["mapped"])

    total_ticks = len(employees) * 2 + n_mapped + 1
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    ticks_done = 0

    def _tick(stage: str, i: int, total: int, employee_id: str) -> None:
        nonlocal ticks_done
        ticks_done += 1
        progress_bar.progress(min(ticks_done / total_ticks, 1.0))
        status_text.caption(f"{stage}: {i}/{total} ({employee_id})")

    decisions: dict = {}
    scope_profiles: dict = {}
    negotiation_results: dict = {}
    modeling_result = None
    modeling_excluded: dict = {}

    try:
        decisions = run_leveling_stage(
            employees, source_org_context, progress_cb=lambda i, t, e: _tick("Stage 1/4 — Leveling", i, t, e)
        )
        scope_profiles = run_scope_extraction_stage(
            employees, progress_cb=lambda i, t, e: _tick("Stage 2/4 — Extracting scope evidence", i, t, e)
        )
        negotiation_results = run_negotiation_stage(
            employees,
            decisions,
            mappings,
            source_org_context,
            progress_cb=lambda i, t, e: _tick("Stage 3/4 — Negotiating contested mappings", i, t, e),
        )
        status_text.caption("Stage 4/4 — Modeling cost & retention...")
        population, modeling_excluded = build_modeling_population(employees, decisions, mappings, negotiation_results)
        modeling_result = run_modeling_stage(population)
        ticks_done += 1
        progress_bar.progress(1.0)
    except BudgetExceededError as e:
        st.error(f"Run stopped — spend limit exceeded: {e}")
    except DemoModeCacheMissError as e:
        st.error(
            f"Run stopped — this needs a live API call that demo mode blocks: {e} "
            "Enter the unlock password in the sidebar to run live."
        )

    status_text.empty()
    progress_bar.empty()

    st.session_state["employees"] = employees
    st.session_state["mappings"] = mappings
    st.session_state["decisions"] = decisions
    st.session_state["scope_profiles"] = scope_profiles
    st.session_state["negotiation_results"] = negotiation_results
    st.session_state["modeling_result"] = modeling_result
    st.session_state["modeling_excluded"] = modeling_excluded
    st.session_state["has_run"] = True


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def _chart_layout(fig: go.Figure, title: str, yaxis_title: str) -> None:
    fig.update_layout(
        title=title,
        plot_bgcolor=CHART_CHROME["surface"],
        paper_bgcolor=CHART_CHROME["surface"],
        font_color=CHART_CHROME["primary_ink"],
        bargap=0.35,
        bargroupgap=0.1,
        margin=dict(t=40, b=30, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
    )
    fig.update_xaxes(showgrid=False, linecolor=CHART_CHROME["muted"])
    fig.update_yaxes(
        title=yaxis_title, gridcolor=CHART_CHROME["gridline"], gridwidth=1, zeroline=False, title_font_color=CHART_CHROME["secondary_ink"]
    )


def _level_distribution_chart() -> go.Figure:
    titles = load_level_titles()
    level_order = sorted(titles.keys(), key=lambda lc: titles[lc]["sort_order"])

    before_counts = {lc: 0 for lc in level_order}
    after_counts = {lc: 0 for lc in level_order}
    for emp in st.session_state["employees"]:
        emp_id = emp["employee_id"]
        decision = st.session_state["decisions"][emp_id]
        if "error" in decision:
            continue
        crosswalk_level = decision["assigned_level"]
        before_counts[crosswalk_level] += 1

        neg = st.session_state["negotiation_results"].get(emp_id)
        final_level = neg["final_level"] if neg and "error" not in neg else crosswalk_level
        after_counts[final_level] += 1

    fig = go.Figure()
    fig.add_bar(name="Crosswalk proposal", x=level_order, y=[before_counts[lc] for lc in level_order], marker_color=COLOR_BLUE)
    fig.add_bar(name="Final negotiated level", x=level_order, y=[after_counts[lc] for lc in level_order], marker_color=COLOR_ORANGE)
    fig.update_layout(barmode="group")
    _chart_layout(fig, "Level distribution — before vs. after negotiation", "Employees")
    return fig


def _confidence_histogram() -> go.Figure:
    confidences = [
        d["confidence"] for d in st.session_state["decisions"].values() if "error" not in d
    ]
    fig = go.Figure()
    fig.add_histogram(x=confidences, xbins=dict(start=0.0, end=1.0, size=0.1), marker_color=COLOR_BLUE)
    _chart_layout(fig, "Leveling confidence distribution", "Employees")
    fig.update_xaxes(title="Confidence", range=[0, 1])
    return fig


def _cost_by_provider_chart() -> go.Figure:
    cost_by_provider = get_session_stats().summary()["cost_by_provider"]
    providers = list(cost_by_provider.keys())
    colors = [COLOR_BLUE, COLOR_ORANGE]
    fig = go.Figure()
    fig.add_bar(
        x=providers,
        y=[cost_by_provider[p] for p in providers],
        marker_color=[colors[i % len(colors)] for i in range(len(providers))],
    )
    _chart_layout(fig, "Session cost by provider", "Cost (USD)")
    fig.update_yaxes(tickprefix="$")
    return fig


def _render_charts() -> None:
    st.subheader("Run charts")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(_level_distribution_chart(), width="stretch", theme=None)
    with col2:
        st.plotly_chart(_confidence_histogram(), width="stretch", theme=None)
    with col3:
        if get_session_stats().summary()["cost_by_provider"]:
            st.plotly_chart(_cost_by_provider_chart(), width="stretch", theme=None)
        else:
            st.caption("No billed calls this run (fully cached) — nothing to chart.")


# ---------------------------------------------------------------------------
# Per-employee card: Nyx side | scope evidence | Meridian side, plus reasoning/negotiation
# ---------------------------------------------------------------------------


def _render_scope_finding(label: str, finding: dict) -> None:
    if finding.get("stated"):
        st.markdown(f"- **{label}:** Stated: {finding['value']}")
    else:
        st.markdown(f"- **{label}:** Not mentioned in the description")


def _render_nyx_side(emp: dict) -> None:
    st.markdown("**What Nyx sent us**")
    st.markdown(f"- **Employee:** {emp['employee_id']}")
    st.markdown(f"- **Raw title:** {emp['job_title']}")
    st.markdown(f"- **MTS level:** {emp['nyx_level']}")
    st.markdown(f"- **Location:** {emp['location']}")
    st.markdown(f"- **Role summary:** {emp['role_summary']}")


def _render_evidence(employee_id: str) -> None:
    st.markdown("**What we found in the description**")
    profile = st.session_state["scope_profiles"].get(employee_id)
    if profile is None:
        st.caption("Not extracted.")
        return
    if "error" in profile:
        st.error("Couldn't extract — needs a human read")
        return
    _render_scope_finding("Team size", profile["span_of_control"])
    _render_scope_finding("Budget authority", profile["budget_authority"])
    ownership = profile["ownership_scope"] or "Not mentioned in the description"
    st.markdown(f"- **What they own:** {ownership}")


def _render_meridian_side(employee_id: str) -> None:
    st.markdown("**Where they land here**")
    decision = st.session_state["decisions"][employee_id]
    if "error" in decision:
        st.error(f"Leveling failed: {decision['error']}")
        return
    titles = load_level_titles()
    level = decision["assigned_level"]
    level_title = titles.get(level, {}).get("title", "—")
    st.markdown(f"- **Level:** {level}")
    st.markdown(f"- **Level title:** {level_title}")
    st.markdown(f"- **How certain:** {decision['confidence']:.2f}")
    st.markdown(f"- **Escalate:** {'Yes' if decision['escalate'] else 'No'}")
    st.markdown(f"- **Rule applied:** {decision['governing_rule']}")


def _render_full_reasoning(employee_id: str) -> None:
    decision = st.session_state["decisions"][employee_id]
    if "error" in decision:
        st.error(f"Leveling failed for {employee_id}: {decision['error']}")
        return

    st.markdown(f"**Track:** {decision['track']}  \n**Factor 5 variant:** {decision['factor5_variant_applied']}")
    st.markdown(f"**Reasoning:** {decision['reasoning']}")
    if decision.get("escalation_factor"):
        st.markdown(f"**Escalation factor:** {decision['escalation_factor']}")
    if decision.get("alternative_level"):
        st.markdown(
            f"**Alternative considered:** {decision['alternative_level']} — {decision['alternative_reasoning']}"
        )

    st.markdown("**Factor ratings**")
    factor_df = pd.DataFrame(decision["factor_ratings"])
    st.dataframe(factor_df, hide_index=True, width="stretch")


# Section 7's admissible argument_basis values, shortened for a one-line summary --
# CrosswalkArgumentBasis in agents/negotiation_schemas.py is the source of truth for the
# full strings; this is display-only.
_BASIS_SHORT_LABELS = {
    "scope evidence not reflected in the mapping": "scope evidence",
    "misapplied factor variant": "factor variant",
    "misread factor anchor": "factor anchor",
    "Meridian precedent": "precedent",
}
_VERDICT_LABELS = {"upheld": "upheld", "revised": "revised", "red_circled": "red-circled", "escalated": "escalated"}
_VERDICT_CONNECTOR = {"upheld": "at", "revised": "to", "red_circled": "at", "escalated": "at"}


def _short_rule_citation(governing_rule: str) -> str:
    """The leading citation from a governing_rule string -- 'rule 4' from 'rule 4: external
    recognition required for L7+', or 'section 6 rule 3' from 'section 6 rule 3: platform
    dependency'. Falls back to the full string if there's no colon to split on."""
    return governing_rule.split(":", 1)[0].strip()


def _advocate_summary_line(advocate_output: dict) -> str:
    if not advocate_output["argument_basis"]:
        return "**Nyx's case:** declines to contest — mapping stands."
    basis = _BASIS_SHORT_LABELS.get(advocate_output["argument_basis"], advocate_output["argument_basis"])
    return f"**Nyx's case:** contests, proposes {advocate_output['proposed_level']} on {basis}."


def _arbiter_summary_line(round_number: int, ruling: dict) -> str:
    label = _VERDICT_LABELS.get(ruling["verdict"], ruling["verdict"])
    connector = _VERDICT_CONNECTOR.get(ruling["verdict"], "at")
    rule = _short_rule_citation(ruling["governing_rule"])
    return f"**Our ruling (round {round_number}):** {label} {connector} {ruling['final_level']}, {rule}."


def _equity_gate_summary_line(round_number: int, gate: dict) -> str:
    if gate["passed"]:
        return f"**Equity gate (round {round_number}):** passed."
    n = len(gate["conflicting_incumbents"])
    return f"**Equity gate (round {round_number}):** failed — conflicts with {n} incumbent(s)."


def _render_negotiation_detail(employee_id: str) -> None:
    mapping = st.session_state["mappings"][employee_id]
    if not mapping["mapped"]:
        st.info(f"Not negotiated — needs human review: {mapping['reason']}")
        return

    neg = st.session_state["negotiation_results"].get(employee_id)
    if neg is None:
        st.info("Not negotiated (leveling failed for this employee).")
        return
    if "error" in neg:
        st.error(f"Negotiation failed for {employee_id}: {neg['error']}")
        return
    if not neg["contested"]:
        st.success(_advocate_summary_line(neg["advocate_output"]))
        return

    # Final verdict + cited rule up top, visible without scrolling -- the round-by-round
    # transcript below explains how it got there, but this is the answer.
    verdict = neg["final_verdict"]
    verdict_label = _VERDICT_LABELS.get(verdict, verdict)
    final_rule = _short_rule_citation(neg["exception_register_entry"]["governing_rule_cited"])
    plural = "s" if neg["round_count"] != 1 else ""
    verdict_line = (
        f"**Final verdict: {verdict_label} at {neg['final_level']}** — {final_rule} "
        f"({neg['round_count']} round{plural})"
    )
    if verdict in ("upheld", "red_circled"):
        st.success(verdict_line)
    elif verdict == "revised":
        st.info(verdict_line)
    else:  # escalated
        st.warning(verdict_line)

    st.divider()

    adv = neg["advocate_output"]
    st.markdown(_advocate_summary_line(adv))
    with st.expander("Full reasoning — Nyx's case"):
        st.markdown(f"- **Evidence:** {adv['evidence_cited']}")
        st.markdown(f"- **Source:** {adv['framework_section']}")

    for round_entry in neg["rounds"]:
        n = round_entry["round"]
        ruling = round_entry["ruling"]
        st.markdown(_arbiter_summary_line(n, ruling))
        with st.expander(f"Full reasoning — round {n} — Our ruling"):
            st.markdown(f"- **Governing rule:** {ruling['governing_rule']}")
            st.markdown(f"- **Reasoning:** {ruling['reasoning']}")

        gate_entry = next((g for g in neg["gate_checks"] if g["round"] == n), None)
        if gate_entry is not None:
            gate = gate_entry["result"]
            st.markdown(_equity_gate_summary_line(n, gate))
            with st.expander(f"Full reasoning — round {n} equity gate"):
                st.markdown(f"- **Reasoning:** {gate['reasoning']}")
                if gate["conflicting_incumbents"]:
                    st.markdown(f"- **Conflicting incumbents:** {', '.join(gate['conflicting_incumbents'])}")

    if neg["exception_register_entry"]:
        with st.expander("Exception register entry (raw)"):
            st.json(neg["exception_register_entry"])


_STATUS_LABELS_BY_VERDICT = {
    "upheld": "Contested — original stands",
    "red_circled": "Contested — level held, pay protected",
    "revised": "Contested — level changed",
    "escalated": "Needs your review",
}


def _employee_status(employee_id: str) -> str:
    decision = st.session_state["decisions"][employee_id]
    mapping = st.session_state["mappings"][employee_id]
    neg = st.session_state["negotiation_results"].get(employee_id)

    if "error" in decision:
        return "Leveling failed"
    if not mapping["mapped"]:
        return "No equivalent role"
    if neg is None:
        return "Not negotiated"
    if "error" in neg:
        return "Negotiation failed"
    if neg["contested"]:
        return _STATUS_LABELS_BY_VERDICT.get(neg["final_verdict"], neg["final_verdict"])
    return "Agreed"


def _render_employee_card(emp: dict) -> None:
    employee_id = emp["employee_id"]
    decision = st.session_state["decisions"][employee_id]
    if "error" in decision:
        level, confidence = "—", "—"
    else:
        level = decision["assigned_level"]
        confidence = f"{decision['confidence']:.2f}"
    status = _employee_status(employee_id)

    header = f"{employee_id} · {emp['job_title']} · {emp['nyx_level']} · {level} · {confidence} · {status}"
    with st.expander(header):
        col_nyx, col_evidence, col_meridian = st.columns(3)
        with col_nyx:
            _render_nyx_side(emp)
        with col_evidence:
            _render_evidence(employee_id)
        with col_meridian:
            _render_meridian_side(employee_id)

        st.divider()
        tab_reasoning, tab_negotiation = st.tabs(["Full reasoning", "How this was decided"])
        with tab_reasoning:
            _render_full_reasoning(employee_id)
        with tab_negotiation:
            _render_negotiation_detail(employee_id)


def _render_results() -> None:
    st.subheader("Where each person lands")
    st.caption("Employee · Their title · Their level · Our level · Confidence · Status")
    st.caption("Click a row to expand the full mapping — both sides, the evidence, and how it was decided.")
    for emp in st.session_state["employees"]:
        _render_employee_card(emp)


# ---------------------------------------------------------------------------
# Cost / retention / synthesis
# ---------------------------------------------------------------------------


def _render_modeling_summary() -> None:
    st.subheader("What this costs to fix")
    result = st.session_state["modeling_result"]
    excluded = st.session_state["modeling_excluded"]

    if result is None:
        st.info("No employees reached modeling — nothing to summarize.")
    else:
        cost = result["cost_assessment"]
        retention = result["retention_assessment"]
        synthesis = result["synthesis"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total day-one cost", f"${cost['total_day_one_cost']:,.0f} {cost['reporting_currency']}")
        col2.metric(
            "Total retention award", f"${retention['total_award_day_one']:,.0f} {retention['reporting_currency']}"
        )
        underwater_count = sum(1 for e in retention["employees"] if e["underwater"])
        col3.metric("Underwater employees", f"{underwater_count}/{len(retention['employees'])}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Cost recommendation ({cost['recommendation']['strategy']})**")
            st.markdown(cost["recommendation"]["reasoning"])
        with col2:
            st.markdown("**Retention judgment**")
            if retention["judgment"]["critical_employee_ids"]:
                st.markdown(f"Critical: {', '.join(retention['judgment']['critical_employee_ids'])}")
            st.markdown(retention["judgment"]["reasoning"])

        st.markdown("**Synthesis**")
        if synthesis["requires_human_judgment"]:
            st.warning("Requires human judgment — cost and retention positions genuinely conflict.")
        st.markdown(synthesis["recommended_plan"])
        for conflict in synthesis["conflicts"]:
            with st.expander(f"Conflict: {conflict['description']}"):
                st.markdown(f"- **Cost position:** {conflict['cost_position']}")
                st.markdown(f"- **Retention position:** {conflict['retention_position']}")
                st.markdown(f"- **Affected:** {', '.join(conflict['affected_employee_ids'])}")

        with st.expander("Employee-level cost & retention lines"):
            cost_df = pd.DataFrame(cost["employees"])[
                ["employee_id", "job_id", "current_pay", "target_pay", "cost_gap_reporting_currency"]
            ]
            retention_df = pd.DataFrame(retention["employees"])[
                ["employee_id", "compa_ratio", "underwater", "retention_award_reporting_currency"]
            ]
            st.markdown("Cost")
            st.dataframe(cost_df, hide_index=True, width="stretch")
            st.markdown("Retention")
            st.dataframe(retention_df, hide_index=True, width="stretch")

    if excluded:
        with st.expander(f"Excluded from modeling ({len(excluded)}) — needs human review"):
            for emp_id, reason in excluded.items():
                st.markdown(f"- **{emp_id}:** {reason}")


def _render_needs_human_summary() -> None:
    unmapped = {
        emp["employee_id"]: st.session_state["mappings"][emp["employee_id"]]["reason"]
        for emp in st.session_state["employees"]
        if not st.session_state["mappings"][emp["employee_id"]]["mapped"]
    }
    if not unmapped:
        return
    with st.expander(f"Needs human review — no Meridian mapping ({len(unmapped)})"):
        for emp_id, reason in unmapped.items():
            st.markdown(f"- **{emp_id}:** {reason}")


def _render_leveling_failures_summary() -> None:
    """error_handling_backlog.md entry 2 / ASSIGNMENT.md's error-handling contract: "a single
    employee failing to parse ... surface in the UI as a review item." A parsing/leveling
    failure and a negotiation escalation are different problems -- one means the system
    couldn't produce a decision at all, the other means it produced one but the two sides
    disagreed -- so this gets its own heading and its own section, never folded into
    escalations (which show inline per employee card as "Needs your review", not here)."""
    failed = {
        emp["employee_id"]: st.session_state["decisions"][emp["employee_id"]]["error"]
        for emp in st.session_state["employees"]
        if "error" in st.session_state["decisions"][emp["employee_id"]]
    }
    if not failed:
        return
    with st.expander(f"Needs human review — failed to level, not escalated ({len(failed)})"):
        for emp_id, error in failed.items():
            st.markdown(f"- **{emp_id}:** {error}")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def main() -> None:
    mode = render_and_apply_mode_control()

    st.title("Meridian Crosswalk")
    st.caption("Map an acquired workforce into your job architecture — and see the reasoning behind every placement.")

    _render_framework_reference()
    st.divider()

    # Cheap local file read, not an API call -- needed here (not just inside _run_pipeline)
    # so the Live-mode cost estimate has real job-description text to size against before
    # anyone clicks anything.
    employees_for_estimate, _ = load_census()
    budget_cap, run_clicked, cost_metrics_slot = _render_sidebar(mode, employees_for_estimate)

    if run_clicked:
        if mode == CACHE_MODE_LIVE:
            st.session_state["pending_live_confirmation"] = True
        else:
            with st.container():
                _run_pipeline(budget_cap)

    if st.session_state.get("pending_live_confirmation"):
        with st.container():
            estimate = estimate_live_run_cost(employees_for_estimate)
            st.warning(
                f"**Live mode will make real API calls against your configured keys.** "
                # Escaped \$ -- Streamlit's markdown renderer treats a pair of bare $ as a
                # LaTeX math span and mangles everything between them (confirmed directly:
                # an earlier unescaped version of this string rendered as garbled italics).
                f"Estimated cost: **~\\${estimate:.2f}** (approximate — negotiation and modeling "
                f"calls depend on how many mappings end up contested). Your spend limit "
                f"(\\${budget_cap:.2f}) is still enforced as a hard cap regardless of this estimate."
            )
            col_confirm, col_cancel = st.columns([1, 1])
            with col_confirm:
                if st.button("Confirm and run live", type="primary"):
                    st.session_state["pending_live_confirmation"] = False
                    _run_pipeline(budget_cap)
            with col_cancel:
                if st.button("Cancel"):
                    st.session_state["pending_live_confirmation"] = False
                    st.rerun()

    if not st.session_state.get("has_run"):
        st.info(
            "Nothing mapped yet. Run the crosswalk to see where each of the 25 acquired "
            "employees lands in our architecture."
        )
        return

    with cost_metrics_slot:
        st.divider()
        _render_cost_metrics()

    _render_charts()
    st.divider()
    _render_results()
    st.divider()
    _render_modeling_summary()
    _render_needs_human_summary()
    _render_leveling_failures_summary()


main()
