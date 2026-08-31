"""Streamlit page for gate 2 (build order item 5): forced human review when a role has no
equivalent in Meridian's architecture. ASSIGNMENT.md's one-liner: "It hands off to a human
when a role has no equivalent in our architecture."

Reads employees off the crosswalk run held in st.session_state (produced by the Home page --
this page renders no pipeline logic of its own, same split as app/pipeline.py vs. Home.py,
and the same convention app/pages/1_Approvals.py already established). Every employee
app/pipeline.py's resolve_mapping marked mapped=False (Nyx's Photonics group, "Engineering
Manager" titles -- neither has an established Meridian mapping anywhere in this codebase)
shows up here. Reviewing calls agents.no_equivalent_gate.start_no_equivalent_review, which
pauses at interrupt() -- nothing is decided until a human resumes with a verdict:
"escalated" (confirm there really is no equivalent -- this employee is handed off entirely)
or "manually_mapped" (the human supplies a real family_group/geo_code/job_prefix from the
actual job architecture, and app.pipeline.apply_manual_mapping re-runs negotiation and
modeling so the employee re-enters the normal results on the Home page).

Per-employee review state (thread_id, pause payload, resolved verdict) lives in
st.session_state["no_equivalent_reviews"], keyed by employee_id, same reason
1_Approvals.py's st.session_state["approvals"] does: a Streamlit rerun happens on every
widget interaction, and this dict is what survives it.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agents.no_equivalent_gate import DEFAULT_REVIEW_LOG_PATH, resume_no_equivalent_review
from agents.secrets import sync_secrets_to_env
from app.demo_mode import render_and_apply_mode_control
from app.pipeline import (
    apply_manual_mapping,
    load_manual_mapping_options,
    start_no_equivalent_review_for_employee,
)

sync_secrets_to_env()

st.set_page_config(page_title="No Equivalent Reviews — Meridian Crosswalk", layout="wide")

render_and_apply_mode_control()

st.title("No Equivalent Reviews")
st.caption(
    "Gate 2: an employee whose role has no established Meridian mapping is never forced "
    "into one. The graph pauses here and waits for you to either confirm there's genuinely "
    "no equivalent, or supply one yourself from the real job architecture."
)

if "no_equivalent_reviews" not in st.session_state:
    st.session_state["no_equivalent_reviews"] = {}  # employee_id -> {thread_id, pause_payload, resolved}


def _review_relevant_employees() -> list[dict]:
    """Every employee still needing review, plus every employee already reviewed on this
    page this run -- a manually-mapped employee's mappings[...]["mapped"] flips to True the
    moment it's resolved, which would otherwise drop its card off this page entirely on the
    very next rerun (the "Manually mapped" success message would be computed and then never
    shown -- confirmed live: the card just vanished instead of ever displaying it). Census
    order, not dict-insertion order, so the list doesn't jump around as reviews resolve."""
    relevant_ids = {
        emp["employee_id"]
        for emp in st.session_state["employees"]
        if not st.session_state["mappings"][emp["employee_id"]]["mapped"]
    }
    relevant_ids |= set(st.session_state["no_equivalent_reviews"].keys())
    return [emp for emp in st.session_state["employees"] if emp["employee_id"] in relevant_ids]


def _render_pause_payload(emp: dict, payload: dict) -> None:
    st.markdown(f"**{emp['employee_id']} · {emp['job_title']}**")
    st.markdown(f"- **Department:** {payload['dept']}")
    st.markdown(f"- **Sub-family:** {payload['sub_family']}")
    st.markdown(f"- **Why it stopped here:** {payload['reason']}")


def _render_decision_controls(emp_id: str, thread_id: str) -> None:
    options = load_manual_mapping_options()

    st.markdown("**Confirm there's no equivalent**")
    if st.button("Confirm — hand off entirely", key=f"escalate-{emp_id}"):
        result = resume_no_equivalent_review(thread_id, verdict="escalated")
        st.session_state["no_equivalent_reviews"][emp_id]["resolved"] = result
        st.rerun()

    st.divider()
    st.markdown("**Or map it yourself** — every option below is a real, existing part of the job architecture.")
    col_fg, col_jp, col_geo = st.columns(3)
    with col_fg:
        family_group = st.selectbox("Family group", options["family_groups"], key=f"fg-{emp_id}")
    with col_jp:
        job_prefix = st.selectbox("Closest job code", options["job_prefixes"], key=f"jp-{emp_id}")
    with col_geo:
        geo_code = st.selectbox("Geo", options["geo_codes"], key=f"geo-{emp_id}")

    if st.button("Manually map to this", key=f"map-{emp_id}", type="primary"):
        manual_mapping = {"family_group": family_group, "job_prefix": job_prefix, "geo_code": geo_code}
        result = resume_no_equivalent_review(thread_id, verdict="manually_mapped", manual_mapping=manual_mapping)
        st.session_state["no_equivalent_reviews"][emp_id]["resolved"] = result

        mappings, negotiation_results, modeling_result, modeling_excluded = apply_manual_mapping(
            employee_id=emp_id,
            manual_mapping=manual_mapping,
            employees=st.session_state["employees"],
            decisions=st.session_state["decisions"],
            mappings=st.session_state["mappings"],
            negotiation_results=st.session_state["negotiation_results"],
            source_org_context=st.session_state["source_org_context"],
        )
        st.session_state["mappings"] = mappings
        st.session_state["negotiation_results"] = negotiation_results
        st.session_state["modeling_result"] = modeling_result
        st.session_state["modeling_excluded"] = modeling_excluded
        st.rerun()


def _render_resolved(emp_id: str, resolved: dict) -> None:
    entry = resolved["review_entry"]
    if entry["verdict"] == "escalated":
        st.warning(f"**Escalated** — {emp_id} was confirmed to have no Meridian equivalent. Handed off, not mapped.")
    else:
        m = entry["manual_mapping"]
        st.success(
            f"**Manually mapped** — {emp_id} → {m['family_group']} / {m['job_prefix']} / {m['geo_code']}. "
            "Negotiation and modeling were re-run; see the Home page for the updated result."
        )


def _render_employee_review(emp: dict) -> None:
    emp_id = emp["employee_id"]
    state = st.session_state["no_equivalent_reviews"].get(emp_id)

    with st.container(border=True):
        if state is not None and state.get("resolved") is not None:
            st.markdown(f"**{emp_id} · {emp['job_title']}**")
            _render_resolved(emp_id, state["resolved"])
            return

        if state is None:
            mapping = st.session_state["mappings"][emp_id]
            st.markdown(f"**{emp_id} · {emp['job_title']}** — {mapping['reason']}")
            if st.button("Send for review", key=f"start-{emp_id}"):
                thread_id = f"no-equivalent-{emp_id}-{uuid.uuid4()}"
                result = start_no_equivalent_review_for_employee(emp, mapping, thread_id)
                st.session_state["no_equivalent_reviews"][emp_id] = {
                    "thread_id": thread_id,
                    "pause_payload": result["__interrupt__"][0].value,
                }
                st.rerun()
            return

        # Paused, waiting on a verdict.
        _render_pause_payload(emp, state["pause_payload"])
        _render_decision_controls(emp_id, state["thread_id"])


def _render_review_log() -> None:
    st.subheader("What's actually been recorded")
    st.caption("Every review, regardless of verdict — proves nothing is decided silently.")
    if not DEFAULT_REVIEW_LOG_PATH.exists():
        st.info("No reviews logged yet.")
        return
    entries = [json.loads(line) for line in DEFAULT_REVIEW_LOG_PATH.read_text().splitlines() if line.strip()]
    st.dataframe(pd.DataFrame(entries), hide_index=True, width="stretch")


if not st.session_state.get("has_run"):
    st.info("Run the crosswalk on the Home page first — reviews work against that run's mappings.")
else:
    employees = _review_relevant_employees()
    if not employees:
        st.success("Every employee in this run mapped cleanly — nothing needs review here.")
    else:
        n_pending = sum(
            1 for emp in employees
            if st.session_state["no_equivalent_reviews"].get(emp["employee_id"], {}).get("resolved") is None
        )
        st.caption(f"{len(employees)} employee(s) had no established Meridian mapping — {n_pending} still need review.")
        for emp in employees:
            _render_employee_review(emp)

st.divider()
_render_review_log()
