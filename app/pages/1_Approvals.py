"""Streamlit page for gate 4 (build order item 5): human approval before any write to
leveling_decisions. ASSIGNMENT.md's framework table: "final approval before any write."

Reads employees off the crosswalk run held in st.session_state (produced by the Home page --
this page renders no pipeline logic of its own, same split as app/pipeline.py vs. Home.py).
For a contested employee, "final" means the negotiated level (agents/negotiation_graph.py's
final_level); for an uncontested one, the crosswalk level stands untouched. Approving calls
agents.approval_graph.start_approval, which pauses at interrupt() -- nothing is written to
data/comp.duckdb until a human resumes with a verdict.

Per-employee approval state (thread_id, pause payload, resolved verdict) lives in
st.session_state["approvals"], keyed by employee_id, so a Streamlit rerun (which happens on
every widget interaction) doesn't lose track of who's mid-approval or forget a resolved one.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agents.approval_graph import resume_approval, start_approval
from agents.secrets import sync_secrets_to_env
from app.demo_mode import render_and_apply_mode_control
from app.pipeline import load_level_titles
from tools.decisions import DEFAULT_DB_PATH

sync_secrets_to_env()

st.set_page_config(page_title="Approvals — Meridian Crosswalk", layout="wide")

render_and_apply_mode_control()

st.title("Approvals")
st.caption(
    "Gate 4: no mapping reaches leveling_decisions without a human approving it here. "
    "The graph pauses, waits for your verdict, and only then writes."
)

if "approvals" not in st.session_state:
    st.session_state["approvals"] = {}  # employee_id -> {thread_id, pause_payload, resolved}


def _eligible_employees() -> list[dict]:
    """Employees with a usable leveling decision -- error rows are excluded (nothing to
    approve), everything else is, including unmapped ones: leveling still ran for them even
    though they never reached negotiation or modeling."""
    out = []
    for emp in st.session_state["employees"]:
        emp_id = emp["employee_id"]
        decision = st.session_state["decisions"][emp_id]
        if "error" in decision:
            continue
        out.append(emp)
    return out


def _proposal_for(emp: dict) -> dict:
    """The (assigned_level, confidence, governing_rule, negotiation_context) this employee's
    approval gate should show -- the negotiated outcome when contested, the crosswalk decision
    untouched otherwise. Confidence and factor_ratings always come from the crosswalk decision:
    the arbiter's ruling (agents/negotiation_schemas.ArbiterRuling) carries a verdict and a
    governing rule, not its own confidence score or factor evidence."""
    emp_id = emp["employee_id"]
    decision = st.session_state["decisions"][emp_id]
    neg = st.session_state["negotiation_results"].get(emp_id)

    negotiation_context = None
    assigned_level = decision["assigned_level"]
    governing_rule = decision["governing_rule"]

    if neg is not None and "error" not in neg and neg["contested"]:
        assigned_level = neg["final_level"]
        governing_rule = neg["exception_register_entry"]["governing_rule_cited"] if neg["exception_register_entry"] else governing_rule
        negotiation_context = {
            "nyx_level": emp["nyx_level"],
            "crosswalk_level": decision["assigned_level"],
            "advocate_proposed_level": neg["advocate_output"]["proposed_level"],
            "final_verdict": neg["final_verdict"],
            "round_count": neg["round_count"],
        }

    return {
        "job_or_employee_ref": emp_id,
        "assigned_level": assigned_level,
        "confidence": decision["confidence"],
        "factor_ratings": decision["factor_ratings"],
        "factor5_variant_applied": decision["factor5_variant_applied"],
        "alternative_considered": decision.get("alternative_level"),
        "governing_rule": governing_rule,
        "source_document_hash": None,
        "negotiation_context": negotiation_context,
        "reviewer_verdict": None,
        "final_level": None,
        "written_record": None,
    }


def _render_pause_payload(emp: dict, payload: dict) -> None:
    st.markdown(f"**{emp['employee_id']} · {emp['job_title']}**")
    st.caption(
        "The system worked this out on its own — nothing has been saved yet. Read what it "
        "found below, then approve it, change the level yourself, or reject it."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"- **Proposed level:** {payload['assigned_level']}")
        st.markdown(
            f"- **Confidence:** {payload['confidence']:.2f} "
            "(how sure the system is, from 0 to 1 — the closer to 1, the more sure)"
        )
        st.markdown(f"- **Why this level:** {payload['governing_rule']}")
        if payload["alternative_considered"]:
            st.markdown(
                f"- **Also considered:** {payload['alternative_considered']} "
                "(a level it weighed but decided against)"
            )
    with col2:
        if payload["negotiation_context"]:
            ctx = payload["negotiation_context"]
            st.markdown("**Both sides' positions** — Nyx pushed back on this mapping, so here's how each side saw it:")
            st.markdown(f"- Nyx's own title says: {ctx['nyx_level']}")
            st.markdown(f"- We originally proposed: {ctx['crosswalk_level']}")
            st.markdown(f"- Nyx argued it should be: {ctx['advocate_proposed_level']}")
            st.markdown(f"- Where it landed: {ctx['final_verdict']} (after {ctx['round_count']} round(s) of back-and-forth)")
        else:
            st.markdown("**No dispute** — Nyx didn't push back on this mapping.")

    with st.expander("See the detailed evidence"):
        st.caption(
            "Each row is one factor from the leveling framework (scope, autonomy, complexity, "
            "and so on) and the level that factor's evidence points to on its own — this is "
            "what the proposed level above is built from."
        )
        st.dataframe(pd.DataFrame(payload["factor_ratings"]), hide_index=True, width="stretch")


def _render_decision_controls(emp_id: str, thread_id: str, payload: dict) -> None:
    titles = load_level_titles()
    level_codes = sorted(titles.keys(), key=lambda lc: titles[lc]["sort_order"])

    col_approve, col_override, col_reject = st.columns([1, 2, 1])
    with col_approve:
        if st.button("Approve", key=f"approve-{emp_id}", type="primary"):
            result = resume_approval(thread_id, verdict="approved")
            st.session_state["approvals"][emp_id]["resolved"] = result
            st.rerun()
    with col_override:
        override_level = st.selectbox(
            "Override to", level_codes, index=level_codes.index(payload["assigned_level"]), key=f"override-select-{emp_id}"
        )
        if st.button("Approve with override", key=f"override-{emp_id}"):
            result = resume_approval(thread_id, verdict="approved_with_override", override_level=override_level)
            st.session_state["approvals"][emp_id]["resolved"] = result
            st.rerun()
    with col_reject:
        if st.button("Reject", key=f"reject-{emp_id}"):
            result = resume_approval(thread_id, verdict="rejected")
            st.session_state["approvals"][emp_id]["resolved"] = result
            st.rerun()


def _render_resolved(emp_id: str, resolved: dict) -> None:
    if resolved["written_record"] is not None:
        record = resolved["written_record"]
        st.success(
            f"**{resolved['reviewer_verdict']}** at {resolved['final_level']} — written to "
            f"leveling_decisions (decision_id `{record['decision_id']}`)."
        )
    else:
        st.warning(f"**Rejected** — {emp_id} was not written to leveling_decisions.")


def _render_employee_approval(emp: dict) -> None:
    emp_id = emp["employee_id"]
    state = st.session_state["approvals"].get(emp_id)

    with st.container(border=True):
        if state is not None and state.get("resolved") is not None:
            st.markdown(f"**{emp_id} · {emp['job_title']}**")
            _render_resolved(emp_id, state["resolved"])
            return

        if state is None:
            proposal = _proposal_for(emp)
            st.markdown(f"**{emp_id} · {emp['job_title']}** — {proposal['assigned_level']}, confidence {proposal['confidence']:.2f}")
            if st.button("Send for approval", key=f"start-{emp_id}"):
                thread_id = f"approval-{emp_id}-{uuid.uuid4()}"
                result = start_approval(proposal, thread_id)
                st.session_state["approvals"][emp_id] = {
                    "thread_id": thread_id,
                    "pause_payload": result["__interrupt__"][0].value,
                }
                st.rerun()
            return

        # Paused, waiting on a verdict.
        _render_pause_payload(emp, state["pause_payload"])
        _render_decision_controls(emp_id, state["thread_id"], state["pause_payload"])


def _render_persisted_table() -> None:
    st.subheader("What's actually been recorded")
    st.caption("A live look at every approved mapping — proves nothing gets written until you approve it.")
    if not DEFAULT_DB_PATH.exists():
        st.info("No decisions written yet.")
        return
    con = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    df = con.execute(
        "SELECT decision_id, job_or_employee_ref, assigned_level, reviewer_verdict, governing_rule, created_at "
        "FROM leveling_decisions ORDER BY created_at DESC"
    ).fetchdf()
    con.close()
    st.dataframe(df, hide_index=True, width="stretch")


if not st.session_state.get("has_run"):
    st.info("Run the crosswalk on the Home page first — approvals work against that run's decisions.")
else:
    employees = _eligible_employees()
    st.caption(f"{len(employees)} employee(s) have a leveling decision ready for approval.")
    for emp in employees:
        _render_employee_approval(emp)

st.divider()
_render_persisted_table()
