"""Streamlit page for gate 3 (build order item 5): forced pause when a contested crosswalk
mapping hits the negotiation round limit. ASSIGNMENT.md's framework table: "forced
escalation ... when negotiation hits the round limit ... Humans intervene by overriding the
level or accepting the escalation."

Unlike gate 2's page (app/pages/2_No_Equivalent.py), there is no "send for review" step here
-- app/pipeline.py's negotiation stage already runs the advocate/arbiter/equity-gate subgraph
during the main crosswalk run, and a case that's unresolved after two rounds is left paused
at agents/negotiation_graph.py's round_limit_gate_node with its thread_id and interrupt
payload sitting in st.session_state["negotiation_results"][employee_id] (see
app/pipeline.py's _run_negotiation_for_employee: {"paused": True, "thread_id": ...,
"interrupt_payload": ...}). This page's only job is to show both positions and resume that
same thread_id once a human decides.

Per-employee resolution state (resolved verdict/level) lives in
st.session_state["round_limit_reviews"], keyed by employee_id -- same reason
1_Approvals.py's st.session_state["approvals"] and 2_No_Equivalent.py's
st.session_state["no_equivalent_reviews"] do: a Streamlit rerun happens on every widget
interaction, and this dict is what survives it. _review_relevant_employees below follows
2_No_Equivalent.py's exact fix for the disappearing-employee bug: once a case resolves,
negotiation_results[employee_id]["paused"] goes away (the resumed result has a real
final_verdict instead), so a card keyed only off "paused" would vanish the instant it
resolves, before ever showing the outcome. Employees already tracked in
round_limit_reviews are kept in the relevant set for that reason.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agents.secrets import sync_secrets_to_env
from app.demo_mode import render_and_apply_mode_control
from app.pipeline import apply_round_limit_resolution, load_level_titles

sync_secrets_to_env()

st.set_page_config(page_title="Round Limit — Meridian Crosswalk", layout="wide")

render_and_apply_mode_control()

st.title("Round Limit Reviews")
st.caption(
    "Gate 3: a contested mapping that's still unresolved after two rounds of advocate/"
    "arbiter back-and-forth pauses here instead of auto-escalating. Read both positions, "
    "then either pick the level yourself or accept the escalation."
)

if "round_limit_reviews" not in st.session_state:
    st.session_state["round_limit_reviews"] = {}  # employee_id -> {"resolved": dict | None}


def _review_relevant_employees() -> list[dict]:
    relevant_ids = {
        emp["employee_id"]
        for emp in st.session_state["employees"]
        if (st.session_state["negotiation_results"].get(emp["employee_id"]) or {}).get("paused")
    }
    relevant_ids |= set(st.session_state["round_limit_reviews"].keys())
    return [emp for emp in st.session_state["employees"] if emp["employee_id"] in relevant_ids]


def _render_positions(emp: dict, neg_result: dict) -> None:
    payload = neg_result["interrupt_payload"]
    st.markdown(f"**{emp['employee_id']} · {emp['job_title']}**")
    plural = "s" if payload["round_count"] != 1 else ""
    st.caption(
        f"Unresolved after {payload['round_count']} round{plural} — nothing has been decided. "
        "Read both sides below, then pick a level or accept the escalation."
    )

    col_nyx, col_meridian = st.columns(2)
    with col_nyx:
        st.markdown("**Nyx's case (advocate)**")
        st.markdown(f"- **Original crosswalk level:** {payload['crosswalk_level']}")
        argument = payload["advocate_argument"]
        if argument is not None:
            st.markdown(f"- **Argues for:** {payload['advocate_position']}")
            st.markdown(f"- **Basis:** {argument['argument_basis']}")
            st.markdown(f"- **Evidence:** {argument['evidence_cited']}")
            st.markdown(f"- **Cites:** {argument['framework_section']}")
        else:
            st.caption("No argument on record for this round.")

    with col_meridian:
        st.markdown("**Our ruling (arbiter)**")
        ruling = payload["last_arbiter_ruling"]
        st.markdown(f"- **Latest verdict:** {ruling['verdict']} at {ruling['final_level']}")
        st.markdown(f"- **Rule cited:** {ruling['governing_rule']}")
        st.markdown(f"- **Reasoning:** {ruling['reasoning']}")
        gate = payload["last_equity_gate_result"]
        if gate is not None:
            gate_status = "passed" if gate["passed"] else "failed"
            st.markdown(f"- **Equity gate:** {gate_status} — {gate['reasoning']}")
            if gate["conflicting_incumbents"]:
                st.markdown(f"- **Conflicts with:** {', '.join(gate['conflicting_incumbents'])}")

    with st.expander("What each round changed"):
        for round_entry in neg_result["rounds"]:
            n = round_entry["round"]
            ruling = round_entry["ruling"]
            st.markdown(f"**Round {n}:** {ruling['verdict']} at {ruling['final_level']} — {ruling['governing_rule']}")
            st.caption(ruling["reasoning"])
            gate_entry = next((g for g in neg_result["gate_checks"] if g["round"] == n), None)
            if gate_entry is not None:
                gate = gate_entry["result"]
                gate_status = "passed" if gate["passed"] else "failed"
                st.markdown(f"Equity gate ({gate_status}): {gate['reasoning']}")


def _render_decision_controls(emp_id: str, thread_id: str, payload: dict) -> None:
    titles = load_level_titles()
    level_codes = sorted(titles.keys(), key=lambda lc: titles[lc]["sort_order"])
    latest_level = payload["last_arbiter_ruling"]["final_level"]

    st.markdown("**Pick a level**")
    col_pick, col_apply = st.columns([3, 1])
    with col_pick:
        chosen_level = st.selectbox(
            "Level",
            level_codes,
            index=level_codes.index(latest_level) if latest_level in level_codes else 0,
            key=f"round-limit-level-{emp_id}",
            label_visibility="collapsed",
        )
    with col_apply:
        if st.button("Apply this level", key=f"round-limit-apply-{emp_id}", type="primary"):
            negotiation_results, modeling_result, modeling_excluded = apply_round_limit_resolution(
                employee_id=emp_id,
                thread_id=thread_id,
                verdict="overridden",
                override_level=chosen_level,
                employees=st.session_state["employees"],
                decisions=st.session_state["decisions"],
                mappings=st.session_state["mappings"],
                negotiation_results=st.session_state["negotiation_results"],
            )
            st.session_state["negotiation_results"] = negotiation_results
            st.session_state["modeling_result"] = modeling_result
            st.session_state["modeling_excluded"] = modeling_excluded
            st.session_state["round_limit_reviews"][emp_id] = {
                "resolved": negotiation_results[emp_id],
            }
            st.rerun()

    st.divider()
    st.markdown("**Or, if neither side should be overridden**")
    if st.button("Accept escalation — hand off, keep the original level", key=f"round-limit-escalate-{emp_id}"):
        negotiation_results, modeling_result, modeling_excluded = apply_round_limit_resolution(
            employee_id=emp_id,
            thread_id=thread_id,
            verdict="accepted_escalation",
            override_level=None,
            employees=st.session_state["employees"],
            decisions=st.session_state["decisions"],
            mappings=st.session_state["mappings"],
            negotiation_results=st.session_state["negotiation_results"],
        )
        st.session_state["negotiation_results"] = negotiation_results
        st.session_state["modeling_result"] = modeling_result
        st.session_state["modeling_excluded"] = modeling_excluded
        st.session_state["round_limit_reviews"][emp_id] = {
            "resolved": negotiation_results[emp_id],
        }
        st.rerun()


def _render_resolved(emp_id: str, resolved: dict) -> None:
    if resolved["final_verdict"] == "human_overridden":
        st.success(f"**Overridden** — {emp_id} set to {resolved['final_level']} by a human after the round limit.")
    else:
        st.warning(
            f"**Escalated** — {emp_id} was handed off; the original crosswalk level "
            f"({resolved['final_level']}) stands pending further review."
        )


def _render_employee_review(emp: dict) -> None:
    emp_id = emp["employee_id"]
    state = st.session_state["round_limit_reviews"].get(emp_id)

    with st.container(border=True):
        if state is not None and state.get("resolved") is not None:
            _render_resolved(emp_id, state["resolved"])
            return

        neg_result = st.session_state["negotiation_results"].get(emp_id)
        if neg_result is None or not neg_result.get("paused"):
            # Resolved by some other route (or the run changed under us) without this page
            # ever recording it -- nothing left to review.
            return

        _render_positions(emp, neg_result)
        _render_decision_controls(emp_id, neg_result["thread_id"], neg_result["interrupt_payload"])


if not st.session_state.get("has_run"):
    st.info("Run the crosswalk on the Home page first — reviews work against that run's negotiations.")
else:
    employees = _review_relevant_employees()
    if not employees:
        st.success("No negotiation in this run hit the round limit — nothing needs review here.")
    else:
        n_pending = sum(
            1 for emp in employees
            if st.session_state["round_limit_reviews"].get(emp["employee_id"], {}).get("resolved") is None
        )
        st.caption(f"{len(employees)} case(s) hit the round limit — {n_pending} still need review.")
        for emp in employees:
            _render_employee_review(emp)
