"""Streamlit sidebar control for demo/live mode. Demo mode is the default everywhere this
app is reachable -- a Streamlit Community Cloud deployment is publicly reachable by anyone
with the URL, and without this gate a visitor's page load would spend the deployed owner's
own Anthropic/Nebius API budget. Demo mode is enforced at the model layer
(agents.instrumented_model's cache-only switch), not just by hoping the warmed cache covers
everything -- a cache miss in demo mode raises rather than silently making a real call.

Unlocking live mode requires the password configured at DEMO_UNLOCK_PASSWORD (st.secrets
when deployed, .env locally, via agents.secrets). With no password configured at all, live
mode can never be unlocked -- the safe state for a fresh deployment that hasn't set one yet.

render_and_apply_gate() must run at the top of every Streamlit page that could reach a model
call, before that page does anything else -- agents.instrumented_model.is_cache_only()
defaults to False (cache-only off) precisely so every non-Streamlit call path is unaffected,
which means nothing else makes this page safe by default. Same "you must call this"
discipline this codebase already expects of reset_session_stats()/reset_default_budget() at
the top of a pipeline run.
"""

from __future__ import annotations

import streamlit as st

from agents.instrumented_model import set_cache_only
from agents.secrets import get_secret


def render_and_apply_gate() -> bool:
    """Renders the sidebar control and applies its result to the current script run's
    cache-only switch. Returns True if live mode is unlocked for this session.

    st.session_state persists "unlocked" across reruns within one browser session (Streamlit
    keeps session_state per session), but agents.instrumented_model's cache-only flag is
    thread-local and does NOT persist on its own -- it must be re-applied every single
    script run, which is exactly what calling this at the top of every page does.
    """
    configured_password = get_secret("DEMO_UNLOCK_PASSWORD")
    unlocked = st.session_state.get("live_mode_unlocked", False)

    with st.sidebar:
        st.divider()
        if unlocked:
            st.success("Live mode unlocked — real API calls allowed.")
            if st.button("Return to demo mode"):
                st.session_state["live_mode_unlocked"] = False
                st.rerun()
        else:
            st.caption("🔒 Demo mode — served from a pre-warmed cache. No live API calls.")
            if configured_password is None:
                st.caption("No unlock password configured — live mode is unavailable.")
            else:
                entered = st.text_input("Unlock live mode", type="password", key="live_mode_password_input")
                if entered:
                    if entered == configured_password:
                        st.session_state["live_mode_unlocked"] = True
                        st.rerun()
                    else:
                        st.error("Incorrect password.")

    unlocked = st.session_state.get("live_mode_unlocked", False)
    set_cache_only(not unlocked)
    return unlocked
