"""Streamlit sidebar control for the three-way cache mode (agents/instrumented_model.py):

  Demo (cached) -- default. Reads cache only; a miss shows "not cached" rather than ever
      calling an API. Zero cost, zero API dependency, guaranteed reproducible -- the safe
      state for a public Streamlit Cloud deployment reachable by anyone with the URL.
  Live -- bypasses the cache entirely, makes real calls, writes results back to cache.
  Fill gaps -- reads cache where available, calls the API only on a miss. Warms the cache
      after a data change without re-paying for what's already there.

Live and Fill gaps can both spend real money (fill gaps calls the API on any cache miss,
same as live does on every call), so both are gated behind DEMO_UNLOCK_PASSWORD -- only Demo
is safe-by-construction for a visitor with no password. With no password configured at all,
Demo is the only selectable mode.

render_and_apply_mode_control() must run at the top of every Streamlit page that could reach
a model call, before that page does anything else -- agents.instrumented_model.get_cache_mode()
defaults to "fill" precisely so every non-Streamlit call path is unaffected, which means
nothing else makes this page safe by default. Same "you must call this" discipline this
codebase already expects of reset_session_stats()/reset_default_budget() at the top of a run.
"""

from __future__ import annotations

import streamlit as st

from agents.instrumented_model import CACHE_MODE_DEMO, CACHE_MODE_FILL, CACHE_MODE_LIVE, set_cache_mode
from agents.secrets import get_secret

_MODE_ORDER = [CACHE_MODE_DEMO, CACHE_MODE_FILL, CACHE_MODE_LIVE]
_MODE_LABELS = {
    CACHE_MODE_DEMO: "Demo (cached)",
    CACHE_MODE_FILL: "Fill gaps",
    CACHE_MODE_LIVE: "Live",
}
_BADGE_TEXT = {
    CACHE_MODE_DEMO: "🔒 DEMO — cached only",
    CACHE_MODE_FILL: "🟡 FILL GAPS — cache + API on miss",
    CACHE_MODE_LIVE: "🔴 LIVE — real API calls",
}


def render_and_apply_mode_control() -> str:
    """Renders the sidebar mode control, applies the chosen mode to this script run's
    thread-local cache mode, and returns the active mode string ("demo"/"fill"/"live").

    st.session_state persists the unlocked/chosen-mode state across reruns within one
    browser session, but agents.instrumented_model's cache mode is thread-local and does
    NOT persist on its own -- it must be re-applied every single script run, which is
    exactly what calling this at the top of every page does.
    """
    configured_password = get_secret("DEMO_UNLOCK_PASSWORD")
    unlocked = st.session_state.get("live_mode_unlocked", False)
    mode = st.session_state.get("cache_mode", CACHE_MODE_DEMO)

    with st.sidebar:
        st.divider()
        st.caption("**Cache mode**")

        if not unlocked:
            st.caption("Demo (cached) — served from a pre-warmed cache. No live API calls.")
            if configured_password is None:
                st.caption("No unlock password configured — Fill gaps and Live are unavailable.")
            else:
                entered = st.text_input("Unlock Fill gaps / Live", type="password", key="cache_mode_password_input")
                if entered:
                    if entered == configured_password:
                        st.session_state["live_mode_unlocked"] = True
                        st.rerun()
                    else:
                        st.error("Incorrect password.")
            mode = CACHE_MODE_DEMO
        else:
            mode = st.radio(
                "Cache mode",
                _MODE_ORDER,
                format_func=lambda m: _MODE_LABELS[m],
                index=_MODE_ORDER.index(mode) if mode in _MODE_ORDER else 0,
                key="cache_mode_radio",
                label_visibility="collapsed",
            )
            if st.button("Lock back to demo mode"):
                st.session_state["live_mode_unlocked"] = False
                st.session_state["cache_mode"] = CACHE_MODE_DEMO
                st.rerun()

    st.session_state["cache_mode"] = mode
    set_cache_mode(mode)
    return mode


def render_mode_badge(mode: str) -> None:
    """A small, always-visible badge -- render this right above/next to the run button, not
    only in the sidebar's mode control above, so someone watching a demo can see at a glance
    whether a run was live or cached without hunting through a menu for it."""
    st.caption(f"**Mode: {_BADGE_TEXT[mode]}**")
