"""Resolves a secret from Streamlit's deployed st.secrets first, falling back to os.environ
(populated locally by python-dotenv's load_dotenv() against .env). Lives in agents/, not
app/, so agents/model_router.py can depend on it directly without introducing an agents/ ->
app/ layering inversion -- app/ already depends on agents/, never the reverse.

Streamlit Community Cloud has no .env file; secrets are configured through its own Secrets
manager, which surfaces as st.secrets, not environment variables. Neither ChatAnthropic nor
ChatOpenAI in this codebase is ever constructed with an explicit api_key= -- both read their
key from os.environ internally -- so sync_secrets_to_env() bridges st.secrets into os.environ
once, early, so that unchanged internal lookup keeps working whether running locally against
.env or deployed against st.secrets.
"""

from __future__ import annotations

import os

# Every secret this codebase reads anywhere, by name -- kept in one place so a new one only
# needs adding here, not re-derived at each call site.
KNOWN_SECRET_KEYS = [
    "ANTHROPIC_API_KEY",
    "NEBIUS_API_KEY",
    "PINECONE_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGSMITH_TRACING",
    "LANGSMITH_PROJECT",
    "DEMO_UNLOCK_PASSWORD",
]


def get_secret(key: str, default: str | None = None) -> str | None:
    """st.secrets first, then os.environ. Streamlit's own secrets object can raise or behave
    oddly when no secrets.toml exists at all (every local dev machine that hasn't created
    one, and importing streamlit at all when this is called from a non-Streamlit script) --
    caught broadly so a missing secrets file is exactly equivalent to an empty one, never a
    crash. Also safe to call from plain scripts/tests that never touch Streamlit."""
    try:
        import streamlit as st

        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


def sync_secrets_to_env() -> None:
    """Copies every known secret into os.environ if it isn't already set there. Call once,
    early -- before agents.model_router.get_model() or anything that constructs a model
    client -- on every Streamlit page (Streamlit's multipage app can enter on any page
    script directly, not only Home.py, so this can't live in just one entry point).

    setdefault, not overwrite: a real local .env value wins over a same-named Streamlit
    secret if both exist, since .env is the more deliberate, closer-to-the-code source in
    that case (shouldn't normally happen -- a machine with both isn't the common setup).
    """
    for key in KNOWN_SECRET_KEYS:
        value = get_secret(key)
        if value is not None:
            os.environ.setdefault(key, str(value))
