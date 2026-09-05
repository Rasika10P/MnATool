"""Guards against max_tokens and the request timeout silently disappearing from a model
instantiation. An unbounded response is the most likely way to burn credits unexpectedly on
a call that should be short (a leveling decision doesn't need a long output). A missing
timeout is worse than a cost risk: without one, ChatAnthropic/ChatOpenAI fall back to their
SDKs' own ~10-minute defaults, which from inside this codebase is indistinguishable from a
genuine hang -- agents/instrumented_model.py's retry-with-backoff (tests/test_instrumented_model.py)
only ever gets a chance to run once a stuck call actually gives up and raises."""

import pytest

from agents.model_router import REQUEST_TIMEOUT_SECONDS, get_embedding_model, get_model


def test_judgment_model_has_a_finite_max_tokens():
    llm = get_model("judgment")
    assert llm.max_tokens is not None
    assert 0 < llm.max_tokens <= 8000  # generous upper bound; a leveling decision is short


def test_unknown_tier_raises():
    with pytest.raises(ValueError):
        get_model("not-a-real-tier")


def test_judgment_model_has_an_explicit_timeout():
    llm = get_model("judgment")
    assert llm.default_request_timeout == REQUEST_TIMEOUT_SECONDS


def test_volume_model_has_an_explicit_timeout():
    llm = get_model("volume")
    assert llm.request_timeout == REQUEST_TIMEOUT_SECONDS


def test_embedding_model_has_an_explicit_timeout():
    llm = get_embedding_model()
    assert llm.request_timeout == REQUEST_TIMEOUT_SECONDS
