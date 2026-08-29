"""Guards against max_tokens silently disappearing from a model instantiation -- an
unbounded response is the most likely way to burn credits unexpectedly on a call that
should be short (a leveling decision doesn't need a long output)."""

import pytest

from agents.model_router import get_model


def test_judgment_model_has_a_finite_max_tokens():
    llm = get_model("judgment")
    assert llm.max_tokens is not None
    assert 0 < llm.max_tokens <= 8000  # generous upper bound; a leveling decision is short


def test_unknown_tier_raises():
    with pytest.raises(ValueError):
        get_model("not-a-real-tier")
