"""Shared pytest fixtures. Autouse-isolates the LLM cache, cost log, session stats,
spend budget, and the negotiation exception register to a per-test temp directory / fresh
state, so running the test suite never reads or writes data/llm_cache, data/llm_cost_log.jsonl
or data/exception_register.jsonl (those should reflect only real usage, never test runs) and
no test's budget/call counts/register entries leak into another's."""

import pytest

import agents.cost_logging as cost_logging
import agents.llm_cache as llm_cache
import agents.negotiation_graph as negotiation_graph
import agents.spend_guard as spend_guard


@pytest.fixture(autouse=True)
def isolate_llm_cache_and_cost_log(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_cache, "DEFAULT_CACHE_DIR", tmp_path / "llm_cache")
    monkeypatch.setattr(cost_logging, "DEFAULT_LOG_PATH", tmp_path / "llm_cost_log.jsonl")
    monkeypatch.setattr(negotiation_graph, "DEFAULT_EXCEPTION_REGISTER_PATH", tmp_path / "exception_register.jsonl")
    cost_logging.reset_session_stats()
    spend_guard.reset_default_budget()
