from agents.cost_logging import compute_cost, log_call, summarize_log


def test_compute_cost_known_model():
    # claude-sonnet-5: $2.00 / $10.00 per 1M input/output tokens
    cost = compute_cost("claude-sonnet-5", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == 12.00


def test_compute_cost_unknown_model_returns_none():
    assert compute_cost("some-model-not-in-pricing-table", 1000, 1000) is None


def test_log_call_writes_jsonl_line(tmp_path):
    log_path = tmp_path / "log.jsonl"
    entry = log_call("claude-sonnet-5", 1000, 500, cached=False, context="test", log_path=log_path)
    assert log_path.exists()
    assert entry["model"] == "claude-sonnet-5"
    assert entry["input_tokens"] == 1000
    assert entry["output_tokens"] == 500
    assert entry["cached"] is False
    assert entry["cost_usd"] == compute_cost("claude-sonnet-5", 1000, 500)


def test_cached_call_logs_zero_cost_but_real_token_counts(tmp_path):
    log_path = tmp_path / "log.jsonl"
    entry = log_call("claude-sonnet-5", 1000, 500, cached=True, context="test", log_path=log_path)
    assert entry["cost_usd"] == 0.0
    assert entry["input_tokens"] == 1000  # still recorded, just no charge
    assert entry["output_tokens"] == 500


def test_unpriced_model_logs_with_none_cost_not_dropped(tmp_path):
    log_path = tmp_path / "log.jsonl"
    entry = log_call("unpriced-model", 100, 100, cached=False, context="test", log_path=log_path)
    assert entry["cost_usd"] is None
    assert entry["model"] == "unpriced-model"


def test_summarize_log_aggregates_across_calls(tmp_path):
    log_path = tmp_path / "log.jsonl"
    log_call("claude-sonnet-5", 1000, 500, cached=False, context="a", log_path=log_path)
    log_call("claude-sonnet-5", 1000, 500, cached=True, context="b", log_path=log_path)

    summary = summarize_log(log_path=log_path)
    assert summary["calls"] == 2
    assert summary["cache_hits"] == 1
    assert summary["total_input_tokens"] == 2000
    assert summary["total_output_tokens"] == 1000
    assert summary["total_cost_usd"] == compute_cost("claude-sonnet-5", 1000, 500)  # only the uncached call


def test_summarize_log_empty_file(tmp_path):
    summary = summarize_log(log_path=tmp_path / "does_not_exist.jsonl")
    assert summary["calls"] == 0
    assert summary["total_cost_usd"] == 0.0
