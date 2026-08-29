"""Per-call token and cost logging, appended to a JSONL file -- so spend is visible after
the fact ("where did the money actually go") without needing the Anthropic console.

Pricing is $/1M tokens, current as of this file's writing (see the claude-api skill for
the authoritative live Anthropic table -- update PRICING if it changes; Nebius entries get
added once a model is confirmed via the standalone smoke test). A model missing from
PRICING still gets logged with cost_usd=None rather than silently skipped, so an unpriced
model doesn't disappear from the log.

Also tracks an in-memory per-process SessionStats (calls, cache hits, cost by provider) --
distinct from the persistent JSONL file, which accumulates across every run ever made.
Scripts print session_summary() at the end so "what did *this run* cost" doesn't require
diffing the log file before and after.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "llm_cost_log.jsonl"

# $ per 1 million tokens.
PRICING = {
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    # Nebius Token Factory, confirmed directly (not a third-party markup) via
    # artificialanalysis.ai's per-provider breakdown, 2026-08-28.
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {"input": 0.10, "output": 0.30},
}


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    rates = PRICING.get(model)
    if rates is None:
        return None
    return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]


class SessionStats:
    def __init__(self):
        self.calls = 0
        self.cache_hits = 0
        self.retries = 0
        self.cost_by_provider: dict[str, float] = {}
        self._lock = threading.Lock()

    def record(self, provider: str, cached: bool, cost_usd: float | None, attempt: int = 1) -> None:
        with self._lock:
            self.calls += 1
            if cached:
                self.cache_hits += 1
            if attempt > 1:
                self.retries += 1
            self.cost_by_provider[provider] = self.cost_by_provider.get(provider, 0.0) + (cost_usd or 0.0)

    def summary(self) -> dict:
        with self._lock:
            return {
                "calls": self.calls,
                "cache_hits": self.cache_hits,
                "retries": self.retries,
                "cost_by_provider": {k: round(v, 4) for k, v in self.cost_by_provider.items()},
                "total_cost_usd": round(sum(self.cost_by_provider.values()), 4),
            }


_session_stats: SessionStats | None = None
_session_stats_lock = threading.Lock()


def get_session_stats() -> SessionStats:
    global _session_stats
    with _session_stats_lock:
        if _session_stats is None:
            _session_stats = SessionStats()
        return _session_stats


def reset_session_stats() -> SessionStats:
    """Starts a fresh session counter. Call once at the top of a script's main(); tests do
    the same via the conftest.py autouse fixture, so counts never leak between runs."""
    global _session_stats
    with _session_stats_lock:
        _session_stats = SessionStats()
        return _session_stats


def log_call(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached: bool,
    context: str,
    provider: str = "unknown",
    log_path: Path | None = None,
    attempt: int = 1,
) -> dict:
    """Appends one JSON line, updates the in-memory session counter, and returns the logged
    entry. A cached call still logs its (stored) token counts for visibility, but cost_usd
    is 0.0 -- no API call happened.

    `attempt` is 1 for a normal call, and 2+ when agents.instrumented_model's retry loop is
    re-attempting a structured-output call that failed to validate on an earlier try (see
    docs/error_handling_backlog.md). Every attempt gets logged, retries included -- each is a
    real, billed API call -- so `attempt > 1` in the persistent JSONL log is how the retry
    rate is visible after the fact, not just in a single run's printed summary."""
    # Resolved at call time so tests can redirect by monkeypatching DEFAULT_LOG_PATH.
    log_path = log_path if log_path is not None else DEFAULT_LOG_PATH
    cost = 0.0 if cached else compute_cost(model, input_tokens, output_tokens)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "provider": provider,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached": cached,
        "cost_usd": cost,
        "context": context,
        "attempt": attempt,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    get_session_stats().record(provider, cached, cost, attempt)
    return entry


def summarize_log(log_path: Path | None = None) -> dict:
    """Aggregate view of the full persistent log (every run ever made), for
    `python -m agents.cost_logging` or ad hoc inspection. For just the current run, use
    get_session_stats().summary() instead."""
    log_path = log_path if log_path is not None else DEFAULT_LOG_PATH
    if not log_path.exists():
        return {
            "calls": 0, "cache_hits": 0, "retries": 0,
            "total_cost_usd": 0.0, "total_input_tokens": 0, "total_output_tokens": 0,
        }
    calls = 0
    cache_hits = 0
    retries = 0
    total_cost = 0.0
    total_input = 0
    total_output = 0
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line)
            calls += 1
            if entry["cached"]:
                cache_hits += 1
            if entry.get("attempt", 1) > 1:  # .get: entries logged before this field existed default to 1
                retries += 1
            total_cost += entry["cost_usd"] or 0.0
            total_input += entry["input_tokens"]
            total_output += entry["output_tokens"]
    return {
        "calls": calls,
        "cache_hits": cache_hits,
        "retries": retries,
        "total_cost_usd": round(total_cost, 4),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
    }


if __name__ == "__main__":
    print(json.dumps(summarize_log(), indent=2))
