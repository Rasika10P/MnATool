"""Per-run spend ceiling. A run aborts cleanly (a caught, specific exception with a clear
message) the instant a call would push cumulative spend past its cap -- not after the fact.

Thread-safe: the fan-out (agents/leveling_batch_graph.py) calls into this from multiple
worker threads concurrently, so both the check and the spend update take a lock.
"""

from __future__ import annotations

import os
import threading

from agents.cost_logging import PRICING

DEFAULT_CAP_USD = float(os.environ.get("LEVELING_BUDGET_CAP_USD", "2.0"))


class BudgetExceededError(RuntimeError):
    """Raised when a call's projected cost would push cumulative spend past the run's cap.
    Scripts catch this at the top level and print a clean message instead of a traceback."""


class Budget:
    def __init__(self, cap_usd: float = DEFAULT_CAP_USD):
        self.cap_usd = cap_usd
        self.spent_usd = 0.0
        self._lock = threading.Lock()

    def project(self, model: str, estimated_input_tokens: int, max_output_tokens: int) -> float:
        rates = PRICING.get(model)
        if rates is None:
            return 0.0  # unpriced model -- nothing to project, so nothing to block on
        return (estimated_input_tokens / 1_000_000) * rates["input"] + (max_output_tokens / 1_000_000) * rates["output"]

    def check_before_call(self, model: str, prompt_parts: list[str], max_output_tokens: int) -> None:
        # Rough chars/4 heuristic for input tokens (no tokenizer dependency); max_output_tokens
        # is the call's own max_tokens setting, so this is a genuine worst-case projection, not
        # an average -- a rambling response can't silently blow past what was projected.
        estimated_input_tokens = sum(len(p) for p in prompt_parts) // 4
        projected = self.project(model, estimated_input_tokens, max_output_tokens)
        with self._lock:
            projected_total = self.spent_usd + projected
            if projected_total > self.cap_usd:
                raise BudgetExceededError(
                    f"Run cap exceeded: already spent ${self.spent_usd:.4f}, this call is "
                    f"projected at up to ${projected:.4f} more (${projected_total:.4f} total), "
                    f"over the ${self.cap_usd:.2f} cap. Raise the cap with --budget, or use "
                    f"--limit to run fewer items."
                )

    def record(self, cost_usd: float | None) -> None:
        if cost_usd is None:
            return
        with self._lock:
            self.spent_usd += cost_usd


_default_budget: Budget | None = None
_default_budget_lock = threading.Lock()


def get_default_budget() -> Budget:
    global _default_budget
    with _default_budget_lock:
        if _default_budget is None:
            _default_budget = Budget(DEFAULT_CAP_USD)
        return _default_budget


def reset_default_budget(cap_usd: float = DEFAULT_CAP_USD) -> Budget:
    """Starts a fresh budget (new cap, zeroed spend). Call this once at the top of a
    script's main() so each run gets a clean cap instead of accumulating across runs within
    the same process; tests do the same via the conftest.py autouse fixture."""
    global _default_budget
    with _default_budget_lock:
        _default_budget = Budget(cap_usd)
        return _default_budget
