"""Proves run_with_budget_guard actually aborts cleanly (a caught message, no traceback
reaching the caller) and still prints a session summary -- the script-level half of the
budget guardrail, as opposed to tests/test_instrumented_model.py which proves the
model-layer raise. Uses InstrumentedModel(FakeModel(...)) as the model= override, matching
how a real agent gets it (via get_model()) -- a bare FakeModel bypasses instrumentation
entirely by design, so it wouldn't exercise the budget guard at all."""

import io
from contextlib import redirect_stdout

from agents.instrumented_model import InstrumentedModel
from agents.leveling import level_role
from agents.schemas import FactorRating, LevelingDecision
from scripts._cli_common import run_with_budget_guard
from tests.fakes import FakeModel


def _decision() -> LevelingDecision:
    return LevelingDecision(
        track="IC",
        assigned_level="L4",
        factor_ratings=[FactorRating(factor="scope_of_impact", level_indicated="L4", evidence="owns a subsystem")],
        factor5_variant_applied="5a",
        confidence=0.8,
        governing_rule="rule 1",
        reasoning="test",
    )


def test_budget_exceeded_aborts_without_raising_to_the_caller():
    fake_model = InstrumentedModel(FakeModel(_decision(), model_name="claude-sonnet-5"))

    def do_work():
        level_role("A role that will blow a tiny budget.", model=fake_model)

    # Must not raise -- run_with_budget_guard catches BudgetExceededError itself.
    run_with_budget_guard(cap_usd=0.0000001, fn=do_work)


def test_abort_message_is_clean_and_summary_still_prints():
    fake_model = InstrumentedModel(FakeModel(_decision(), model_name="claude-sonnet-5"))
    buf = io.StringIO()

    with redirect_stdout(buf):
        run_with_budget_guard(cap_usd=0.0000001, fn=lambda: level_role("x", model=fake_model))

    output = buf.getvalue()
    assert "RUN ABORTED" in output
    assert "BUDGET EXCEEDED" in output
    assert "Traceback" not in output
    assert "SESSION SUMMARY" in output


def test_successful_run_still_prints_summary_without_abort_message():
    fake_model = InstrumentedModel(FakeModel(_decision(), model_name="claude-sonnet-5"))
    buf = io.StringIO()

    with redirect_stdout(buf):
        run_with_budget_guard(cap_usd=2.0, fn=lambda: level_role("a healthy-budget role", model=fake_model))

    output = buf.getvalue()
    assert "RUN ABORTED" not in output
    assert "SESSION SUMMARY" in output
