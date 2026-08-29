from agents.arbiter import _build_human_message, rule
from agents.negotiation_schemas import ArbiterRuling, CrosswalkArgument, EquityGateResult
from agents.schemas import FactorRating, LevelingDecision
from tests.fakes import FakeModel


def _crosswalk_decision(**overrides) -> LevelingDecision:
    fields = dict(
        track="IC",
        assigned_level="L6",
        factor_ratings=[
            FactorRating(factor="scope_of_impact", level_indicated="L6", evidence="business-unit impact")
        ],
        factor5_variant_applied="5a",
        confidence=0.72,
        governing_rule="rule 4: external recognition required for L7+",
        reasoning="Capped at L6 for lack of external recognition.",
    )
    fields.update(overrides)
    return LevelingDecision(**fields)


def _argument(**overrides) -> CrosswalkArgument:
    fields = dict(
        argument_basis="scope evidence not reflected in the mapping",
        proposed_level="L7",
        evidence_cited="company-wide final authority across the entire roadmap",
        framework_section="nyx_level_framework.md section 4",
    )
    fields.update(overrides)
    return CrosswalkArgument(**fields)


def _ruling(**overrides) -> ArbiterRuling:
    fields = dict(
        verdict="red_circled",
        governing_rule="rule 4: external recognition required for L7+",
        final_level="L6",
        reasoning="Scope evidence supports L7 under the ordinary anchors, but rule 4 bars it "
        "without external recognition; level stays at L6 with pay protection flagged.",
    )
    fields.update(overrides)
    return ArbiterRuling(**fields)


def test_human_message_includes_crosswalk_and_argument_content():
    message = _build_human_message(_crosswalk_decision(), _argument())
    assert "assigned_level: L6" in message
    assert "rule 4: external recognition required for L7+" in message
    assert "proposed_level: L7" in message
    assert "company-wide final authority" in message
    assert "nyx_level_framework.md section 4" in message


def test_rule_returns_validated_ruling():
    fake = FakeModel(_ruling(), schema=ArbiterRuling)
    ruling = rule(_crosswalk_decision(), _argument(), model=fake)
    assert ruling.verdict == "red_circled"
    assert ruling.final_level == "L6"
    assert "rule 4" in ruling.governing_rule


def test_rule_passes_through_upheld_verdict():
    fake = FakeModel(
        _ruling(verdict="upheld", final_level="L6", governing_rule="rule 4: external recognition required for L7+"),
        schema=ArbiterRuling,
    )
    ruling = rule(_crosswalk_decision(), _argument(), model=fake)
    assert ruling.verdict == "upheld"


def test_rule_passes_through_revised_verdict():
    fake = FakeModel(
        _ruling(verdict="revised", final_level="L7", governing_rule="rule 1: scope of impact is primary"),
        schema=ArbiterRuling,
    )
    ruling = rule(_crosswalk_decision(), _argument(), model=fake)
    assert ruling.verdict == "revised"
    assert ruling.final_level == "L7"


def test_human_message_without_prior_rejection_has_no_second_round_section():
    message = _build_human_message(_crosswalk_decision(), _argument())
    assert "second round" not in message


def test_human_message_with_prior_rejection_includes_gate_feedback():
    rejection = EquityGateResult(
        passed=False,
        conflicting_incumbents=["MER-0234", "MER-0235"],
        reasoning="Candidate compa-ratio exceeds every existing L7 incumbent in engineering.",
    )
    message = _build_human_message(_crosswalk_decision(), _argument(), prior_equity_gate_rejection=rejection)
    assert "second round" in message
    assert "MER-0234" in message
    assert "Candidate compa-ratio exceeds every existing L7 incumbent" in message
    assert "final round" in message


def test_rule_passes_prior_rejection_through_to_the_prompt():
    # A tiny local capturing double, since tests.fakes.FakeModel doesn't record what it was
    # called with -- this confirms rule() actually threads prior_equity_gate_rejection into
    # the sent message rather than silently dropping it before the call.
    captured = {}

    class _CapturingStructuredModel:
        def invoke(self, messages):
            captured["messages"] = messages
            return _ruling(verdict="red_circled", final_level="L6")

    class _CapturingModel:
        model = "capturing-fake"
        max_tokens = 2048

        def with_structured_output(self, schema, include_raw=False):
            assert schema is ArbiterRuling
            return _CapturingStructuredModel()

    rejection = EquityGateResult(
        passed=False, conflicting_incumbents=["MER-0236"], reasoning="Exceeds MER-0236's compa-ratio."
    )
    rule(_crosswalk_decision(), _argument(), model=_CapturingModel(), prior_equity_gate_rejection=rejection)

    sent_content = captured["messages"][1]["content"]
    assert "MER-0236" in sent_content
    assert "second round" in sent_content
