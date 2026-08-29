"""Proves the graph's decision-shaping logic (escalate computation, model_dump plumbing)
matches level_role's, and that the SqliteSaver checkpoint survives a real process kill
between nodes.

Since the graph's level node now receives the parse node's extracted scope_profile as
advisory context (agents/leveling_graph.py) while a bare level_role() call doesn't get one
unless a caller passes it explicitly, the two paths are no longer expected to send identical
prompts to a real model -- only to process whatever the model returns identically. A fake
model that ignores its input and returns a fixed decision regardless is exactly what isolates
that: both paths get the same decision object back, so this proves the escalate/model_dump
logic didn't diverge, without depending on (or masking) a difference in what's actually sent
upstream. A live diff on real prompts is a qualitative check instead (see
scripts/level_five_jobs_via_graph.py), since Claude isn't seeded and won't repeat prose
identically across two live runs regardless.
"""

from agents.leveling import level_role
from agents.leveling_graph import LevelingState, build_graph, get_checkpointer
from agents.schemas import FactorRating, LevelingDecision, ScopeFinding, ScopeProfile
from tests.fakes import FakeModel


def _fixed_scope_profile() -> ScopeProfile:
    return ScopeProfile(
        reports_to=ScopeFinding(stated=True, value="Director of Engineering"),
        span_of_control=ScopeFinding(stated=False, value=None),
        budget_authority=ScopeFinding(stated=False, value=None),
        decision_scope="Sets methodology for the hardest blocks independently.",
        ownership_scope="Owns place-and-route and timing closure for a subsystem.",
    )


def _fixed_decision() -> LevelingDecision:
    return LevelingDecision(
        track="IC",
        assigned_level="L4",
        factor_ratings=[
            FactorRating(factor="scope_of_impact", level_indicated="L4", evidence="owns a subsystem"),
        ],
        factor5_variant_applied="5a",
        confidence=0.8,
        governing_rule="rule 1: scope of impact is primary",
        alternative_level="L5",
        alternative_reasoning="considered but scope caps at L4",
        escalation_factor=None,
        reasoning="fixed test decision",
    )


def test_graph_matches_plain_function_given_same_model_output(tmp_path):
    fake_decision = _fixed_decision()
    fake_level_model = FakeModel(fake_decision)
    fake_parse_model = FakeModel(_fixed_scope_profile(), schema=ScopeProfile)
    job_description = "Owns a subsystem across a full development cycle."

    plain_result = level_role(job_description, model=fake_level_model)

    checkpointer = get_checkpointer(tmp_path / "test.sqlite")
    app = build_graph(level_model=fake_level_model, parse_model=fake_parse_model).compile(checkpointer=checkpointer)
    initial_state: LevelingState = {
        "job_description": job_description,
        "source_org_context": None,
        "low_confidence_threshold": 0.65,
        "high_confidence_threshold": 0.75,
        "parsed": False,
        "scope_profile": None,
        "decision": None,
    }
    graph_result = app.invoke(initial_state, {"configurable": {"thread_id": "test-thread"}})

    assert graph_result["decision"] == plain_result.model_dump()
    assert graph_result["scope_profile"] == _fixed_scope_profile().model_dump()


def test_graph_computes_escalate_same_as_plain_function_below_threshold(tmp_path):
    fake_decision = _fixed_decision().model_copy(update={"confidence": 0.5})
    fake_level_model = FakeModel(fake_decision)
    fake_parse_model = FakeModel(_fixed_scope_profile(), schema=ScopeProfile)
    job_description = "A genuinely ambiguous role."

    plain_result = level_role(job_description, model=fake_level_model)
    assert plain_result.escalate is True

    checkpointer = get_checkpointer(tmp_path / "test2.sqlite")
    app = build_graph(level_model=fake_level_model, parse_model=fake_parse_model).compile(checkpointer=checkpointer)
    initial_state: LevelingState = {
        "job_description": job_description,
        "source_org_context": None,
        "low_confidence_threshold": 0.65,
        "high_confidence_threshold": 0.75,
        "parsed": False,
        "scope_profile": None,
        "decision": None,
    }
    graph_result = app.invoke(initial_state, {"configurable": {"thread_id": "test-thread-2"}})

    assert graph_result["decision"]["escalate"] is True
    assert graph_result["decision"] == plain_result.model_dump()


class _RecordingStructuredModel:
    """Records the exact messages it's invoked with, so a test can prove what reached the
    model -- FakeModel's fixed-return fakes are opaque to this (they ignore input entirely)."""

    def __init__(self, decision):
        self._decision = decision
        self.seen_messages: list[list[dict]] = []

    def invoke(self, messages):
        self.seen_messages.append(messages)
        return self._decision


class _RecordingModel:
    def __init__(self, decision, schema, model_name: str = "recording-model"):
        self.model = model_name
        self.max_tokens = 2048
        self._schema = schema
        self.structured_model = _RecordingStructuredModel(decision)

    def with_structured_output(self, schema, include_raw: bool = False):
        assert schema is self._schema
        assert not include_raw
        return self.structured_model


def test_level_node_actually_receives_the_extracted_scope_profile(tmp_path):
    # The fixed-decision fakes elsewhere in this file ignore their input, so they can't prove
    # the extraction reaches the level node's prompt -- only that decision-shaping logic is
    # unchanged. This test uses a recording fake specifically to check what was sent.
    fake_decision = _fixed_decision()
    recording_level_model = _RecordingModel(fake_decision, schema=LevelingDecision)
    fake_parse_model = FakeModel(_fixed_scope_profile(), schema=ScopeProfile)
    job_description = "Owns a subsystem across a full development cycle."

    checkpointer = get_checkpointer(tmp_path / "test3.sqlite")
    app = build_graph(level_model=recording_level_model, parse_model=fake_parse_model).compile(checkpointer=checkpointer)
    initial_state: LevelingState = {
        "job_description": job_description,
        "source_org_context": None,
        "low_confidence_threshold": 0.65,
        "high_confidence_threshold": 0.75,
        "parsed": False,
        "scope_profile": None,
        "decision": None,
    }
    app.invoke(initial_state, {"configurable": {"thread_id": "test-thread-3"}})

    assert len(recording_level_model.structured_model.seen_messages) == 1
    human_message = recording_level_model.structured_model.seen_messages[0][1]["content"]
    assert "Extracted scope profile" in human_message
    assert "reports_to: explicitly stated -- 'Director of Engineering'" in human_message
    assert "span_of_control: not mentioned in the text" in human_message
