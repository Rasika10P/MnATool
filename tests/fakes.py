"""Shared fake model test doubles. Match the real ChatAnthropic + with_structured_output
contract for both call shapes production code actually uses:
- include_raw=False (the default): .invoke() returns the parsed object directly, or raises
  the parsing error -- what agents/leveling.py's _run_leveling_call calls now that caching
  etc. live in InstrumentedModel instead.
- include_raw=True: .invoke() returns {"raw", "parsed", "parsing_error"} -- what
  InstrumentedModel itself calls internally (agents/instrumented_model.py).

Wrap a FakeModel in InstrumentedModel (agents.instrumented_model) when a test needs caching/
cost logging/budget behavior; pass a bare FakeModel when a test is only about decision
content or graph plumbing and instrumentation would just be incidental noise.
"""

from __future__ import annotations

from agents.schemas import LevelingDecision


class FakeRawMessage:
    def __init__(self, input_tokens: int = 100, output_tokens: int = 50):
        self.usage_metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens}


class FakeStructuredModel:
    """`sequence`, when given, is a list of (decision, parsing_error) pairs, one per call --
    the Nth call returns sequence[N-1], and once exhausted the last entry repeats. Lets a
    test simulate "fails twice then succeeds" without a bespoke fake per retry scenario. When
    `sequence` is None (the default), every call returns the fixed decision/parsing_error
    from __init__, exactly as before -- existing single-shot tests are unaffected."""

    def __init__(
        self,
        decision: LevelingDecision | None,
        parsing_error: Exception | None = None,
        include_raw: bool = False,
        sequence: list[tuple] | None = None,
    ):
        self._decision = decision
        self._parsing_error = parsing_error
        self._include_raw = include_raw
        self._sequence = sequence
        self.call_count = 0

    def invoke(self, messages):
        if self._sequence is not None:
            index = min(self.call_count, len(self._sequence) - 1)
            decision, parsing_error = self._sequence[index]
        else:
            decision, parsing_error = self._decision, self._parsing_error
        self.call_count += 1
        if self._include_raw:
            return {
                "raw": FakeRawMessage(),
                "parsed": decision,
                "parsing_error": parsing_error,
            }
        if parsing_error is not None:
            raise parsing_error
        return decision


class FakeModel:
    """`schema` defaults to LevelingDecision (every leveling-agent call site); pass the
    schema explicitly (e.g. ScopeProfile) to fake a different structured-output call, such
    as agents.scope_extraction.extract_scope_profile's parse-node call."""

    def __init__(
        self,
        decision,
        model_name: str = "fake-model",
        parsing_error: Exception | None = None,
        schema=LevelingDecision,
        sequence: list[tuple] | None = None,
    ):
        self.model = model_name  # matches ChatAnthropic's .model attribute
        self.max_tokens = 2048  # matches ChatAnthropic's configured attribute
        self._decision = decision
        self._parsing_error = parsing_error
        self._schema = schema
        # Two independent counters, one per call shape, since real code (via
        # InstrumentedModel) only ever requests include_raw=True, while a bare FakeModel
        # used directly (no InstrumentedModel wrapper) only ever sees include_raw=False.
        self.structured_model = FakeStructuredModel(decision, parsing_error, include_raw=False, sequence=sequence)
        self.raw_structured_model = FakeStructuredModel(decision, parsing_error, include_raw=True, sequence=sequence)

    def with_structured_output(self, schema, include_raw: bool = False):
        assert schema is self._schema
        return self.raw_structured_model if include_raw else self.structured_model
