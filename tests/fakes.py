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
    def __init__(self, input_tokens: int = 100, output_tokens: int = 50, tool_call_args: dict | None = None):
        self.usage_metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        # tool_call_args mirrors what a real ChatAnthropic response's raw.tool_calls[0]["args"]
        # looks like (agents/instrumented_model.py's _raw_tool_call_args) -- None (the
        # default) leaves .tool_calls unset entirely, matching every pre-existing fake that
        # never simulated it, so tests written before the tag-leak check keep working
        # unchanged (agents.instrumented_model._raw_tool_call_args treats a missing
        # attribute as "nothing to scan," not an error).
        if tool_call_args is not None:
            self.tool_calls = [{"name": "FakeSchema", "args": tool_call_args, "id": "fake-call-id"}]


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
        tool_call_args_sequence: list[dict | None] | None = None,
    ):
        self._decision = decision
        self._parsing_error = parsing_error
        self._include_raw = include_raw
        self._sequence = sequence
        # Independent of `sequence` (a separate parameter rather than a third tuple element)
        # so every existing 2-tuple (decision, parsing_error) call site is untouched --
        # indexed the same way: the Nth call gets tool_call_args_sequence[N-1], clamped once
        # exhausted. None (the default) means "no raw tool_calls to scan," same as omitting
        # the parameter to FakeRawMessage directly.
        self._tool_call_args_sequence = tool_call_args_sequence
        self.call_count = 0

    def invoke(self, messages):
        if self._sequence is not None:
            index = min(self.call_count, len(self._sequence) - 1)
            decision, parsing_error = self._sequence[index]
        else:
            decision, parsing_error = self._decision, self._parsing_error

        tool_call_args = None
        if self._tool_call_args_sequence is not None:
            index = min(self.call_count, len(self._tool_call_args_sequence) - 1)
            tool_call_args = self._tool_call_args_sequence[index]

        self.call_count += 1
        if self._include_raw:
            return {
                "raw": FakeRawMessage(tool_call_args=tool_call_args),
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
        tool_call_args_sequence: list[dict | None] | None = None,
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
        self.raw_structured_model = FakeStructuredModel(
            decision, parsing_error, include_raw=True, sequence=sequence,
            tool_call_args_sequence=tool_call_args_sequence,
        )

    def with_structured_output(self, schema, include_raw: bool = False):
        assert schema is self._schema
        return self.raw_structured_model if include_raw else self.structured_model


class FakeNetworkFlakyModel:
    """with_structured_output(schema, include_raw=True).invoke(...) -- the shape
    agents/instrumented_model.py's _InstrumentedStructuredRunnable actually calls -- raises
    `errors[i]` on the i-th call (0-indexed) instead of returning, simulating a network
    error (a timeout, a dropped connection) rather than a parsing failure. Returns `decision`
    (via the normal include_raw=True dict shape) once `errors` is exhausted.

    Pass `errors=[e] * N` to simulate "never recovers" (exhausts every retry); a test that
    wants "recovers on the last allowed attempt" passes exactly MAX_ATTEMPTS - 1 errors.
    """

    def __init__(self, decision, errors: list[Exception], model_name: str = "fake-network-flaky-model"):
        self.model = model_name
        self.max_tokens = 2048
        self._decision = decision
        self._errors = errors
        self.call_count = 0

    def with_structured_output(self, schema, include_raw: bool = False):
        return self

    def invoke(self, messages):
        index = self.call_count
        self.call_count += 1
        if index < len(self._errors):
            raise self._errors[index]
        return {"raw": FakeRawMessage(), "parsed": self._decision, "parsing_error": None}


class FakeFaultInjectingModel:
    """with_structured_output(LevelingDecision).invoke(messages) raises `failure` whenever
    `fail_when_content_contains` appears in the outgoing messages, and returns `decision`
    otherwise. Lets a batch-fan-out test (tests/test_leveling_batch_graph.py) force exactly
    one employee among many to fail, by giving that one employee's job_description a marker
    string the others don't have -- no bespoke per-employee fake needed, and safe under real
    concurrent Send tasks since each call only inspects its own messages."""

    def __init__(self, decision: LevelingDecision, failure: Exception, fail_when_content_contains: str):
        self.model = "fake-fault-injecting-model"
        self.max_tokens = 2048
        self._decision = decision
        self._failure = failure
        self._trigger = fail_when_content_contains

    def with_structured_output(self, schema, include_raw: bool = False):
        assert not include_raw, "agents.leveling._run_leveling_call always uses include_raw=False"
        return self

    def invoke(self, messages):
        content = " ".join(m["content"] if isinstance(m, dict) else str(m) for m in messages)
        if self._trigger in content:
            raise self._failure
        return self._decision


class FakeAIMessage:
    """Minimal double for what agents/pricing_agent.py reads off a bind_tools response:
    .content and .tool_calls (a list of {"name", "args", "id"} dicts -- empty once the model
    is done calling tools, matching a real AIMessage with no tool calls)."""

    def __init__(self, tool_calls: list[dict] | None = None, content: str = ""):
        self.tool_calls = tool_calls or []
        self.content = content
        self.usage_metadata = {"input_tokens": 100, "output_tokens": 50}


class FakeBoundTools:
    """Returned by FakeToolCallingModel.bind_tools. `responses` is one FakeAIMessage per
    tool-calling turn, in order; the last entry (no tool_calls) repeats if invoked past the
    end, so a test doesn't need to predict exactly how many turns a loop will take."""

    def __init__(self, responses: list["FakeAIMessage"]):
        self._responses = responses
        self.call_count = 0

    def invoke(self, messages):
        response = self._responses[min(self.call_count, len(self._responses) - 1)]
        self.call_count += 1
        return response


class FakeToolCallingModel:
    """Fakes both halves of a bind_tools agent (agents/pricing_agent.py): the tool-calling
    loop (a fixed sequence of FakeAIMessage turns, ending in one with no tool_calls) and the
    follow-up with_structured_output call for the final judgment."""

    def __init__(self, tool_call_responses: list[FakeAIMessage], judgment):
        self.model = "fake-tool-calling-model"
        self.max_tokens = 2048
        self._tool_call_responses = tool_call_responses
        self._judgment = judgment

    def bind_tools(self, tools, context: str | None = None):
        return FakeBoundTools(self._tool_call_responses)

    def with_structured_output(self, schema):
        return FakeStructuredModel(self._judgment, include_raw=False)
