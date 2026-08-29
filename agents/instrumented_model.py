"""Wraps a chat model so caching, cost logging, session stats, retry-on-malformed-output,
and the spend budget apply to every structured-output call automatically -- any agent that
gets its model from get_model() inherits all five by construction, without writing or
remembering to wire up any of this itself. This is the single place those concerns live;
agents/leveling.py and every future agent just call .with_structured_output(schema).invoke(
messages) like plain LangChain code.

Matches plain LangChain's with_structured_output(schema) (include_raw defaulted False)
return shape exactly: .invoke(messages) returns the parsed Pydantic object, or raises an
exception. include_raw isn't offered as a caller-facing toggle -- nothing in this codebase
needs the raw message once caching/logging have already extracted what they need from it,
and offering a mode that skips instrumentation would defeat the point.

Retry: docs/error_handling_backlog.md documents repeated cases (across LevelingDecision,
ScopeProfile, and AdvocateOutput) of the model's tool call not exactly matching the schema
-- a leaked tag, a missing field, an extra nesting level -- that a second identical call
then returned cleanly. MAX_ATTEMPTS retries the same call on any such parsing failure before
giving up; every attempt (successful or not) gets its own cost-log entry via log_call's
`attempt` parameter, so the retry rate is visible in the persistent log, not just inferred
from a raised exception. This subsumes backlog entries 2 and 4's shared root cause (nothing
retried a malformed structured-output call) -- but not entry 2's other half: a caller that
wants one bad row to become a per-item failure record instead of a raised exception (e.g.
the batch fan-out continuing past one employee) still has to catch StructuredOutputError
itself. This wrapper's job ends at "retry, then raise clearly."

Entry 1 specifically -- a leaked tag corrupting a field *without* raising a ValidationError
at all (its own example: the tag ate the JSON structure and alternative_level came back null
instead of raising) -- is not caught by the parsing-error retry above, since Pydantic never
objects. `invoke` below checks the raw tool-call arguments for leaked tag syntax
independently of whether parsing succeeded, and retries on that too. Every schema's own
prose fields also run a field validator that strips any leaked syntax on construction
(agents/text_sanitization.py) -- an independent guarantee, so a value that still leaks after
every retry is exhausted is used anyway (sanitized) rather than dropping the decision
entirely, consistent with how a parsing-error retry that never clears still gets a decision
recorded (StructuredOutputError) rather than silently vanishing.
"""

from __future__ import annotations

import threading

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

from agents.cost_logging import log_call
from agents.llm_cache import get_cached, set_cached
from agents.spend_guard import get_default_budget
from agents.text_sanitization import find_leaked_tag_strings

MAX_ATTEMPTS = 3

# Demo-mode enforcement: when active, a cache miss raises instead of making a real call --
# not a soft warning, since demo mode's whole guarantee (a public Streamlit Cloud visitor
# with no unlock password can never spend the deployed owner's API budget) depends on this
# being unconditional. Thread-local, not a plain module global: Streamlit runs each active
# session's script execution in its own thread, and a plain global would leak one user's
# unlocked live mode into every other concurrent user's session on the same deployment.
#
# Defaults to False (cache-only OFF) so every existing script, test, and non-Streamlit call
# path keeps its original behavior unchanged -- this is an opt-in Streamlit-deployment safety
# feature, not a system-wide default. app/demo_mode.py's render_and_apply_gate() is what
# turns it on, and it must run at the top of every Streamlit page that could reach a model
# call, before that page does anything else -- the same "you must call this" discipline this
# codebase already expects of reset_session_stats()/reset_default_budget() at the top of a run.
_thread_local = threading.local()


def set_cache_only(enabled: bool) -> None:
    _thread_local.cache_only = enabled


def is_cache_only() -> bool:
    return getattr(_thread_local, "cache_only", False)


class DemoModeCacheMissError(RuntimeError):
    """Raised instead of making a real model call when cache-only ("demo") mode is active
    and this exact (model, schema-or-tools, messages) isn't already in the warmed cache."""


class StructuredOutputError(Exception):
    """Raised when a structured-output call still fails to validate after MAX_ATTEMPTS
    consecutive tries. `attempts` holds every parsing exception encountered, oldest first,
    for whoever's debugging this -- not just the last one, since docs/error_handling_backlog.md
    entries 1 and 4 show the failure shape itself can differ attempt to attempt."""

    def __init__(self, schema_name: str, model_name: str, attempts: list[Exception]):
        self.schema_name = schema_name
        self.model_name = model_name
        self.attempts = attempts
        super().__init__(
            f"{schema_name} structured output failed to validate on all {len(attempts)} "
            f"attempts against {model_name}. Last error: {attempts[-1]!r}"
        )


class TagLeakDetected(Exception):
    """Recorded in an attempt's error list (module docstring's entry-1 fix) when the raw
    tool call's arguments contained tag-like syntax, even though Pydantic validation itself
    succeeded. Never raised past `invoke` -- it's collected the same way a genuine
    ValidationError is, purely so the retry loop below has a uniform reason to point to and
    StructuredOutputError.attempts stays complete if every attempt ends up exhausted."""

    def __init__(self, schema_name: str, leaked_strings: list[str]):
        self.schema_name = schema_name
        self.leaked_strings = leaked_strings
        super().__init__(f"{schema_name}: tag-like syntax leaked into {len(leaked_strings)} field(s): {leaked_strings!r}")


def _detect_provider(llm) -> str:
    if isinstance(llm, ChatAnthropic):
        return "anthropic"
    if isinstance(llm, (ChatOpenAI, OpenAIEmbeddings)):
        return "nebius"  # the only thing ChatOpenAI/OpenAIEmbeddings is ever pointed at here
    return "unknown"


def _message_content(message) -> str:
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", message))


def _raw_tool_call_args(raw_message) -> dict:
    """The raw (pre-Pydantic) arguments dict from a structured-output call's underlying tool
    call, for scanning independent of whatever the schema's own field validators later do to
    the parsed object -- confirmed directly against a real ChatAnthropic call:
    with_structured_output(..., include_raw=True)'s `raw` AIMessage carries `.tool_calls`, a
    list of {"name", "args", "id"} dicts where `args` is already a parsed dict. Defensive:
    a raw message with no tool_calls (a shape this wrapper hasn't seen in production, or a
    test double that doesn't set one) has nothing to scan, not an error."""
    tool_calls = getattr(raw_message, "tool_calls", None) or []
    if not tool_calls:
        return {}
    return tool_calls[0].get("args", {})


def _cache_key_parts(schema: type[BaseModel], messages: list) -> list[str]:
    # schema name is part of the key: the same messages requested against a different
    # schema is a different question and must not collide in the cache.
    return [schema.__name__] + [_message_content(m) for m in messages]


def would_hit_cache(llm, schema: type[BaseModel], messages: list) -> bool:
    """Reports whether this exact (model, schema, messages) would be served from cache,
    without making any call or touching the budget -- what --dry-run reports on."""
    model_name = getattr(llm, "model", None) or getattr(llm, "model_name", None) or "unknown"
    return get_cached(model_name, _cache_key_parts(schema, messages)) is not None


class _InstrumentedStructuredRunnable:
    def __init__(self, llm, schema: type[BaseModel]):
        self._structured_llm = llm.with_structured_output(schema, include_raw=True)
        self._schema = schema
        self._model_name = getattr(llm, "model", None) or getattr(llm, "model_name", None) or "unknown"
        self._provider = _detect_provider(llm)
        self._max_output_tokens = getattr(llm, "max_tokens", None) or 2048
        self._context = schema.__name__

    def invoke(self, messages, *args, **kwargs):
        prompt_parts = _cache_key_parts(self._schema, messages)

        cached = get_cached(self._model_name, prompt_parts)
        if cached is not None:
            log_call(
                self._model_name, cached["input_tokens"], cached["output_tokens"],
                cached=True, context=self._context, provider=self._provider,
            )
            return self._schema(**cached["decision"])

        if is_cache_only():
            raise DemoModeCacheMissError(
                f"Demo mode is active (no live API calls allowed) and this exact "
                f"{self._schema.__name__} call against {self._model_name} isn't in the "
                "warmed cache. Unlock live mode to run it for real."
            )

        errors: list[Exception] = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            # Checked before every attempt, retries included -- each is a real, separately
            # billed call, so a retry can push a run over budget just like any other call.
            get_default_budget().check_before_call(self._model_name, prompt_parts, self._max_output_tokens)

            result = self._structured_llm.invoke(messages, *args, **kwargs)
            usage = result["raw"].usage_metadata or {}
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            logged = log_call(
                self._model_name, input_tokens, output_tokens,
                cached=False, context=self._context, provider=self._provider, attempt=attempt,
            )
            get_default_budget().record(logged["cost_usd"])

            if result["parsing_error"] is not None:
                errors.append(result["parsing_error"])
                continue

            leaked = find_leaked_tag_strings(_raw_tool_call_args(result["raw"]))
            is_last_attempt = attempt == MAX_ATTEMPTS
            if leaked:
                # module docstring's entry-1 fix: Pydantic validated cleanly, but the raw
                # tool call still leaked tag syntax -- a fresh retry is strictly better than
                # accepting a response known to have glitched mid-generation, so this is
                # treated the same as a parsing failure for retry purposes.
                errors.append(TagLeakDetected(self._schema.__name__, leaked))
                if not is_last_attempt:
                    continue
                print(
                    f"[instrumented_model] {self._schema.__name__}: tag-like syntax leaked "
                    f"into the raw tool call on the final attempt ({MAX_ATTEMPTS}/{MAX_ATTEMPTS}) "
                    "-- accepting the field-validator-sanitized result rather than dropping "
                    "this decision.",
                    flush=True,
                )

            decision = result["parsed"]
            set_cached(
                self._model_name, prompt_parts,
                {"decision": decision.model_dump(mode="json"), "input_tokens": input_tokens, "output_tokens": output_tokens},
            )
            return decision

        raise StructuredOutputError(self._schema.__name__, self._model_name, errors)


class _InstrumentedToolCallingRunnable:
    """bind_tools' counterpart to _InstrumentedStructuredRunnable above. No schema to
    validate against and so no retry-on-malformed-output loop (a tool-calling turn is just an
    AIMessage; there's nothing to parse) -- but the same budget check, cost log entry, and
    disk cache apply, keyed on (model, full message list) exactly like the structured path,
    so a repeated tool-calling loop (e.g. re-running a Streamlit demo against the same inputs)
    is served from cache turn-for-turn rather than re-billed.
    """

    def __init__(self, llm, tools, context: str):
        self._bound_llm = llm.bind_tools(tools)
        self._model_name = getattr(llm, "model", None) or getattr(llm, "model_name", None) or "unknown"
        self._provider = _detect_provider(llm)
        self._max_output_tokens = getattr(llm, "max_tokens", None) or 2048
        self._context = context

    def invoke(self, messages, *args, **kwargs):
        from langchain_core.messages import AIMessage

        prompt_parts = [_message_content(m) for m in messages]

        cached = get_cached(self._model_name, prompt_parts)
        if cached is not None:
            log_call(
                self._model_name, cached["input_tokens"], cached["output_tokens"],
                cached=True, context=self._context, provider=self._provider,
            )
            return AIMessage(content=cached["content"], tool_calls=cached["tool_calls"])

        if is_cache_only():
            raise DemoModeCacheMissError(
                f"Demo mode is active (no live API calls allowed) and this tool-calling turn "
                f"against {self._model_name} isn't in the warmed cache. Unlock live mode to "
                "run it for real."
            )

        get_default_budget().check_before_call(self._model_name, prompt_parts, self._max_output_tokens)

        result = self._bound_llm.invoke(messages, *args, **kwargs)
        usage = result.usage_metadata or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        logged = log_call(
            self._model_name, input_tokens, output_tokens,
            cached=False, context=self._context, provider=self._provider,
        )
        get_default_budget().record(logged["cost_usd"])

        set_cached(
            self._model_name, prompt_parts,
            {
                "content": result.content,
                "tool_calls": result.tool_calls,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )
        return result


class _InstrumentedEmbeddingsRunnable:
    """Embeddings' counterpart to the two runnables above: same cache-by-(model, texts),
    demo-mode cache-only guard, and cost log entry. No retry-on-malformed-output (a vector
    has nothing to validate against a schema) and no tool_calls -- embed_documents just
    returns list[list[float]], one vector per input text.

    Token counts are the same chars/4 heuristic agents/spend_guard.py already uses for
    pre-call budget projection, not a real usage figure -- LangChain's OpenAIEmbeddings
    interface (unlike ChatOpenAI's AIMessage) discards the API response's own usage block,
    so there's nothing exact to log without bypassing it. Acceptable here the same way it's
    acceptable for spend_guard's own projections: an approximation used consistently, not a
    regression from some more precise number this call shape ever had.
    """

    def __init__(self, embeddings_client, context: str):
        self._client = embeddings_client
        self._model_name = getattr(embeddings_client, "model", None) or "unknown"
        self._provider = _detect_provider(embeddings_client)
        self._context = context

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prompt_parts = [self._model_name] + list(texts)

        cached = get_cached(self._model_name, prompt_parts)
        if cached is not None:
            log_call(
                self._model_name, cached["input_tokens"], 0,
                cached=True, context=self._context, provider=self._provider,
            )
            return cached["vectors"]

        if is_cache_only():
            raise DemoModeCacheMissError(
                f"Demo mode is active (no live API calls allowed) and this embedding call "
                f"against {self._model_name} ({len(texts)} text(s)) isn't in the warmed cache. "
                "Unlock live mode to run it for real."
            )

        estimated_input_tokens = sum(len(t) for t in texts) // 4
        get_default_budget().check_before_call(self._model_name, prompt_parts, max_output_tokens=0)

        vectors = self._client.embed_documents(texts)
        logged = log_call(
            self._model_name, estimated_input_tokens, 0,
            cached=False, context=self._context, provider=self._provider,
        )
        get_default_budget().record(logged["cost_usd"])

        set_cached(self._model_name, prompt_parts, {"vectors": vectors, "input_tokens": estimated_input_tokens})
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class InstrumentedModel:
    """Drop-in wrapper: with_structured_output(schema).invoke(messages) gets caching, cost
    logging, session stats and the spend budget for free. Any other attribute (`.model`,
    `.max_tokens`, provider-specific methods, ...) falls through to the wrapped model
    unchanged, so this can stand in anywhere a plain chat model is expected.
    """

    def __init__(self, llm):
        self._llm = llm

    def with_structured_output(self, schema: type[BaseModel], **kwargs):
        return _InstrumentedStructuredRunnable(self._llm, schema)

    def bind_tools(self, tools, context: str | None = None, **kwargs):
        context = context or "+".join(sorted(getattr(t, "name", str(t)) for t in tools))
        return _InstrumentedToolCallingRunnable(self._llm, tools, context)

    def embed_documents(self, texts: list[str], context: str = "embeddings") -> list[list[float]]:
        return _InstrumentedEmbeddingsRunnable(self._llm, context).embed_documents(texts)

    def embed_query(self, text: str, context: str = "embeddings") -> list[float]:
        return _InstrumentedEmbeddingsRunnable(self._llm, context).embed_query(text)

    def __getattr__(self, name):
        return getattr(self._llm, name)
