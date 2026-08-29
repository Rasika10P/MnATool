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
from a raised exception. This subsumes backlog entries 1, 2 and 4's shared root cause
(nothing retried a malformed structured-output call) -- but not entry 2's other half: a
caller that wants one bad row to become a per-item failure record instead of a raised
exception (e.g. the batch fan-out continuing past one employee) still has to catch
StructuredOutputError itself. This wrapper's job ends at "retry, then raise clearly."
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from agents.cost_logging import log_call
from agents.llm_cache import get_cached, set_cached
from agents.spend_guard import get_default_budget

MAX_ATTEMPTS = 3


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


def _detect_provider(llm) -> str:
    if isinstance(llm, ChatAnthropic):
        return "anthropic"
    if isinstance(llm, ChatOpenAI):
        return "nebius"  # the only thing ChatOpenAI is ever pointed at in this codebase
    return "unknown"


def _message_content(message) -> str:
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", message))


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

            if result["parsing_error"] is None:
                decision = result["parsed"]
                set_cached(
                    self._model_name, prompt_parts,
                    {"decision": decision.model_dump(mode="json"), "input_tokens": input_tokens, "output_tokens": output_tokens},
                )
                return decision

            errors.append(result["parsing_error"])

        raise StructuredOutputError(self._schema.__name__, self._model_name, errors)


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

    def __getattr__(self, name):
        return getattr(self._llm, name)
