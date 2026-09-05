"""One function decides provider per CLAUDE.md's model routing section, so provider choice
is never scattered through agent code.

Every model returned here is wrapped in InstrumentedModel (agents/instrumented_model.py):
caching, cost logging, session stats, and the spend budget apply to any agent that gets its
model through get_model(), automatically -- an agent author writes plain
with_structured_output(schema).invoke(messages) and inherits all four without knowing they
exist. That's deliberate: the alternative is every new agent needing to remember to wire
this up itself, which is exactly the kind of thing that quietly doesn't happen.

"volume" -> Nebius Token Factory (job description parsing, title normalization, survey
match candidates, first-pass batch leveling). "judgment" -> Claude (leveling adjudication on
low-confidence cases, pricing judgment, reviewer, crosswalk arbitration, M&A synthesis).
base_url, model name and pricing were all verified live against Nebius, not assumed --
scripts/smoke_test_nebius.py and agents/cost_logging.py's PRICING comment carry that trail.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from agents.instrumented_model import InstrumentedModel
from agents.secrets import get_secret

# Leveling adjudication, pricing judgment, the reviewer agent, crosswalk arbitration, and
# M&A synthesis all stay on Claude per CLAUDE.md's model routing table.
_JUDGMENT_MODEL = "claude-sonnet-5"

# Confirmed live via scripts/smoke_test_nebius.py: base_url moved off api.studio.nebius.ai
# to Token Factory, and this model id was checked against a live GET /v1/models rather than
# the docs' own (stale, since-retired) quickstart example.
_NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
_VOLUME_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# Every client this router constructs gets this explicit request timeout -- both SDKs
# (anthropic, openai) otherwise fall back to their own internal defaults (10 minutes for
# each, confirmed against both packages' client source), which is indistinguishable from a
# genuine hang for anything in this codebase's own call path: a Streamlit page or a batch
# script waiting 10 minutes on one stuck request looks identical to one that will never
# return. 30s is long enough for a real structured-output call with reasoning (observed
# live calls finish in low single-digit seconds) and short enough that a stuck connection
# fails fast into agents.instrumented_model's retry-with-backoff instead of stalling the
# whole run on it. Passed as `timeout=` on every client (ChatAnthropic, ChatOpenAI,
# OpenAIEmbeddings all accept it, confirmed directly -- LangChain aliases it to each
# provider's own field name, default_request_timeout for Anthropic and request_timeout for
# OpenAI, so one constant covers both without needing a per-provider kwarg name).
REQUEST_TIMEOUT_SECONDS = 30

# CLAUDE.md's stack table names BAAI/bge-en-icl -- confirmed via a live GET /v1/models call
# (this session) that it does not exist on the current Token Factory catalog (404). The only
# embedding model actually live there right now is Qwen/Qwen3-Embedding-8B, confirmed to
# return exactly the 4096-dim vectors the locked stack's Pinecone index config requires (same
# "verify against live, not docs" drift this file's chat model name already documents).
_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBEDDING_DIMENSION = 4096


def get_model(tier: str):
    if tier == "judgment":
        return InstrumentedModel(
            ChatAnthropic(model=_JUDGMENT_MODEL, max_tokens=2048, timeout=REQUEST_TIMEOUT_SECONDS)
        )
    if tier == "volume":
        # ChatOpenAI (unlike ChatAnthropic) raises OpenAIError at construction time -- not
        # at the first real call -- when it has no api_key and no OPENAI_API_KEY env var at
        # all. Demo mode's whole point is running with zero API keys present, and its
        # cache-only guard (agents.instrumented_model) only intercepts a call at .invoke()
        # time, after this constructor has already run -- so a placeholder string here (used
        # only if no real key is configured) keeps construction from crashing before demo
        # mode ever gets a chance to serve a cache hit or refuse a miss. The placeholder is
        # never sent anywhere: cache-only mode raises before this client's transport is used
        # on a miss, and any cache hit never touches the network at all.
        return InstrumentedModel(
            ChatOpenAI(
                base_url=_NEBIUS_BASE_URL,
                api_key=get_secret("NEBIUS_API_KEY", default="demo-mode-no-key-configured"),
                model=_VOLUME_MODEL,
                max_tokens=2048,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        )
    raise ValueError(f"Unknown model tier: {tier!r}")


def get_embedding_model() -> InstrumentedModel:
    """Nebius Qwen3-Embedding-8B, wrapped for the same caching/cost-logging/demo-mode
    guard as every other model this router returns -- a live query embedding at pricing
    time must not be a hole in the demo-mode guarantee that no live API call happens
    without the unlock password (agents.instrumented_model / app.demo_mode).

    check_embedding_ctx_length=False is required, not optional, against this endpoint:
    LangChain's OpenAIEmbeddings default pre-tokenizes long inputs with tiktoken and sends
    integer token arrays instead of raw text; Nebius's endpoint rejects that with "Tokenized
    input is not supported" (confirmed live). Disabling it sends plain strings instead.
    """
    return InstrumentedModel(
        OpenAIEmbeddings(
            base_url=_NEBIUS_BASE_URL,
            api_key=get_secret("NEBIUS_API_KEY", default="demo-mode-no-key-configured"),
            model=_EMBEDDING_MODEL,
            check_embedding_ctx_length=False,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    )
