"""Retrieval-as-candidate-generation over the survey-job corpus (build order item 11:
Pinecone via the bare SDK -- see tools/vector_store.py's docstring for why not
langchain-pinecone). A separate module from tools/agent_tools.py's six ASSIGNMENT.md tools:
this is a seventh, later addition, not a member of that fixed enumeration.

Retrieval narrows; it does not decide. This tool returns Pinecone's own top-K nearest
neighbors by cosine similarity over Nebius embeddings -- the calling agent is the one that
judges which (if any) of the returned candidates is actually the right match, the same way a
human comp analyst would treat a set of survey-cut suggestions as a starting point, not an
answer. Confirmed live (scripts/smoke_test_pinecone.py, scripts/embed_survey_jobs.py against
the real 120-job corpus) that cosine similarity over these embeddings reliably narrows by
discipline/topic but does not reliably rank by seniority within a discipline (a query for a
senior/principal analog role's top match came back an Associate Engineer posting, all five
matches scoring within 0.01 of each other) -- exactly why final judgment stays with the
agent, not this function.

Cache mode note: agents.model_router.get_embedding_model() already applies the active cache
mode to the embedding half of this call, but Pinecone's own query is a second live network
call with no such guard of its own -- InstrumentedModel wraps model calls, not arbitrary tool
functions. This module caches and mode-gates the whole (query_text, top_k) result as one
unit, directly against agents.llm_cache/agents.instrumented_model's primitives, so a demo-mode
visitor can't reach a live Pinecone call just because the embedding half happened to be a
cache hit, and so live mode genuinely bypasses both halves rather than just one.
"""

from __future__ import annotations

from langchain_core.tools import tool

from agents.instrumented_model import (
    CACHE_MODE_DEMO,
    DemoModeCacheMissError,
    get_cache_mode,
    read_cache_for_mode,
)
from agents.llm_cache import set_cached
from agents.model_router import get_embedding_model
from agents.secrets import get_secret
from tools.vector_store import get_client, query_similar

CORPUS_INDEX_NAME = "meridian-survey-jobs"
_CACHE_MODEL_KEY = f"pinecone:{CORPUS_INDEX_NAME}"


def _retrieve(query_text: str, top_k: int) -> list[dict]:
    prompt_parts = [query_text, str(top_k)]
    mode = get_cache_mode()

    cached = read_cache_for_mode(_CACHE_MODEL_KEY, prompt_parts, mode)
    if cached is not None:
        return cached["candidates"]

    if mode == CACHE_MODE_DEMO:
        raise DemoModeCacheMissError(
            f"Demo mode is active (no live API calls allowed) and this retrieval query "
            f"({query_text!r}, top_k={top_k}) isn't in the warmed cache. Switch to Live "
            "to run it for real."
        )

    embeddings = get_embedding_model()
    query_vector = embeddings.embed_query(query_text)

    client = get_client(get_secret("PINECONE_API_KEY", default="demo-mode-no-key-configured"))
    matches = query_similar(client, CORPUS_INDEX_NAME, query_vector, top_k=top_k)

    candidates = [
        {
            "survey_code": m["id"],
            "score": m["score"],
            "survey_job_title": m["metadata"].get("survey_job_title"),
            "discipline": m["metadata"].get("discipline"),
            "survey_level_label": m["metadata"].get("survey_level_label"),
            "survey_job_description": m["metadata"].get("survey_job_description"),
        }
        for m in matches
    ]
    set_cached(_CACHE_MODEL_KEY, prompt_parts, {"candidates": candidates})
    return candidates


@tool
def retrieve_similar_survey_jobs(query_text: str, top_k: int = 5) -> list[dict]:
    """Retrieve the top_k survey job descriptions most similar to query_text, by cosine
    similarity over Nebius embeddings against the ~120-job survey corpus. Returns candidates
    to consider, not an answer -- narrows the corpus down to a handful; judging which one (if
    any) actually matches, and at what confidence, is still yours to do, the same as a human
    comp analyst would treat a market survey cut someone pulled up as a starting point, not a
    verdict. Similarity reliably narrows by discipline/topic but not reliably by seniority
    within a discipline -- don't assume the top-scored candidate is the best-leveled one."""
    return _retrieve(query_text, top_k)
