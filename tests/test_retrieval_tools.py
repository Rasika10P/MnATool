"""Tests for tools/retrieval_tools.py -- specifically the cache + demo-mode guard around
the combined (embed, Pinecone query) call, since that's the part with no test coverage
elsewhere (agents/instrumented_model.py's own tests cover the embedding half in isolation).
Nebius and Pinecone are both faked; conftest.py's autouse fixture already isolates the disk
cache this module reads/writes through to a per-test tmp_path.
"""

import pytest

import tools.retrieval_tools as retrieval_tools
from agents.instrumented_model import DemoModeCacheMissError, set_cache_only

CANDIDATE_MATCH = {
    "id": "SYN-001-ANA-AD-L1",
    "score": 0.6,
    "metadata": {
        "survey_job_title": "Associate Engineer - Analog Design",
        "discipline": "Analog Design",
        "survey_level_label": "Radleigh-L1",
        "survey_job_description": "test description",
    },
}


class _FakeEmbeddings:
    def __init__(self):
        self.call_count = 0

    def embed_query(self, text):
        self.call_count += 1
        return [0.1, 0.2, 0.3]


@pytest.fixture(autouse=True)
def reset_cache_only():
    set_cache_only(False)
    yield
    set_cache_only(False)


def _patch_live_calls(monkeypatch, fake_embeddings, query_calls):
    monkeypatch.setattr(retrieval_tools, "get_embedding_model", lambda: fake_embeddings)
    monkeypatch.setattr(retrieval_tools, "get_client", lambda api_key: "fake-client")

    def fake_query_similar(client, index_name, query_vector, top_k):
        query_calls.append((index_name, query_vector, top_k))
        return [CANDIDATE_MATCH]

    monkeypatch.setattr(retrieval_tools, "query_similar", fake_query_similar)


def test_live_call_shapes_candidates_from_pinecone_matches(monkeypatch):
    fake_embeddings = _FakeEmbeddings()
    query_calls = []
    _patch_live_calls(monkeypatch, fake_embeddings, query_calls)

    result = retrieval_tools.retrieve_similar_survey_jobs.invoke({"query_text": "senior analog design", "top_k": 3})

    assert query_calls == [(retrieval_tools.CORPUS_INDEX_NAME, [0.1, 0.2, 0.3], 3)]
    assert result == [
        {
            "survey_code": "SYN-001-ANA-AD-L1",
            "score": 0.6,
            "survey_job_title": "Associate Engineer - Analog Design",
            "discipline": "Analog Design",
            "survey_level_label": "Radleigh-L1",
            "survey_job_description": "test description",
        }
    ]


def test_second_identical_call_hits_cache_not_pinecone_or_nebius(monkeypatch):
    fake_embeddings = _FakeEmbeddings()
    query_calls = []
    _patch_live_calls(monkeypatch, fake_embeddings, query_calls)

    retrieval_tools.retrieve_similar_survey_jobs.invoke({"query_text": "senior analog design", "top_k": 3})
    retrieval_tools.retrieve_similar_survey_jobs.invoke({"query_text": "senior analog design", "top_k": 3})

    assert fake_embeddings.call_count == 1, "second call should be served from the tool-level cache"
    assert len(query_calls) == 1


def test_different_top_k_is_a_different_cache_entry(monkeypatch):
    fake_embeddings = _FakeEmbeddings()
    query_calls = []
    _patch_live_calls(monkeypatch, fake_embeddings, query_calls)

    retrieval_tools.retrieve_similar_survey_jobs.invoke({"query_text": "senior analog design", "top_k": 3})
    retrieval_tools.retrieve_similar_survey_jobs.invoke({"query_text": "senior analog design", "top_k": 5})

    assert len(query_calls) == 2


def test_demo_mode_blocks_a_cache_miss_without_calling_nebius_or_pinecone(monkeypatch):
    fake_embeddings = _FakeEmbeddings()
    query_calls = []
    _patch_live_calls(monkeypatch, fake_embeddings, query_calls)

    set_cache_only(True)
    with pytest.raises(DemoModeCacheMissError):
        retrieval_tools.retrieve_similar_survey_jobs.invoke({"query_text": "never cached", "top_k": 3})

    assert fake_embeddings.call_count == 0
    assert query_calls == []


def test_demo_mode_still_serves_a_warm_cache_hit(monkeypatch):
    fake_embeddings = _FakeEmbeddings()
    query_calls = []
    _patch_live_calls(monkeypatch, fake_embeddings, query_calls)

    retrieval_tools.retrieve_similar_survey_jobs.invoke({"query_text": "warm query", "top_k": 3})
    set_cache_only(True)
    result = retrieval_tools.retrieve_similar_survey_jobs.invoke({"query_text": "warm query", "top_k": 3})

    assert result[0]["survey_code"] == "SYN-001-ANA-AD-L1"
    assert fake_embeddings.call_count == 1
