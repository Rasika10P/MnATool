"""Unit tests for tools/vector_store.py against a fake Pinecone client -- the wrapper logic
(idempotent ensure_index, the dimension/metric mismatch guard, response reshaping) is
testable without a real Pinecone account. scripts/smoke_test_pinecone.py is the live,
real-account version of this same claim.
"""

from dataclasses import dataclass, field

import pytest

from tools.vector_store import delete_index, ensure_index, query_similar, upsert_vectors


@dataclass
class _FakeIndexInfo:
    name: str
    dimension: int
    metric: str


@dataclass
class _FakeUpsertResponse:
    upserted_count: int


@dataclass
class _FakeMatch:
    id: str
    score: float
    metadata: dict


@dataclass
class _FakeQueryResponse:
    matches: list


class _FakeIndex:
    def __init__(self, store: dict):
        self._store = store

    def upsert(self, vectors):
        for v in vectors:
            self._store[v["id"]] = v
        return _FakeUpsertResponse(upserted_count=len(vectors))

    def query(self, vector, top_k, include_metadata=False):
        # Deterministic "similarity": exact stored vector match scores 1.0, else 0.0 --
        # enough to test ordering/shaping without a real distance metric.
        scored = [
            _FakeMatch(id=vid, score=1.0 if v["values"] == vector else 0.0, metadata=v.get("metadata"))
            for vid, v in self._store.items()
        ]
        scored.sort(key=lambda m: m.score, reverse=True)
        return _FakeQueryResponse(matches=scored[:top_k])


class FakePineconeClient:
    def __init__(self, existing_indexes: list[_FakeIndexInfo] | None = None):
        self._indexes = {idx.name: idx for idx in (existing_indexes or [])}
        self._stores: dict[str, dict] = {name: {} for name in self._indexes}
        self.created = []
        self.deleted = []

    def list_indexes(self):
        return list(self._indexes.values())

    def create_index(self, name, dimension, metric, spec):
        self._indexes[name] = _FakeIndexInfo(name=name, dimension=dimension, metric=metric)
        self._stores[name] = {}
        self.created.append((name, dimension, metric))

    def Index(self, name):
        return _FakeIndex(self._stores[name])

    def delete_index(self, name):
        del self._indexes[name]
        del self._stores[name]
        self.deleted.append(name)


def test_ensure_index_creates_when_absent():
    client = FakePineconeClient()
    ensure_index(client, "test-index", dimension=4096, metric="cosine")
    assert client.created == [("test-index", 4096, "cosine")]


def test_ensure_index_is_a_noop_when_matching_index_already_exists():
    client = FakePineconeClient([_FakeIndexInfo("test-index", 4096, "cosine")])
    ensure_index(client, "test-index", dimension=4096, metric="cosine")
    assert client.created == []


def test_ensure_index_raises_on_dimension_mismatch():
    client = FakePineconeClient([_FakeIndexInfo("test-index", 1536, "cosine")])
    with pytest.raises(ValueError, match="dimension=1536"):
        ensure_index(client, "test-index", dimension=4096, metric="cosine")


def test_ensure_index_raises_on_metric_mismatch():
    client = FakePineconeClient([_FakeIndexInfo("test-index", 4096, "euclidean")])
    with pytest.raises(ValueError, match="metric='euclidean'"):
        ensure_index(client, "test-index", dimension=4096, metric="cosine")


def test_upsert_and_query_round_trip():
    client = FakePineconeClient([_FakeIndexInfo("test-index", 3, "cosine")])
    vectors = [
        {"id": "a", "values": [1.0, 0.0, 0.0], "metadata": {"title": "Job A"}},
        {"id": "b", "values": [0.0, 1.0, 0.0], "metadata": {"title": "Job B"}},
    ]
    result = upsert_vectors(client, "test-index", vectors)
    assert result == {"upserted_count": 2}

    matches = query_similar(client, "test-index", query_vector=[1.0, 0.0, 0.0], top_k=2)
    assert matches[0]["id"] == "a"
    assert matches[0]["score"] == 1.0
    assert matches[0]["metadata"] == {"title": "Job A"}


def test_delete_index():
    client = FakePineconeClient([_FakeIndexInfo("test-index", 4096, "cosine")])
    delete_index(client, "test-index")
    assert client.deleted == ["test-index"]
    assert "test-index" not in {idx.name for idx in client.list_indexes()}
