"""Pinecone index management for the survey-job retrieval corpus (CLAUDE.md's locked
stack: "Vector | Pinecone... dimension 4096, cosine, serverless").

Uses the bare `pinecone` SDK directly, not `langchain-pinecone`'s LangChain wrapper: this
codebase doesn't use LangChain's retriever/VectorStore abstractions anywhere else (every
model call already goes through plain with_structured_output/bind_tools, not a chain), and
langchain-pinecone's real releases cap at Requires-Python <3.14 -- confirmed by a live pip
install attempt failing on this Python 3.14 environment with no compatible wheel for its
pinned simsimd dependency, even after checking the latest release's actual requirement
(simsimd>=5.9.11, satisfiable on its own) against the package's own Requires-Python metadata
(<3.14, the real blocker). The bare SDK has no such constraint and is a strict functional
equivalent for what this codebase needs: create an index, upsert vectors with metadata,
query by vector, delete. Revisit if a langchain-pinecone release drops the <3.14 cap.

Every function takes an explicit Pinecone client rather than constructing one internally, so
a caller controls where the api_key comes from (agents.secrets.get_secret, matching every
other credential in this codebase) and a test can pass a fake client instead.
"""

from __future__ import annotations

from pinecone import Pinecone, ServerlessSpec


def get_client(api_key: str) -> Pinecone:
    return Pinecone(api_key=api_key)


def ensure_index(
    client: Pinecone,
    index_name: str,
    dimension: int,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
) -> None:
    """Creates the index if it doesn't already exist. Idempotent, so the embedding script
    (scripts/embed_survey_jobs.py) can be re-run safely, per SETUP.md's "write the embedding
    script so it's re-runnable." Raises ValueError if an index by this name already exists
    with a different dimension or metric -- the whole retrieval feature depends on both
    matching the embedding model exactly, and a silent mismatch would corrupt every query
    with no error at all until someone notices the results look wrong.
    """
    existing = {idx.name: idx for idx in client.list_indexes()}
    if index_name in existing:
        info = existing[index_name]
        if info.dimension != dimension or info.metric != metric:
            raise ValueError(
                f"Index {index_name!r} already exists with dimension={info.dimension} "
                f"metric={info.metric!r}, not the requested dimension={dimension} "
                f"metric={metric!r}. Delete it first if you actually want to change these."
            )
        return
    client.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )


def upsert_vectors(client: Pinecone, index_name: str, vectors: list[dict]) -> dict:
    """vectors: [{"id": ..., "values": [...], "metadata": {...}}, ...]. Returns
    {"upserted_count": int} -- Pinecone's own confirmation of what was actually written,
    same provenance discipline as tools/decisions.py's write returning its persisted record."""
    index = client.Index(index_name)
    response = index.upsert(vectors=vectors)
    return {"upserted_count": response.upserted_count}


def query_similar(client: Pinecone, index_name: str, query_vector: list[float], top_k: int = 5) -> list[dict]:
    """Returns plain dicts: [{"id", "score", "metadata"}, ...], ordered by score descending
    (Pinecone's own order) -- the closest top_k matches, not a judgment about which to use."""
    index = client.Index(index_name)
    response = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return [{"id": m.id, "score": m.score, "metadata": m.metadata or {}} for m in response.matches]


def delete_index(client: Pinecone, index_name: str) -> None:
    client.delete_index(index_name)
