"""Standalone Pinecone smoke test (SETUP.md step E): create the index, upsert three
vectors, query, delete. Proves the key, index config, and Nebius embeddings work together
before wiring retrieval into any agent -- does not import anything from agents/pricing_agent.py
or touch the real survey-job corpus index (scripts/embed_survey_jobs.py's index name).

Uses three real embeddings (via agents.model_router.get_embedding_model(), already verified
live: Qwen/Qwen3-Embedding-8B, 4096-dim -- see that module's docstring for why this isn't
BAAI/bge-en-icl despite CLAUDE.md's stack table naming it) rather than random vectors, so a
clean run here is evidence the whole embed -> index -> query path works, not just that
Pinecone's API responds to arbitrary floats.
"""

import os

from dotenv import load_dotenv

load_dotenv()

from agents.model_router import EMBEDDING_DIMENSION, get_embedding_model
from tools.vector_store import delete_index, ensure_index, get_client, query_similar, upsert_vectors

SMOKE_TEST_INDEX = "meridian-smoke-test"

# Three deliberately distinct roles so a query against one of them should retrieve itself
# as the closest match, not one of the other two -- a real signal the embeddings and the
# index are actually doing semantic matching, not just accepting arbitrary vectors.
SAMPLE_JOBS = {
    "job-1": "Senior analog design engineer, owns the bandgap reference and LDO regulator circuits for a power management IC.",
    "job-2": "Engineering manager leading a six-person embedded firmware team building RTOS integration for a sensor platform.",
    "job-3": "Principal RTL design engineer, sets timing closure methodology and sign-off criteria across every tapeout company-wide.",
}


def main():
    client = get_client(os.environ["PINECONE_API_KEY"])
    embeddings = get_embedding_model()

    print(f"1. Creating index {SMOKE_TEST_INDEX!r} (dimension={EMBEDDING_DIMENSION}, metric=cosine, serverless)...")
    ensure_index(client, SMOKE_TEST_INDEX, dimension=EMBEDDING_DIMENSION)
    print("   done.")

    print("2. Embedding 3 sample job descriptions and upserting...")
    vectors = [
        {"id": job_id, "values": embeddings.embed_query(text), "metadata": {"text": text}}
        for job_id, text in SAMPLE_JOBS.items()
    ]
    result = upsert_vectors(client, SMOKE_TEST_INDEX, vectors)
    print(f"   upserted_count: {result['upserted_count']}")

    print("3. Querying with job-3's own text -- expecting job-3 as the top match...")
    query_vector = embeddings.embed_query(SAMPLE_JOBS["job-3"])
    matches = query_similar(client, SMOKE_TEST_INDEX, query_vector, top_k=3)
    for m in matches:
        print(f"   {m['id']}: score={m['score']:.4f}")
    top_match = matches[0]["id"] if matches else None
    print(f"   top match: {top_match} -- {'PASS' if top_match == 'job-3' else 'UNEXPECTED'}")

    print(f"4. Deleting index {SMOKE_TEST_INDEX!r}...")
    delete_index(client, SMOKE_TEST_INDEX)
    print("   done.")

    print("\nSMOKE TEST COMPLETE.")


if __name__ == "__main__":
    main()
