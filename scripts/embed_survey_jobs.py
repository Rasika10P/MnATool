"""Embeds every survey_jobs description into the real, persistent Pinecone corpus index
(SETUP.md step E: "Embed the ~400 survey job descriptions once... write the embedding script
so it's re-runnable"). Re-runnable and idempotent: ensure_index is a no-op if the index
already exists with matching dimension/metric, and upserting by survey_code overwrites the
same vector rather than duplicating it, so running this again after a data regeneration
just refreshes every vector cleanly.

Commits nothing: the vectors live in Pinecone, not in this repo (CLAUDE.md non-negotiable 3
-- survey_jobs.parquet itself is the synthetic source of truth already committed; this script
only projects it into the vector index).

Standalone -- does not import agents/pricing_agent.py or anything that would make this
runnable from the deployed Streamlit app. This is a maintenance job the deployment owner
runs locally with their own real Pinecone/Nebius keys, the same category as data/generate.py,
not a live agent call subject to demo mode's cache-only guard.
"""

import argparse
import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from agents.model_router import EMBEDDING_DIMENSION, get_embedding_model
from tools.vector_store import ensure_index, get_client, upsert_vectors

SURVEY_JOBS_PATH = "data/parquet/survey_jobs.parquet"
CORPUS_INDEX_NAME = "meridian-survey-jobs"
BATCH_SIZE = 20  # keeps each upsert request small and each progress print meaningful


def main(limit: int | None, batch_size: int) -> None:
    df = pd.read_parquet(SURVEY_JOBS_PATH)
    if limit:
        df = df.head(limit)

    client = get_client(os.environ["PINECONE_API_KEY"])
    embeddings = get_embedding_model()

    print(f"Ensuring index {CORPUS_INDEX_NAME!r} (dimension={EMBEDDING_DIMENSION}, metric=cosine, serverless)...")
    ensure_index(client, CORPUS_INDEX_NAME, dimension=EMBEDDING_DIMENSION)

    total_upserted = 0
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start : start + batch_size]
        texts = batch["survey_job_description"].tolist()
        vectors_values = embeddings.embed_documents(texts)
        vectors = [
            {
                "id": row.survey_code,
                "values": values,
                "metadata": {
                    "survey_job_title": row.survey_job_title,
                    "survey_source": row.survey_source,
                    "discipline": row.discipline,
                    "survey_level_label": row.survey_level_label,
                    "survey_job_description": row.survey_job_description,
                },
            }
            for row, values in zip(batch.itertuples(index=False), vectors_values)
        ]
        result = upsert_vectors(client, CORPUS_INDEX_NAME, vectors)
        total_upserted += result["upserted_count"]
        print(f"  {start + len(batch)}/{len(df)} embedded and upserted (running total: {total_upserted})")

    print(f"\nDone. {total_upserted} vectors in {CORPUS_INDEX_NAME!r}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="max survey_jobs rows to embed (default: all 120)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    main(args.limit, args.batch_size)
