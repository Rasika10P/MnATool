"""Standalone Nebius smoke test (SETUP.md step D). Proves the key, base_url and model name
work before touching any agent code -- does not import anything from agents/.

base_url and model confirmed live, not assumed: the endpoint moved from api.studio.nebius.ai
to Token Factory, re-confirmed by direct fetch of docs.tokenfactory.nebius.com (not just a
search snippet). The model name is verified against a live GET /v1/models call with this
key, not the docs' own quickstart example -- that example model (deepseek-ai/DeepSeek-R1-0528)
has since been retired from the live catalog, which is exactly the kind of drift "verify,
don't assume" is guarding against. Qwen3-30B-A3B-Instruct-2507 is confirmed live and is a
reasonable size/type for the "volume" tier's intended use (fast first-pass work), not a
heavy reasoning model.
"""

import os

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
NEBIUS_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"  # confirmed live via GET /v1/models

llm = ChatOpenAI(
    base_url=NEBIUS_BASE_URL,
    api_key=os.environ["NEBIUS_API_KEY"],
    model=NEBIUS_MODEL,
    max_tokens=100,  # trivial reply expected; an unfamiliar open model rambling is exactly
                     # the unbounded-response cost risk this guards against on a first call
)
response = llm.invoke("Reply with the single word: working")
print(response.content)
