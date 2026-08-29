# SETUP.md — Services and integration order

Companion to `CLAUDE.md`. Follow this in order. Do not wire two services in the same session.

---

## 0. Before anything else — protect the keys

The repo is public. A committed API key is scraped by bots within minutes.

Create `.gitignore` **first**, before any key exists on your machine:

```
.env
.env.*
*.duckdb
__pycache__/
.venv/
.streamlit/secrets.toml
```

Then commit `.gitignore` on its own. Only after that commit lands should you create `.env`.

Create `.env.example` with empty values and commit that instead — it documents what's needed without exposing anything:

```
ANTHROPIC_API_KEY=
NEBIUS_API_KEY=
PINECONE_API_KEY=
LANGSMITH_API_KEY=
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=meridian-comp-agents
```

If a key ever does get committed, rotating it is the fix. Deleting the commit is not — it stays in git history.

---

## 1. Keys you need

| Service | Where | Notes |
|---|---|---|
| Anthropic API | console.anthropic.com | **Separate from your Claude Code subscription.** Claude Code uses your subscription; the agents inside your app need an API key with credits. |
| Nebius | Nebius AI Studio / Token Factory | Your credits live here. OpenAI-compatible endpoint. |
| Pinecone | app.pinecone.io | Your credits live here. |
| LangSmith | smith.langchain.com | Free tier is enough. Debugging only. |

---

## 2. Dependencies

```
python-dotenv
pandas
numpy
pyarrow
duckdb
statsmodels
openpyxl

langgraph
langgraph-checkpoint-sqlite
langchain-core
langchain-anthropic
langchain-openai
langchain-pinecone
pinecone

streamlit
plotly

pytest
ruff
```

Two gotchas: `langgraph-checkpoint-sqlite` is a separate package from `langgraph`, and `langchain-openai` is what talks to Nebius — there is no Nebius-specific LangChain package needed.

Load keys once at app entry:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## 3. Integration order

Each step gets its own session and its own commit. **Run the smoke test before wiring the service into any agent.** A smoke test that fails tells you the key or endpoint is wrong; a failing agent tells you nothing.

### Step A — Anthropic only

Smoke test:

```python
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=200)
print(llm.invoke("Reply with the single word: working").content)
```

Then build the leveling agent as a plain function — no graph yet. Job description in, structured decision out, validated against a Pydantic model.

### Step B — LangGraph

Do not introduce LangGraph and a new model provider together. Convert the working leveling agent into a two-node graph (parse → level) with a `SqliteSaver` checkpointer. Same behaviour, new plumbing. If output changes, the graph is wrong.

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
```

Verify checkpointing works before moving on: run a graph, kill it mid-way, resume from the same thread id.

### Step C — LangSmith

One env var, no code change. Set `LANGSMITH_TRACING=true` and confirm runs appear in the dashboard. If they don't, fix it now — this is your debugging tool for everything after.

### Step D — Nebius

Smoke test first, standalone:

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1",
    api_key=os.environ["NEBIUS_API_KEY"],
    model="<pick from the current Nebius model list>",
)
print(llm.invoke("Reply with the single word: working").content)
```

Confirm the base URL and model name against current Nebius docs — the endpoint moved from `api.studio.nebius.ai` to Token Factory and the model list changes.

Then route **only the parser** to Nebius. Leave everything else on Claude. Compare parser output before and after on the same five job descriptions; if extraction quality dropped, try a larger open model before concluding it doesn't work.

Put routing behind one function so provider choice is never scattered through agent code:

```python
def get_model(tier: str):
    return CLAUDE if tier == "judgment" else NEBIUS
```

### Step E — Pinecone

Smoke test: create the index, upsert three vectors, query, delete.

Index config that must match your embeddings:
- **Dimension 4096** for `BAAI/bge-en-icl`
- Metric: cosine
- Serverless

Embeddings come from Nebius, not Pinecone — Nebius embeds, Pinecone indexes and searches. Embed the ~400 survey job descriptions once, commit nothing (vectors live in Pinecone), and write the embedding script so it's re-runnable.

Then wire retrieval into the pricing agent as candidate generation only. The agent still judges which match to use and how to weight it.

---

## 4. Deployment secrets

Neither host reads `.env`.

- **Streamlit Community Cloud** — paste keys into the app's Secrets panel, read with `st.secrets`.
- **Replit** — use the Secrets tab, read with `os.environ`.

Write a small helper that checks `st.secrets` first and falls back to `os.environ`, so the same code runs locally and on both hosts.

**Before deploying**, add the public-demo protections: default to pre-computed demo runs, gate live runs behind a password or a bring-your-own-key field, and cap the M&A crosswalk to a subset. A public URL wired to your API key is an open tab on your credit card.

---

## 5. Session prompts

One service per session. Rough shape:

- *"Read CLAUDE.md and SETUP.md. We're on step A. Write the Anthropic smoke test and run it."*
- *"Step B. Convert the leveling agent into a two-node LangGraph with a SqliteSaver checkpointer. Behaviour must not change."*
- *"Step D. Nebius smoke test only. Do not touch the agents yet."*

If a session ends with something working, commit before closing it.
