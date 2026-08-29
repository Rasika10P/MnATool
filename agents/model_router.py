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

import os

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from agents.instrumented_model import InstrumentedModel

# Leveling adjudication, pricing judgment, the reviewer agent, crosswalk arbitration, and
# M&A synthesis all stay on Claude per CLAUDE.md's model routing table.
_JUDGMENT_MODEL = "claude-sonnet-5"

# Confirmed live via scripts/smoke_test_nebius.py: base_url moved off api.studio.nebius.ai
# to Token Factory, and this model id was checked against a live GET /v1/models rather than
# the docs' own (stale, since-retired) quickstart example.
_NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
_VOLUME_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def get_model(tier: str):
    if tier == "judgment":
        return InstrumentedModel(ChatAnthropic(model=_JUDGMENT_MODEL, max_tokens=2048))
    if tier == "volume":
        return InstrumentedModel(
            ChatOpenAI(
                base_url=_NEBIUS_BASE_URL,
                api_key=os.environ["NEBIUS_API_KEY"],
                model=_VOLUME_MODEL,
                max_tokens=2048,
            )
        )
    raise ValueError(f"Unknown model tier: {tier!r}")
