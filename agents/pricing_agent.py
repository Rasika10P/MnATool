"""The one agent in this codebase where a model actually chooses which tool to call, in what
order, with what arguments -- real LangChain tool-calling (bind_tools), not a graph node
calling a fixed deterministic function ahead of time. Every other agent here (equity_gate,
cost_model, leveling, ...) has its Python code decide which deterministic function applies at
a given point and calls it directly, then asks the model for judgment on the result; there's
no ambiguity for an LLM to resolve in those cases. Here there is: pricing a candidate needs
some subset of the tools depending on whether their currency already matches the market
data's, whether the exact market slice exists at all, and whether checking comparable survey
postings is worth the extra step -- so the model decides for itself which to call and when.

Five of the six tools are agents.pricing_agent's own imports of the @tool wrappers in
tools/agent_tools.py; the sixth, retrieve_similar_survey_jobs (tools/retrieval_tools.py,
build order item 11), is retrieval as candidate generation, not as an answer: it returns
Pinecone's nearest neighbors by embedding similarity over the ~120-job survey corpus, and the
model is the one that judges whether any of them are actually relevant, the same discipline
CLAUDE.md non-negotiable 1 already applies to every other tool here -- retrieval computes a
similarity ranking, it does not decide a price or a level. write_mapping_decision is
deliberately excluded from the bound tool list. CLAUDE.md's "write actions get a human"
restricts every write to agents/approval_graph.py's interrupt() gate; an autonomous
tool-calling loop must never be handed a tool that writes.

Provenance (non-negotiable 2) needs active protection in a tool-calling design specifically:
it would be easy for the model's final natural-language answer to restate a number wrong.
PricingJudgment below carries no numeric fields at all -- every number in the returned
PricingAssessment comes from tool_calls, the actual (tool, args, result) triples captured
during the loop, assembled by this module's own code and never retyped by the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from agents.cost_model import ANNUAL_MARKET_GROWTH_RATE
from agents.model_router import get_model
from tools.agent_tools import (
    check_internal_equity,
    compute_pay_metrics,
    convert_currency,
    lookup_market_data,
    read_job_architecture,
)
from tools.retrieval_tools import retrieve_similar_survey_jobs

MAX_TOOL_TURNS = 6

TOOLS = [
    read_job_architecture,
    lookup_market_data,
    convert_currency,
    compute_pay_metrics,
    check_internal_equity,
    retrieve_similar_survey_jobs,
]
_TOOLS_BY_NAME = {t.name: t for t in TOOLS}

_SYSTEM_PROMPT = """You are Meridian Silicon's pricing agent. Decide for yourself which of \
your tools to call, in what order, to assess whether {candidate_salary} {candidate_currency} \
is defensible for job_id={job_id} in geo_code={geo_code}, as of {as_of_date}.

Your tools: read_job_architecture, lookup_market_data, convert_currency, compute_pay_metrics, \
check_internal_equity, retrieve_similar_survey_jobs. You likely need read_job_architecture \
first, to learn this job's family_group and level_code before lookup_market_data can use \
them. You only need convert_currency if the candidate's currency differs from the market \
data's currency (USD). Pass annual_growth_rate={annual_growth_rate} to lookup_market_data.

retrieve_similar_survey_jobs is optional, and it is candidate generation only -- it returns \
survey postings similar to a query you write (e.g. the job title you learned from \
read_job_architecture), ranked by text similarity. It narrows a field of ~120 survey jobs to \
a handful; it does not tell you which one is right, and it does not reliably rank by \
seniority within a discipline -- do not treat its top-scored result as authoritative. Use it \
when lookup_market_data comes back empty for the exact slice, or when you want a second \
signal on whether this job_id's assumed level looks right against how similar roles are \
described in the market -- never as a substitute for lookup_market_data's own number when \
that number exists.

If a tool call fails (e.g. no market data for this exact slice), do not guess or substitute a \
number -- say so in your final summary instead.

When you have what you need, respond with no further tool calls and a short plain-text \
summary of what you found. Do not restate a final dollar figure or compa-ratio yourself in \
that summary -- those are read back from your own tool results afterward, not from your \
words."""


class PricingJudgment(BaseModel):
    """The one thing this agent is actually asked to judge. Every field is qualitative --
    see module docstring for why no numeric field belongs here."""

    is_offer_defensible: bool = Field(description="Whether candidate_salary holds up against market and internal equity")
    primary_concern: str | None = Field(default=None, description="The single biggest risk in this placement, if any")
    reasoning: str = Field(description="Brief rationale citing what the tool calls actually found")
    recommended_next_step: str = Field(description="e.g. 'proceed', 'escalate to comp lead', 'add a retention award'")


@dataclass
class ToolCallRecord:
    tool_name: str
    args: dict
    result: dict | None
    error: str | None


@dataclass
class PricingAssessment:
    tool_calls: list[ToolCallRecord]
    judgment: PricingJudgment
    # True only when the tool-calling loop hit MAX_TOOL_TURNS without the model ever
    # declaring itself done -- a structural flag, not something inferable from judgment's
    # own prose, so a caller can gate on it (app/Home.py, a batch script) without parsing
    # reasoning text. See _degraded_judgment below for what judgment looks like in this case.
    degraded: bool = False


def _degraded_judgment(job_id: str, geo_code: str) -> PricingJudgment:
    """Built in plain Python, no model call -- MAX_TOOL_TURNS turns of tool-calling without
    a final answer means something is already wrong (a loop, a confused tool sequence, an
    unresolvable ambiguity in the data); asking the same model for one more judgment call on
    top of that is more likely to compound the problem than resolve it. is_offer_defensible
    is deliberately False, never a guess -- the whole point of the cap is "stop trusting this
    run," and a defensible-by-default fallback would silently defeat that."""
    return PricingJudgment(
        is_offer_defensible=False,
        primary_concern=f"Tool-calling loop hit the {MAX_TOOL_TURNS}-turn cap without reaching a final answer",
        reasoning=(
            f"The pricing agent used all {MAX_TOOL_TURNS} tool-calling turns assessing "
            f"job_id={job_id!r} geo_code={geo_code!r} without the model ever declaring "
            "itself done. Treating this as inconclusive rather than accepting a possibly "
            "incomplete or looping tool-call sequence as a real judgment."
        ),
        recommended_next_step="escalate to comp lead",
    )


def _execute_tool_call(call: dict) -> ToolCallRecord:
    tool = _TOOLS_BY_NAME.get(call["name"])
    if tool is None:
        return ToolCallRecord(call["name"], call["args"], None, f"Unknown tool: {call['name']}")
    try:
        return ToolCallRecord(call["name"], call["args"], tool.invoke(call["args"]), None)
    except Exception as e:
        return ToolCallRecord(call["name"], call["args"], None, str(e))


def price_role(
    job_id: str,
    geo_code: str,
    candidate_salary: float,
    candidate_currency: str,
    as_of_date: str,
    model=None,
) -> PricingAssessment:
    """Runs the tool-calling loop to completion, then asks for a structured judgment over the
    transcript. `model` is an override for tests (a fake with .bind_tools/.with_structured_
    output) -- same convention as every other agent function in this repo."""
    llm = model or get_model("judgment")

    system = _SYSTEM_PROMPT.format(
        candidate_salary=candidate_salary,
        candidate_currency=candidate_currency,
        job_id=job_id,
        geo_code=geo_code,
        as_of_date=as_of_date,
        annual_growth_rate=ANNUAL_MARKET_GROWTH_RATE,
    )
    messages = [SystemMessage(system), HumanMessage("Assess this placement.")]
    tool_calls: list[ToolCallRecord] = []
    bound = llm.bind_tools(TOOLS, context="pricing_agent_tool_call")

    for _ in range(MAX_TOOL_TURNS):
        response = bound.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for call in response.tool_calls:
            record = _execute_tool_call(call)
            tool_calls.append(record)
            content = {"result": record.result} if record.error is None else {"error": record.error}
            messages.append(ToolMessage(content=json.dumps(content, default=str), tool_call_id=call["id"]))
    else:
        # Never raise here -- a tool-turn cap is a normal, expected outcome (a confused
        # sequence, a genuinely hard case), not a bug worth crashing the caller over. Return
        # a result marked degraded and pointed at a human instead, the same "flag and wait"
        # discipline CLAUDE.md applies to every other judgment call this codebase won't make
        # silently. No further model call: see _degraded_judgment's own docstring for why.
        return PricingAssessment(
            tool_calls=tool_calls, judgment=_degraded_judgment(job_id, geo_code), degraded=True
        )

    judgment = llm.with_structured_output(PricingJudgment).invoke(
        messages + [HumanMessage("Give your final judgment now, as the structured schema.")]
    )
    return PricingAssessment(tool_calls=tool_calls, judgment=judgment)
