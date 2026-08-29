"""The one agent in this codebase where a model actually chooses which tool to call, in what
order, with what arguments -- real LangChain tool-calling (bind_tools), not a graph node
calling a fixed deterministic function ahead of time. Every other agent here (equity_gate,
cost_model, leveling, ...) has its Python code decide which deterministic function applies at
a given point and calls it directly, then asks the model for judgment on the result; there's
no ambiguity for an LLM to resolve in those cases. Here there is: pricing a candidate needs
some subset of the five read tools depending on whether their currency already matches the
market data's, so the model decides for itself which of the five to call and in what order.

The five tools are agents.pricing_agent's own imports of the @tool wrappers in
tools/agent_tools.py -- write_mapping_decision is deliberately excluded from the bound tool
list. CLAUDE.md's "write actions get a human" restricts every write to
agents/approval_graph.py's interrupt() gate; an autonomous tool-calling loop must never be
handed a tool that writes.

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

MAX_TOOL_TURNS = 6

TOOLS = [read_job_architecture, lookup_market_data, convert_currency, compute_pay_metrics, check_internal_equity]
_TOOLS_BY_NAME = {t.name: t for t in TOOLS}

_SYSTEM_PROMPT = """You are Meridian Silicon's pricing agent. Decide for yourself which of \
your tools to call, in what order, to assess whether {candidate_salary} {candidate_currency} \
is defensible for job_id={job_id} in geo_code={geo_code}, as of {as_of_date}.

Your tools: read_job_architecture, lookup_market_data, convert_currency, compute_pay_metrics, \
check_internal_equity. You likely need read_job_architecture first, to learn this job's \
family_group and level_code before lookup_market_data can use them. You only need \
convert_currency if the candidate's currency differs from the market data's currency (USD). \
Pass annual_growth_rate={annual_growth_rate} to lookup_market_data.

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
        raise RuntimeError(f"Pricing agent exceeded {MAX_TOOL_TURNS} tool-calling turns without finishing")

    judgment = llm.with_structured_output(PricingJudgment).invoke(
        messages + [HumanMessage("Give your final judgment now, as the structured schema.")]
    )
    return PricingAssessment(tool_calls=tool_calls, judgment=judgment)
