"""Static reference page: the problem, the agents, the frameworks, what's synthetic. No
pipeline calls here -- content only.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from app.demo_mode import render_and_apply_mode_control

st.set_page_config(page_title="About — Meridian Crosswalk", layout="wide")

render_and_apply_mode_control()

st.title("About")

st.markdown(
    """
## The problem

When an acquisition closes, the compensation team has about ninety days to answer a question that sounds simple: where does each acquired employee belong in our job architecture?

It isn't simple. The acquired company has its own levels, its own titles, and its own idea of what "Principal" means. Nobody's titles line up. There's rarely a level column in the file you're handed. And every answer has consequences — get someone a level too high and you've broken internal equity with people who've been here for years; too low and you lose the person you paid to acquire.

Today this is done by hand. A comp analyst reads role descriptions one at a time, forms a view, checks it against internal peers, and defends it in a meeting with the acquired company's leadership, who have their own framework and their own view. Three to four weeks of spreadsheet work, and the reasoning behind each decision lives in someone's head.

This tool does that work, and shows its reasoning.

## What it does

It takes an acquired company's employee census — the messy kind that arrives from a data room, with inconsistent titles and three date formats and blank currency fields — and maps each person into Meridian's job architecture.

For each employee it reads the role description, extracts what the person actually does (team size, budget authority, what they own), and assigns a level by applying Meridian's leveling framework rather than by reading their title. Every decision cites the specific rule that governed it.

Where the mapping is contested, it runs a negotiation. One agent argues the acquired company's case from *their* framework. Another rules from Meridian's. A third checks that any revision doesn't break internal equity with existing employees. Contested cases resolve one of four ways: the original mapping stands, the level changes, the level holds but pay is protected, or it goes to a human.

Then it models what the integration costs and who is at risk of leaving.

## Why two frameworks matter

Meridian uses eight individual-contributor levels plus a separate manager track. Nyx Semiconductor — the acquired company — uses five, with no manager track at all, plus a "Fellow" title that sits outside the ladder entirely. (Nyx's Photonics group goes further still: it has no equivalent anywhere in Meridian's architecture, and is handed to a human rather than forced into a mapping.)

Five levels don't divide into eight. Every Nyx level straddles two Meridian levels, so the ambiguity is structural, not a data quality problem. That's what the negotiation exists for.

The two frameworks also disagree on substance. Meridian caps deep specialists who lack cross-domain breadth. Nyx doesn't — depth alone can carry someone to the top of their ladder. So an engineer who is genuinely the company's last word on one technical domain sits at the top under one framework and a level lower under the other. Both frameworks are internally coherent. They just reach different answers, and someone has to decide.

## What a human decides

The system never finalizes anything on its own. It stops for a person when a role has no equivalent in the architecture, when the two sides can't agree after two rounds, and before any decision is written.

It also refuses to make certain arguments. Retention risk, current pay, and job title are not leveling arguments — they're real concerns, but you solve them with money, not by moving someone's level. That constraint is enforced in the system, not just suggested.

## What's real and what isn't

Everything here is synthetic. Meridian Silicon is a fictional fabless semiconductor company; Nyx Semiconductor is a fictional acquisition. The employees, salaries, and market data are generated, not sourced from any real company or compensation survey.

That's deliberate. Real survey data is licensed and can't be used in a system like this, and real employee compensation data is confidential. The job architecture, leveling framework, and compensation logic reflect how this work is actually done — the data underneath does not describe anyone.

## How it's built

Agents handle judgment: leveling, arbitration, cost and retention analysis. Deterministic Python functions handle every calculation — compa-ratio, range placement, currency conversion, cost roll-ups. No model computes a pay figure, and every number traces back to a source row.

Built with LangGraph for orchestration, Claude for judgment, and open models on Nebius for extraction work.
"""
)
