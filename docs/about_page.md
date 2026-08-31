# About this project

## The problem

When an acquisition closes, the compensation team has about ninety days to answer a question that sounds simple: where does each acquired employee belong in our job architecture?

It isn't simple. The acquired company has its own levels, its own titles, and its own idea of what "Principal" means. Nobody's titles line up. There's rarely a level column in the file you're handed. And every answer has consequences — get someone a level too high and you've broken internal equity with people who've been here for years; too low and you lose the person you paid to acquire.

Today this is done by hand: three to four weeks of spreadsheet work, defended in a meeting with the acquired company's leadership, who have their own framework and their own view. The reasoning behind each decision lives in someone's head.

This tool does that work, and shows its reasoning.

---

## The agent, in one line

> This agent helps a compensation manager map an acquired company's workforce into our job architecture, replacing the three to four weeks of manual spreadsheet leveling that follows every acquisition. It parses each employee's role from a messy census, levels it against our framework, and negotiates contested mappings between two incompatible frameworks on its own using six tools. It hands off to a human when a role has no equivalent, when the two sides can't agree after two rounds, and before any decision is written.

---

## The framework

| Field | |
|---|---|
| **Agent goal** | Takes an acquired company's employee census and produces an approved mapping into our job architecture, with cost and retention impact modeled. |
| **Where people use it** | A Streamlit web app. Comp managers live in spreadsheets; this replaces the spreadsheet. |
| **Steps, in order** | 1. Ingest census. 2. Propose column mapping → **human confirms**. 3. Normalize currency, dates, locations, titles. 4. Extract a scope profile from each role description. 5. Assign a level against the framework. 6. Contested cases go to the negotiation loop. 7. Roles with no equivalent → **human decides**. 8. Cost and retention run in parallel. 9. Synthesis reconciles them. 10. **Human approves** before anything is written. |
| **What it can do** | `read_job_architecture` *(read)* · `lookup_market_data` *(read)* · `find_survey_matches` *(read, vector retrieval)* · `convert_currency` *(read, dated FX)* · `compute_pay_metrics` *(read — compa-ratio, range penetration, cost to minimum)* · `check_internal_equity` *(read)* · `write_mapping_decision` *(**write** — gated behind human approval)* |
| **What it remembers** | Within a run: full state via a checkpointer, so a crash resumes rather than restarts. Across sessions: leveling decisions persist and are retrieved as precedent, so the system levels consistently over time. |
| **What it must never do** | Never finalize a mapping that fails the internal equity check. Never move a level based on retention risk, title, or current pay — those are compensation remedies, not leveling arguments. Never invent a market figure. Never accept real employee data. |
| **Human-in-the-loop** | Four gates: column mapping before ingest, roles with no equivalent, negotiation deadlock after two rounds, and final approval before any write. Humans resolve by overriding the level, mapping manually from the real architecture, or accepting the escalation. |
| **When something breaks** | Tool error → retry with backoff. Extraction fails on the open model → fall back to Claude and log it. One employee fails → mark that result failed, **continue the batch**, surface it as a review item. No market data for a job → decline rather than guess, and escalate. Never fail the whole run for one bad row. |
| **How I know it worked** | Leveling matches expert labels on a held-out set *(see the evals page for current accuracy)*, and a full 25-employee integration completes end to end with zero unhandled exceptions. |

---

## The agents

Eight agents. Judgment work runs on Claude; extraction runs on an open model.

| Agent | Model | What it does |
|---|---|---|
| **Parser** | Nebius | Reads a free-text role description and extracts structured evidence — team size, budget authority, what the person owns. Extraction only; it never assigns a level. |
| **Leveler** | Claude | Assigns a level by applying the framework to that evidence, citing the rule that governed the decision and naming the level it rejected. |
| **Advocate** | Claude | Argues the acquired company's case, from *their* framework. May only argue from evidence — title, retention risk and current pay are rejected before they reach the arbiter. |
| **Arbiter** | Claude | Rules on contested mappings from our framework, citing a rule by number. Splitting the difference is not an available verdict. |
| **Equity gate** | Claude | Checks any revision against existing employees at that level. Can veto the arbiter. |
| **Cost** | Claude | Models what harmonization costs, immediately and phased. Calls tools for every figure. |
| **Retention** | Claude | Identifies who lands underwater and who is a genuine flight risk. |
| **Synthesis** | Claude | Reconciles cost and retention. Where they conflict, it surfaces the conflict rather than averaging it away. |

## Where a human decides

| Gate | When it fires | What you can do |
|---|---|---|
| **Column mapping** | On upload, before ingest | Confirm or correct the proposed mapping |
| **No equivalent role** | The role has no match in our architecture — photonics, or the Fellow honorific | Hand it off, or map manually from the real job codes |
| **Negotiation deadlock** | Advocate and arbiter unresolved after two rounds | Decide, with both positions in front of you |
| **Final approval** | Before any decision is written | Approve or send back |

Manual mapping is deliberately constrained: you pick from job families, codes and locations that actually exist. An override can't point at a combination that isn't real. Once mapped, that employee runs back through negotiation and the cost and retention figures update for the whole population.

Every review, either way, is logged.

---

## Why two frameworks matter

Meridian uses eight individual-contributor levels and a separate manager track. Nyx Semiconductor uses five, with no manager track, plus a "Fellow" title outside the ladder entirely.

Five levels don't divide into eight. Every Nyx level straddles two Meridian levels, so the ambiguity is structural, not a data quality problem. That's what the negotiation exists for.

The two frameworks also disagree on substance. Meridian caps deep specialists who lack cross-domain breadth; Nyx doesn't — depth alone can carry someone to the top. Both frameworks are internally coherent. They just reach different answers, and someone has to decide.

## What's real and what isn't

Everything here is synthetic. Meridian Silicon is a fictional fabless semiconductor company; Nyx Semiconductor is a fictional acquisition. The employees, salaries, and market data are generated, not sourced from any real company or compensation survey.

That's deliberate. Real survey data is licensed and can't be used in a system like this, and real employee compensation data is confidential. The architecture, framework and compensation logic reflect how this work is actually done — the data underneath does not describe anyone.

## How it's built

Orchestration is LangGraph, checkpointed so a run can be interrupted and resumed rather than restarted. No model computes a pay figure anywhere: compa-ratio, range placement, currency conversion and cost roll-ups are deterministic functions with tests, and every number traces to a source row.

Runs can be replayed from cache at no cost, which is how the demo works. A live run makes real model calls and shows what it spent.
