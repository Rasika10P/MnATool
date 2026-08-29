# ASSIGNMENT.md — Week 3 submission pack

Track 2 (LangChain + LangGraph). Submission: Google Doc + 5-minute video + GitHub link.

---

## Part 1 — The one-liner

> My agent helps a compensation manager map an acquired company's workforce into our job architecture in a web app, replacing the three to four weeks of manual spreadsheet leveling that follows every acquisition. It parses each employee's role from a messy census, levels it against our framework, prices it to market, and negotiates contested mappings between the two companies' incompatible frameworks on its own using six tools. It hands off to a human when a role has no equivalent in our architecture, when the two sides can't agree after two rounds, and before any mapping is finalized. I'll know it works when a comp manager can get a defensible mapping and cost model for 25 employees in under ten minutes, with leveling that matches expert judgment eight times out of ten.

### Against the three rules

- **Task completion, not single-shot accuracy.** Success is a finished, approved integration plan — not one good leveling call. Measured end to end.
- **State.** LangGraph `SqliteSaver` holds run state across the fan-out and survives interruption. Leveling decisions persist to DuckDB and are retrievable as precedent across sessions.
- **Write actions get a human.** Every read is autonomous. Every write — final mapping, cost plan, exception register — requires approval.

---

## Part 2 — The framework

| Field | Answer |
|---|---|
| **Agent goal** | Takes an acquired company's employee census and produces an approved mapping into our job architecture, with cost and retention impact modeled. |
| **Where people use it** | Streamlit web app. Comp managers live in spreadsheets; this is the surface that replaces the spreadsheet. |
| **Steps, in order** | 1. Ingest census. 2. Propose column mapping → **human confirms**. 3. Normalize (currency, dates, locations, titles). 4. Parse each role into a scope profile. 5. Level each role against the framework. 6. Contested cases go to the negotiation loop. 7. Cost and retention agents run in parallel. 8. Synthesis reconciles them. 9. **Human approves** the plan. |
| **What it can do (6 tools)** | `read_job_architecture` (read) · `lookup_market_data` (read) · `convert_currency` (read, dated FX) · `compute_pay_metrics` (read — compa-ratio, range penetration, cost to minimum) · `check_internal_equity` (read) · `write_mapping_decision` (**write** — gated) |
| **What it remembers** | Within a run: full state via `SqliteSaver`, resumable after failure. Across sessions: leveling decisions persist to DuckDB and are retrieved as precedent so the system levels consistently over time. |
| **What it must never do** | Never finalize a mapping that fails the internal equity check. Never move a level based on retention risk, title, or current pay — those are compensation remedies, not leveling arguments. Never invent a market figure; every number traces to a source row. Never accept real employee data. |
| **Human-in-the-loop** | Three gates: column mapping confirmation before ingest, forced escalation when a role has no equivalent or negotiation hits the round limit, and final approval before any write. Humans intervene by overriding the level or accepting the escalation. |
| **When something breaks** | Tool error → retry once with backoff. Nebius unavailable → fall back to Claude for that call and log the fallback. A single employee failing to parse → mark as failed, **continue the batch**, surface in the UI as a review item. Empty market data for a job → the pricing agent declines rather than guessing, and the case escalates. Never fail the whole run for one bad row. |
| **How I know it worked** | Leveling matches expert labels on 20 held-out cases 8 times out of 10, and a full 25-employee integration completes end to end in under ten minutes with zero unhandled exceptions. |

---

## Scope — full architecture, reduced volume

Nothing is cut from the architecture. Data volume is the only dial turned down for submission week, and it's a parameter in `generate.py`, not a design decision. After submission, change `N` and regenerate.

| Dimension | Submission | Portfolio (after Sunday) |
|---|---|---|
| Job catalog | 273 | 273 |
| Survey corpus | 120 descriptions | 400 |
| Market rows | ~1,200 | ~6,000 |
| Incumbents | 300 | 1,500 |
| Nyx census | 25 | 104 |
| Geographies | US + India | US, India, EU, LATAM |
| Planted problems | 6 | 12 |
| Deployment | none | Replit + Streamlit Cloud |

Everything else ships: both workflows, all agents, the negotiation subgraph, Pinecone retrieval, sales OTE modeling, pay equity regression.

### Build order

Build in this order. Anything unfinished on Sunday morning does not ship Sunday — it ships the following week for the portfolio version. This way what's missing is the least important thing, not whatever happened to be in progress.

1. Data, tools, tests, provider smoke tests
2. Leveling agent → LangGraph → checkpointer
3. Parallel fan-out over the acquired population
4. Negotiation subgraph — advocate, arbiter, equity gate
5. `interrupt()` gates: column mapping, unmappable roles, final approval
6. Cost + retention + synthesis agents
7. Streamlit UI, both workflows
8. Error handling on every failure path
9. Eval set + calibration
10. Trace view
11. Pinecone retrieval
12. Compliance agent
13. Pay equity regression
14. Sales OTE modeling

Items 1–10 are the submission. Items 11–14 are the stretch, and each is genuinely additive rather than load-bearing — the system runs correctly without any of them.

---

## Four-day plan

### Thursday — foundation, no AI
- Repo, `.gitignore`, `.env`, `prompts.md`
- Generator parts 2 and 3 at submission volume
- Six deterministic tools with pytest
- Smoke tests: Anthropic, then Nebius (satisfies the Nebius requirement)
- **Gate:** data loads, tools tested, both providers reachable. If tools aren't tested, do not start Friday — untested math poisons everything downstream.

### Friday — the agent core
- Leveling agent as a plain function, validated against Pydantic
- Convert to LangGraph, add `SqliteSaver`, verify resume-after-kill
- Parser routed to Nebius, leveling stays on Claude
- Fan-out over the acquired population with `Send`
- **Gate:** 25 employees leveled in parallel, run resumable. If `Send` is fighting you by evening, use a sequential loop and move on — it's still multi-agent and you can convert later.

### Saturday — what makes it multi-agent
- Negotiation subgraph: advocate, arbiter, equity gate, two-round limit
- `interrupt()` on column mapping, unmappable roles, final approval
- Cost, retention, synthesis agents
- Streamlit: upload → mapping → run → results, both workflows
- **Gate:** full workflow runs end to end with the negotiation visible.
- **Record a rough five-minute walkthrough before bed, whatever state it's in.** Fifteen minutes, and it removes the worst outcome on Sunday.

### Sunday — the part the rubric names
- Morning: error handling per the framework table; test each failure path deliberately
- Midday: eval set, accuracy and calibration, trace view
- Afternoon: record the real video, write the Google Doc
- **No new features after Sunday morning.** Anything unfinished moves to the portfolio backlog.

---

## Capture as you go

Three deliverables can't be reconstructed on Sunday:

- **`prompts.md`** — every Claude Code prompt, with one line on what happened. Start tonight.
- **`iterations.md`** — what you tried that didn't work. The rubric asks for this and failures are more interesting than successes. The Nebius-vs-Claude parser comparison is a good one.
- **`learnings.md`** — running notes.

---

## Video plan (5 minutes)

The brief requires four things: walk through the application, explain what you built, **describe how you used AI coding tools**, and demonstrate the result live.

| Time | Content |
|---|---|
| 0:00–0:30 | The problem: acquisition closes, 25 people, incompatible frameworks, three weeks of spreadsheets |
| 0:30–1:00 | Architecture diagram, one breath |
| 1:00–2:10 | Live run: upload messy census → column mapping gate → parallel leveling |
| 2:10–3:10 | **The negotiation** — advocate argues, arbiter rules citing rule 3, equity agent vetoes. Slow down here. |
| 3:10–3:35 | Human interrupt on the unmappable Fellow |
| 3:35–4:05 | **How I used AI coding tools** — `CLAUDE.md` as persistent context, one service per session, the "explain it before I accept it" habit, and one thing Claude Code got wrong that I caught |
| 4:05–4:30 | Trace view + eval numbers + cost split across providers |
| 4:30–5:00 | Failure handling: kill a tool, show it degrade rather than crash |

Two segments carry disproportionate weight. The failure demo, because almost nobody will break their own app on purpose. And the AI-tools segment, because most people will say "I used Claude Code and it was fast" — whereas you can show a briefing file, a prompt log, and a specific case where domain knowledge caught a plausible-looking mistake. That's the more interesting answer.
