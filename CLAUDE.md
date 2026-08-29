# CLAUDE.md — Meridian Comp Agents

Read this first, every session.

## The one-liner

> My agent helps a compensation manager map an acquired company's workforce into our job architecture in a web app, replacing the three to four weeks of manual spreadsheet leveling that follows every acquisition. It parses each employee's role from a messy census, levels it against our framework, prices it to market, and negotiates contested mappings between the two companies' incompatible frameworks on its own using six tools. It hands off to a human when a role has no equivalent in our architecture, when the two sides can't agree after two rounds, and before any mapping is finalized. I'll know it works when a comp manager can get a defensible mapping and cost model for 25 employees in under ten minutes, with leveling that matches expert judgment eight times out of ten.

## What this is

A multi-agent compensation system built on a shared job-and-pay spine. Two workflows sit on top of it:

1. **M&A integration war room** — an acquired company's census comes in, agents crosswalk the population into our architecture in parallel, contested mappings go to a negotiation loop, then specialists model cost and retention before a synthesis agent reconciles them.
2. **Market pricing desk** — a job description comes in, agents level it, price it against market, check internal equity, and produce a recommendation packet. Shares the same agents as the crosswalk.

It is both a course submission and a portfolio project. The submission audience is engineers, instructors and product managers; the end user is a compensation manager. **The architecture leads; the comp domain is what makes the decomposition defensible rather than arbitrary.**

## Non-negotiables

These are not preferences. Breaking them breaks the project.

1. **Math in code, judgment in agents.** Compa-ratio, range penetration, aging market data, geo differentials, cost roll-ups — all deterministic Python functions with unit tests. An LLM must never compute a pay figure. Agents decide *which* survey cut, *what* level, *whether* a gap is defensible. Tools compute.
2. **Every number carries provenance.** No dollar figure anywhere in the system may be unattributable to a source row. Tools return their inputs alongside their outputs. This is a validation step, not a prompt instruction.
3. **Synthetic data only.** No licensed survey data, no real employee data, ever. The repo is public.
4. **Deterministic generation.** The generator is seeded. Same seed, same company, every run.

## Stack (locked — do not substitute)

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| Orchestration | LangGraph state machines |
| Tools | LangChain `@tool` wrapping deterministic comp functions |
| Models | `ChatAnthropic` (judgment) + `ChatOpenAI` pointed at Nebius base_url (volume) |
| Persistence | LangGraph SqliteSaver checkpointer |
| Human-in-loop | LangGraph `interrupt()` |
| Tracing | LangSmith for debugging + a custom Streamlit trace view for the product |
| Vector | Pinecone via `langchain-pinecone`; embeddings from Nebius `BAAI/bge-en-icl` (4096-dim) |
| Data | DuckDB + Parquet |
| Stats | statsmodels (pay equity regression only) |
| UI | Streamlit, multipage |
| Charts | Plotly |

Do not add LangChain beyond `@tool` and the model clients. Do not add a framework not listed here without asking.

## Model routing

- **Nebius open models** — job description parsing (structured scope extraction: reports-to, span of control, budget authority, decision scope, ownership scope), title normalization, survey match candidate generation.
- **Claude** — all leveling and adjudication, pricing judgment, the reviewer agent, crosswalk arbitration, M&A synthesis.

Put routing behind one function so provider choice is never scattered through agent code. Log per-agent token spend so the cost split can be reported on the evals page.

Nebius was tried on first-pass leveling and dropped: on 5 test cases it leveled one notch high on 4/5, reporting confidence 0.85–0.95 on all five — including the one Claude flagged for escalation. That's a calibration problem, not an uncertainty problem, so no confidence threshold could route around it. `agents/leveling.py`'s `level_role_routed` and `scripts/level_five_jobs_nebius_vs_claude.py` stay in the repo, unused, as the evidence for this decision. Leveling is Claude-only now; Nebius does extraction (`agents/scope_extraction.py`), which doesn't call for the same adjudication.

## Reference documents

- `ASSIGNMENT.md` — the one-liner, the agent framework table, scope, build order and timeline. **The build order there is binding.**
- `SETUP.md` — services, keys, dependencies, and the integration order. One service per session, smoke test before wiring.
- `docs/level_framework.md` — the leveling rubric. The leveling agent applies this document literally. Rules in section 5 are binding, especially rule 2 (lower level governs a split), rule 3 (deep-but-narrow caps at L5) and rule 9 (escalate below confidence threshold). Section 6 covers source-organization calibration; section 7 covers the negotiation.
- `docs/data_model_spec.md` — every table schema, generation rules, and the planted problems.
- `docs/nyx_level_framework.md` — the acquired company's framework. Deliberately incompatible: five MTS levels (MTS I, MTS II, Senior MTS, Principal MTS, Distinguished MTS), no manager track, plus a Fellow honorific that sits outside the ladder entirely.
- `docs/comp_philosophy.md` — target percentiles, geo strategy, pay mix.
- `docs/error_handling_backlog.md` — failure modes found during earlier build steps, to implement in build order item 8.

## Build order

See `ASSIGNMENT.md`. Items 1–10 ship this week; 11–14 are stretch. Do not reorder without asking.

**Add one external service at a time.** Never wire two new services in the same session — if it breaks, there is no way to tell which one.

## Current status

- [x] Level framework v0.2
- [x] Data model spec v0.1
- [x] Nyx level framework (`docs/nyx_level_framework.md`) — five-level MTS ladder, no manager track, Fellow honorific
- [x] Generator part 1 — level_definitions, geo_locations, fx_rates, job_catalog (273 jobs, verified)
- [x] Generator part 2 — survey_jobs (120 descriptions), survey_data (2,520 rows)
- [x] Generator part 3 — salary_structures, incumbents (300)
- [x] Nyx census (25) — regenerated against the real five-level Nyx framework (`NYX_ROSTER` in `data/generate.py`), not derived from Meridian's own L-codes: 5-person Photonics group with no Meridian equivalent, 2 MTS holders with managerial scope, 1 Fellow, deliberate within-level ambiguity between adjacent Meridian levels. Deterministic (content-identical across runs; only the `.xlsx` file's own embedded metadata varies).
- [x] Deterministic comp functions + tests
- [x] Build order item 2 — leveling agent as a plain function (`agents/leveling.py`), then LangGraph with a `parse`/`level` node split and `SqliteSaver` checkpointer (`agents/leveling_graph.py`), proven via kill/resume
- [x] Build order item 3 — parallel fan-out over the 25-employee Nyx census via `Send` (`agents/leveling_batch_graph.py`), proven via wall-clock comparison and kill/resume at scale
- [x] Model layer — caching, cost logging, session stats and the spend budget guard live in one wrapper (`agents/instrumented_model.py`) so any agent using `get_model()` inherits all four by construction; `--dry-run`, `--limit`, `--budget` on every population script
- [x] Nebius routing (`agents/model_router.py`) — tried on first-pass leveling and dropped (`agents/leveling.py`'s `level_role_routed`, kept unused as evidence: leveled one notch high on 4/5 test cases at high confidence, including the one Claude flagged for escalation — a calibration problem, not an uncertainty one). Now handles structured scope extraction only (`agents/scope_extraction.py`, wired into the graph's parse node as advisory context for the level node, raw job description stays authoritative)
- [x] `ScopeFinding` — distinguishes "not mentioned" from "explicitly stated as none" in extracted scope fields (rule 6 evidence, e.g. an explicit "no direct reports")
- [x] Build order item 4 — negotiation subgraph, all 5 steps done: (1) Pydantic schemas (`agents/negotiation_schemas.py` — `CrosswalkArgument`, `AdvocateOutput`, `ArbiterRuling`, `EquityGateResult`, `ExceptionRegisterEntry`), (2) advocate agent (`agents/advocate.py`, scoped to `docs/nyx_level_framework.md` only, never Meridian's own framework), (3) arbiter agent (`agents/arbiter.py`, applies `level_framework.md` sections 5 and 7 only, red-circling is a first-class verdict, takes an optional prior-round equity-gate rejection for round 2), (4) equity gate (`agents/equity_gate.py` — fully deterministic, no model call: peer group is family-wide across geos via compa-ratio, not the narrow sub-family+geo `check_internal_equity` alone matches; see `learnings.md`), (5) wired into a LangGraph subgraph (`agents/negotiation_graph.py`): advocate → arbiter → conditional on verdict, `revised` → equity gate → a gate failure loops back to the arbiter (not the advocate), round counter caps at 2 then forces `escalated`. Every contested case writes an `ExceptionRegisterEntry` to `data/exception_register.jsonl` regardless of verdict; an uncontested case writes nothing. Demoed live end to end on NYX-011 via `scripts/negotiation_nyx_011.py --employee-id <id>` (full transcript: crosswalk → advocate argument → arbiter ruling → exception register entry) and on NYX-009 via `scripts/arbiter_nyx_011.py` (weak contest, plain upheld citing rule 6). NYX-011 consistently resolves via red-circling (rule 4 is a hard bar), so it never live-exercises the equity-gate step of the graph itself — that path is covered by `tests/test_negotiation_graph.py`'s synthetic engineering/L7 scenario (gate pass, gate fail + round 2, forced escalation after 2 vetoed rounds) and by the equity gate's own live L7/Austin demo (piece 4).
- [x] Build order item 6 — cost, retention and synthesis agents (skipped ahead of item 5's interrupt() gates, confirmed). New deterministic tools first: `tools/comp_math.py` gained `interpolate_percentile` (P60/P65 etc. — survey_data only carries p25/p50/p75/p90), `compute_pay_gap`, `phase_amount`, `flag_underwater`; `tools/data_access.py` gained `lookup_market_percentile`, scoped to an exact `job_id` rather than `lookup_market_data`'s `family_group`, which was found to silently blend market data across sub-families (see `learnings.md`). `agents/cost_model.py` prices every employee to comp_philosophy.md's target percentile (+5 at L6-and-above via `level_definitions.sort_order`, not a hardcoded level set) and models day-one vs. phased (50/50 over 2 years, confirmed) funding — every figure deterministic, the one model call recommends a strategy only. `agents/retention_model.py` flags compa-ratio < 0.85 (confirmed) as underwater, computes a deterministic retention award, and asks the model only which underwater employees are genuinely critical given role scope. `agents/synthesis.py` reconciles both — no numeric fields of its own, and explicitly does not average away a real tension (schema: `SynthesisResult.conflicts`, tested and demoed live producing `requires_human_judgment=True` when a critical, underwater employee gets left exposed by phasing). Wired in parallel via `agents/modeling_graph.py` (cost + retention both edge from START, synthesis waits on both — no `Send` needed, this isn't per-employee dispatch). Demoed live on 3 real USD-denominated Nyx employees (`scripts/modeling_demo.py`) and on a constructed critical-vs-junior pair that produces a genuine, correctly-surfaced conflict.
- [x] Follow-up — Nyx census generator fixed to use real dated FX (`tools.currency.convert_currency` against `fx_rates.parquet` at the deal reference date) instead of the approximate multiplier that produced non-USD "Base" figures nowhere near real currency scale; regenerated (`python3 -m data.generate --seed 42`). Fixing it surfaced a second bug: the cost/retention agents' population totals were naively summing `cost_gap`/`retention_award` across whatever currency each employee happened to be in. Fixed with per-employee `cost_gap_reporting_currency`/`retention_award_reporting_currency` (converted at the deal reference date) and an explicit `reporting_currency` field on both `CostAssessment` and `RetentionAssessment`. Confirmed live on a real USD/INR/EUR population: naive sum 1,103,384.03 vs. correct total 71,388.78 USD. Full writeup in `learnings.md`.
- [ ] Build order item 5 — `interrupt()` gates, 1 of 4 done: gate 4, "final approval before any write to leveling_decisions" (`agents/approval_graph.py` — a two-node graph, `gate` then `write`, `SqliteSaver`-checkpointed like the other graphs; `gate` calls `interrupt()` with the employee, proposed level, confidence, governing rule, factor-rating evidence, and both positions when the decision came out of a contested negotiation; a human's `Command(resume=...)` carries `approved` / `approved_with_override` / `rejected`, and only a non-reject verdict reaches `write`, which calls `tools/decisions.write_mapping_decision`). Demoed live end to end via Streamlit (`app/pages/2_Approvals.py`, reading the crosswalk run held in session state from the Home page): pause with full context shown → human clicks Approve → resume → row lands in `leveling_decisions` with a real `decision_id`, confirmed both in the same session and via a fresh page load's live DuckDB query. Rejection confirmed to never touch the database (`tests/test_approval_graph.py`, 5 tests). Remaining 3 gates — column mapping confirmation before ingest, no-Meridian-equivalent escalation (Photonics, the Nyx Fellow), and negotiation's two-round limit — not yet built.
- [ ] `@tool` wiring — CLAUDE.md's stack table names `@tool` as the locked tool-wrapping mechanism, but no file in the repo used it (confirmed by grep: zero `@tool` decorators anywhere, including in `tools/`). Every deterministic function was being called directly by orchestration/graph-node code; no agent ever let a model choose which tool to call. Fixed in two pieces: (1) `tools/agent_tools.py` wraps all six ASSIGNMENT.md tools (`read_job_architecture`, `lookup_market_data`, `convert_currency`, `compute_pay_metrics`, `check_internal_equity`, `write_mapping_decision`) as thin `@tool` functions with JSON-primitive signatures, without touching the original plain functions or any of their dozens of existing direct call sites. (2) `agents/pricing_agent.py` — new, additive, doesn't touch the crosswalk pipeline — is the one agent that actually binds tools to a model (`agents/instrumented_model.py` gained a `bind_tools` method alongside its existing `with_structured_output`, so tool-calling turns get the same caching/cost-logging/budget-guard treatment as every other model call, tracked under context `"pricing_agent_tool_call"`) and lets it choose which of the five *read* tools to call, in what order and with what arguments, to assess whether a candidate's pay is defensible. `write_mapping_decision` is deliberately never bound to a model — the write path stays exclusively behind `agents/approval_graph.py`'s gate. Numeric provenance is enforced structurally: `PricingJudgment` (the model's output) has no numeric fields at all; every number in the returned `PricingAssessment` comes from the actual tool call log, not the model's words. Verified live (not just against `tests/test_pricing_agent.py`'s faked model): given an INR salary, the model independently called `read_job_architecture` → `lookup_market_data` → `convert_currency` → `compute_pay_metrics` → `check_internal_equity`, in that order, unprompted as to which tools it needed, then correctly flagged the offer as below the p25 floor with zero internal peers to check equity against ($0.058, 5 calls).
- [x] `error_handling_backlog.md` entry 2 closed — `agents/leveling_batch_graph.py`'s `level_employee` node now catches any per-employee exception and returns a `{"employee_id": ..., "error": str(e)}` decisions entry instead of raising, so one bad employee's structured-output failure no longer takes the rest of the 25-employee `Send` fan-out down with it (previously it did: confirmed directly by reverting the fix and re-running the new test, which lost all 24 other employees along with the forced failure). `tests/test_leveling_batch_graph.py` forces employee 12 of 25 to fail via a fault-injecting fake model and confirms all 25 Send tasks still complete (24 real decisions + 1 error entry) in one `run_batch` invocation. `app/Home.py` gained a "Needs human review — failed to level, not escalated" summary section, kept textually and structurally separate from negotiation-escalation status (verified via Streamlit's `AppTest` harness, not just unit tests).
- [ ] Build order item 9 (eval harness) — started. `evals/labeled_cases.jsonl`: 20 cases, 5 pre-filled from the existing Claude baselines (`scripts/level_five_jobs_via_graph.py`'s `CASES`, already-reviewed structural results treated as ground truth), 15 more drawn from the real Nyx census (NYX-001..NYX-015, deterministic, first-15-in-order) with `expected_level`/`expected_escalate` left `null` — **awaiting the user's actual labels**, not invented (leveling-decision domain calls are the user's per CLAUDE.md's "what NOT to decide on your own"). `evals/scoring.py` — pure functions (exact-match rate, within-one-level rate via a dense rank over `level_definitions.sort_order`, escalation precision/recall, markdown table), unit-tested directly (`tests/test_evals_scoring.py`, 10 tests) with no model or file I/O. `evals/run_evals.py` — CLI harness, writes `evals/results.md`; `--fake` runs `FixedFakeModel` (always returns L4, confidence 0.9, no API call) so the harness's own plumbing can be verified by hand before spending anything on a real Claude run. Run live: `python3 -m evals.run_evals --fake` produced exact-match 40%, within-one 80%, escalation precision n/a, recall 0% — hand-verified correct against the 5 labeled cases and the fixed model's known output. Still open: the real run once the 15 Nyx cases are labeled, and the within-one-level/escalation metrics' real (non-fake) values.
- [ ] Everything else

See `learnings.md` for the reasoning behind the two dropped/reworked decisions above (Nebius-off-leveling, the source_org_context confidence-inversion finding), a data-generator bug worth knowing about (the old messy-title generator's silent no-op), and the equity gate's peer-group-scope decision.

Update this section as work completes.

## How to work with me

- **Small pieces.** One table, one function, one node. If a request would produce 300 lines I cannot evaluate, it was too big — break it up and say so.
- **Explain before I accept.** After writing code, say in plain language what it does. I am reviewing the explanation, not the syntax.
- **Run it.** Do not hand me code that has not been executed. Show the output.
- **Commit when it works**, with a message describing what works.
- **Log the prompt.** After each meaningful piece, remind me to paste the prompt into `prompts.md` — it is a graded deliverable.

## What NOT to decide on your own

I am a compensation professional. The domain decisions are mine, and getting them wrong silently is worse than stopping to ask. **Flag and wait** rather than inventing an answer when you hit:

- Anything about how levels map, what a factor means, or how a leveling rule should resolve
- Range width, geo differential, pay mix, or target percentile values
- What counts as a pay equity gap, compression, or inversion
- Anything involving the negotiation rules or admissible arguments
- New job families, sub-families, or level definitions
- Whether a planted problem is realistic

Engineering choices inside the locked stack are yours. Comp choices are not.

## Known open decisions

1. Product Management currently inherits factor variant 5b from the `corporate` family group; it arguably owns product lifecycle (5a). Proposed fix: assign the variant at sub-family level rather than family level.
2. Several L8 jobs read implausibly (e.g. Fellow for Applications Engineering). Trim the L8 ceiling on families where it isn't credible.
3. Salary structure midpoints: derive from market at target percentile, then age Physical Design US by 24 months to create the structure-drift planted problem. Do not hand-pick drift values.
