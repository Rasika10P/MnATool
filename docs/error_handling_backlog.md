# Error-handling backlog

Companion to `ASSIGNMENT.md` build order item 8 ("Error handling on every failure path").
Failure modes discovered during earlier build steps land here as they're found, so item 8
starts from real observed behavior instead of a guess at what could go wrong.

Each entry: what was observed, why current validation doesn't catch it, and the planned
handling. Update an entry's status once item 8 actually implements it.

---

## 1. Malformed structured output that passes validation

**Status:** Open -- NOT covered by the entry 2/4 retry fix below

**Update:** `agents/instrumented_model.py` now retries automatically (see entry 2/4's
update), but only on an actual `pydantic.ValidationError` -- this entry's failure mode is
specifically the case where Pydantic does *not* raise (a corrupted or silently-missing value
inside an otherwise-valid decision), so the new retry loop never fires for it. Steps 1-3
below are still unimplemented.

**Observed:** Running the leveling agent (`agents/leveling.py`) on the deep-but-narrow
adversarial case (see prompt log), one of two identical live re-runs returned a
`LevelingDecision` where:
- `reasoning` (free text) ended with leaked tag-like syntax: `...Senior Staff Engineer).
  </reasoning>\n<parameter name="alternative_level">L6`
- `alternative_level` (typed `LevelCode | None`) came back `null`, despite the model's own
  `alternative_reasoning` text making clear it meant to set it to `"L6"`

The second identical run was completely clean. This is a generation-time glitch, not a
reproducible bug in the schema or prompt.

**Why Pydantic doesn't catch it:** `reasoning` and `alternative_reasoning` are free prose --
an enum or type constraint can't validate the internal consistency of a text field. The
`LevelCode` Literal on `alternative_level` did its job (a real code would have validated);
the failure mode is that the value never made it into the right field in the first place.

**Planned handling (item 8):**
1. After receiving a structured decision, scan free-text fields (`reasoning`,
   `alternative_reasoning`, `escalation_factor`) for tag-like syntax (e.g. a regex for
   `</?[a-zA-Z_]+>` or `<parameter name=`).
2. If found, retry the call once.
3. If the retry is also malformed, don't drop the decision -- mark it degraded (e.g. a
   `degraded: bool` field or a wrapping result type) and surface it rather than silently
   returning something that looks clean but isn't fully trustworthy. A degraded decision
   still carries its `assigned_level` and factor ratings; what's suspect is specifically the
   free-text explanation, so downstream consumers can decide whether that's acceptable for
   their use (e.g. fine for a batch run, not fine for something a human is about to approve
   verbatim).
4. Log the occurrence (which field, which case) so the eval set (item 9) can track how often
   this happens in practice -- one glitch in two runs isn't enough data to know if it's rare
   or not.

---

## 2. Structured output missing a required field entirely

**Status:** Closed -- both halves now fixed

**Update 2:** `level_employee` (`agents/leveling_batch_graph.py`) now wraps its whole body
(the empty-`job_description` check and the `_run_leveling_call` invocation) in a bare
`except Exception`, not scoped to `StructuredOutputError`/`ValidationError` specifically --
any per-employee failure, whatever its type, becomes a `{"employee_id": ..., "error": str(e)}`
decisions entry instead of propagating. Returning a normal state update (never raising) means
the Send task checkpoints as completed like any other, satisfying this entry's step 3 request
for a regression test: `tests/test_leveling_batch_graph.py::test_one_forced_failure_among_25_does_not_take_down_the_batch`
forces employee 12 of 25 to fail via a fault-injecting fake model, confirmed to fail against
the pre-fix code (the other 24 employees' results were lost along with it) and pass against
the fix (all 25 complete: 24 real decisions, 1 `error` entry, in one `run_batch` invocation).
`app/pipeline.py` and `app/Home.py` already expected this exact `"error"` key shape (they'd
been synthesizing it themselves at the whole-batch-abort level as a stopgap); `app/Home.py`
gained a dedicated "Needs human review — failed to level, not escalated" summary section,
separate from any negotiation-escalation status, per ASSIGNMENT.md's "surface in the UI as a
review item."

**Update 1:** `agents/instrumented_model.py`'s `_InstrumentedStructuredRunnable.invoke` now
retries any `pydantic.ValidationError` from `with_structured_output` up to `MAX_ATTEMPTS`
(3) before raising -- this entry's exact failure (a required field missing from the tool
call) is a `ValidationError`, so it's covered. Every attempt, retries included, gets its own
`agents.cost_logging.log_call` entry with an `attempt` number, so the retry rate is visible
in the persistent JSONL log and in a run's printed session summary (`retries` count),
covering this entry's step 1 for every agent generically -- not just leveling -- since it
lives at the model-wrapper layer that every `get_model()` caller already goes through.

**Observed:** During the batch fan-out kill/resume demo (build order item 3,
`scripts/batch_kill_demo.py`), a resumed run crashed with an unhandled Pydantic
`ValidationError` mid-batch:

```
1 validation error for LevelingDecision
reasoning
  Field required [type=missing, input_value={'track': 'IC', 'assigned...ther this is L4 or L5."}, input_type=dict]
```

The model's tool call omitted `reasoning` (a required field, no default) entirely, rather
than returning it malformed. Unlike entry 1 (a corrupted-but-present field), this is a
field that never showed up in the tool-call arguments at all. Because `level_employee`
(`agents/leveling_batch_graph.py`) doesn't catch exceptions per task, this single employee's
failure raised all the way up through LangGraph's task runner and aborted the *entire*
fan-out invocation -- every other employee still in flight in that same call was lost too,
not just the one with the bad output. This directly contradicts ASSIGNMENT.md's stated
error-handling contract: "A single employee failing to parse -> mark as failed, **continue
the batch**, surface in the UI as a review item. Never fail the whole run for one bad row."

**Why Pydantic doesn't catch it upstream:** it *does* catch it -- that's what raised the
`ValidationError`. The gap is that nothing in `level_employee` catches *that* exception and
converts it into a per-employee failure record; it's left to propagate and take the rest of
the batch down with it.

**Planned handling (item 8):**
1. Wrap the `_run_leveling_call` invocation inside `level_employee` in a try/except for
   `pydantic.ValidationError` (and the retry-then-degrade path from entry 1, once that
   exists).
2. On failure, return a `decisions` entry for that employee marked as failed (e.g.
   `{"employee_id": ..., "status": "failed", "error": str(e)}`) instead of raising --
   matching the "continue the batch" contract instead of the current all-or-nothing crash.
3. Since Send-task completions are per-task writes under `durability="sync"` (see the
   durability note in `agents/leveling_batch_graph.py`), a converted-to-failure task should
   check-point cleanly like any other completed task, rather than aborting the invocation --
   worth a regression test once implemented, since this is exactly the kind of thing that's
   easy to silently regress.

---

## 3. Resume occasionally re-executes an already-completed Send task at scale

**Status:** Open -- root cause not confirmed, workaround not applied

**Observed:** `scripts/batch_kill_demo.py`, real 25-employee fan-out. `durability="sync"`
(entry-1-adjacent fix, see `agents/leveling_batch_graph.py`) makes resume reliably skip
already-completed employees in a clean, isolated 8-employee test (killed mid-flight after 2
completions, resumed with zero re-execution, verified twice). At the full 25-employee
scale, the same setup re-executed one already-completed employee (`NYX-002`) on resume, in
two separate attempts.

**Suspected mechanism (not confirmed):** across every real run this session, only ~12 of
the 25 `Send`-dispatched tasks ever print their "executing" marker immediately; the rest
appear staggered, one at a time, as earlier tasks finish. That's consistent with a bounded
thread pool (plausibly Python's `ThreadPoolExecutor` default sizing, `min(32, cpu_count() +
4)`) queuing the excess tasks rather than truly running all 25 concurrently. The 8-employee
test that worked cleanly never exceeded a pool of that size; the 25-employee case does. The
hypothesis is that queuing interacts with how resume matches newly re-dispatched tasks
against prior completed writes, but this hasn't been isolated further -- it could instead be
something entirely separate that happens to correlate with employee count.

**Practical impact:** `decisions` is collected via an additive reducer
(`Annotated[list[dict], operator.add]`), so a spuriously re-executed employee doesn't
overwrite anything -- worst case is a wasted API call and a **duplicate entry for that
employee_id** in the final list. Not silent data loss, but real callers need to dedupe by
employee_id (keeping the latest) rather than assuming one entry per employee.

**Planned handling (item 8):**
1. Confirm or rule out the thread-pool-queuing hypothesis directly (e.g. instrument or
   patch the executor to log actual concurrent-task count, or test whether explicitly
   capping concurrency at or below the suspected pool size eliminates the re-execution).
2. If confirmed, either raise the pool size to comfortably exceed realistic batch sizes, or
   cap `Send` fan-out concurrency to a known-safe level (LangGraph's `max_concurrency` config
   is the natural knob) and accept the resulting wall-clock cost.
3. Regardless of root cause, dedupe `decisions` by `employee_id` (keep latest) before
   treating the batch result as final -- cheap, and correct even if the underlying cause
   turns out to be something else entirely.
4. Re-run the 25-employee kill/resume demo clean (zero re-execution) as the exit criterion
   for closing this entry.

---

## 4. A third shape of malformed structured output: extra nesting

**Status:** Root cause (no retry) fixed generically; two follow-ups still open

**Observed:** Re-running the 25-employee fan-out (after adding Role Summary text, richer
prompts than the earlier bare-title runs), one task's tool call came back with the entire
payload wrapped under an extra key instead of flat fields:
`{'LevelingDecision': {'track': ..., ...}}` rather than `{'track': ..., ...}`. Pydantic
correctly rejected it (7 "Field required" errors, one per top-level field), and -- same gap
as entry 2 -- nothing catches that exception locally, so it took down the whole batch
invocation again.

Three distinct malformed-output shapes now observed across this session: leaked tag syntax
inside a prose field (entry 1), a single field silently omitted (entry 2), and now the
entire response nested one level too deep. This is enough data points to say the failure
mode itself is the recurring issue -- "the model's tool call doesn't exactly match the
schema" -- not any one specific shape of it. Item 8's retry/catch logic (entries 1-2) should
be written against that general failure mode (`pydantic.ValidationError` from
`with_structured_output`, whatever its specific shape), not pattern-matched to leaked tags
or missing fields specifically -- a naive regex-for-tags check (entry 1's step 1) would miss
both entry 2 and this one.

**Workaround used for the build-order-item-3.5 Role Summary demo:** none at the per-task
level (that's item 8's job) -- the demo script simply retried the *entire* batch invocation
from scratch on any exception, up to 4 attempts. Cheap and effective for a one-off
demonstration run; not a substitute for the real per-task catch/retry/degrade design in
entries 1-2, which avoids re-running 24 already-correct employees to recover from 1 bad one.

**Update, scope extraction (`agents/scope_extraction.py`), Claude:** running the 5-case
Nebius-vs-Claude extraction comparison (`scripts/parse_five_jobs_nebius_vs_claude.py`), one
specific case ("3. Acquired -- Director of Analog Design") triggered this exact shape on
**Claude**, not just Nebius/open models -- three separate times across three separate
attempts, each with a *different* wrapper key (`parameter`, then `ScopeProfile`, then
`parameters`), before a fourth attempt finally returned flat fields. All other 4 cases on
both providers, and this same case on Nebius, never showed it. A single case failing this
consistently while everything else is clean points at something about that specific
input/schema combination provoking it, not pure random glitch rate -- worth a closer look
at that case's prompt once item 8's general catch/retry exists, rather than assuming
uniform glitch probability across cases.

**Update, advocate agent (`agents/advocate.py`), Claude:** running `scripts/arbiter_nyx_011.py
--employee-id NYX-009`, the advocate's `AdvocateOutput` call (`contests: bool` +
`argument: CrosswalkArgument | None`) failed **6 consecutive times** before succeeding on
the 7th -- the worst run of this failure mode observed so far (entry 4's "Director of Analog
Design" case took 4 attempts). Diagnosed by inspecting the raw `ValidationError` across
three of the six failures rather than just the exception type:

- Attempts 1 and 3: `argument` came back as a correctly-typed nested dict with all of
  `CrosswalkArgument`'s own fields intact -- but the top-level `contests` field was missing
  entirely.
- Attempt 2: `argument` came back as a *string* containing the escaped JSON for the whole
  object, **including a `"contests": true` key inside it**, plus a trailing leaked
  `</invoke>` tag (entry 1's tag-leak pattern, now co-occurring with entry 4's nesting
  pattern in the same response).

Across all sampled failures, `contests` is the field that goes missing or gets swallowed
into `argument`, never the reverse -- consistent with the model treating "whether this
contests" and "what the contest argument is" as one decision it's trying to express in a
single value, then failing to split that back into the two separate fields the schema
actually wants. This is a sharper, more specific version of entry 4's general "nested
Optional[BaseModel] field is a recurring trigger" note: here it looks specifically like
*a bool flag paired with an Optional[BaseModel] gated by that same flag* (the
`ScopeFinding`-style `stated`/`value` pattern used throughout this codebase's schemas) is
what's provoking it, not just nesting on its own. Worth item 8 checking whether this
schema shape specifically (flag + conditionally-required nested model) has a higher glitch
rate than plain nested-but-always-required fields, once the general retry/degrade path
exists to gather that data systematically instead of one-off manual retries like this.

**Update -- two fixes landed:**

1. **The general retry path now exists.** `agents/instrumented_model.py`'s
   `_InstrumentedStructuredRunnable.invoke` retries any `pydantic.ValidationError` up to
   `MAX_ATTEMPTS` (3) -- written against the general failure mode as this entry asked for,
   not pattern-matched to any one shape. Covers every occurrence in this entry (the
   "Director of Analog Design" 4-attempt case, the advocate's 6-consecutive-failure case)
   automatically for every agent, since it lives in the shared model wrapper every
   `get_model()` caller goes through. Still open from this entry's own "planned handling":
   nothing yet catches the *exhausted-retries* case (now `StructuredOutputError`, see entry
   2's update) to convert it into a per-item failure record instead of raising -- entry 2's
   batch-continuation gap is unchanged by this.

2. **The specific flag+Optional[BaseModel] schema shape this update identified was removed**
   from `AdvocateOutput` (`agents/negotiation_schemas.py`): `contests: bool` +
   `argument: CrosswalkArgument | None` is now four flat, independently-optional fields
   (`argument_basis`, `proposed_level`, `evidence_cited`, `framework_section`) that must all
   be null or all be set, validated by a single cross-field check instead of a flag gating a
   nested object. `ArbiterRuling` and `EquityGateResult` were audited for the same shape --
   `ArbiterRuling` never had it; `EquityGateResult`'s `passed`/`conflicting_incumbents` pair
   was judged different enough (a `list[str]`, not a nested `BaseModel`) to leave as is, per
   that schema's own docstring note. This doesn't prove the flag+nesting shape is *the*
   cause -- retry-then-succeed doesn't distinguish "this schema shape is risky" from "this
   was random glitching that happened to occur on a schema with this shape" -- but removing
   it costs nothing here and was cheap to do while the diagnosis was fresh.

---

## 5. Open-model repetition loop overruns max_tokens and kills the whole extraction

**Status:** Open -- confirmed NOT covered by the entry 2/4 retry fix, for the reason this
entry already predicted

**Update:** `agents/instrumented_model.py`'s new retry loop only catches
`result["parsing_error"]` -- LangChain's own `ValidationError` handling on a response it did
receive and could parse the JSON of. This entry's `LengthFinishReasonError` is raised
directly out of `self._structured_llm.invoke(...)` itself, before that point, exactly as
predicted below ("this fails before Pydantic validation ever runs"). It still propagates
uncaught. Still needs its own catch, as planned.

**Observed:** Running `agents/scope_extraction.py`'s `extract_scope_profile` (Nebius,
`Qwen/Qwen3-30B-A3B-Instruct-2507`) on the 5 recorded test cases, `ownership_scope` and
`decision_scope` -- the two open free-text fields -- repeatedly degenerated into long
runs of near-duplicate clauses (case 5's `ownership_scope` alone: several hundred words of
"no role in X; no role in Y; ..." repeated with minor variation dozens of times). This isn't
a one-off: the same input (case 3, "Acquired -- Director of Analog Design") produced a
clean, concise extraction on one run and, on another run of the identical prompt, rambled
long enough to hit `max_tokens=2048` before finishing valid JSON -- `openai.
LengthFinishReasonError: Could not parse response content as the length limit was reached`,
raised out of `agents/instrumented_model.py`'s `.invoke()` uncaught. `max_tokens` did what
it's for (bounded spend on a rambling response, guardrail item from earlier this session)
but as a side effect turns a quality problem into a hard, non-deterministic crash rather
than a degraded-but-usable result.

**Why nothing upstream catches it:** this fails before Pydantic validation ever runs --
`with_structured_output`'s underlying OpenAI-compatible parser can't parse a response that
never finished, so it's a client-library exception, not a `ValidationError`. Entries 1-2's
planned validation-error retry/degrade path wouldn't fire on this at all; it needs its own
catch.

**Planned handling (item 8):**
1. Catch `openai.LengthFinishReasonError` (and, more generally, any truncated/incomplete
   response) around Nebius extraction calls specifically -- this is a volume-tier,
   open-model failure mode; Claude has not shown this behavior in this codebase's other
   agents so far.
2. On catch, retry once. If it repeats, mark the profile degraded (same pattern as entry 1)
   rather than raising -- a scope profile with a truncated `ownership_scope` is still useful
   for its other four fields.
3. Independent of the crash case: even when it *doesn't* truncate, the repetition itself
   (case 5) is a real quality problem for a field meant to summarize scope concisely -- worth
   deciding whether the prompt needs an explicit conciseness/no-repetition instruction, or a
   generation parameter (e.g. a repetition penalty, if the Nebius API exposes one) before this
   feeds anything downstream. Comp-domain call on how much extraction quality matters here is
   the user's, not mine to resolve unprompted.
