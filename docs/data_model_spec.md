# Data Model Spec — Meridian Silicon

**Version:** 0.1
**Companion to:** `level_framework.md` v0.2
**Purpose:** Defines every table in the canonical data layer, how the synthetic generator produces it, and what problems are deliberately planted in it.

**Non-negotiable:** all data is synthetic. No licensed survey data, no real employee data, ever. The generator is seeded and deterministic — the same seed produces the same company every run.

---

## 1. Table inventory

| Table | Rows (approx) | Purpose |
|---|---|---|
| `level_definitions` | 13 | Machine-readable level metadata |
| `job_catalog` | ~220 | Every job at Meridian |
| `geo_locations` | 8 | Sites, tiers, currencies |
| `salary_structures` | ~900 | Ranges by level and geo |
| `survey_jobs` | ~400 | Survey job descriptions — the embedded corpus |
| `survey_data` | ~6,000 | Market data points |
| `incumbents` | 1,500 | The employee population |
| `fx_rates` | ~500 | Monthly rates by pair |
| `nyx_census` | 104 | The acquired population (messy, as .xlsx) |
| `acquisition_context` | 1 | Source org calibration record |
| `leveling_decisions` | grows | Agent output, persisted for precedent retrieval |
| `exception_register` | grows | Contested crosswalk cases |

Documents that are not tables: `level_framework.md`, `nyx_level_framework.md`, `comp_philosophy.md`.

---

## 2. Schemas

### `level_definitions`
| Column | Type | Notes |
|---|---|---|
| `level_code` | text PK | L1–L8, M3–M7 |
| `track` | text | IC, MGR |
| `level_title` | text | Staff Engineer, Director |
| `ic_equivalent` | text | For manager rows; M3 → L4 |
| `sort_order` | int | For ordering across both tracks |
| `target_bonus_pct` | float | Standard mix; overridden for GTM |
| `equity_tier` | text | limited, standard, enhanced, executive |

### `job_catalog`
| Column | Type | Notes |
|---|---|---|
| `job_id` | text PK | e.g. `ANA-RF-L5` |
| `job_title` | text | Canonical title |
| `family` | text | One of the twelve |
| `sub_family` | text | |
| `family_group` | text | engineering, corporate, gtm |
| `level_code` | text FK | |
| `factor5_variant` | text | 5a, 5b, 5c — derived from family_group |
| `is_quota_carrying` | bool | Drives OTE vs base comparison |
| `pay_basis` | text | `base` or `OTE` |
| `survey_code_primary` | text FK | |
| `survey_code_secondary` | text FK | Nullable |
| `blend_weight_primary` | float | 1.0 when no secondary |

### `geo_locations`
| Column | Type | Notes |
|---|---|---|
| `geo_code` | text PK | `US-SJC`, `IN-BLR`, `EU-EIN`, `LATAM-GDL` |
| `country` | text | |
| `city` | text | |
| `currency` | text | USD, INR, EUR, MXN |
| `differential` | float | vs US national = 1.00 |
| `tier` | text | Used for structure assignment |

Differentials: US-SJC 1.15, US-AUS 1.00, IN-BLR 0.34, IN-HYD 0.31, EU-EIN 0.78, EU-MUC 0.82, LATAM-GDL 0.42.

### `salary_structures`
| Column | Type | Notes |
|---|---|---|
| `structure_id` | text PK | |
| `geo_code` | text FK | |
| `level_code` | text FK | |
| `family_group` | text | Engineering and GTM carry separate structures |
| `currency` | text | Local currency, not USD |
| `pay_basis` | text | `base` or `OTE` |
| `range_min` / `range_mid` / `range_max` | numeric | |
| `effective_date` | date | |

Range width: 40% at L1–L3, 50% at L4–L6, 60% at L7–L8.

### `survey_jobs` — the embedded corpus
| Column | Type | Notes |
|---|---|---|
| `survey_code` | text PK | e.g. `RAD-SEM-4412` |
| `survey_source` | text | Three fake sources — see §4 |
| `survey_job_title` | text | |
| `survey_job_description` | text | 120–200 words. **This is what gets embedded into Pinecone.** |
| `survey_level_label` | text | The survey's own scheme, not Meridian's |
| `discipline` | text | |

### `survey_data`
| Column | Type | Notes |
|---|---|---|
| `survey_code` | text FK | |
| `geo_code` | text | |
| `currency` | text | |
| `pay_element` | text | `base`, `TCC`, `OTE` |
| `p25` / `p50` / `p75` / `p90` | numeric | |
| `incumbent_count` | int | Low counts must be flagged by the pricing agent |
| `effective_date` | date | |

### `incumbents`
| Column | Type | Notes |
|---|---|---|
| `employee_id` | text PK | |
| `display_name` | text | Synthetic |
| `job_id` | text FK | |
| `level_code` | text FK | |
| `manager_id` | text FK | Nullable at the top |
| `geo_code` | text FK | |
| `currency` | text | |
| `hire_date` | date | |
| `level_effective_date` | date | Drives time-in-level |
| `base_salary` | numeric | Local currency |
| `target_variable_pct` | float | Bonus % or variable side of the split |
| `target_variable` | numeric | |
| `actual_variable_last_cycle` | numeric | Diverges from target for GTM |
| `ote` | numeric | base + target_variable, for quota roles |
| `equity_annual_grant_value` | numeric | USD, grant value |
| `equity_unvested_value` | numeric | USD |
| `performance_rating` | text | 5-point scale on a curve |
| `is_red_circled` | bool | |
| `source` | text | `organic` or `acquired:<deal_id>` |
| `gender` | text | Required for the pay equity regression |

`gender` exists solely to make the pay equity analysis real. Note in the README that this is synthetic and used only to demonstrate regression methodology.

### `fx_rates`
| Column | Type | Notes |
|---|---|---|
| `from_currency` / `to_currency` | text | |
| `rate` | numeric | |
| `rate_month` | date | First of month |

Monthly granularity, 24 months of history. Every conversion returns the rate and month used. The acquisition cost model runs against a declared **deal reference date** so demo output is reproducible.

### `acquisition_context`
Per §6 of the framework: `source_headcount`, `source_stage`, `source_type`, `parent_headcount`, `org_depth`, `platform_dependency`.

Nyx values: 104 headcount, late-stage private, whole company, org_depth 4, platform_dependency low.

### `leveling_decisions`
Persisted agent output. `decision_id`, `job_or_employee_ref`, `assigned_level`, `confidence`, per-factor ratings, `factor5_variant_applied`, `alternative_considered`, `governing_rule`, `reviewer_verdict`, `source_document_hash`, `created_at`.

**This table is the precedent corpus.** It gets embedded into Pinecone so the leveling agent can retrieve prior decisions on similar roles — the consistency feature from the original design.

### `exception_register`
`case_id`, `employee_id`, `crosswalk_level`, `advocate_position`, `advocate_argument`, `arbiter_ruling`, `governing_rule_cited`, `equity_gate_result`, `verdict`, `round_count`.

---

## 3. Population shape

Level pyramid across 1,500 incumbents:

| Level | Count |
|---|---|
| L1 | 45 |
| L2 | 160 |
| L3 | 340 |
| L4 | 310 |
| L5 | 215 |
| L6 | 95 |
| L7 | 14 |
| L8 | 1 |
| M3 | 130 |
| M4 | 105 |
| M5 | 62 |
| M6 | 19 |
| M7 | 4 |

Geographic split per framework §10: US 40%, India 35%, EU 15%, LATAM 10%. India skews toward Design Verification and Physical Design; EU toward Analog and RF; LATAM toward Application Software.

### Generation rules
- **Compa-ratio** drawn from a normal distribution centred at 0.98, σ 0.11, clipped to [0.72, 1.28]. Correlated positively with time-in-level and performance rating.
- **Performance ratings** on a curve: 5% top, 20% high, 60% mid, 12% low, 3% bottom.
- **Time in level** log-normal, longer at higher levels.
- **Manager spans** consistent with framework §4 anchors — the org must actually resolve into a valid reporting tree.
- **Actual variable** for GTM roles drawn from an attainment distribution: median ~96% of target, long right tail to 180%, floor at 0.
- **Titles are deliberately inconsistent** in the `nyx_census` only. Meridian's own catalog is clean — that contrast is the point.

---

## 4. Survey design

Three fictional sources with different characteristics, so the pricing agent has real blending decisions:

| Source | Coverage | Refresh | Character |
|---|---|---|---|
| `Radleigh Semiconductor Survey` | Deep in silicon disciplines, thin in corporate | Current | The anchor source |
| `Vantis Global Technology` | Broad, all families, all geos | Current | Good coverage, shallow samples in niche disciplines |
| `Corbin General Industry` | Corporate and GTM only, no engineering | **20 months stale** | Must be aged; the agent should flag it |

Each source uses its own level scheme — Radleigh uses P1–P6/M1–M5, Vantis uses numeric grades 8–20, Corbin uses broad career bands. **None of them match Meridian's L1–L8.** This is what makes survey matching an actual task rather than a join.

`survey_jobs` descriptions are generated once with an LLM from the family/sub-family/level grid, then **committed to the repo and never regenerated**. Reproducibility matters more than variety.

---

## 5. Planted problems

Each is seeded deliberately and maps to an agent that should catch it. This table is your demo script and your eval set.

| # | Problem | Where | Agent that should catch it |
|---|---|---|---|
| 1 | Two L5 Analog incumbents in IN-BLR sit below the current market offer point | `incumbents` | Equity agent — compression flag |
| 2 | US Physical Design structure is 6% below market; not refreshed in 2 cycles | `salary_structures` | Pricing agent — structure drift |
| 3 | ~4.5% unexplained gender pay gap in Design Verification after controls | `incumbents` | Pay equity regression |
| 4 | One M3 in Embedded Software paid below their highest-paid L5 report | `incumbents` | Equity agent — inversion |
| 5 | Corbin survey is 20 months stale | `survey_data` | Pricing agent — must age and flag |
| 6 | A Solutions Architect role that is 60% Systems Architecture, 40% Sales Engineering | `job_catalog` | Pricing agent — hybrid, no clean match |
| 7 | A senior salesperson whose base is at P25 but OTE is at P60 | `incumbents` | Pricing agent — must compare on OTE not base |
| 8 | L8 population of exactly one | `incumbents` | Equity agent — must not treat n=1 as a distribution |
| 9 | Three L4 Verification roles with inconsistent internal titles | `job_catalog` | Job parser — normalization |
| 10 | Photonics has no Meridian equivalent | `nyx_census` | Crosswalk — escalation path |
| 11 | Nyx Fellow honorific sits outside the MTS ladder entirely | `nyx_census` | Crosswalk — no mapping, forced human interrupt |
| 12 | A Nyx Principal MTS with deep-but-narrow scope | `nyx_census` | Negotiation — rule 3 test case |

Problems 1–9 are Meridian-internal and drive the pricing desk demo. Problems 10–12 drive the M&A demo.

---

## 6. The Nyx census

Delivered as `.xlsx`, deliberately unclean — this is what actually arrives from a data room.

Columns: `Emp ID`, `Job Title`, `Dept`, `Location`, `Curr`, `Base`, `Bonus`, `Unvested Options`, `Start`, `Role Summary`.

Built-in mess:
- **No level column.** Nyx levels are embedded in free-text titles.
- **Three date formats** — `3/14/22`, `2021-08-02`, `15/06/23`.
- **Three currencies** — USD, INR, EUR — with `Curr` blank on some rows.
- **Bonus expressed inconsistently** — `15%` on some rows, `15` on others.
- **Location strings vary** — `San Jose, CA`, `San Jose`, `Bangalore`, `Eindhoven`.
- **Title variants for the same level** — `Sr MTS`, `Senior MTS`, `Senior Member of Technical Staff`.
- **`Role Summary` quality is deliberately inconsistent** — 2-3 sentences of free text per
  employee, generated once with Claude from each row's real (family, sub_family, level) and
  committed (never regenerated — same policy as `survey_jobs` descriptions, section 4). Most
  rows carry enough absolute scope evidence (section 6 of `level_framework.md`) to level
  confidently; a handful are too vague to level from at all (`"Good team player in the RTL
  org."`), matching how a real HR export mixes carefully-written and rushed fields. Two rows
  are bespoke rather than tiered, matching their title overrides: the row behind "Principal
  MTS — Systems Architecture" is written as a genuine deep-but-narrow case (planted problem
  12), and the row behind "Fellow — Photonics" is written with strong, confident evidence in
  a domain Meridian doesn't work in at all (planted problems 10 and 11).

Ingest runs: column mapping (human-confirmed) → normalization → crosswalk fan-out. Generated by the same seeded generator so the population is internally consistent and plausibly mappable, apart from the deliberate exceptions.

A template file with the expected shape ships alongside for users who want to upload their own.

---

## 7. File layout

```
data/
  generate.py                 seeded generator, one entry point
  comp.duckdb                 built artifact
  parquet/
    level_definitions.parquet
    job_catalog.parquet
    geo_locations.parquet
    salary_structures.parquet
    survey_jobs.parquet
    survey_data.parquet
    incumbents.parquet
    fx_rates.parquet
  acquisitions/
    nyx_photonics_census.xlsx
    nyx_level_framework.md
    acquisition_context.json
    census_template.xlsx
  docs/
    comp_philosophy.md
    level_framework.md
```

`generate.py` takes a seed and rebuilds everything from scratch. Parquet files are committed so the app runs without regenerating. DuckDB is opened **read-only** with a cached connection.

---

## 8. Open items

1. **Names in `incumbents`** — use a neutral generator and avoid any real-looking name collisions. Worth a sanity check before the repo goes public.
2. **Range midpoint derivation** — set midpoints to the market composite at the target percentile from `comp_philosophy.md`, or set them independently and let the drift in planted problem 2 emerge? Deriving them is cleaner but makes the drift harder to plant.
3. **Nyx equity** — options with a strike price, or simplify to unvested grant value? Options are more realistic for a private company and matter for the change-of-control acceleration story.
