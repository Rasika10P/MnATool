"""
Synthetic data generator for Meridian Silicon.

Seeded and deterministic: the same seed always produces the same company.
Run: python generate.py [--seed 42]

Part 1 of 3 — reference tables and job catalog.
  Part 2 will add survey_jobs and survey_data.
  Part 3 will add salary_structures, incumbents and the Nyx census.

All data here is synthetic. No licensed survey data, no real employee data.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tools.currency import convert_currency

SEED = 42
REFERENCE_DATE = pd.Timestamp("2026-08-01")
OUT = Path(__file__).parent / "parquet"


# ---------------------------------------------------------------- levels

LEVELS = [
    # code, track, title, ic_equivalent, sort, bonus_pct, equity_tier
    ("L1", "IC", "Associate Engineer", None, 10, 0.05, "limited"),
    ("L2", "IC", "Engineer", None, 20, 0.05, "limited"),
    ("L3", "IC", "Senior Engineer", None, 30, 0.10, "standard"),
    ("L4", "IC", "Staff Engineer", None, 40, 0.10, "standard"),
    ("L5", "IC", "Senior Staff Engineer", None, 50, 0.15, "standard"),
    ("L6", "IC", "Principal Engineer", None, 60, 0.20, "enhanced"),
    ("L7", "IC", "Distinguished Engineer", None, 70, 0.25, "enhanced"),
    ("L8", "IC", "Fellow", None, 80, 0.35, "executive"),
    ("M3", "MGR", "Manager", "L4", 45, 0.10, "standard"),
    ("M4", "MGR", "Senior Manager", "L5", 55, 0.15, "standard"),
    ("M5", "MGR", "Director", "L6", 65, 0.20, "enhanced"),
    ("M6", "MGR", "Senior Director", "L7", 75, 0.25, "enhanced"),
    ("M7", "MGR", "Vice President", "L8", 85, 0.35, "executive"),
]

IC_EQUIVALENT = {code: ic_eq for code, _, _, ic_eq, *_ in LEVELS if ic_eq}


def build_level_definitions() -> pd.DataFrame:
    return pd.DataFrame(
        LEVELS,
        columns=[
            "level_code",
            "track",
            "level_title",
            "ic_equivalent",
            "sort_order",
            "target_bonus_pct",
            "equity_tier",
        ],
    )


# ------------------------------------------------------------ geography

GEOS = [
    # code, country, city, currency, differential, tier
    ("US-SJC", "United States", "San Jose", "USD", 1.15, "us_premium"),
    ("US-AUS", "United States", "Austin", "USD", 1.00, "us_standard"),
    ("EU-EIN", "Netherlands", "Eindhoven", "EUR", 0.78, "eu_standard"),
    ("EU-MUC", "Germany", "Munich", "EUR", 0.82, "eu_standard"),
    ("IN-BLR", "India", "Bangalore", "INR", 0.34, "india_premium"),
    ("IN-HYD", "India", "Hyderabad", "INR", 0.31, "india_standard"),
    ("LATAM-GDL", "Mexico", "Guadalajara", "MXN", 0.42, "latam_standard"),
]


def build_geo_locations() -> pd.DataFrame:
    return pd.DataFrame(
        GEOS,
        columns=["geo_code", "country", "city", "currency", "differential", "tier"],
    )


# --------------------------------------------------------------- fx

# Anchor rates at REFERENCE_DATE, walked backwards 24 months.
FX_ANCHORS = {("USD", "INR"): 87.4, ("USD", "EUR"): 0.918, ("USD", "MXN"): 18.6}
FX_VOL = {("USD", "INR"): 0.011, ("USD", "EUR"): 0.016, ("USD", "MXN"): 0.021}


def build_fx_rates(rng: np.random.Generator) -> pd.DataFrame:
    months = pd.date_range(end=REFERENCE_DATE, periods=24, freq="MS")
    rows = []
    for pair, anchor in FX_ANCHORS.items():
        base, quote = pair
        # Walk backwards from the anchor so the reference month is exact.
        shocks = rng.normal(0.0, FX_VOL[pair], size=len(months))
        series = [anchor]
        for shock in reversed(shocks[:-1]):
            series.append(series[-1] * (1.0 - shock))
        series = list(reversed(series))
        for month, rate in zip(months, series):
            rows.append((base, quote, round(rate, 6), month))
            rows.append((quote, base, round(1.0 / rate, 6), month))
    for month in months:
        rows.append(("USD", "USD", 1.0, month))
    return pd.DataFrame(rows, columns=["from_currency", "to_currency", "rate", "rate_month"])


# ------------------------------------------------------------ job catalog

# family, family_group, code, ic_levels, mgr_levels, [sub-families]
FAMILIES = [
    ("Digital Design", "engineering", "DD", (1, 8), (3, 6),
     [("RTL Design", "RTL"), ("Microarchitecture", "UARCH")]),
    ("Analog & Mixed-Signal", "engineering", "ANA", (1, 8), (3, 6),
     [("Analog Design", "AD"), ("RF", "RF"), ("Power Management", "PMIC")]),
    ("Physical Design", "engineering", "PD", (1, 7), (3, 5),
     [("Place & Route", "PNR"), ("Timing", "STA"), ("Signal Integrity", "SI")]),
    ("Design Verification", "engineering", "DV", (1, 7), (3, 6),
     [("Functional Verification", "FV"), ("Formal", "FML"), ("Emulation", "EMU")]),
    ("Silicon Validation & DFT", "engineering", "SV", (1, 7), (3, 5),
     [("Post-Silicon Validation", "PSV"), ("DFT", "DFT"), ("Test Engineering", "TE")]),
    ("Embedded Software", "engineering", "EMB", (1, 7), (3, 5),
     [("Firmware", "FW"), ("Device Drivers", "DRV"), ("RTOS", "RTOS")]),
    ("Platform Software", "engineering", "PSW", (1, 8), (3, 6),
     [("Compilers & Toolchains", "CMP"), ("SDK", "SDK"), ("Systems Software", "SYS")]),
    ("Application Software", "engineering", "ASW", (1, 6), (3, 5),
     [("Applications", "APP"), ("Web Platform", "WEB"), ("Software QA", "SQA")]),
    ("Systems & Architecture", "engineering", "SA", (3, 8), (4, 6),
     [("Systems Architecture", "ARCH"), ("Applications Engineering", "FAE")]),
    ("Product & Program", "corporate", "PP", (2, 7), (3, 6),
     [("Product Management", "PM"), ("Technical Program Management", "TPM")]),
    ("Go-to-Market", "gtm", "GTM", (2, 6), (4, 7),
     [("Sales", "SLS"), ("Sales Engineering", "SE"), ("Marketing", "MKT")]),
    ("Corporate", "corporate", "CORP", (2, 6), (3, 6),
     [("Finance", "FIN"), ("HR", "HR"), ("Legal", "LGL"), ("IT", "IT"), ("Operations", "OPS")]),
]

FACTOR5 = {"engineering": "5a", "corporate": "5b", "gtm": "5c"}

IC_TITLE_STEM = {
    "engineering": {
        1: "Associate Engineer", 2: "Engineer", 3: "Senior Engineer",
        4: "Staff Engineer", 5: "Senior Staff Engineer", 6: "Principal Engineer",
        7: "Distinguished Engineer", 8: "Fellow",
    },
    "corporate": {
        2: "Analyst", 3: "Senior Analyst", 4: "Lead", 5: "Principal",
        6: "Senior Principal", 7: "Distinguished Principal",
    },
    "gtm": {
        2: "Associate", 3: "Specialist", 4: "Senior Specialist",
        5: "Principal", 6: "Senior Principal",
    },
}

# Planted problem 9: three L4 Design Verification jobs with inconsistent titles.
TITLE_OVERRIDES = {
    "DV-FV-L4": "Staff Engineer - Functional Verification",
    "DV-FML-L4": "Staff DV Engineer, Formal",
    "DV-EMU-L4": "Verification Engineer IV - Emulation",
}

QUOTA_SUBFAMILIES = {"Sales"}


def build_job_catalog() -> pd.DataFrame:
    rows = []
    for family, group, fcode, ic_range, mgr_range, subs in FAMILIES:
        for sub_name, scode in subs:
            for n in range(ic_range[0], ic_range[1] + 1):
                level = f"L{n}"
                job_id = f"{fcode}-{scode}-{level}"
                stem = IC_TITLE_STEM[group].get(n)
                if stem is None:
                    continue
                title = TITLE_OVERRIDES.get(job_id, f"{stem} - {sub_name}")
                quota = sub_name in QUOTA_SUBFAMILIES and n >= 3
                rows.append({
                    "job_id": job_id,
                    "job_title": title,
                    "family": family,
                    "sub_family": sub_name,
                    "family_group": group,
                    "level_code": level,
                    "factor5_variant": FACTOR5[group],
                    "is_quota_carrying": quota,
                    "pay_basis": "OTE" if quota else "base",
                })
        for n in range(mgr_range[0], mgr_range[1] + 1):
            level = f"M{n}"
            mgr_title = dict((c, t) for c, _, t, *_ in LEVELS)[level]
            quota = group == "gtm" and n >= 4
            rows.append({
                "job_id": f"{fcode}-MGR-{level}",
                "job_title": f"{mgr_title}, {family}",
                "family": family,
                "sub_family": "Management",
                "family_group": group,
                "level_code": level,
                "factor5_variant": FACTOR5[group],
                "is_quota_carrying": quota,
                "pay_basis": "OTE" if quota else "base",
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------- market data

SUBMISSION_SURVEY_JOBS = 120
SUBMISSION_INHABITANTS = 300
SUBMISSION_NYX = 25


def _level_number(level_code: str) -> int:
    return int(level_code[1:])


def _survey_code(job_id: str, index: int) -> str:
    return f"SYN-{index + 1:03d}-{job_id}"


# Local-currency units per USD equivalent -- NOT a real FX rate, just the fixed scale used to
# generate synthetic local-currency figures.
CURRENCY_FACTOR = {"USD": 1.0, "EUR": 0.93, "INR": 83.0, "MXN": 17.2}


def _local_market_p50(level_code: str, family_group: str, geo_code: str) -> float:
    """Return a reproducible local-currency market P50 used by structures and incumbents.

    Manager levels price against their ic_equivalent (M3 -> L4, not the digit in "M3"),
    per level_definitions.
    """
    level = _level_number(IC_EQUIVALENT.get(level_code, level_code))
    group_factor = {"engineering": 1.0, "corporate": 0.78, "gtm": 0.92}[family_group]
    geo_factor = dict((code, differential) for code, _, _, _, differential, _ in GEOS)[geo_code]
    currency = dict((code, currency) for code, _, _, currency, _, _ in GEOS)[geo_code]
    return 104_000 * (1.16 ** (level - 1)) * group_factor * geo_factor * CURRENCY_FACTOR[currency]


def _stratified_survey_sample(job_catalog: pd.DataFrame, total: int) -> pd.DataFrame:
    """Select survey benchmark jobs with guaranteed coverage of every (family_group, level_code)
    combination in the catalog, spending the remaining budget proportional to stratum size."""
    strata = list(job_catalog.groupby(["family_group", "level_code"], sort=False))
    sizes = {key: len(frame) for key, frame in strata}
    quota = {key: 1 for key in sizes}
    remaining = total - len(strata)
    total_size = sum(sizes.values())
    shares = {key: remaining * size / total_size for key, size in sizes.items()}
    for key in sizes:
        quota[key] += int(shares[key])
    short = total - sum(quota.values())
    by_remainder = sorted(sizes, key=lambda k: shares[k] - int(shares[k]), reverse=True)
    for key in by_remainder[:max(short, 0)]:
        quota[key] += 1
    selected = [
        frame.sort_values("job_id").head(min(quota[key], len(frame)))
        for key, frame in strata
    ]
    return pd.concat(selected, ignore_index=True)


def build_survey_jobs(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    meta = []
    for index, job in selected.iterrows():
        if job.family_group == "engineering":
            source = "Radleigh Semiconductor Survey"
        elif index % 3 == 0:
            source = "Corbin General Industry"
        else:
            source = "Vantis Global Technology"
        level = job.level_code
        survey_code = _survey_code(job.job_id, index)
        description = (
            f"This benchmark role leads work in {job.sub_family} within the {job.family} discipline. "
            f"At the {level} scope, the role owns technical direction, delivery quality, and decisions "
            "that affect partner teams and the reliability of the product. The incumbent translates "
            "ambiguous requirements into an operating plan, sets measurable outcomes, and resolves "
            "risks using sound professional judgment. Typical work includes defining priorities, reviewing "
            "complex deliverables, coaching colleagues, documenting tradeoffs, and communicating status "
            "to technical and business stakeholders. The role is expected to understand relevant methods "
            "and tools deeply enough to challenge assumptions and improve the way work is performed. "
            "Scope expands with level: senior roles influence multiple projects, while the highest roles "
            "shape standards, architecture, and long-range capability. Success is measured by durable "
            "technical outcomes, predictable execution, and the quality of decisions made under uncertainty."
        )
        rows.append({
            "survey_code": survey_code,
            "survey_source": source,
            "survey_job_title": job.job_title,
            "survey_job_description": description,
            "survey_level_label": f"{source.split()[0]}-{level}",
            "discipline": job.sub_family,
        })
        meta.append({"survey_code": survey_code, "family_group": job.family_group, "level_code": level})
    return pd.DataFrame(rows), pd.DataFrame(meta)


def build_survey_data(survey_jobs: pd.DataFrame, survey_meta: pd.DataFrame, geo_locations: pd.DataFrame) -> pd.DataFrame:
    jobs = survey_jobs.merge(survey_meta, on="survey_code")
    rows = []
    for _, job in jobs.iterrows():
        for _, geo in geo_locations.iterrows():
            p50 = _local_market_p50(job.level_code, job.family_group, geo.geo_code)
            stale = job.survey_source == "Corbin General Industry"
            effective_date = REFERENCE_DATE - pd.DateOffset(months=20 if stale else 1)
            for pay_element in ("base", "TCC", "OTE"):
                multiplier = {"base": 1.0, "TCC": 1.08, "OTE": 1.22}[pay_element]
                rows.append({
                    "survey_code": job.survey_code,
                    "geo_code": geo.geo_code,
                    "currency": geo.currency,
                    "pay_element": pay_element,
                    "p25": round(p50 * multiplier * 0.84, 2),
                    "p50": round(p50 * multiplier, 2),
                    "p75": round(p50 * multiplier * 1.18, 2),
                    "p90": round(p50 * multiplier * 1.34, 2),
                    "incumbent_count": 6 if geo.geo_code in {"IN-BLR", "IN-HYD"} else 18,
                    "effective_date": effective_date,
                })
    return pd.DataFrame(rows)


def build_salary_structures(
    geo_locations: pd.DataFrame, survey_data: pd.DataFrame, survey_meta: pd.DataFrame
) -> pd.DataFrame:
    """Derive each structure's midpoint from the market P50 of the survey rows that actually
    cover its (family_group, level_code), rather than computing it independently of market data."""
    market = survey_data.merge(survey_meta, on="survey_code")
    rows = []
    for _, geo in geo_locations.iterrows():
        for (family_group, level), _ in survey_meta.groupby(["family_group", "level_code"]):
            pay_element = "OTE" if family_group == "gtm" else "base"
            cut = market[
                (market.geo_code == geo.geo_code)
                & (market.family_group == family_group)
                & (market.level_code == level)
                & (market.pay_element == pay_element)
            ]
            midpoint = cut.p50.mean()
            if geo.geo_code == "US-AUS" and family_group == "engineering" and level in {"L4", "L5", "L6", "L7"}:
                midpoint *= 0.94
            width = 0.40 if _level_number(level) <= 3 else 0.50 if _level_number(level) <= 6 else 0.60
            rows.append({
                "structure_id": f"{geo.geo_code}-{family_group}-{level}",
                "geo_code": geo.geo_code,
                "level_code": level,
                "family_group": family_group,
                "currency": geo.currency,
                "pay_basis": "OTE" if family_group == "gtm" else "base",
                "range_min": round(midpoint * (1 - width / 2), 2),
                "range_mid": round(midpoint, 2),
                "range_max": round(midpoint * (1 + width / 2), 2),
                "effective_date": REFERENCE_DATE,
            })
    return pd.DataFrame(rows)


def _submission_level_counts() -> dict[str, int]:
    return {
        "L1": 9, "L2": 32, "L3": 68, "L4": 62, "L5": 43, "L6": 19,
        "L7": 3, "L8": 1, "M3": 26, "M4": 21, "M5": 12, "M6": 3, "M7": 1,
    }


def build_incumbents(
    job_catalog: pd.DataFrame,
    geo_locations: pd.DataFrame,
    structures: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    jobs_by_level = {level: frame.reset_index(drop=True) for level, frame in job_catalog.groupby("level_code")}
    structure_map = structures.set_index(["geo_code", "level_code", "family_group"])
    geo_codes = geo_locations.geo_code.tolist()
    geo_weights = [0.20, 0.20, 0.175, 0.175, 0.15, 0.05, 0.05]
    rows = []
    employee_number = 1
    for level, count in _submission_level_counts().items():
        for _ in range(count):
            job = jobs_by_level[level].iloc[(employee_number * 7) % len(jobs_by_level[level])]
            geo_code = rng.choice(geo_codes, p=geo_weights)
            family_group = job.family_group
            structure = structure_map.loc[(geo_code, level, family_group)]
            compa_ratio = float(np.clip(rng.normal(0.98, 0.09), 0.72, 1.28))
            base_salary = structure.range_mid * compa_ratio
            target_variable_pct = 0.12 if job.is_quota_carrying else dict((code, bonus) for code, _, _, _, _, bonus, _ in LEVELS)[level]
            target_variable = base_salary * target_variable_pct
            actual_variable = target_variable * (rng.lognormal(np.log(0.96), 0.22) if job.is_quota_carrying else 1.0)
            performance = rng.choice(["5", "4", "3", "2", "1"], p=[0.05, 0.20, 0.60, 0.12, 0.03])
            start = REFERENCE_DATE - pd.DateOffset(months=int(rng.lognormal(np.log(36), 0.55)))
            level_date = REFERENCE_DATE - pd.DateOffset(months=int(rng.lognormal(np.log(18), 0.65)))
            rows.append({
                "employee_id": f"MER-{employee_number:04d}",
                "display_name": f"Employee {employee_number:04d}",
                "job_id": job.job_id,
                "level_code": level,
                "manager_id": None,
                "geo_code": geo_code,
                "currency": dict(zip(geo_locations.geo_code, geo_locations.currency))[geo_code],
                "hire_date": start,
                "level_effective_date": level_date,
                "base_salary": round(base_salary, 2),
                "target_variable_pct": target_variable_pct,
                "target_variable": round(target_variable, 2),
                "actual_variable_last_cycle": round(actual_variable, 2),
                "ote": round(base_salary + target_variable, 2),
                "equity_annual_grant_value": round(base_salary * rng.uniform(0.04, 0.22), 2),
                "equity_unvested_value": round(base_salary * rng.uniform(0.08, 0.65), 2),
                "performance_rating": performance,
                "is_red_circled": False,
                "source": "organic",
                "gender": rng.choice(["F", "M", "X"], p=[0.35, 0.62, 0.03]),
                "_family_group": family_group,
            })
            employee_number += 1

    incumbents = pd.DataFrame(rows)
    incumbents = _assign_managers(incumbents, rng)
    _ensure_no_empty_managers(incumbents)

    # Keep the demo cases deterministic and obvious to downstream agents.
    analog = incumbents[(incumbents.job_id.str.startswith("ANA-")) & (incumbents.level_code == "L5") & (incumbents.geo_code == "IN-BLR")]
    incumbents.loc[analog.index[:2], "base_salary"] = incumbents.loc[analog.index[:2], "base_salary"] * 0.82
    verification = incumbents[incumbents.job_id.str.startswith("DV-")]
    incumbents.loc[verification.index[verification.gender == "F"], "base_salary"] *= 0.955

    # Resolve every naturally-occurring inversion first (capped, so some legitimate compression
    # cases remain), then plant the one deliberate case last so nothing runs afterward to
    # disturb it.
    _resolve_unintended_inversions(incumbents, structure_map)
    planted_pair = _plant_manager_inversion(incumbents)
    _verify_org_tree(incumbents)
    _print_org_audit(incumbents, planted_pair)

    _plant_red_circled_cases(incumbents, structure_map, exclude={planted_pair[0], planted_pair[1]})

    # Red-circle anyone whose final base salary landed above their range max -- computed last
    # so it reflects every adjustment above (inversion fixes can push pay above range).
    keys = list(zip(incumbents.geo_code, incumbents.level_code, incumbents._family_group))
    range_mid = [structure_map.range_mid.loc[k] for k in keys]
    range_max = [structure_map.range_max.loc[k] for k in keys]
    range_min = [structure_map.range_min.loc[k] for k in keys]
    incumbents["is_red_circled"] = [b > m for b, m in zip(incumbents.base_salary, range_max)]

    _print_compa_audit(incumbents, range_mid, range_min, range_max)
    incumbents = incumbents.drop(columns="_family_group")
    return incumbents


# ------------------------------------------------------ manager hierarchy

MANAGER_LEVELS = ["M3", "M4", "M5", "M6", "M7"]


SAME_GEO_SHARE = 0.87  # target share of reporting relationships kept within one geo


def _assign_with_geo_bias(
    incumbents: pd.DataFrame, child_ids: list[str], manager_ids: list[str], rng: np.random.Generator,
    same_geo_share: float = SAME_GEO_SHARE,
) -> None:
    """Deal `child_ids` out across `manager_ids`, biased so ~`same_geo_share` land with a
    manager in their own geo and the rest land cross-geo -- each pool (same-geo per geo,
    cross-geo per geo) is dealt round-robin independently so it doesn't cluster onto whichever
    manager a shared offset happens to hit. Falls back to whichever pool is non-empty if the
    preferred one has no candidate (e.g. a geo with no manager at this level)."""
    geo_of = dict(zip(incumbents.employee_id, incumbents.geo_code))
    managers_in_geo: dict[str, list[str]] = {}
    for m in manager_ids:
        managers_in_geo.setdefault(geo_of[m], []).append(m)

    cursors: dict[tuple, int] = {}

    def next_from(pool: list[str], key: tuple) -> str:
        i = cursors.get(key, 0)
        cursors[key] = i + 1
        return pool[i % len(pool)]

    for child_id in child_ids:
        geo = geo_of[child_id]
        same_pool = managers_in_geo.get(geo, [])
        other_pool = [m for m in manager_ids if geo_of[m] != geo]
        prefer_same = rng.random() < same_geo_share
        if prefer_same and same_pool:
            manager = next_from(same_pool, (geo, "same"))
        elif other_pool:
            manager = next_from(other_pool, (geo, "other"))
        else:
            manager = next_from(same_pool, (geo, "same"))
        incumbents.loc[incumbents.employee_id == child_id, "manager_id"] = manager


def _assign_managers(incumbents: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Build the reporting tree top-down: M7 is the sole root, and each manager level from
    M6 down to M3 takes reports from the eligible pool immediately below it -- its own IC
    level (per ic_equivalent) plus the manager level directly beneath. Every IC lands with
    the lowest-level manager that can hold it, so the hierarchy has real depth instead of
    the old per-geo random rotation, which ignored level entirely. Assignment is biased
    ~80/20 toward same-geo reporting lines (see SAME_GEO_SHARE) -- comparing pay across labor
    markets is a category error, so pay-equity checks only evaluate same-geo pairs, and most
    of the org should actually be in one.
    """
    incumbents = incumbents.copy()
    incumbents["manager_id"] = None
    by_level = {
        level: incumbents.loc[incumbents.level_code == level, "employee_id"].sort_values().tolist()
        for level in ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"] + MANAGER_LEVELS
    }

    def assign(children: list[str], managers: list[str]) -> None:
        _assign_with_geo_bias(incumbents, children, managers, rng)

    # M3 -- "4-10 direct reports, all individual contributors" (level_framework.md, span & budget)
    assign(by_level["L1"] + by_level["L2"] + by_level["L3"] + by_level["L4"], by_level["M3"])
    # M4 -- its own IC level (L5) plus the M3 layer
    assign(by_level["L5"], by_level["M4"])
    assign(by_level["M3"], by_level["M4"])
    # M5 -- L6 plus the M4 layer
    assign(by_level["L6"], by_level["M5"])
    assign(by_level["M4"], by_level["M5"])
    # M6 -- L7 plus the M5 layer
    assign(by_level["L7"], by_level["M6"])
    assign(by_level["M5"], by_level["M6"])
    # M7 -- sole root; L8 (nothing else can hold it) plus the M6 layer report here
    assign(by_level["L8"], by_level["M7"])
    assign(by_level["M6"], by_level["M7"])

    return incumbents


def _ensure_no_empty_managers(incumbents: pd.DataFrame) -> None:
    """The geo-biased assignment can, by chance, leave a manager with zero direct reports.
    For each one, take a report from whichever manager at the same level currently has the
    most (same level, so the ic_equivalent ceiling still holds for whoever moves)."""
    for level in MANAGER_LEVELS:
        managers = incumbents.loc[incumbents.level_code == level, "employee_id"].sort_values().tolist()
        counts = incumbents.manager_id.value_counts()
        for empty_mgr in [m for m in managers if counts.get(m, 0) == 0]:
            counts = incumbents.manager_id.value_counts()
            donor = max((m for m in managers if m != empty_mgr), key=lambda m: counts.get(m, 0))
            if counts.get(donor, 0) <= 1:
                continue  # nobody at this level has a spare report to give
            moved = incumbents[incumbents.manager_id == donor].sort_values("employee_id").iloc[-1]
            incumbents.loc[incumbents.employee_id == moved.employee_id, "manager_id"] = empty_mgr


def _plant_manager_inversion(incumbents: pd.DataFrame) -> tuple[str, str]:
    """Planted problem 4: one M3 in Embedded Software paid below their highest-paid same-geo
    report. M3's normal ceiling is L4 (see _assign_managers); this is the single deliberate
    exception where an L5 is moved under an M3, with pay set so the manager is clearly
    out-earned. Comp-policy call: inversion is a within-geo, within-currency check -- an India
    M3 and a US L4 are different labor markets, so the planted pair must share a geo too, or
    it wouldn't even register as an inversion under the new rule. Gap targets 5-8% -- a $1,132
    (0.6%) gap tried here first is too subtle to read in a demo.
    """
    GAP_CHOICES = (0.08, 0.075, 0.07, 0.065, 0.06, 0.055, 0.05)

    emb_m3 = incumbents[(incumbents.level_code == "M3") & incumbents.job_id.str.startswith("EMB-MGR")].sort_values(
        "employee_id"
    )
    emb_l5 = incumbents[(incumbents.level_code == "L5") & incumbents.job_id.str.startswith("EMB-")]

    # Try every EMB M3 in a deterministic order, and for each, every same-geo EMB L5 candidate
    # highest-paid first, and for each of those the widest gap (8% down to 5%) that also clears
    # the M3's other same-geo reports -- so the planted case doesn't incidentally create a
    # second, untracked inversion. EMB only has one M3 job_id company-wide, so with just a
    # couple of EMB M3 incumbents this clean placement can be infeasible; fall back to whichever
    # (M3, L5, gap) gives the widest gap regardless of the M3's other reports; anything that
    # spills over becomes a natural inversion, which is an accepted outcome as of this policy.
    def find(require_clears_floor: bool):
        for _, candidate_m3 in emb_m3.iterrows():
            same_geo_l5 = emb_l5[emb_l5.geo_code == candidate_m3.geo_code].sort_values(
                "base_salary", ascending=False
            )
            if same_geo_l5.empty:
                continue
            other_reports = incumbents[
                (incumbents.manager_id == candidate_m3.employee_id)
                & (incumbents.geo_code == candidate_m3.geo_code)
            ]
            floor = other_reports.base_salary.max() if not other_reports.empty else 0
            for _, candidate_l5 in same_geo_l5.iterrows():
                for gap in GAP_CHOICES:
                    candidate_pay = candidate_l5.base_salary / (1 + gap)
                    if not require_clears_floor or candidate_pay > floor:
                        return candidate_m3, candidate_l5, candidate_pay
        return None

    found = find(require_clears_floor=True) or find(require_clears_floor=False)
    assert found is not None, "no EMB M3/L5 same-geo pairing at all -- cannot plant the inversion"
    target_m3, target_l5, m3_pay = found

    incumbents.loc[incumbents.employee_id == target_l5.employee_id, "manager_id"] = target_m3.employee_id
    incumbents.loc[incumbents.employee_id == target_m3.employee_id, "base_salary"] = round(m3_pay, 2)
    return target_m3.employee_id, target_l5.employee_id


def _resolve_unintended_inversions(incumbents: pd.DataFrame, structure_map: pd.DataFrame) -> None:
    """Bump any manager's pay above its highest-paid same-geo direct report. Cross-geo reports
    are never evaluated -- comparing pay across labor markets is a category error, not a
    lenient check. Same geo implies same currency, so raw pay is directly comparable once
    scoped this way. Runs low-to-high through the manager levels so a raise at M3 can cascade
    into a needed raise at M4, and so on up. Must run before _plant_manager_inversion -- it has
    no notion of a deliberately-planted case, so if it ran after, it would try to "fix" that one
    too.

    The bump is capped at the manager's own range max, and never lets compa-ratio exceed 1.28
    for anyone -- whichever cap is tighter. If that cap isn't enough to clear the report's pay,
    the manager is raised as far as the cap allows and the inversion is left in place: that's
    legitimate compression, not something to paper over by breaking the manager's own range.
    """
    for level in MANAGER_LEVELS:
        for emp_id in incumbents.loc[incumbents.level_code == level, "employee_id"]:
            mgr_row = incumbents.loc[incumbents.employee_id == emp_id].iloc[0]
            reports = incumbents[(incumbents.manager_id == emp_id) & (incumbents.geo_code == mgr_row.geo_code)]
            if reports.empty:
                continue
            max_report_pay = reports.base_salary.max()
            if max_report_pay >= mgr_row.base_salary:
                structure = structure_map.loc[(mgr_row.geo_code, level, mgr_row._family_group)]
                cap = min(structure.range_max, 1.28 * structure.range_mid)
                new_pay = min(max_report_pay * 1.03, cap)
                if new_pay > mgr_row.base_salary:
                    incumbents.loc[incumbents.employee_id == emp_id, "base_salary"] = round(new_pay, 2)


def _plant_red_circled_cases(incumbents: pd.DataFrame, structure_map: pd.DataFrame, exclude: set[str]) -> None:
    """Plant 5 deliberate above-range-maximum cases, spread across geos and levels: people
    down-leveled in a past restructure whose pay was protected rather than cut to the new
    level's range. Each lands at 1.05-1.15x their range max -- red-circling is a real state
    the negotiation depends on, and until now nothing in the population ever triggered it.
    """
    incumbents["red_circle_reason"] = None

    combos = incumbents[~incumbents.employee_id.isin(exclude)][["geo_code", "level_code"]].drop_duplicates()
    combos = combos.sort_values(["geo_code", "level_code"]).reset_index(drop=True)
    step = max(1, len(combos) // 5)
    picks = combos.iloc[::step].head(5)

    ratios = [1.05, 1.08, 1.10, 1.12, 1.15]
    for (_, combo), ratio in zip(picks.iterrows(), ratios):
        candidates = incumbents[
            (incumbents.geo_code == combo.geo_code)
            & (incumbents.level_code == combo.level_code)
            & (~incumbents.employee_id.isin(exclude))
        ].sort_values("employee_id")
        target = candidates.iloc[0]
        structure = structure_map.loc[(combo.geo_code, combo.level_code, target._family_group)]
        new_pay = round(structure.range_max * ratio, 2)
        incumbents.loc[incumbents.employee_id == target.employee_id, "base_salary"] = new_pay
        incumbents.loc[incumbents.employee_id == target.employee_id, "red_circle_reason"] = (
            f"Pay retained above the {combo.level_code} range following a prior-cycle restructure"
        )


def _verify_org_tree(incumbents: pd.DataFrame) -> None:
    non_root = incumbents[incumbents.level_code != "M7"]
    assert non_root.manager_id.notna().all(), "every non-M7 employee must have exactly one manager"

    manager_of = dict(zip(incumbents.employee_id, incumbents.manager_id))
    for emp_id in incumbents.employee_id:
        seen = set()
        current = emp_id
        for _ in range(len(MANAGER_LEVELS) + 2):
            current = manager_of.get(current)
            if pd.isna(current):
                break
            assert current not in seen, f"cycle detected in reporting chain starting at {emp_id}"
            seen.add(current)
        else:
            raise AssertionError(f"reporting chain from {emp_id} does not resolve to a root within expected depth")


def _print_org_audit(incumbents: pd.DataFrame, planted_pair: tuple[str, str]) -> None:
    mgrs = incumbents[incumbents.level_code.str.startswith("M")].set_index("employee_id")
    reports = incumbents[incumbents.manager_id.notna()]
    planted_manager, planted_report = planted_pair

    natural = []  # (manager_id, report_id, gap $, gap %)
    cross_geo = 0
    for emp_id, row in reports.iterrows():
        mgr = mgrs.loc[row.manager_id]
        if mgr.geo_code != row.geo_code:
            cross_geo += 1
            continue  # different labor market -- never evaluated for inversion
        if row.base_salary > mgr.base_salary:
            if row.employee_id == planted_report and row.manager_id == planted_manager:
                continue  # counted separately as the planted case
            gap = row.base_salary - mgr.base_salary
            natural.append((row.manager_id, row.employee_id, gap, gap / mgr.base_salary))

    planted_mgr = incumbents.loc[incumbents.employee_id == planted_manager].iloc[0]
    planted_rep = incumbents.loc[incumbents.employee_id == planted_report].iloc[0]
    planted_gap = planted_rep.base_salary - planted_mgr.base_salary

    print("\n--- org chart audit ---")
    print(f"planted inversions: 1 -> {planted_pair}  gap {planted_gap:,.2f} ({planted_gap / planted_mgr.base_salary:.1%})")
    print(f"natural inversions (compression left by the range-max/1.28 cap): {len(natural)}")
    for mgr_id, rep_id, gap, gap_pct in natural:
        print(f"  {mgr_id} < {rep_id}: gap {gap:,.2f} ({gap_pct:.1%})")
    print(f"total inversions: {1 + len(natural)}")
    print(f"cross-geo reporting relationships: {cross_geo}/{len(reports)} ({cross_geo / len(reports):.1%})")

    org_size: dict[str, int] = {}
    for level in MANAGER_LEVELS:
        for emp_id in incumbents.loc[incumbents.level_code == level, "employee_id"]:
            direct = incumbents[incumbents.manager_id == emp_id]
            org_size[emp_id] = sum(1 + org_size.get(r, 0) for r in direct.employee_id)

    zero_report_managers = 0
    for level in MANAGER_LEVELS:
        sizes = [org_size[e] for e in incumbents.loc[incumbents.level_code == level, "employee_id"]]
        n_zero = sum(1 for s in sizes if s == 0)
        zero_report_managers += n_zero
        print(f"  {level}: n={len(sizes):>3}  span min={min(sizes):>3} mean={sum(sizes)/len(sizes):>6.1f} max={max(sizes):>3}")
    print(f"managers with zero direct reports: {zero_report_managers}")

    roots = incumbents[incumbents.manager_id.isna()].employee_id.tolist()
    print(f"tree root(s): {roots} -> {'single root confirmed' if len(roots) == 1 else 'MULTIPLE ROOTS -- BUG'}")


def _print_compa_audit(
    incumbents: pd.DataFrame, range_mid: list[float], range_min: list[float], range_max: list[float]
) -> None:
    compa_ratio = incumbents.base_salary.to_numpy() / np.array(range_mid)
    below_min = sum(b < m for b, m in zip(incumbents.base_salary, range_min))
    above_max = sum(b > m for b, m in zip(incumbents.base_salary, range_max))

    print("\n--- compa-ratio audit ---")
    print(
        f"compa-ratio  min={compa_ratio.min():.3f}  p25={np.percentile(compa_ratio, 25):.3f}  "
        f"median={np.median(compa_ratio):.3f}  p75={np.percentile(compa_ratio, 75):.3f}  max={compa_ratio.max():.3f}"
    )
    print(f"below range minimum: {below_min}")
    print(f"above range maximum: {above_max}")
    print(f"red-circled: {int(incumbents.is_red_circled.sum())}")


# Nyx's own five-level ladder (docs/nyx_level_framework.md) -- MTS I through Distinguished
# MTS, no manager track, Fellow as an honorific outside the ladder. This replaces an earlier
# version of this roster that derived Nyx titles directly from Meridian's own L-codes
# (effectively "MTS N = L N"), which baked the crosswalk answer into the input data instead
# of requiring it to be evidence-derived -- exactly what a genuinely incompatible source
# framework should not do.
#
# Families: Digital Design (RTL Design, Microarchitecture) and Analog & Mixed-Signal
# (Analog Design, RF) exist in Meridian's own catalog -- _verify_nyx_family_coverage checks
# this at generation time. Photonics does not, and is checked to confirm it doesn't
# (planted problem 10, data_model_spec.md section 5). Two rows carry a functional manager
# label despite Nyx having no manager track (docs/nyx_level_framework.md section 5); one row
# is the Fellow honorific (section 6, planted problem 11).
#
# Role summaries generated once with Claude from each row's (family, sub_family, MTS level)
# -- see prompts.md -- then committed here and never regenerated, same policy as
# survey_jobs descriptions (data_model_spec.md section 4). Deliberately spread within each
# MTS level so some rows read toward the lower plausible Meridian level and some toward the
# higher one -- that ambiguity, not sparseness, is the point of this pass (an earlier pass
# already covers deliberately-too-vague-to-level rows for the fan-out demo).
NYX_ROSTER = [
    # (family, sub_family, title, pay_level, role_summary)
    ("Digital Design", "RTL Design", "MTS I - RTL Design", "MTS I",
     "Joined as a new-grad RTL designer on the NX-100 SerDes PHY controller block; wrote and verified the CDC synchronizer and register-map RTL under supervision, carrying it through 1 tapeout (NX-100 A0). Individual contributor, no direct reports; scope limited to a single sub-block owned start to finish."),
    ("Digital Design", "RTL Design", "MTS II - RTL Design", "MTS II",
     "Owns the register-transfer implementation of the NX-210 interconnect fabric end to end, including arbitration and clock-domain-crossing logic, and has carried it through 2 tapeouts without escalation. Increasingly the person other RTL engineers check designs with before sign-off, though this hasn't been made official."),
    ("Digital Design", "RTL Design", "Sr MTS - RTL Design", "Senior MTS",
     "Solid RTL engineer on the NX-300 team, has taken primary responsibility for the descriptor-fetch pipeline this generation. Coordinates with two other engineers on integration but hasn't yet had a tapeout where the block was entirely his own from spec through sign-off."),
    ("Digital Design", "RTL Design", "Senior MTS, Engineering Manager", "Senior MTS",
     "Engineering Manager for a 4-person RTL sub-team on the NX-400 digital front-end, responsible for sprint planning, code review sign-off, and performance reviews for the group. Continues to personally own the clock/reset architecture RTL for the block; no budget authority beyond headcount requisitions."),
    ("Digital Design", "RTL Design", "Principal MTS, RTL Design", "Principal MTS",
     "Owns the clock-tree synthesis and skew-budgeting methodology used across every NX-400 and NX-500 physical design handoff, a role that started as one person's personal scripts and is now the mandatory sign-off gate company-wide for CTS closure. Has never worked outside this one methodology area, and has no patents or external publications."),
    ("Digital Design", "Microarchitecture", "MTS I - Microarchitecture", "MTS I",
     "First full year on the team, built and owns the branch predictor sizing model for the NX-500 core, which directly set the BTB capacity used in the final tapeout. Review from senior architects was light-touch this cycle compared to the previous new-grad's onboarding."),
    ("Digital Design", "Microarchitecture", "MTS 2 - Microarchitecture", "MTS II",
     "Strong contributor to the microarchitecture team, has been involved in cache coherence modeling discussions for the current core project."),
    ("Digital Design", "Microarchitecture", "Senior Member of Technical Staff - Microarchitecture", "Senior MTS",
     "Owns the out-of-order load-store microarchitecture for the NX-400 core family across 2 tapeouts, including a new memory disambiguation scheme adopted for NX-410 after measuring an 8% IPC gain in simulation. Other engineers on adjacent blocks now route their interface questions through her rather than the architecture lead."),
    ("Digital Design", "Microarchitecture", "Principal MTS - Microarchitecture", "Principal MTS",
     "Principal-level title carried since a reorg two years ago; day to day the work looks like ordinary core-team microarchitecture spec work on the current NX-500 generation, shared with two peers at the Senior MTS level doing comparable scope."),
    ("Digital Design", "Microarchitecture", "Principal MTS, Engineering Manager", "Principal MTS",
     "Manages the 7-person microarchitecture team for the NX-500/NX-600 core roadmap, with hire/fire authority and a discretionary tooling budget of roughly $150K/year; also holds final sign-off on performance model gates before RTL freeze. Delegated most hands-on modeling work to the team two tapeouts ago."),
    ("Digital Design", "Microarchitecture", "Distinguished MTS - Microarchitecture", "Distinguished MTS",
     "Company-wide final authority on core performance/power/area tradeoffs across the entire NX-400 through NX-600 roadmap; every core team's performance model gets his sign-off before tapeout. Has never published, patented, or presented outside Nyx -- the reputation is entirely internal, built over eight years in the same role."),
    ("Analog & Mixed-Signal", "Analog Design", "MTS I - Analog Design", "MTS I",
     "Designed the bandgap reference and LDO regulator circuits for the NX-100 power management IC, taking the block through 1 tapeout with first-pass silicon success. Individual contributor scope, single analog block owned end to end."),
    ("Analog & Mixed-Signal", "Analog Design", "MTS II - Analog Design", "MTS II",
     "Strong analog designer, has been picking up more responsibility on the PMIC team including some layout review."),
    ("Analog & Mixed-Signal", "Analog Design", "Sr. MTS - Analog Design", "Senior MTS",
     "Owned the high-speed SerDes transmitter driver and equalizer circuits for the NX-210/NX-220 PHYs across 3 tapeouts, closing timing and jitter to a 25Gbps target with margin verified in silicon. Increasingly asked to review other engineers' high-speed IO work before tapeout sign-off, informally."),
    ("Analog & Mixed-Signal", "Analog Design", "Principal MTS - Analog Design", "Principal MTS",
     "Principal analog architect for all PLL and clock generation IP reused across the NX-400, NX-500 and NX-600 product lines (6 tapeouts total), owning the phase-noise budget methodology adopted company-wide. Directly mentors 4 analog designers and holds final sign-off authority on PLL closure for every tapeout in that set."),
    ("Analog & Mixed-Signal", "Analog Design", "Distinguished MTS - Analog Design", "Distinguished MTS",
     "Senior analog design leader with 11 years at Nyx, regarded across the company as the person to bring in when a tapeout's analog closure is stuck; has personally unblocked sign-off on 6 of the last 8 difficult closure escalations company-wide. No patents or publications -- has never had time to write anything up, by his own account."),
    ("Analog & Mixed-Signal", "RF", "MTS II - RF", "MTS II",
     "Valued contributor to the RF design team."),
    ("Analog & Mixed-Signal", "RF", "Senior MTS - RF", "Senior MTS",
     "Owns the RF transceiver front-end for the NX-220 PHY, taking the LNA and mixer design through 2 tapeouts with the target noise figure met on first silicon. Consulted by the layout and test teams on RF-specific design-for-test tradeoffs, but has not yet led a design independent of the senior architect's floor plan."),
    ("Analog & Mixed-Signal", "RF", "Distinguished MTS, RF", "Distinguished MTS",
     "Distinguished MTS title held for three years; current work is leading RF front-end design for the NX-600 PHY, a single product's RF subsystem, coordinating with two other RF engineers. Hasn't taken on anything spanning multiple product lines since the NX-400 generation."),
    ("Analog & Mixed-Signal", "RF", "Fellow - RF", "Distinguished MTS",
     "Recognized externally for pioneering work on high-speed SerDes equalization, having led design of an adaptive DFE architecture presented at two major industry conferences and covered in a trade-press writeup on Nyx's NX-220 PHY. Holds 3 patents in equalizer architecture, and is Nyx's go-to authority on link-budget modeling for every high-speed interface program company-wide."),
    ("Photonics", "Photonics", "MTS I - Photonics", "MTS I",
     "Joined the Photonics group nine months ago straight out of a PhD program; running characterization sweeps on grating-coupler test structures for the co-packaged optics program under the group lead's direction. First assignment, closely supervised."),
    ("Photonics", "Photonics", "MTS II - Photonics", "MTS II",
     "Owns the waveguide loss characterization flow for the co-packaged optics program end to end, including the test-structure design and the measurement methodology now used by the rest of the group. Works independently within the group's current experimental scope."),
    ("Photonics", "Photonics", "Senior MTS - Photonics", "Senior MTS",
     "Leads the optical-electrical co-design work between the photonics group and the SerDes team for the co-packaged optics program, the first cross-group technical integration the Photonics team has attempted. Makes the calls on link-budget tradeoffs without needing sign-off from the group lead."),
    ("Photonics", "Photonics", "Principal MTS - Photonics", "Principal MTS",
     "The most senior technical contributor in the Photonics group after its founder; owns the overall optical link architecture for the co-packaged optics program and has final say on every design tradeoff the group makes. Two more junior photonics engineers route technical questions through her."),
    ("Photonics", "Photonics", "Distinguished MTS - Photonics", "Distinguished MTS",
     "Founded and leads Nyx's Photonics group; every technical decision the group has made in its two years of existence has gone through her. Came from a research background before Nyx and has no engineering track record here beyond this one program, which itself has not yet shipped in a product."),
]

# $/year, Nyx's own internal pay banding -- independent of any Meridian level or lookup, so
# the crosswalk answer is never accidentally encoded into the census's own pay data (the
# earlier version of this roster derived pay from a Meridian-level market lookup, which
# leaked the "correct" mapping into an input the crosswalk is supposed to determine).
NYX_LEVEL_BASE_USD = {
    "MTS I": 105_000, "MTS II": 128_000, "Senior MTS": 158_000,
    "Principal MTS": 195_000, "Distinguished MTS": 240_000,
}

# Blank Curr and a bare "San Jose" location string are both deliberate mess
# (data_model_spec.md section 6); indices are arbitrary, not tied to any planted level story.
# _BARE_LOCATION_INDEX must land on an index%3==0 row (the San Jose/USD slot in the location
# cycle below) -- otherwise the override fights the cycle and produces a San Jose location
# paired with a non-USD currency, which isn't the intended mess.
_BLANK_CURRENCY_INDICES = {7, 17}
_BARE_LOCATION_INDEX = 6


def _verify_nyx_family_coverage(job_catalog: pd.DataFrame) -> None:
    catalog_sub_families = set(job_catalog.sub_family)
    nyx_families = {sub_family for _, sub_family, *_ in NYX_ROSTER}
    meridian_equivalent = nyx_families - {"Photonics"}
    missing = meridian_equivalent - catalog_sub_families
    assert not missing, f"Nyx sub-families expected to have a Meridian equivalent are missing from job_catalog: {missing}"
    assert "Photonics" not in catalog_sub_families, (
        "Photonics must have no Meridian equivalent (planted problem 10) -- "
        "it just appeared in job_catalog, which would silently defeat that story."
    )


def build_nyx_census(job_catalog: pd.DataFrame, fx_rates: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    _verify_nyx_family_coverage(job_catalog)
    assert len(NYX_ROSTER) == SUBMISSION_NYX

    locations = [("San Jose, CA", "USD"), ("Bangalore", "INR"), ("Eindhoven", "EUR")]
    rows = []
    for index, (family, sub_family, title, pay_level, role_summary) in enumerate(NYX_ROSTER):
        location, currency = locations[index % len(locations)]
        if index == _BARE_LOCATION_INDEX:
            location = "San Jose"
        # Nyx pays each level the same USD-equivalent figure everywhere (NYX_LEVEL_BASE_USD
        # is deliberately independent of Meridian's geo differentials -- see that constant's
        # own comment); the local-currency figure is a real dated FX conversion at the deal
        # reference date, not an approximate multiplier standing in for one. Previously this
        # multiplied by the same 0.34/0.93 numbers salary_structures uses as a market-level
        # differential, which produced INR/EUR figures nowhere near real currency scale (see
        # learnings.md) -- convert_currency against fx_rates replaces that entirely.
        base_usd = NYX_LEVEL_BASE_USD[pay_level] * rng.uniform(0.92, 1.08)
        base = round(
            convert_currency(base_usd, "USD", currency, REFERENCE_DATE, fx_rates)["converted_amount"], 2
        )
        bonus = f"{15 + index % 5}%" if index % 2 == 0 else 15 + index % 5
        start = ["3/14/22", "2021-08-02", "15/06/23"][index % 3]
        rows.append({
            "Emp ID": f"NYX-{index + 1:03d}", "Job Title": title,
            "Dept": family, "Location": location,
            "Curr": "" if index in _BLANK_CURRENCY_INDICES else currency,
            "Base": base, "Bonus": bonus, "Unvested Options": round(base * rng.uniform(0.2, 1.2), 2),
            "Start": start, "Role Summary": role_summary,
        })
    return pd.DataFrame(rows)


def build_acquisition_context() -> pd.DataFrame:
    return pd.DataFrame([{
        "source_headcount": 104, "source_stage": "late-stage private",
        "source_type": "whole company", "parent_headcount": 300,
        "org_depth": 4, "platform_dependency": "low",
    }])


def attach_survey_matches(
    job_catalog: pd.DataFrame, selected: pd.DataFrame, survey_jobs: pd.DataFrame
) -> pd.DataFrame:
    catalog = job_catalog.copy()
    survey_map = dict(zip(selected.job_id, survey_jobs.survey_code))
    catalog["survey_code_primary"] = catalog.job_id.map(survey_map)
    catalog["survey_code_secondary"] = None
    catalog["blend_weight_primary"] = catalog.survey_code_primary.notna().astype(float)
    return catalog


# ---------------------------------------------------------------- main

def main(seed: int = SEED) -> None:
    rng = np.random.default_rng(seed)
    OUT.mkdir(parents=True, exist_ok=True)

    geo_locations = build_geo_locations()
    job_catalog = build_job_catalog()
    selected = _stratified_survey_sample(job_catalog, SUBMISSION_SURVEY_JOBS)
    survey_jobs, survey_meta = build_survey_jobs(selected)
    survey_data = build_survey_data(survey_jobs, survey_meta, geo_locations)
    salary_structures = build_salary_structures(geo_locations, survey_data, survey_meta)
    tables = {
        "level_definitions": build_level_definitions(),
        "geo_locations": geo_locations,
        "fx_rates": build_fx_rates(rng),
        "job_catalog": attach_survey_matches(job_catalog, selected, survey_jobs),
        "survey_jobs": survey_jobs,
        "survey_data": survey_data,
        "salary_structures": salary_structures,
        "incumbents": build_incumbents(job_catalog, geo_locations, salary_structures, rng),
        "acquisition_context": build_acquisition_context(),
    }
    nyx_census = build_nyx_census(job_catalog, tables["fx_rates"], rng)

    for name, frame in tables.items():
        frame.to_parquet(OUT / f"{name}.parquet", index=False)
        print(f"{name:22} {len(frame):>6} rows")
    nyx_census.to_excel(OUT / "nyx_census.xlsx", index=False)
    print(f"{'nyx_census.xlsx':22} {len(nyx_census):>6} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    main(parser.parse_args().seed)
